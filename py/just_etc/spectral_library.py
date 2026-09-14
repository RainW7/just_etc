"""Simulate spectral libraries and write Redrock-compatible FITS files."""

from pathlib import Path
import multiprocessing as mp
import os
import time

import numpy as np
from astropy.io import fits
from astropy.table import Table

from .just_etc_api import JUSTExposureTimeCalculator
from .redrock import _flux_and_ivar


_BANDS = ("b", "r", "z")
_FLUX_UNIT = "10**-17 erg/(s cm**2 Angstrom)"
_IVAR_UNIT = "10**34 (s**2 cm**4 Angstrom**2)/erg**2"


def _process_chunk(args):
    """Simulate one chunk of spectra in a process worker."""
    (
        chunk_index,
        wave_aa,
        flux_chunk,
        target_ids,
        t_exp,
        n_exp,
        seeing_fwhm,
        zenith_angle,
        calc_mode,
    ) = args

    etc = JUSTExposureTimeCalculator(calc_mode=calc_mode)
    etc.set_obs_conditions(
        seeing_fwhm_800=seeing_fwhm,
        zenith_angle=zenith_angle,
    )

    flux_by_band = {band: [] for band in _BANDS}
    ivar_by_band = {band: [] for band in _BANDS}
    resolution_by_band = {band: [] for band in _BANDS}
    wave_by_band = {}

    for flux_1e17, target_id in zip(flux_chunk, target_ids):
        flux_flam = np.nan_to_num(
            np.asarray(flux_1e17, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ) * 1.0e-17
        flux_flam = np.maximum(flux_flam, 0.0)

        mock_observation = etc.simulate_mock_observation(
            wave_aa,
            flux_flam,
            t_exp=t_exp,
            n_exp=n_exp,
            seed=int(target_id) % (2**31 - 1),
        )
        arm_indices = [int(result["arm"]) for result in mock_observation]
        if sorted(arm_indices) != list(range(len(_BANDS))):
            raise ValueError(f"expected one result for each JUST arm, got {arm_indices}")

        for arm_result in mock_observation:
            arm_index = int(arm_result["arm"])
            if arm_index < 0 or arm_index >= len(_BANDS):
                raise ValueError(f"unexpected spectrograph arm index: {arm_index}")
            band = _BANDS[arm_index]
            wave_arm = np.asarray(arm_result["wave_aa"], dtype=np.float64)
            flux_arm, ivar_arm = _flux_and_ivar(arm_result)
            if wave_arm.ndim != 1 or flux_arm.shape != wave_arm.shape:
                raise ValueError(
                    f"arm {arm_index} returned inconsistent wavelength and flux arrays"
                )
            if not np.all(np.isfinite(wave_arm)) or not np.all(np.diff(wave_arm) > 0.0):
                raise ValueError(f"arm {arm_index} wavelength grid is not finite and increasing")

            if band not in wave_by_band:
                wave_by_band[band] = wave_arm
            elif not np.array_equal(wave_by_band[band], wave_arm):
                raise ValueError(f"wavelength grid changed within arm {band.upper()}")

            flux_by_band[band].append(flux_arm)
            ivar_by_band[band].append(ivar_arm)
            # The ETC mock spectrum is sampled on the detector wavelength grid.
            # This identity kernel preserves the convention of the batch example;
            # it does not model the JUST line-spread function.
            resolution_by_band[band].append(
                np.ones((1, wave_arm.size), dtype=np.float32)
            )

    return (
        chunk_index,
        wave_by_band,
        flux_by_band,
        ivar_by_band,
        resolution_by_band,
    )


def _validate_target_ids(target_ids, n_spectra):
    if target_ids is None:
        return np.arange(300000001, 300000001 + n_spectra, dtype=np.int64)

    raw_ids = np.asarray(target_ids)
    if raw_ids.ndim != 1 or raw_ids.size != n_spectra:
        raise ValueError(f"target_ids must have shape ({n_spectra},)")
    if raw_ids.dtype.kind not in "iu":
        try:
            numeric_ids = np.asarray(raw_ids, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_ids must contain integer values") from exc
        if not np.all(np.isfinite(numeric_ids)) or not np.all(numeric_ids == np.floor(numeric_ids)):
            raise ValueError("target_ids must contain finite integer values")
    try:
        ids = raw_ids.astype(np.int64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("target_ids must be representable as int64") from exc
    if np.unique(ids).size != n_spectra:
        raise ValueError("target_ids must be unique")
    return ids


def _make_fibermap(target_ids):
    n_spectra = target_ids.size
    index = np.arange(n_spectra, dtype=np.int64)
    return Table({
        "TARGETID": target_ids,
        "RA": 150.0 + (index % 1000) * 0.001,
        "DEC": 2.0 + (index // 1000) * 0.001,
        "FIBER": (index % 4000).astype(np.int32),
        "SPECTROGRAPH": ((index // 500) % 8).astype(np.int16),
        "OBJTYPE": np.full(n_spectra, "TGT", dtype="U3"),
    })


def convet_to_redrock_format(
    wave,
    flux,
    output,
    t_exp=900.0,
    n_exp=4,
    seeing_fwhm=0.8,
    zenith_angle=45.0,
    nproc=None,
    target_ids=None,
    calc_mode="fast",
    verbose=False,
):
    """Simulate a spectral library with JUST and write Redrock-compatible FITS.

    Parameters
    ----------
    wave : array-like, shape (n_wave,)
        Strictly increasing observed-frame wavelength grid in Angstroms.
    flux : array-like, shape (n_spectra, n_wave) or (n_wave,)
        Input spectra in the Redrock scale of
        ``10^-17 erg s^-1 cm^-2 Angstrom^-1``. A one-dimensional spectrum is
        accepted and written as a one-target library. Non-finite input fluxes
        are set to zero and negative fluxes are clipped to zero, matching the
        behavior of the original batch example.
    output : str or pathlib.Path
        Output FITS path. An existing file at this path is overwritten.
    t_exp : float
        Exposure time per exposure in seconds.
    n_exp : int
        Number of co-added exposures.
    seeing_fwhm : float
        Seeing FWHM at 800 nm in arcseconds.
    zenith_angle : float
        Zenith angle in degrees.
    nproc : int or None
        Worker process count. ``None`` or a non-positive value uses the
        available CPU count, capped at the number of spectra. Use ``1`` for a
        serial run or when calling from an interactive notebook.
    target_ids : array-like, optional
        Unique integer TARGETID for each input spectrum. Defaults to IDs
        beginning at 300000001.
    calc_mode : {"fast", "balanced", "accurate"}
        ETC calculation mode.
    verbose : bool
        Print progress and output information.

    Returns
    -------
    pathlib.Path
        The output FITS path.

    Notes
    -----
    The public function name retains the spelling requested for the issue:
    ``convet_to_redrock_format``. The output resolution arrays use an identity
    kernel, as in the original batch example; this does not represent a
    measured or modeled JUST line-spread function.
    """
    wave_aa = np.asarray(wave, dtype=np.float64)
    flux_library = np.asarray(flux, dtype=np.float64)
    if wave_aa.ndim != 1 or wave_aa.size < 2:
        raise ValueError("wave must be a one-dimensional array with at least two samples")
    if not np.all(np.isfinite(wave_aa)) or not np.all(np.diff(wave_aa) > 0.0):
        raise ValueError("wave must be finite and strictly increasing")
    if np.any(wave_aa <= 0.0):
        raise ValueError("wave must contain positive wavelengths")
    if flux_library.ndim == 1:
        flux_library = flux_library[np.newaxis, :]
    if flux_library.ndim != 2 or flux_library.shape[1] != wave_aa.size:
        raise ValueError("flux must have shape (n_spectra, len(wave)) or (len(wave),)")
    if flux_library.shape[0] < 1:
        raise ValueError("flux must contain at least one spectrum")
    if not np.isfinite(t_exp) or t_exp <= 0.0:
        raise ValueError("t_exp must be finite and positive")
    if isinstance(n_exp, bool) or not isinstance(n_exp, (int, np.integer)) or n_exp < 1:
        raise ValueError("n_exp must be a positive integer")
    if not np.isfinite(seeing_fwhm) or seeing_fwhm <= 0.0:
        raise ValueError("seeing_fwhm must be finite and positive")
    if not np.isfinite(zenith_angle) or zenith_angle < 0.0:
        raise ValueError("zenith_angle must be finite and non-negative")

    n_spectra = flux_library.shape[0]
    target_ids = _validate_target_ids(target_ids, n_spectra)
    if nproc is None or nproc <= 0:
        nproc = max(1, os.cpu_count() or 1)
    elif isinstance(nproc, bool) or not isinstance(nproc, (int, np.integer)):
        raise ValueError("nproc must be an integer, None, or non-positive for automatic selection")
    nproc = min(int(nproc), n_spectra)
    if nproc < 1:
        raise ValueError("nproc must resolve to at least one worker")

    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    index_chunks = np.array_split(np.arange(n_spectra), nproc)
    tasks = [
        (
            chunk_index,
            wave_aa,
            flux_library[indices],
            target_ids[indices],
            float(t_exp),
            int(n_exp),
            float(seeing_fwhm),
            float(zenith_angle),
            calc_mode,
        )
        for chunk_index, indices in enumerate(index_chunks)
    ]

    if verbose:
        print(
            f"Simulating {n_spectra} spectra with JUST ETC "
            f"({n_exp} x {t_exp:g} s; {nproc} worker(s))"
        )

    chunk_results = {}
    if nproc == 1:
        chunk_results[0] = _process_chunk(tasks[0])
    else:
        with mp.get_context("spawn").Pool(processes=nproc) as pool:
            for result in pool.imap_unordered(_process_chunk, tasks):
                chunk_results[result[0]] = result
                if verbose:
                    print(
                        f"Completed {len(chunk_results)}/{len(tasks)} chunks",
                        end="\r",
                    )
        if verbose:
            print()

    wave_by_band = {}
    flux_by_band = {}
    ivar_by_band = {}
    resolution_by_band = {}
    for band in _BANDS:
        ordered_results = [chunk_results[index] for index in range(len(tasks))]
        wave_by_band[band] = ordered_results[0][1][band]
        flux_by_band[band] = np.concatenate(
            [np.asarray(result[2][band]) for result in ordered_results], axis=0
        )
        ivar_by_band[band] = np.concatenate(
            [np.asarray(result[3][band]) for result in ordered_results], axis=0
        )
        resolution_by_band[band] = np.concatenate(
            [np.asarray(result[4][band]) for result in ordered_results], axis=0
        )

    primary = fits.PrimaryHDU()
    primary.header["ORIGIN"] = "JUST ETC"
    primary.header["CONTENT"] = "Redrock-compatible simulated spectral library"
    primary.header["EXPTIME"] = (float(t_exp), "Seconds per exposure")
    primary.header["NEXP"] = (int(n_exp), "Number of co-added exposures")
    primary.header["SEEING"] = (float(seeing_fwhm), "FWHM at 800 nm [arcsec]")
    primary.header["ZENITH"] = (float(zenith_angle), "Zenith angle [deg]")
    hdus = [
        primary,
        fits.BinTableHDU(_make_fibermap(target_ids), name="FIBERMAP"),
    ]

    scores = Table()
    scores["TARGETID"] = target_ids
    for band in _BANDS:
        upper_band = band.upper()
        wave_hdu = fits.ImageHDU(wave_by_band[band], name=f"{upper_band}_WAVELENGTH")
        wave_hdu.header["BUNIT"] = "Angstrom"
        flux_hdu = fits.ImageHDU(
            flux_by_band[band].astype(np.float32), name=f"{upper_band}_FLUX"
        )
        flux_hdu.header["BUNIT"] = _FLUX_UNIT
        ivar_hdu = fits.ImageHDU(
            ivar_by_band[band].astype(np.float32), name=f"{upper_band}_IVAR"
        )
        ivar_hdu.header["BUNIT"] = _IVAR_UNIT
        resolution_hdu = fits.ImageHDU(
            resolution_by_band[band].astype(np.float32),
            name=f"{upper_band}_RESOLUTION",
        )
        hdus.extend((wave_hdu, flux_hdu, ivar_hdu, resolution_hdu))

        scores[f"INTEG_RAW_FLUX_{upper_band}"] = np.mean(
            flux_by_band[band], axis=1
        )
        scores[f"MEDIAN_RAW_FLUX_{upper_band}"] = np.median(
            flux_by_band[band], axis=1
        )
        scores[f"SNR_{upper_band}"] = np.mean(
            flux_by_band[band] * np.sqrt(np.maximum(0.0, ivar_by_band[band])),
            axis=1,
        )

    hdus.append(fits.BinTableHDU(scores, name="SCORES"))
    fits.HDUList(hdus).writeto(output, overwrite=True, checksum=True)

    if verbose:
        elapsed = time.time() - start_time
        size_mb = output.stat().st_size / (1024.0 * 1024.0)
        print(
            f"Saved {output.resolve()} ({size_mb:.2f} MB) in {elapsed:.1f} s "
            f"({n_spectra / elapsed:.1f} spectra/s)"
        )
    return output
