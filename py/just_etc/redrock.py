"""Helpers for writing JUST mock observations in a Redrock FITS layout."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table


ARM_BANDS = ("b", "r", "z")
FLUX_UNIT = "10**-17 erg/(s cm**2 Angstrom)"
IVAR_UNIT = "10**34 (s**2 cm**4 Angstrom**2)/erg**2"


def _flux_and_ivar(arm_result):
    """Convert one ETC mock arm from physical FLAM to Redrock units."""
    intrinsic = np.asarray(arm_result["flux_intrinsic"], dtype=np.float64)
    mock_flux = np.asarray(arm_result["flux_mock"], dtype=np.float64)
    signal_e = np.asarray(arm_result["signal_e"], dtype=np.float64)
    noise_e = np.asarray(arm_result["noise_e"], dtype=np.float64)

    if not (intrinsic.shape == mock_flux.shape == signal_e.shape == noise_e.shape):
        raise ValueError("ETC arm flux, signal, and noise arrays must have matching shapes")

    # signal_e / intrinsic is the electron response to physical FLAM.  Its
    # inverse converts the ETC electron-space noise back to sigma(F_lambda).
    sigma_flam = np.full_like(intrinsic, np.inf)
    valid = (
        np.isfinite(intrinsic)
        & np.isfinite(signal_e)
        & np.isfinite(noise_e)
        & (intrinsic > 0.0)
        & (signal_e > 0.0)
        & (noise_e > 0.0)
    )
    sigma_flam[valid] = noise_e[valid] * intrinsic[valid] / signal_e[valid]

    flux = mock_flux * 1.0e17
    sigma = sigma_flam * 1.0e17
    ivar = np.zeros_like(sigma)
    finite_sigma = np.isfinite(sigma) & (sigma > 0.0)
    ivar[finite_sigma] = 1.0 / sigma[finite_sigma] ** 2

    bad_flux = ~np.isfinite(flux)
    flux[bad_flux] = 0.0
    ivar[bad_flux | ~np.isfinite(ivar)] = 0.0
    return flux.astype(np.float32), ivar.astype(np.float32)


def _build_fibermap(target_id, ra, dec):
    """Return the minimal DESI-style target metadata used by ``rrdesi``."""
    return Table({
        "TARGETID": np.array([target_id], dtype=np.int64),
        "TARGET_RA": np.array([ra], dtype=np.float64),
        "TARGET_DEC": np.array([dec], dtype=np.float64),
        "RA": np.array([ra], dtype=np.float64),
        "DEC": np.array([dec], dtype=np.float64),
        "OBJTYPE": np.array(["TGT"], dtype="U3"),
        "DESI_TARGET": np.array([1], dtype=np.int64),
        "COADD_FIBERSTATUS": np.array([0], dtype=np.int32),
        "TILEID": np.array([0], dtype=np.int32),
        "PETAL_LOC": np.array([0], dtype=np.int16),
        "FIBER": np.array([0], dtype=np.int32),
        "SPECTROGRAPH": np.array([0], dtype=np.int16),
    })


def write_redrock_mock_spectrum(
    output,
    mock_observation,
    *,
    target_id=10001,
    ra=150.0,
    dec=2.0,
    primary_metadata=None,
):
    """Write one JUST ETC mock observation in a Redrock-compatible layout.

    Parameters
    ----------
    output : str or pathlib.Path
        Destination FITS filename.
    mock_observation : sequence of dict
        Three-arm result returned by
        :meth:`just_etc.JUSTExposureTimeCalculator.simulate_mock_observation`.
    target_id : int
        Unique Redrock target identifier.
    ra, dec : float
        ICRS sky coordinates in degrees.
    primary_metadata : mapping, optional
        Extra FITS primary-header entries. Values may be scalars or
        ``(value, comment)`` tuples accepted by Astropy.

    Returns
    -------
    pathlib.Path
        The output path.
    """
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    if len(mock_observation) != len(ARM_BANDS):
        raise ValueError(
            f"expected {len(ARM_BANDS)} spectrograph arms, got {len(mock_observation)}"
        )

    primary = fits.PrimaryHDU()
    primary.header["ORIGIN"] = "JUST ETC"
    primary.header["CONTENT"] = "Redrock-compatible simulated spectra"
    if primary_metadata is not None:
        for key, value in primary_metadata.items():
            primary.header[key] = value

    hdus = [
        primary,
        fits.BinTableHDU(_build_fibermap(target_id, ra, dec), name="FIBERMAP"),
    ]
    score_columns = {"TARGETID": np.array([target_id], dtype=np.int64)}

    for expected_arm, arm_result in enumerate(mock_observation):
        arm = int(arm_result["arm"])
        if arm != expected_arm:
            raise ValueError(
                f"unexpected spectrograph arm order: {arm} at index {expected_arm}"
            )

        band_name = ARM_BANDS[arm].upper()
        wavelength = np.asarray(arm_result["wave_aa"], dtype=np.float64)
        flux, ivar = _flux_and_ivar(arm_result)
        snr = np.asarray(arm_result["snr"], dtype=np.float64)
        if wavelength.ndim != 1 or wavelength.shape != flux.shape or snr.shape != flux.shape:
            raise ValueError(f"arm {arm} wavelength, flux, and SNR must be matching 1D arrays")
        if not np.all(np.isfinite(wavelength)) or not np.all(np.diff(wavelength) > 0.0):
            raise ValueError(f"arm {arm} wavelength grid must be finite and strictly increasing")

        wave_hdu = fits.ImageHDU(wavelength, name=f"{band_name}_WAVELENGTH")
        wave_hdu.header["BUNIT"] = "Angstrom"
        flux_hdu = fits.ImageHDU(flux[np.newaxis, :], name=f"{band_name}_FLUX")
        flux_hdu.header["BUNIT"] = FLUX_UNIT
        ivar_hdu = fits.ImageHDU(ivar[np.newaxis, :], name=f"{band_name}_IVAR")
        ivar_hdu.header["BUNIT"] = IVAR_UNIT

        # The ETC spectrum is already sampled on the detector grid.  A
        # one-diagonal identity matrix records that no additional LSF is being
        # applied here; replace this when a measured JUST resolution matrix is
        # available.
        resolution = np.ones((1, 1, wavelength.size), dtype=np.float32)
        resolution_hdu = fits.ImageHDU(resolution, name=f"{band_name}_RESOLUTION")
        hdus.extend((wave_hdu, flux_hdu, ivar_hdu, resolution_hdu))

        valid = ivar > 0.0
        score_columns[f"INTEG_RAW_FLUX_{band_name}"] = np.array(
            [np.mean(flux[valid]) if np.any(valid) else 0.0], dtype=np.float32
        )
        score_columns[f"MEDIAN_RAW_FLUX_{band_name}"] = np.array(
            [np.median(flux[valid]) if np.any(valid) else 0.0], dtype=np.float32
        )
        score_columns[f"SNR_{band_name}"] = np.array(
            [np.nanmedian(snr[valid]) if np.any(valid) else 0.0], dtype=np.float32
        )

    hdus.append(fits.BinTableHDU(Table(score_columns), name="SCORES"))
    fits.HDUList(hdus).writeto(output, overwrite=True, checksum=True)
    return output
