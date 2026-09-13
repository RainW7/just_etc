"""Simulate one bundled template and write a Redrock-compatible JUST FITS file.

The input template is shifted to the requested redshift, normalized to an AB
magnitude, and passed through the JUST ETC noise model.  The output follows the
multi-extension layout used by ``archive/process_bgs_10k_redrock.py``.

Example
-------
python examples/simulate_mock_spectrum.py \
    --template starburst/sb4_kinney_fuv_001.fits \
    --redshift 0.2 --magnitude 20.5 \
    --exposure-time 900 --exposures 4 \
    --output output/just_mock_spectrum.fits
"""

from argparse import ArgumentParser
from pathlib import Path
import warnings

import numpy as np
from astropy.io import fits
from astropy.table import Table

from just_etc import JUSTExposureTimeCalculator, load_template, normalize_to_mag


ARM_BANDS = ("b", "r", "z")
FLUX_UNIT = "10**-17 erg/(s cm**2 Angstrom)"
IVAR_UNIT = "10**34 (s**2 cm**4 Angstrom**2)/erg**2"


def _flux_and_ivar(arm_result):
    """Convert a mock arm from cgs FLAM to Redrock flux and inverse variance."""
    intrinsic = np.asarray(arm_result["flux_intrinsic"], dtype=np.float64)
    mock_flux = np.asarray(arm_result["flux_mock"], dtype=np.float64)
    signal_e = np.asarray(arm_result["signal_e"], dtype=np.float64)
    noise_e = np.asarray(arm_result["noise_e"], dtype=np.float64)

    # The electron-to-FLAM response is signal_e / intrinsic.  Applying its
    # inverse to noise_e gives sigma(F_lambda), including source, sky, detector,
    # and systematic terms already evaluated by the ETC.
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

    bad = ~np.isfinite(flux) | ~np.isfinite(ivar)
    flux[~np.isfinite(flux)] = 0.0
    ivar[bad] = 0.0
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


def simulate_mock_spectrum(
    template="starburst/sb4_kinney_fuv_001.fits",
    redshift=0.2,
    magnitude=20.5,
    band="r",
    exposure_time=900.0,
    exposures=4,
    seeing=0.8,
    zenith_angle=45.0,
    effective_radius=0.0,
    ebv=0.03,
    seed=42,
    target_id=10001,
    ra=150.0,
    dec=2.0,
    calc_mode="fast",
    output=Path("output/just_mock_spectrum.fits"),
):
    """Generate one noisy JUST spectrum in a Redrock-compatible FITS layout.

    Wavelengths are in Angstrom, input and simulated spectra use FLAM in
    erg s^-1 cm^-2 Angstrom^-1, and output fluxes use the standard Redrock
    scale of 10^-17 erg s^-1 cm^-2 Angstrom^-1.
    """
    if redshift <= -1.0:
        raise ValueError("redshift must be greater than -1")
    if exposure_time <= 0.0:
        raise ValueError("exposure_time must be positive")
    if exposures < 1:
        raise ValueError("exposures must be at least 1")
    if seeing <= 0.0:
        raise ValueError("seeing must be positive")
    if effective_radius < 0.0:
        raise ValueError("effective_radius cannot be negative")

    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    wave_rest, flux_rest = load_template(template)
    if not np.all(np.isfinite(wave_rest)) or not np.all(np.diff(wave_rest) > 0.0):
        raise ValueError("template wavelength grid must be finite and strictly increasing")
    if not np.all(np.isfinite(flux_rest)):
        raise ValueError("template flux contains non-finite values")
    if np.any(flux_rest < 0.0):
        warnings.warn("negative template flux values were clipped to zero", RuntimeWarning)
        flux_rest = np.maximum(flux_rest, 0.0)

    wave_observed = wave_rest * (1.0 + redshift)
    # F_lambda transforms with 1/(1+z).  The later AB normalization sets the
    # absolute scale, while this factor keeps the redshift operation explicit.
    flux_observed = flux_rest / (1.0 + redshift)
    flux_normalized, scale_factor = normalize_to_mag(
        wave_observed, flux_observed, target_mag=magnitude, band=band
    )

    calculator = JUSTExposureTimeCalculator(calc_mode=calc_mode)
    calculator.set_obs_conditions(
        seeing_fwhm_800=seeing,
        zenith_angle=zenith_angle,
        r_eff=effective_radius,
        ebv=ebv,
    )
    mock_observation = calculator.simulate_mock_observation(
        wave_observed,
        flux_normalized,
        t_exp=exposure_time,
        n_exp=exposures,
        seed=seed,
    )
    if len(mock_observation) != len(ARM_BANDS):
        raise RuntimeError(
            f"expected {len(ARM_BANDS)} spectrograph arms, got {len(mock_observation)}"
        )

    primary = fits.PrimaryHDU()
    primary.header["ORIGIN"] = "JUST ETC"
    primary.header["CONTENT"] = "Redrock-compatible simulated spectra"
    primary.header["TEMPLATE"] = str(template)
    primary.header["REDSHIFT"] = (redshift, "Template redshift")
    primary.header["MAG"] = (magnitude, "Target AB magnitude")
    primary.header["MAGBAND"] = (band, "Normalization band")
    primary.header["EXPTIME"] = (exposure_time, "Seconds per exposure")
    primary.header["NEXP"] = (exposures, "Number of co-added exposures")
    primary.header["SEEING"] = (seeing, "FWHM at 800 nm [arcsec]")
    primary.header["ZENITH"] = (zenith_angle, "Zenith angle [deg]")
    primary.header["REFF"] = (effective_radius, "Effective radius [arcsec]")
    primary.header["EBV"] = (ebv, "Galactic E(B-V)")
    primary.header["RNGSEED"] = (seed, "Random-noise seed")
    primary.header["TARGSCL"] = (scale_factor, "Template flux scale factor")

    hdus = [primary, fits.BinTableHDU(_build_fibermap(target_id, ra, dec), name="FIBERMAP")]
    score_columns = {"TARGETID": np.array([target_id], dtype=np.int64)}

    for expected_arm, arm_result in enumerate(mock_observation):
        arm = int(arm_result["arm"])
        if arm != expected_arm:
            raise RuntimeError(f"unexpected spectrograph arm order: {arm} at index {expected_arm}")
        band_name = ARM_BANDS[arm].upper()
        wavelength = np.asarray(arm_result["wave_aa"], dtype=np.float64)
        flux, ivar = _flux_and_ivar(arm_result)

        wave_hdu = fits.ImageHDU(wavelength, name=f"{band_name}_WAVELENGTH")
        wave_hdu.header["BUNIT"] = "Angstrom"
        flux_hdu = fits.ImageHDU(flux[np.newaxis, :], name=f"{band_name}_FLUX")
        flux_hdu.header["BUNIT"] = FLUX_UNIT
        ivar_hdu = fits.ImageHDU(ivar[np.newaxis, :], name=f"{band_name}_IVAR")
        ivar_hdu.header["BUNIT"] = IVAR_UNIT

        # A one-diagonal identity matrix is the same convention used by the
        # archived batch example: the ETC spectrum is already sampled on the
        # JUST detector grid and no additional line-spread function is applied.
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
        snr = np.asarray(arm_result["snr"], dtype=np.float64)
        score_columns[f"SNR_{band_name}"] = np.array(
            [np.nanmedian(snr[valid]) if np.any(valid) else 0.0], dtype=np.float32
        )

    hdus.append(fits.BinTableHDU(Table(score_columns), name="SCORES"))
    fits.HDUList(hdus).writeto(output, overwrite=True, checksum=True)

    print(f"Saved Redrock-compatible JUST mock spectrum to {output.resolve()}")
    print(
        f"Template: {template} | z={redshift:g} | {band}={magnitude:g} AB | "
        f"{exposures} x {exposure_time:g} s"
    )
    return output


def _parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="starburst/sb4_kinney_fuv_001.fits")
    parser.add_argument("--redshift", type=float, default=0.2)
    parser.add_argument("--magnitude", type=float, default=20.5)
    parser.add_argument("--band", default="r")
    parser.add_argument("--exposure-time", type=float, default=900.0)
    parser.add_argument("--exposures", type=int, default=4)
    parser.add_argument("--seeing", type=float, default=0.8)
    parser.add_argument("--zenith-angle", type=float, default=45.0)
    parser.add_argument("--effective-radius", type=float, default=0.0)
    parser.add_argument("--ebv", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-id", type=int, default=10001)
    parser.add_argument("--ra", type=float, default=150.0)
    parser.add_argument("--dec", type=float, default=2.0)
    parser.add_argument("--calc-mode", choices=("fast", "balanced", "accurate"), default="fast")
    parser.add_argument("--output", type=Path, default=Path("output/just_mock_spectrum.fits"))
    return parser.parse_args()


if __name__ == "__main__":
    simulate_mock_spectrum(**vars(_parse_args()))
