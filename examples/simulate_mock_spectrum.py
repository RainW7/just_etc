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

from just_etc import (
    JUSTExposureTimeCalculator,
    load_template,
    normalize_to_mag,
    write_redrock_mock_spectrum,
)


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
    write_redrock_mock_spectrum(
        output,
        mock_observation,
        target_id=target_id,
        ra=ra,
        dec=dec,
        primary_metadata={
            "TEMPLATE": str(template),
            "REDSHIFT": (redshift, "Template redshift"),
            "MAG": (magnitude, "Target AB magnitude"),
            "MAGBAND": (band, "Normalization band"),
            "EXPTIME": (exposure_time, "Seconds per exposure"),
            "NEXP": (exposures, "Number of co-added exposures"),
            "SEEING": (seeing, "FWHM at 800 nm [arcsec]"),
            "ZENITH": (zenith_angle, "Zenith angle [deg]"),
            "REFF": (effective_radius, "Effective radius [arcsec]"),
            "EBV": (ebv, "Galactic E(B-V)"),
            "RNGSEED": (seed, "Random-noise seed"),
            "TARGSCL": (scale_factor, "Template flux scale factor"),
        },
    )

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
