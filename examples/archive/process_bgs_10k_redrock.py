"""Run the JUST ETC on a FITS spectral library and write Redrock FITS.

This archived command-line wrapper reads the historical two-HDU input format:
the primary HDU contains a wavelength vector in Angstroms, and the next HDU
contains spectra in ``10^-17 erg s^-1 cm^-2 Angstrom^-1``. The reusable
array-based implementation is available as
``just_etc.convert_to_redrock_format``.

Example
-------
python process_bgs_10k_redrock.py \
    --input /path/to/input-spectra.fits \
    --output /path/to/just-redrock-spectra.fits \
    --t_exp 900 --n_exp 4 --nproc 32
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

from just_etc import convert_to_redrock_format


def process_spectral_library(
    input_fits_path,
    output_fits_path,
    t_exp=900.0,
    n_exp=4,
    seeing_fwhm=0.8,
    zenith_angle=45.0,
    nproc=None,
):
    """Read a legacy library FITS file and delegate its simulation to JUST ETC."""
    input_fits_path = Path(input_fits_path).expanduser()
    output_fits_path = Path(output_fits_path).expanduser()
    if not input_fits_path.is_file():
        raise FileNotFoundError(f"Input FITS file not found: {input_fits_path}")
    if input_fits_path.resolve() == output_fits_path.resolve():
        raise ValueError("input and output FITS paths must be different")

    with fits.open(input_fits_path, memmap=True) as hdul:
        if len(hdul) < 2 or hdul[0].data is None or hdul[1].data is None:
            raise ValueError(
                "input FITS must have a wavelength vector in the primary HDU "
                "and a 2D flux array in extension 1"
            )
        wave = np.array(hdul[0].data, dtype=np.float64, copy=True)
        flux = np.array(hdul[1].data, dtype=np.float64, copy=True)

        target_ids = None
        for hdu in hdul[2:]:
            names = getattr(getattr(hdu, "data", None), "names", None)
            if names and "TARGETID" in names:
                target_ids = np.array(hdu.data["TARGETID"], copy=True)
                break

    if target_ids is not None and target_ids.size != flux.shape[0]:
        target_ids = None

    return convert_to_redrock_format(
        wave,
        flux,
        output_fits_path,
        t_exp=t_exp,
        n_exp=n_exp,
        seeing_fwhm=seeing_fwhm,
        zenith_angle=zenith_angle,
        nproc=nproc,
        target_ids=target_ids,
        verbose=True,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate a FITS spectral library with JUST ETC and write Redrock FITS"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input spectral-library FITS")
    parser.add_argument("--output", type=Path, required=True, help="Output Redrock FITS")
    parser.add_argument("--t_exp", type=float, default=900.0, help="Exposure time per frame [s]")
    parser.add_argument("--n_exp", type=int, default=4, help="Number of co-added exposures")
    parser.add_argument("--seeing", type=float, default=0.8, help="Seeing FWHM at 800 nm [arcsec]")
    parser.add_argument("--zenith-angle", type=float, default=45.0, help="Zenith angle [deg]")
    parser.add_argument("--nproc", type=int, default=32, help="Worker process count")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process_spectral_library(
        input_fits_path=args.input,
        output_fits_path=args.output,
        t_exp=args.t_exp,
        n_exp=args.n_exp,
        seeing_fwhm=args.seeing,
        zenith_angle=args.zenith_angle,
        nproc=args.nproc,
    )
