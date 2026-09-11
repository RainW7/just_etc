"""
convert_input_fits_to_redrock.py
================================
Converts an input template/spectrum library FITS file (with HDU 0: WAVELENGTH, 
HDU 1: FLUX in 10^-17 erg/s/cm^2/A) into a DESI/Redrock-standard multi-extension 
FITS file containing simulated JUST spectrograph observations.

Example Input File:
  /Users/rain/JUST_ETC/redrock/input-spectra_10.fits

Output FITS File Structure:
  - B_WAVELENGTH, B_FLUX, B_IVAR, B_RESOLUTION
  - R_WAVELENGTH, R_FLUX, R_IVAR, R_RESOLUTION
  - Z_WAVELENGTH, Z_FLUX, Z_IVAR, Z_RESOLUTION
  - FIBERMAP, SCORES

Usage via Command Line:
  python convert_input_fits_to_redrock.py \
      --input /Users/rain/JUST_ETC/redrock/input-spectra_10.fits \
      --output /Users/rain/JUST_ETC/output/just_redrock_input10_obs.fits \
      --t_exp 900 --n_exp 4 --verify
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table

# Add JUST ETC paths
_JUST_ROOT = Path('/Users/rain/JUST_ETC')
_V1_DIR = _JUST_ROOT / 'ETC_py_v1'
if str(_V1_DIR) not in sys.path:
    sys.path.insert(0, str(_V1_DIR))

# Ensure Redrock template path is set
rr_template_dir = _JUST_ROOT / 'redrock' / 'py' / 'redrock' / 'templates'
os.environ['RR_TEMPLATE_DIR'] = str(rr_template_dir)

import desispec.io.spectra as io_spec
import redrock.templates
import redrock.zfind
import redrock.external.desi as rrdesi
from just_etc_api import JUSTExposureTimeCalculator


def convert_input_spectra_to_just_redrock(
    input_fits_path,
    output_fits_path,
    t_exp=900.0,
    n_exp=4,
    seeing_fwhm=0.8,
    zenith_angle=45.0
):
    """
    Reads input spectra FITS file, simulates JUST observations via ETC, and writes
    a Redrock-standard multi-arm FITS file.

    Parameters
    ----------
    input_fits_path : str or Path
        Path to input FITS file (HDU 0: WAVELENGTH, HDU 1: FLUX in 10^-17 erg/s/cm^2/A).
    output_fits_path : str or Path
        Path to output DESI/Redrock-standard FITS file.
    t_exp : float
        Exposure time per frame in seconds (default: 900s).
    n_exp : int
        Number of co-added exposures (default: 4).
    seeing_fwhm : float
        Atmospheric seeing FWHM at 800nm in arcseconds (default: 0.8").
    zenith_angle : float
        Zenith angle in degrees (default: 45.0 deg).

    Returns
    -------
    output_fits_path : Path
    """
    input_fits_path = Path(input_fits_path)
    output_fits_path = Path(output_fits_path)
    output_fits_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("      JUST Spectrograph Converter: Input Spectra -> Redrock FITS")
    print("=" * 75)
    print(f" Reading Input FITS: {input_fits_path}")

    with fits.open(input_fits_path) as hdul:
        wave_input = np.array(hdul[0].data, dtype=np.float64)  # Wavelength array [Å]
        flux_input_1e17 = np.array(hdul[1].data, dtype=np.float64)  # Flux density [10^-17 cgs]

    n_spectra, n_wave = flux_input_1e17.shape
    print(f" Input File Info: {n_spectra} spectra | {n_wave} wavelength points ({wave_input[0]:.1f} - {wave_input[-1]:.1f} Å)")

    # Initialize JUST ETC
    etc = JUSTExposureTimeCalculator(calc_mode='fast')
    etc.set_obs_conditions(seeing_fwhm_800=seeing_fwhm, zenith_angle=zenith_angle)

    bands = ['b', 'r', 'z']
    wave_dict = {}
    flux_dict = {b: [] for b in bands}
    ivar_dict = {b: [] for b in bands}
    res_dict = {b: [] for b in bands}

    target_ids = []
    ras = []
    decs = []

    print(f"\nSimulating JUST spectrograph observations (t_exp = {n_exp} x {t_exp:.0f}s = {n_exp*t_exp/3600:.2f}h)...")

    for i in range(n_spectra):
        tid = 30001 + i
        f_cgs = flux_input_1e17[i] * 1e-17  # Convert to erg/s/cm²/Å

        # Clean non-finite or negative inputs
        f_cgs = np.nan_to_num(f_cgs, nan=0.0, posinf=0.0, neginf=0.0)
        f_cgs = np.maximum(0.0, f_cgs)

        # Run JUST ETC mock observation
        mock_obs = etc.simulate_mock_observation(wave_input, f_cgs, t_exp=t_exp, n_exp=n_exp, seed=tid)

        for arm in mock_obs:
            ia = arm['arm']
            b = bands[ia]
            w_arm = arm['wave_aa']
            f_mock = arm['flux_mock']
            std_e = arm['noise_e']
            sig_e = arm['signal_e']

            if b not in wave_dict:
                wave_dict[b] = w_arm

            snr = np.where(std_e > 0, sig_e / std_e, 0.0)
            flux_1e17 = f_mock * 1e17
            f_int_1e17 = arm['flux_intrinsic'] * 1e17
            
            sigma_f_1e17 = np.where(snr > 0, np.maximum(f_int_1e17, 1e-5) / snr, 0.0)
            ivar_1e17 = np.where(sigma_f_1e17 > 0, 1.0 / (sigma_f_1e17 ** 2), 0.0)

            bad = (~np.isfinite(flux_1e17)) | (~np.isfinite(ivar_1e17))
            ivar_1e17[bad] = 0.0

            nwave_arm = len(w_arm)
            res_matrix = np.ones((1, nwave_arm))

            flux_dict[b].append(flux_1e17)
            ivar_dict[b].append(ivar_1e17)
            res_dict[b].append(res_matrix)

        target_ids.append(tid)
        ras.append(150.0 + i * 0.05)
        decs.append(2.0 + i * 0.05)
        print(f" -> Processed Spectrum [{i+1:2d}/{n_spectra}] TARGETID: {tid}")

    # Format arrays into DESI 2D/3D shapes
    for b in bands:
        flux_dict[b] = np.array(flux_dict[b])  # shape: (N_targets, N_wave_b)
        ivar_dict[b] = np.array(ivar_dict[b])  # shape: (N_targets, N_wave_b)
        res_dict[b]  = np.array(res_dict[b])   # shape: (N_targets, 1, N_wave_b)

    # Build Fibermap metadata table conforming to DESI standards
    n_targ = len(target_ids)
    fibermap = Table({
        'TARGETID': np.array(target_ids, dtype=np.int64),
        'TARGET_RA': np.array(ras, dtype=np.float64),
        'TARGET_DEC': np.array(decs, dtype=np.float64),
        'OBJTYPE': np.array(['TGT'] * n_targ, dtype='U3'),
        'DESI_TARGET': np.array([1] * n_targ, dtype=np.int64),
        'COADD_FIBERSTATUS': np.array([0] * n_targ, dtype=np.int32),
        'TILEID': np.array([1111] * n_targ, dtype=np.int32),
        'PETAL_LOC': np.array([0] * n_targ, dtype=np.int16),
        'FIBER': np.array(list(range(n_targ)), dtype=np.int32),
    })

    # Build Scores metadata table required by DESI rrdesi reader
    scores = Table({
        'TARGETID': np.array(target_ids, dtype=np.int64)
    })

    # Write Redrock-standard Spectra FITS object
    spec_obj = io_spec.Spectra(
        bands=bands,
        wave=wave_dict,
        flux=flux_dict,
        ivar=ivar_dict,
        resolution_data=res_dict,
        fibermap=fibermap,
        scores=scores
    )

    io_spec.write_spectra(str(output_fits_path), spec_obj)
    print(f"\n✅ Successfully created Redrock-standard JUST spectra FITS file:")
    print(f"   Path: {output_fits_path}")
    print(f"   Total Targets: {len(target_ids)} | Spectrograph Arms: b, r, z")

    return output_fits_path


def run_redrock_verification(fits_path, redrock_out_fits=None):
    """
    Run Redrock redshift fitting on the output FITS file.
    """
    if redrock_out_fits is None:
        redrock_out_fits = Path(fits_path).parent / "just_redrock_input10_zbest.fits"

    print("\n" + "=" * 75)
    print("      Running Redrock Redshift Fitting on Converted FITS File")
    print("=" * 75)

    rrdesi_bin = _JUST_ROOT / "redrock" / "bin" / "rrdesi"
    cmd = f"python -c \"import sys, redrock.external.desi as rrd; sys.argv=['rrdesi', '-i', '{fits_path}', '-o', '{redrock_out_fits}', '--mp', '4']; rrd.rrdesi()\""

    print(f"Executing Command:\n  {cmd}\n")
    ret = os.system(cmd)

    if ret == 0 and Path(redrock_out_fits).exists():
        zbest = Table.read(str(redrock_out_fits), hdu='REDSHIFTS')
        print("\n" + "=" * 75)
        print("                    REDROCK FITTING RESULTS")
        print("=" * 75)
        for row in zbest:
            print(f" Target ID: {row['TARGETID']} | z_fit: {row['Z']:.6f} +/- {row['ZERR']:.6f} | Type: {row['SPECTYPE']:8s} | chi2: {row['CHI2']:.2f}")
        print("=" * 75)
    else:
        print(f"⚠️ Warning: Redrock process completed with code {ret}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert input spectra FITS file into Redrock-compatible JUST observation FITS file.")
    parser.add_argument('--input', type=str, default=str(_JUST_ROOT / 'redrock' / 'input-spectra_10.fits'), help="Path to input FITS file")
    parser.add_argument('--output', type=str, default=str(_JUST_ROOT / 'output' / 'just_redrock_input10_obs.fits'), help="Path to output Redrock FITS file")
    parser.add_argument('--t_exp', type=float, default=900.0, help="Exposure time per frame in seconds")
    parser.add_argument('--n_exp', type=int, default=4, help="Number of co-added exposures")
    parser.add_argument('--verify', action='store_true', help="Run Redrock redshift fitting immediately after conversion")

    args = parser.parse_args()

    out_file = convert_input_spectra_to_just_redrock(args.input, args.output, t_exp=args.t_exp, n_exp=args.n_exp)

    if args.verify:
        run_redrock_verification(out_file)
