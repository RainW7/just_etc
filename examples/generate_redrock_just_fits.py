"""
generate_redrock_just_fits.py
==============================
Generates DESI/Redrock-standard multi-extension FITS files for mock JUST 
spectrograph observations using JUST ETC.

The output FITS file conforms to DESI/Redrock spectra specifications and can
be directly fed into Redrock CLI (`rrdesi`) or Python API for redshift fitting
and target classification.

Requirements:
- Conda Environment: desi
- Environment Variable: RR_TEMPLATE_DIR=/path/to/redrock/templates

Usage:
  python generate_redrock_just_fits.py
"""

from just_etc.resources import DATA_DIR

import sys
import os
from pathlib import Path
import numpy as np
from astropy.table import Table

# Add JUST ETC paths
_JUST_ROOT = Path.cwd()
_V1_DIR = Path.cwd()

# Ensure Redrock template path

import desispec.io.spectra as io_spec
import redrock.templates
import redrock.zfind
import redrock.external.desi as rrdesi
from just_etc import JUSTExposureTimeCalculator, load_template, normalize_to_mag
from just_etc.dwarf_ssp_model import build_dwarf_ssp_spectrum


def generate_just_redrock_fits(output_fits_path, targets_config, t_exp=900.0, n_exp=4):
    """
    Generate a DESI/Redrock-standard FITS file containing simulated JUST spectrograph observations.

    Parameters
    ----------
    output_fits_path : str or Path
        Target filepath for the output FITS file.
    targets_config : list of dict
        List of target dictionary configs.
    t_exp : float, optional
        Exposure time per frame in seconds (default: 900s).
    n_exp : int, optional
        Number of co-added exposures (default: 4).

    Returns
    -------
    output_fits_path : Path
    """
    output_fits_path = Path(output_fits_path)
    output_fits_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("      JUST Spectrograph Mock Spectra Generator for Redrock FITS")
    print("=" * 75)

    etc = JUSTExposureTimeCalculator(calc_mode='accurate')
    etc.set_obs_conditions(seeing_fwhm_800=0.8, zenith_angle=45.0)

    bands = ['b', 'r', 'z']
    wave_dict = {}
    flux_dict = {b: [] for b in bands}
    ivar_dict = {b: [] for b in bands}
    res_dict = {b: [] for b in bands}

    target_ids = []
    ras = []
    decs = []

    print(f"\nProcessing {len(targets_config)} targets for JUST mock observation...")

    for idx, targ in enumerate(targets_config):
        tid = targ.get('targetid', 10001 + idx)
        z_true = targ['z_true']
        mag_r = targ.get('mag_r', 20.0)
        t_type = targ.get('type', 'template')

        print(f" -> Target [{idx+1}/{len(targets_config)}] ID: {tid} | z_true: {z_true:.4f} | r_mag: {mag_r:.1f} mag | Type: {t_type}")

        # 1. Generate Intrinsic Spectrum
        if t_type == 'ssp_dwarf':
            dwarf_kind = targ.get('dwarf_kind', 'dIrr')
            mag_r = targ.get('mag_r', 20.5)
            # build_dwarf_ssp_spectrum already applies redshift to w_obs and f_obs
            w_obs, f_obs, meta = build_dwarf_ssp_spectrum(dwarf_type=dwarf_kind, target_mag=mag_r, redshift=z_true)
            f_norm = f_obs
        else:
            # Load template FITS file
            tpl_path = DATA_DIR / "templates" / targ['template_file']
            w_rest, f_rest = load_template(tpl_path)
            w_obs = w_rest * (1.0 + z_true)
            f_obs = f_rest / (1.0 + z_true)
            f_norm, _ = normalize_to_mag(w_obs, f_obs, target_mag=mag_r, band='r')

        # 2. Simulate JUST 3-arm mock observation
        mock_obs = etc.simulate_mock_observation(w_obs, f_norm, t_exp=t_exp, n_exp=n_exp, seed=tid)

        # 3. Format for Redrock
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
            
            # Physical noise std dev in flux density units: sigma_F = f_int / snr
            sigma_f_1e17 = np.where(snr > 0, np.maximum(f_int_1e17, 1e-5) / snr, 0.0)
            ivar_1e17 = np.where(sigma_f_1e17 > 0, 1.0 / (sigma_f_1e17 ** 2), 0.0)

            bad = (~np.isfinite(flux_1e17)) | (~np.isfinite(ivar_1e17))
            ivar_1e17[bad] = 0.0

            nwave = len(w_arm)
            res_matrix = np.ones((1, nwave))  # 1 diagonal

            flux_dict[b].append(flux_1e17)
            ivar_dict[b].append(ivar_1e17)
            res_dict[b].append(res_matrix)

        target_ids.append(tid)
        ras.append(targ.get('ra', 150.0 + idx * 0.1))
        decs.append(targ.get('dec', 2.0 + idx * 0.1))

    # Format arrays into DESI data shape
    for b in bands:
        flux_dict[b] = np.array(flux_dict[b])   # shape: (N_targets, N_wave_b)
        ivar_dict[b] = np.array(ivar_dict[b])   # shape: (N_targets, N_wave_b)
        res_dict[b]  = np.array(res_dict[b])    # shape: (N_targets, 1, N_wave_b)

    # 4. Build Fibermap metadata table adhering to DESI / rrdesi standards
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

    # 5. Build Scores metadata table required by DESI rrdesi reader
    scores = Table({
        'TARGETID': np.array(target_ids, dtype=np.int64)
    })

    # 6. Build DESI Spectra object and save to FITS
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
    print(f"   Targets: {len(target_ids)} targets | Arms: b, r, z")

    return output_fits_path


def verify_with_redrock(fits_path, redrock_out_h5=None):
    """
    Run Redrock redshift fitting directly on the generated FITS file using rrdesi CLI wrapper.
    """
    if redrock_out_h5 is None:
        redrock_out_h5 = Path(fits_path).parent / "redrock_zbest.fits"

    print("\n" + "=" * 75)
    print("      Verifying FITS File Compatibility with Redrock Engine (rrdesi)")
    print("=" * 75)

    cmd = f"python -c \"import sys, redrock.external.desi as rrd; sys.argv=['rrdesi', '-i', '{fits_path}', '-o', '{redrock_out_h5}', '--mp', '4']; rrd.rrdesi()\""

    print(f"Executing CLI Command:\n  {cmd}\n")
    ret = os.system(cmd)

    if ret == 0 and Path(redrock_out_h5).exists():
        from astropy.table import Table
        zbest = Table.read(str(redrock_out_h5), hdu='REDSHIFTS')
        print("\n" + "=" * 75)
        print("                    REDROCK VERIFICATION RESULTS")
        print("=" * 75)
        for row in zbest:
            print(f" Target ID: {row['TARGETID']} | z_fit: {row['Z']:.6f} +/- {row['ZERR']:.6f} | Type: {row['SPECTYPE']:8s} | chi2: {row['CHI2']:.2f}")
        print("=" * 75)
    else:
        print(f"⚠️ Warning: Redrock execution finished with status code {ret}")


if __name__ == '__main__':
    # Sample Target Configurations
    sample_targets = [
        {
            'targetid': 20001,
            'z_true': 0.20,
            'mag_r': 19.5,
            'type': 'template',
            'template_file': 'starburst/sb4_kinney_fuv_001.fits',
            'ra': 150.05, 'dec': 2.10
        },
        {
            'targetid': 20002,
            'z_true': 0.35,
            'mag_r': 20.2,
            'type': 'template',
            'template_file': 'galaxy/elliptical_001.fits',
            'ra': 150.15, 'dec': 2.20
        },
        {
            'targetid': 20003,
            'z_true': 0.02,
            'type': 'ssp_dwarf',
            'dwarf_kind': 'dIrr',
            'mag_r': 20.5,
            'ra': 150.25, 'dec': 2.30
        }
    ]

    out_file = _JUST_ROOT / 'output' / 'just_redrock_mock_spectra.fits'
    generate_just_redrock_fits(out_file, sample_targets, t_exp=900.0, n_exp=4)
    verify_with_redrock(out_file)
