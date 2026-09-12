"""
process_bgs_10k_redrock.py
==========================
High-Performance Parallel JUST ETC Simulator for Large Spectral Libraries (e.g. 10,000 BGS Spectra).

Simulates JUST spectrograph observations (throughput, atmosphere, photon noise, readout noise)
and outputs a DESI/Redrock-standard multi-extension FITS file suitable for Redrock redshift fitting.

Server Input Path Example:
  /path/to/data/input-spectra.fits

Server Output Path Example:
  /path/to/data/just_redrock_bgs_10k_obs.fits

Usage Example:
  python process_bgs_10k_redrock.py \
      --input /path/to/data/input-spectra.fits \
      --output /path/to/data/just_redrock_bgs_10k_obs.fits \
      --t_exp 900 --n_exp 4 --nproc 32
"""

from just_etc.resources import DATA_DIR

import sys
import os
import time
import argparse
from pathlib import Path
import multiprocessing as mp
import numpy as np
from astropy.io import fits
from astropy.table import Table

# Add local ETC path
_V1_DIR = Path.cwd()
_JUST_ROOT = Path.cwd()

from just_etc import JUSTExposureTimeCalculator


def _process_single_spectrum_chunk(args_tuple):
    """
    Worker task: processes a chunk of spectra index range (start_idx, end_idx).
    """
    (chunk_idx, start_idx, end_idx, wave_input, flux_chunk_1e17, 
     t_exp, n_exp, seeing_fwhm, zenith_angle, target_ids_chunk) = args_tuple

    # Each worker instantiates its own JUSTExposureTimeCalculator
    etc = JUSTExposureTimeCalculator(calc_mode='fast')
    etc.set_obs_conditions(seeing_fwhm_800=seeing_fwhm, zenith_angle=zenith_angle)

    bands = ['b', 'r', 'z']
    chunk_n = end_idx - start_idx

    results_b = {b: [] for b in bands}
    ivar_b = {b: [] for b in bands}
    res_b = {b: [] for b in bands}
    wave_dict = {}

    for i in range(chunk_n):
        tid = target_ids_chunk[i]
        f_cgs = flux_chunk_1e17[i] * 1e-17  # Convert 10^-17 erg/s/cm²/Å → erg/s/cm²/Å
        f_cgs = np.nan_to_num(f_cgs, nan=0.0, posinf=0.0, neginf=0.0)
        f_cgs = np.maximum(0.0, f_cgs)

        mock_obs = etc.simulate_mock_observation(wave_input, f_cgs, t_exp=t_exp, n_exp=n_exp, seed=int(tid % (2**31 - 1)))

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
            res_matrix = np.ones((1, nwave_arm), dtype=np.float32)

            results_b[b].append(flux_1e17.astype(np.float32))
            ivar_b[b].append(ivar_1e17.astype(np.float32))
            res_b[b].append(res_matrix)

    return chunk_idx, wave_dict, results_b, ivar_b, res_b


def process_spectral_library(
    input_fits_path,
    output_fits_path,
    t_exp=900.0,
    n_exp=4,
    seeing_fwhm=0.8,
    zenith_angle=45.0,
    nproc=None
):
    input_fits_path = Path(input_fits_path)
    output_fits_path = Path(output_fits_path)
    output_fits_path.parent.mkdir(parents=True, exist_ok=True)

    if nproc is None or nproc <= 0:
        nproc = max(1, os.cpu_count() or 1)

    t_start = time.time()

    print("=" * 75)
    print("      JUST ETC Parallel Simulator (10,000 Spectra -> Redrock FITS)")
    print("=" * 75)
    print(f" 📂 Input FITS File : {input_fits_path}")
    print(f" 📂 Output FITS File: {output_fits_path}")
    print(f" ⚙️ Parallel Cores  : {nproc} Workers")
    print(f" ⏱️ Observation     : {n_exp} x {t_exp:.0f}s (Total {n_exp * t_exp / 3600:.2f} hrs)")
    print("=" * 75)

    if not input_fits_path.exists():
        print(f"❌ Error: Input FITS file not found: {input_fits_path}")
        sys.exit(1)

    with fits.open(input_fits_path) as hdul:
        wave_input = np.array(hdul[0].data, dtype=np.float64)
        flux_input_1e17 = np.array(hdul[1].data, dtype=np.float32)

        # Check for TARGETID in extra extensions if present
        target_ids = None
        for ext in hdul:
            if hasattr(ext, 'data') and ext.data is not None and 'TARGETID' in getattr(ext.data, 'names', []):
                target_ids = np.array(ext.data['TARGETID'], dtype=np.int64)
                break

    n_spectra, n_wave = flux_input_1e17.shape
    if target_ids is None or len(target_ids) != n_spectra:
        target_ids = np.arange(300000001, 300000001 + n_spectra, dtype=np.int64)

    print(f" ✅ Loaded {n_spectra} spectra with {n_wave} wavelength points ({wave_input[0]:.1f} - {wave_input[-1]:.1f} Å)")

    # Prepare chunks for multiprocessing
    chunk_size = int(np.ceil(n_spectra / nproc))
    chunks = []

    for c in range(nproc):
        s_idx = c * chunk_size
        e_idx = min((c + 1) * chunk_size, n_spectra)
        if s_idx >= n_spectra:
            break
        chunks.append((
            c, s_idx, e_idx, wave_input, flux_input_1e17[s_idx:e_idx],
            t_exp, n_exp, seeing_fwhm, zenith_angle, target_ids[s_idx:e_idx]
        ))

    actual_chunks = len(chunks)
    print(f"\n🚀 Launching {actual_chunks} parallel worker tasks across {nproc} CPU cores...")

    bands = ['b', 'r', 'z']
    flux_dict = {b: [None] * actual_chunks for b in bands}
    ivar_dict = {b: [None] * actual_chunks for b in bands}
    res_dict = {b: [None] * actual_chunks for b in bands}
    wave_dict = {}

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=nproc) as pool:
        for res_tuple in pool.imap_unordered(_process_single_spectrum_chunk, chunks):
            c_idx, w_dict, r_b, i_b, res_b = res_tuple
            if not wave_dict:
                wave_dict = w_dict
            for b in bands:
                flux_dict[b][c_idx] = r_b[b]
                ivar_dict[b][c_idx] = i_b[b]
                res_dict[b][c_idx] = res_b[b]
            done_count = sum(1 for item in flux_dict['b'] if item is not None)
            pct = (done_count / actual_chunks) * 100.0
            print(f" Progress: [{done_count}/{actual_chunks}] chunks completed ({pct:.1f}%)", end='\r')

    print(f"\n✅ All parallel worker tasks finished! Assembling Redrock FITS extensions...")

    # Concatenate chunk arrays into full N_spectra 2D/3D arrays
    assembled_flux = {}
    assembled_ivar = {}
    assembled_res = {}

    for b in bands:
        assembled_flux[b] = np.concatenate([np.array(item) for item in flux_dict[b] if item is not None], axis=0)
        assembled_ivar[b] = np.concatenate([np.array(item) for item in ivar_dict[b] if item is not None], axis=0)
        assembled_res[b]  = np.concatenate([np.array(item) for item in res_dict[b]  if item is not None], axis=0)

    # Build DESI/Redrock Primary HDU & Extensions
    hdus = [fits.PrimaryHDU()]

    # 1. FIBERMAP Table
    fibermap_table = Table()
    fibermap_table['TARGETID'] = target_ids
    fibermap_table['RA'] = 150.0 + (np.arange(n_spectra) % 1000) * 0.001
    fibermap_table['DEC'] = 2.0 + (np.arange(n_spectra) // 1000) * 0.001
    fibermap_table['FIBER'] = np.arange(n_spectra, dtype=np.int32) % 4000
    fibermap_table['SPECTROGRAPH'] = (np.arange(n_spectra, dtype=np.int32) // 500) % 8
    fibermap_table['OBJTYPE'] = np.array(['TGT'] * n_spectra, dtype='S3')
    fibermap_hdr = fits.Header()
    fibermap_hdr['EXTNAME'] = 'FIBERMAP'
    hdus.append(fits.BinTableHDU(fibermap_table, header=fibermap_hdr))

    # 2. B, R, Z Arm Extensions
    for b in bands:
        b_upper = b.upper()
        
        # WAVELENGTH
        wave_hdu = fits.ImageHDU(wave_dict[b].astype(np.float64), name=f'{b_upper}_WAVELENGTH')
        hdus.append(wave_hdu)

        # FLUX
        flux_hdu = fits.ImageHDU(assembled_flux[b].astype(np.float32), name=f'{b_upper}_FLUX')
        flux_hdu.header['BUNIT'] = '10**-17 erg/(s cm**2 Angstrom)'
        hdus.append(flux_hdu)

        # IVAR
        ivar_hdu = fits.ImageHDU(assembled_ivar[b].astype(np.float32), name=f'{b_upper}_IVAR')
        ivar_hdu.header['BUNIT'] = '10**34 (s**2 cm**4 Angstrom**2)/erg**2'
        hdus.append(ivar_hdu)

        # RESOLUTION
        res_hdu = fits.ImageHDU(assembled_res[b].astype(np.float32), name=f'{b_upper}_RESOLUTION')
        hdus.append(res_hdu)

    # 3. SCORES Table
    scores_table = Table()
    scores_table['TARGETID'] = target_ids
    for b in bands:
        b_upper = b.upper()
        scores_table[f'INTEG_RAW_FLUX_{b_upper}'] = np.mean(assembled_flux[b], axis=1)
        scores_table[f'MEDIAN_RAW_FLUX_{b_upper}'] = np.median(assembled_flux[b], axis=1)
        scores_table[f'SNR_{b_upper}'] = np.mean(assembled_flux[b] * np.sqrt(np.maximum(0, assembled_ivar[b])), axis=1)
    
    scores_hdr = fits.Header()
    scores_hdr['EXTNAME'] = 'SCORES'
    hdus.append(fits.BinTableHDU(scores_table, header=scores_hdr))

    # Save to disk
    hdul_out = fits.HDUList(hdus)
    hdul_out.writeto(output_fits_path, overwrite=True)

    elapsed = time.time() - t_start
    file_size_mb = output_fits_path.stat().st_size / (1024 * 1024)

    print("-" * 75)
    print(f"🎉 Successfully generated Redrock FITS file!")
    print(f" 📂 Output Path  : {output_fits_path}")
    print(f" 📊 File Size    : {file_size_mb:.2f} MB")
    print(f" ⏱️ Total Time   : {elapsed:.2f} seconds ({n_spectra / elapsed:.1f} spectra/sec)")
    print("-" * 75)

    return output_fits_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parallel JUST ETC Simulator for Large Spectral Libraries (10,000 Spectra -> Redrock FITS)")
    parser.add_argument('--input', type=str, required=True, help="Path to input spectra FITS file")
    parser.add_argument('--output', type=str, required=True, help="Path to output Redrock FITS file")
    parser.add_argument('--t_exp', type=float, default=900.0, help="Exposure time per frame in seconds")
    parser.add_argument('--n_exp', type=int, default=4, help="Number of co-added exposures")
    parser.add_argument('--seeing', type=float, default=0.8, help="800nm seeing FWHM in arcsec")
    parser.add_argument('--nproc', type=int, default=32, help="Number of parallel worker processes")

    args = parser.parse_args()
    process_spectral_library(
        input_fits_path=args.input,
        output_fits_path=args.output,
        t_exp=args.t_exp,
        n_exp=args.n_exp,
        seeing_fwhm=args.seeing,
        nproc=args.nproc
    )
