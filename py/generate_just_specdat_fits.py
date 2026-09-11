"""
generate_just_specdat_fits.py
==============================
Generates three FITS throughput files (thru-b, thru-r, thru-z) based on the
spectrograph throughput curves in `ETC_py_v1/spec.dat`, formatted exactly like
DESI / justspecsimu throughput FITS files (`thru-{b,r,z}_y1measured.fits`).

Grid: 63,001 wavelength points from 3550.05 Å to 9850.05 Å (step 0.1 Å).
Columns: wavelength [Å], throughput, extinction, fiberinput.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

# Add ETC_py_v1 to python path
v1_dir = Path('/Users/rain/JUST_ETC/ETC_py_v1')
if str(v1_dir) not in sys.path:
    sys.path.insert(0, str(v1_dir))

import ETC_py_optimized as ETC_py

def generate_fits_files():
    print("=" * 60)
    print("Generating JUST Throughput FITS files from spec.dat")
    print("=" * 60)

    # 1. Read spec.dat throughput data
    spec_path = v1_dir / 'spec.dat'
    spectro = ETC_py.SpectroAttrib()
    ETC_py.gsReadSpectrographConfig(str(spec_path), spectro)

    # 2. Template grid matching DESI / justspecsimu (3550.05 to 9850.05 A, step 0.1 A, 63001 points)
    wave_grid = np.arange(3550.05, 9850.15, 0.10)
    n_pts = len(wave_grid)
    print(f"Target wavelength grid: {wave_grid[0]:.2f} - {wave_grid[-1]:.2f} Å ({n_pts} points)")

    # Output directories
    compare_dir = v1_dir / 'compare' / 'justspecsimu_and_etc参数对比'
    output_fits_dir = v1_dir / 'output' / 'fits'
    output_fits_dir.mkdir(parents=True, exist_ok=True)

    arm_names = ['b', 'r', 'z']
    arm_labels = ['Blue (Arm 0: 3650-5680 Å)', 'Green/Red (Arm 1: 5400-7450 Å)', 'Red/NIR (Arm 2: 7200-9250 Å)']

    generated_files = []

    for ia, (arm_name, arm_label) in enumerate(zip(arm_names, arm_labels)):
        imin = spectro.istart[ia]
        imax = spectro.istart[ia + 1]

        w_spec_aa = spectro.l[imin:imax] * 10.0  # nm -> Å
        t_spec = spectro.T[imin:imax]

        # Interpolate throughput onto target wavelength grid (0.0 outside coverage)
        t_grid = np.interp(wave_grid, w_spec_aa, t_spec, left=0.0, right=0.0)
        extinction_grid = np.zeros_like(wave_grid)
        fiberinput_grid = np.zeros_like(wave_grid)

        # Create Table & FITS HDU List matching DESI format
        col_wave = fits.Column(name='wavelength', format='D', unit='Angstrom', array=wave_grid)
        col_thru = fits.Column(name='throughput', format='D', array=t_grid)
        col_ext = fits.Column(name='extinction', format='D', array=extinction_grid)
        col_fib = fits.Column(name='fiberinput', format='D', array=fiberinput_grid)

        cols = fits.ColDefs([col_wave, col_thru, col_ext, col_fib])
        table_hdu = fits.BinTableHDU.from_columns(cols, name='THROUGHPUT')

        primary_hdu = fits.PrimaryHDU()
        hdul = fits.HDUList([primary_hdu, table_hdu])

        # Target file paths
        path_compare = compare_dir / f"thru-{arm_name}_specdat.fits"
        path_output = output_fits_dir / f"thru-{arm_name}.fits"

        hdul.writeto(path_compare, overwrite=True)
        hdul.writeto(path_output, overwrite=True)

        generated_files.append((path_compare, path_output))

        print(f"✅ Generated Arm {ia} ({arm_name.upper()}): {path_compare.name} & {path_output.name}")
        print(f"   Coverage: {w_spec_aa[0]:.1f} - {w_spec_aa[-1]:.1f} Å | Peak Throughput: {t_grid.max():.4f}")

    # 3. Create Verification Plot comparing spec.dat FITS with Y1 Measured FITS
    print("\nCreating verification plot...")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    for ia, (arm_name, color) in enumerate(zip(arm_names, colors)):
        # Y1 measured
        y1_path = compare_dir / f"thru-{arm_name}_y1measured.fits"
        with fits.open(y1_path) as h:
            d_y1 = h[1].data
            ax.plot(d_y1['wavelength'], d_y1['throughput'], color=color, alpha=0.5, lw=1.2,
                    label=f'justspecsimu {arm_name}-arm (Y1 Measured)')

        # Generated spec.dat
        spec_path = compare_dir / f"thru-{arm_name}_specdat.fits"
        with fits.open(spec_path) as h:
            d_spec = h[1].data
            ax.plot(d_spec['wavelength'], d_spec['throughput'], color=color, linestyle='--', lw=2.0,
                    label=f'JUST spec.dat thru-{arm_name}.fits')

    ax.set_xlabel('Wavelength [Å]', fontsize=12)
    ax.set_ylabel('Total Throughput [0 - 1]', fontsize=12)
    ax.set_title('Comparison of Generated JUST spec.dat FITS vs DESI Y1 FITS', fontsize=13)
    ax.set_xlim(3500, 10000)
    ax.set_ylim(0.0, 0.55)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)

    plot_path = compare_dir / 'plots' / 'throughput_specdat_fits_comparison.png'
    plot_path.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Verification plot saved to: {plot_path}")
    print("=" * 60)

if __name__ == '__main__':
    generate_fits_files()
