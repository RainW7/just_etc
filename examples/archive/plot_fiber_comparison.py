
from just_etc.resources import DATA_DIR
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add local logic to path
from just_etc import ETC_py_optimized as ETC_py
from just_etc import JUSTExposureTimeCalculator
from copy import deepcopy

def plot_fiber_comparison():
    base_dir = Path.cwd()
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    print("Initialize ETC configurations...")
    etc = JUSTExposureTimeCalculator(calc_mode='accurate')
    obs = etc._make_obs()
    sp = etc._spectro
    
    # Base parameters
    fa = etc._obs_params['field_angle']
    EFL_val = sp.EFL[0]
    ARCSEC_PER_URAD = 0.206264806247097
    
    # Define fiber sizes
    # Case 1: 1.36" diameter -> 0.68" radius
    r1_arcsec = 1.36 / 2.0
    fiber_ent_rad1 = (r1_arcsec / ARCSEC_PER_URAD) * EFL_val
    
    # Case 2: 1.57" diameter -> 0.785" radius
    r2_arcsec = 1.57 / 2.0
    fiber_ent_rad2 = (r2_arcsec / ARCSEC_PER_URAD) * EFL_val
    
    # Create two copies of the spectrograph configuration
    sp1 = deepcopy(sp)
    sp1.fiber_ent_rad = fiber_ent_rad1
    
    sp2 = deepcopy(sp)
    sp2.fiber_ent_rad = fiber_ent_rad2
    
    # Setup plots
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_lam = axes[0, 0]
    ax_see = axes[0, 1]
    ax_dec = axes[1, 0]
    ax_ext = axes[1, 1]
    
    # Common helper function to compute efficiency
    def get_eff(spectro, obs_state, lam, r_eff, decent):
        return ETC_py.gsGeometricThroughput(spectro, obs_state, lam, r_eff, decent, fa, 0x0)

    label1 = f'1.36" Diameter (r={r1_arcsec:.2f}")'
    label2 = f'1.57" Diameter (r={r2_arcsec:.3f}")'

    # -------------------------------------------------------------
    # Plot 1: Efficiency vs Wavelength
    # -------------------------------------------------------------
    lambda_array = np.linspace(350, 950, 50)
    obs.seeing_fwhm_800 = 0.8
    eff1_lam, eff2_lam = [], []
    for lam in lambda_array:
        eff1_lam.append(get_eff(sp1, obs, lam, 0.0, 0.0))
        eff2_lam.append(get_eff(sp2, obs, lam, 0.0, 0.0))
        
    ax_lam.plot(lambda_array, eff1_lam, lw=2, label=label1, color='blue')
    ax_lam.plot(lambda_array, eff2_lam, lw=2, label=label2, color='red')
    ax_lam.set_xlabel('Wavelength [nm]')
    ax_lam.set_ylabel('Geometric Throughput')
    ax_lam.set_title('Efficiency vs Wavelength\n(Seeing=0.8", Point Source)')
    ax_lam.grid(True, linestyle=':', alpha=0.7)
    ax_lam.legend()

    # -------------------------------------------------------------
    # Plot 2: Efficiency vs Seeing FWHM
    # -------------------------------------------------------------
    seeing_array = np.linspace(0.4, 2.0, 50)
    eff1_see, eff2_see = [], []
    for s in seeing_array:
        obs.seeing_fwhm_800 = s
        eff1_see.append(get_eff(sp1, obs, 600.0, 0.0, 0.0))
        eff2_see.append(get_eff(sp2, obs, 600.0, 0.0, 0.0))
        
    ax_see.plot(seeing_array, eff1_see, lw=2, label=label1, color='blue')
    ax_see.plot(seeing_array, eff2_see, lw=2, label=label2, color='red')
    ax_see.set_xlabel('Seeing FWHM [arcsec]')
    ax_see.set_ylabel('Geometric Throughput')
    ax_see.set_title('Efficiency vs Atmospheric Seeing\n(Wavelength=600nm, Point Source)')
    ax_see.grid(True, linestyle=':', alpha=0.7)
    ax_see.legend()

    # -------------------------------------------------------------
    # Plot 3: Efficiency vs Decenter (Fiber Misalignment)
    # -------------------------------------------------------------
    obs.seeing_fwhm_800 = 0.8
    decent_array = np.linspace(0.0, 1.0, 50)
    eff1_dec, eff2_dec = [], []
    for d in decent_array:
        eff1_dec.append(get_eff(sp1, obs, 600.0, 0.0, d))
        eff2_dec.append(get_eff(sp2, obs, 600.0, 0.0, d))
        
    ax_dec.plot(decent_array, eff1_dec, lw=2, label=label1, color='blue')
    ax_dec.plot(decent_array, eff2_dec, lw=2, label=label2, color='red')
    ax_dec.axvline(r1_arcsec, color='blue', linestyle='--', alpha=0.5, label='1.36" Boundary')
    ax_dec.axvline(r2_arcsec, color='red', linestyle='--', alpha=0.5, label='1.57" Boundary')
    ax_dec.set_xlabel('Fiber Pointing Decenter [arcsec]')
    ax_dec.set_ylabel('Geometric Throughput')
    ax_dec.set_title('Efficiency vs Pointing Decenter\n(Seeing=0.8", 600nm, Point Source)')
    ax_dec.grid(True, linestyle=':', alpha=0.7)
    ax_dec.legend()

    # -------------------------------------------------------------
    # Plot 4: Efficiency vs Extended Source Radius (r_eff)
    # -------------------------------------------------------------
    reff_array = np.linspace(0.0, 2.0, 50)
    eff1_ext, eff2_ext = [], []
    for r in reff_array:
        eff1_ext.append(get_eff(sp1, obs, 600.0, r, 0.0))
        eff2_ext.append(get_eff(sp2, obs, 600.0, r, 0.0))
        
    ax_ext.plot(reff_array, eff1_ext, lw=2, label=label1, color='blue')
    ax_ext.plot(reff_array, eff2_ext, lw=2, label=label2, color='red')
    ax_ext.set_xlabel('Source Effective Radius (r_eff) [arcsec]')
    ax_ext.set_ylabel('Geometric Throughput')
    ax_ext.set_title('Efficiency vs Source Extent\n(Seeing=0.8", 600nm, Centered)')
    ax_ext.grid(True, linestyle=':', alpha=0.7)
    ax_ext.legend()

    plt.tight_layout()
    output_png = output_dir / "fiber_size_comparison.png"
    plt.savefig(output_png, dpi=300)
    plt.close()
    
    print(f"Analysis plot generated: {output_png}")
    
    # Calculate difference at typical values
    obs.seeing_fwhm_800 = 0.8
    eff1_typical = get_eff(sp1, obs, 600.0, 0.0, 0.0)
    eff2_typical = get_eff(sp2, obs, 600.0, 0.0, 0.0)
    
    print(f"\n--- Numeric Comparison at Typical Conditions ---")
    print(f"Seeing = 0.8\", Wavelength = 600 nm, Point Source, Centered:")
    print(f"{label1} Geometric Efficiency: {eff1_typical*100:.2f}%")
    print(f"{label2} Geometric Efficiency: {eff2_typical*100:.2f}%")
    print(f"Absolute Difference: {(eff2_typical - eff1_typical)*100:.2f}%")
    print(f"Relative Gain: {((eff2_typical - eff1_typical) / eff1_typical)*100:.2f}%")

if __name__ == '__main__':
    plot_fiber_comparison()
