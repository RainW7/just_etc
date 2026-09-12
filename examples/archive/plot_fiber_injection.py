
from just_etc.resources import DATA_DIR
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add local logic to path
from just_etc import ETC_py_optimized as ETC_py
from just_etc import JUSTExposureTimeCalculator

def plot_fiber_injection_analysis():
    base_dir = Path.cwd()
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    print("Initialize ETC configurations...")
    etc = JUSTExposureTimeCalculator(calc_mode='accurate')
    obs = etc._make_obs()
    sp = etc._spectro
    
    # Defaults
    fa = etc._obs_params['field_angle']
    lambda_nm = 600.0 # Fix wavelength to 600 nm for the first plots
    r_eff = 0.0 # Point source
    
    # Setup plots
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_seeing = axes[0, 0]
    ax_decent = axes[0, 1]
    ax_lambda = axes[1, 0]
    ax_psf = axes[1, 1]
    
    # -------------------------------------------------------------
    # Plot 1: Efficiency vs Seeing FWHM
    # -------------------------------------------------------------
    seeing_array = np.linspace(0.4, 2.5, 50)
    eff_seeing = []
    
    for s in seeing_array:
        obs.seeing_fwhm_800 = s
        eff = ETC_py.gsGeometricThroughput(sp, obs, lambda_nm, r_eff, 
                                           0.0, # Zero decenter
                                           fa, 0x0)
        eff_seeing.append(eff)
        
    ax_seeing.plot(seeing_array, eff_seeing, lw=2, color='#1f77b4')
    ax_seeing.axvline(1.0, color='gray', linestyle='--')
    ax_seeing.set_xlabel('Seeing FWHM [arcsec]')
    ax_seeing.set_ylabel('Fiber Geometric Throughput [0-1]')
    ax_seeing.set_title('Efficiency vs. Atmospheric Seeing')
    ax_seeing.grid(True, linestyle=':', alpha=0.7)
    
    # -------------------------------------------------------------
    # Plot 2: Efficiency vs Fiber Decenter (Misalignment)
    # -------------------------------------------------------------
    obs.seeing_fwhm_800 = 0.8 # Reset seeing to standard 0.8"
    decent_array = np.linspace(0.0, 1.5, 50)
    eff_decent = []
    
    for d in decent_array:
        eff = ETC_py.gsGeometricThroughput(sp, obs, lambda_nm, r_eff, 
                                           d, # Varying decenter
                                           fa, 0x0)
        eff_decent.append(eff)
        
    ax_decent.plot(decent_array, eff_decent, lw=2, color='#d62728')
    ax_decent.axvline(0.5, color='gray', linestyle='--', label="Fiber Radius boundary (approx 0.5\")")
    ax_decent.set_xlabel('Fiber Target Decenter [arcsec]')
    ax_decent.set_ylabel('Fiber Geometric Throughput [0-1]')
    ax_decent.set_title('Efficiency vs. Pointing Decenter Error')
    ax_decent.grid(True, linestyle=':', alpha=0.7)
    ax_decent.legend(fontsize=10)
    
    # -------------------------------------------------------------
    # Plot 3: Efficiency vs Wavelength
    # -------------------------------------------------------------
    lambda_array = np.linspace(350, 950, 50)
    eff_lambda = []
    
    for lam in lambda_array:
        eff = ETC_py.gsGeometricThroughput(sp, obs, lam, r_eff, 
                                           0.03, # Standard small decenter
                                           fa, 0x0)
        eff_lambda.append(eff)
        
    ax_lambda.plot(lambda_array, eff_lambda, lw=2, color='#2ca02c')
    ax_lambda.set_xlabel('Wavelength [nm]')
    ax_lambda.set_ylabel('Fiber Geometric Throughput [0-1]')
    ax_lambda.set_title('Efficiency vs. Wavelength \n(due to diffractive & seeing wavelength scaling)')
    ax_lambda.grid(True, linestyle=':', alpha=0.7)
    
    # -------------------------------------------------------------
    # Plot 4: Real-space intuitive visualization (Encircled Energy curve)
    # -------------------------------------------------------------
    from copy import deepcopy
    sp_mod = deepcopy(sp)
    
    # Start closer to 0 to capture the core properly
    aperture_array = np.linspace(0.01, 2.5, 100)
    EE_curve = []
    
    EFL_val = sp_mod.EFL[0]
    ARCSEC_PER_URAD = 0.2062648
    
    for ap in aperture_array:
        sp_mod.fiber_ent_rad = (ap / ARCSEC_PER_URAD) * EFL_val
        eff = ETC_py.gsGeometricThroughput(sp_mod, obs, 600.0, 0.0, 0.0, fa, 0x0)
        EE_curve.append(eff)
        
    nominal_fiber_arcsec = (sp.fiber_ent_rad / EFL_val) * ARCSEC_PER_URAD
    
    ax_psf.plot(aperture_array, EE_curve, lw=2, color='purple', label='Encircled Energy (Total Fraction)')
    ax_psf.axvline(nominal_fiber_arcsec, color='orange', lw=2, linestyle='--', label=f'Nominal JUST Fiber (~{nominal_fiber_arcsec:.2f}")')
    
    # True Radial Surface Brightness PSF(r)
    # Since EE(r) is the integral of 2*pi*r * PSF(r) dr,
    # then PSF(r) ~ (1 / r) * d(EE) / dr
    EE_deriv = np.gradient(EE_curve, aperture_array)
    true_psf = EE_deriv / aperture_array
    
    # Normalize for plotting aesthetics (scale to match y-axis 0-1)
    true_psf_plot = true_psf / np.max(true_psf) * 0.95
    
    ax_psf.plot(aperture_array, true_psf_plot, color='gray', linestyle='-', alpha=0.5, label="True Surface Brightness PSF(r)")
    ax_psf.fill_between(aperture_array, 0, true_psf_plot, where=(aperture_array < nominal_fiber_arcsec), color='orange', alpha=0.2)
    
    ax_psf.set_xlabel('Integration Radius (Aperture) [arcsec]')
    ax_psf.set_ylabel('Encircled Energy Fraction [0-1]')
    ax_psf.set_title('Spatial Encircled Energy & Spillage')
    ax_psf.grid(True, linestyle=':', alpha=0.7)
    ax_psf.legend(fontsize=9, loc='lower right')
    
    plt.tight_layout()
    output_png = output_dir / "fiber_injection_analysis.png"
    plt.savefig(output_png, dpi=300)
    plt.close()
    
    print(f"Analysis plot generated: {output_png}")

if __name__ == '__main__':
    plot_fiber_injection_analysis()
