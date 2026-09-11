
from just_etc.resources import DATA_DIR
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from just_etc import JUSTExposureTimeCalculator, load_template, normalize_to_mag

def setup_plot_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })

def plot_seeing_impact(model_path, z, target_mag, t_exp, n_exp):
    """
    Explore the impact of different atmospheric seeing conditions on the final SNR.
    Fixes the target as a point source (r_eff = 0.0) or compact object to isolate seeing effects.
    """
    print("\n--- Running Seeing Impact Analysis ---")
    base_dir = Path.cwd()
    template_path = DATA_DIR / "templates" / model_path
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    wave_aa, flux_flam = load_template(template_path)
    wave_aa_z = wave_aa * (1 + z)
    flux_norm, _ = normalize_to_mag(wave_aa_z, flux_flam, target_mag=target_mag, band='r')
    
    # Typical seeing values at astronomical sites (e.g., Lenghu)
    # 0.6": Excellent, 0.8": Median/Good, 1.2": Poor, 1.5": Very Poor
    seeing_values = [0.6, 0.8, 1.2, 1.5]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    etc = JUSTExposureTimeCalculator(calc_mode='fast')
    
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for seeing, color in zip(seeing_values, colors):
        print(f"Computing for Seeing FWHM = {seeing}\"")
        # Set seeing, fix r_eff to 0 (point source)
        etc.set_obs_conditions(seeing_fwhm_800=seeing, r_eff=0.0)
        results = etc.compute_snr(wave_aa_z, flux_norm, t_exp=t_exp, n_exp=n_exp)
        
        # Plot each arm
        for arm_res in results:
            wave_nm = arm_res['wave_nm']
            snr = arm_res['snr']
            
            # Smooth SNR slightly for visualization
            window = 11
            snr_smooth = np.convolve(snr, np.ones(window)/window, mode='same')
            
            # Only add label for the first arm to avoid legend duplication
            label = f'Seeing {seeing}"' if arm_res['arm'] == 0 else ""
            ax.plot(wave_nm, snr_smooth, color=color, label=label, lw=1.5)

    ax.axhline(5, color='gray', linestyle='--', alpha=0.8, label='S/N=5 Limit')
    ax.set_ylabel('Total S/N per pixel')
    ax.set_xlabel('Wavelength [nm]')
    ax.set_title(f'Impact of Atmospheric Seeing (Target: r={target_mag} AB, Point Source)')
    ax.set_xlim(350, 950)
    ax.legend(loc='upper right', frameon=False)
    
    output_png = output_dir / "etc_seeing_impact.png"
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved seeing impact plot to: {output_png}")

def plot_reff_impact(model_path, z, target_mag, t_exp, n_exp):
    """
    Explore the impact of different galaxy effective radii (r_eff) on the final SNR.
    Fixes the seeing at a median value (e.g., 0.8").
    """
    print("\n--- Running Galaxy Effective Radius Analysis ---")
    base_dir = Path.cwd()
    template_path = DATA_DIR / "templates" / model_path
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    wave_aa, flux_flam = load_template(template_path)
    wave_aa_z = wave_aa * (1 + z)
    flux_norm, _ = normalize_to_mag(wave_aa_z, flux_flam, target_mag=target_mag, band='r')
    
    # Typical effective radii (Re) for different types of targets:
    # 0.0" : Point source (Star, Quasar)
    # 0.3" : Compact galaxy / High-z Lyman Break Galaxy (LBG)
    # 1.5" : Typical Spiral / Disk Galaxy
    # 3.0" : Large Elliptical Galaxy / Extended emission
    reff_values = [
        (0.0, "Point Source (Star/QSO, Re=0.0\")"),
        (0.3, "Compact / High-z LBG (Re=0.3\")"),
        (1.5, "Spiral / Disk Galaxy (Re=1.5\")"),
        (3.0, "Large Elliptical (Re=3.0\")")
    ]
    colors = ['#9467bd', '#1f77b4', '#2ca02c', '#d62728']
    
    etc = JUSTExposureTimeCalculator(calc_mode='fast')
    
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for (reff, label_name), color in zip(reff_values, colors):
        print(f"Computing for {label_name}")
        # Fix seeing to 0.8", vary r_eff
        etc.set_obs_conditions(seeing_fwhm_800=0.8, r_eff=reff)
        results = etc.compute_snr(wave_aa_z, flux_norm, t_exp=t_exp, n_exp=n_exp)
        
        for arm_res in results:
            wave_nm = arm_res['wave_nm']
            snr = arm_res['snr']
            
            # Smooth SNR
            window = 11
            snr_smooth = np.convolve(snr, np.ones(window)/window, mode='same')
            
            label = label_name if arm_res['arm'] == 0 else ""
            ax.plot(wave_nm, snr_smooth, color=color, label=label, lw=1.5)

    ax.axhline(5, color='gray', linestyle='--', alpha=0.8, label='S/N=5 Limit')
    ax.set_ylabel('Total S/N per pixel')
    ax.set_xlabel('Wavelength [nm]')
    ax.set_title(f'Impact of Galaxy Effective Radius (Target: r={target_mag} AB, Seeing=0.8")')
    ax.set_xlim(350, 950)
    ax.legend(loc='upper right', frameon=False)
    
    output_png = output_dir / "etc_reff_impact.png"
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Reff impact plot to: {output_png}")

if __name__ == "__main__":
    # Choose a starburst galaxy template for demonstration
    test_model = "galaxy/sb2_b2004a_001.fits"
    test_z = 0.6
    target_mag = 20.5
    t_exp = 2500
    n_exp = 1
    
    plot_seeing_impact(test_model, test_z, target_mag, t_exp, n_exp)
    plot_reff_impact(test_model, test_z, target_mag, t_exp, n_exp)
