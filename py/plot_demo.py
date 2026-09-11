import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from just_etc_api import JUSTExposureTimeCalculator, load_template, normalize_to_mag

def generate_demo_plots(model_path, z, target_mag, n_exp, t_exp):
    # Setup paths
    base_dir = Path(__file__).resolve().parent
    template_path = base_dir / "templates" / model_path
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 1. Load and prepare spectrum
    print(f"Loading template: {model_path}")
    wave_aa, flux_flam = load_template(template_path)
    wave_aa_z = wave_aa*(1+z)
    
    # Normalize to r=20 AB mag to clearly see noise effects
    print(f"Normalizing to z={target_mag} AB mag")
    flux_norm, _ = normalize_to_mag(wave_aa_z, flux_flam, target_mag=target_mag, band='z')
    
    # 2. Setup ETC and compute
    print("Initializing ETC in 'balanced' mode")
    etc = JUSTExposureTimeCalculator(calc_mode='fast')

    print(f"Computing SNR for {n_exp} exposures of {t_exp}s each (in total of {n_exp * t_exp} sec)...")
    results = etc.compute_snr(wave_aa_z, flux_norm, t_exp=t_exp, n_exp=n_exp)
    
    # 3. Plotting
    print("Generating plot...")
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.subplots_adjust(hspace=0.05)
    
    ax_spec, ax_snr, ax_counts = axes
    
    arm_colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    # Overlay Common Emission Lines
    emission_lines = {
        r'Ly$\alpha$': 1215.67,
        r'[O II]': 3727.0,
        r'H$\beta$': 4861.3,
        r'[O III]': 5006.8,
        r'H$\alpha$': 6562.8,
    }
    for name, w_rest in emission_lines.items():
        w_obs_nm = w_rest * (1 + z) / 10.0
        if 350 <= w_obs_nm <= 950:
            for ax in axes:
                ax.axvline(w_obs_nm, color='gray', linestyle=':', alpha=0.7, lw=1.0, zorder=0)
            # Label on top panel
            ax_spec.text(w_obs_nm, 0.1, name, transform=ax_spec.get_xaxis_transform(),
                         rotation=90, color='dimgray', fontsize=20,
                         va='bottom', ha='center')
    
    # Plot input spectrum on Top panel
    ax_spec.plot(wave_aa_z / 10.0, flux_norm, color='black', lw=1.0)
    ax_spec.set_ylabel(r'Flux [erg s$^{-1}$ cm$^{-2}$ $\rm{\AA}^{-1}$]',fontsize=25)
    ax_spec.set_yscale('log')
    ax_spec.set_title(f'{model_path} (r={target_mag} AB mag at z={z}) - {n_exp}x{t_exp}s exposures',fontsize=25)
    
    # Plot SNR on Middle panel
    for arm_res in results:
        arm_idx = arm_res['arm']
        wave_nm = arm_res['wave_nm']
        snr = arm_res['snr']
        
        # Smooth SNR slightly for visualization
        window = 5
        snr_smooth = np.convolve(snr, np.ones(window)/window, mode='same')
        
        color = arm_colors[arm_idx] if arm_idx < len(arm_colors) else 'gray'
        ax_snr.plot(wave_nm, snr, color=color, alpha=0.3, lw=0.5)
        ax_snr.plot(wave_nm, snr_smooth, color=color, label=f'Arm {arm_idx}')
        
    ax_snr.axhline(5, color='gray', linestyle='--', alpha=0.8, label='S/N=5')
    ax_snr.set_ylabel('Total S/N per pixel')
    ax_snr.legend(loc='upper right', frameon=False)
    
    # Plot Signal and Noise on Bottom panel
    for arm_res in results:
        arm_idx = arm_res['arm']
        wave_nm = arm_res['wave_nm']
        signal = arm_res['signal'] * n_exp # Total signal
        noise = np.sqrt(arm_res['noise_var'] * n_exp) # Total noise
        
        color = arm_colors[arm_idx] if arm_idx < len(arm_colors) else 'gray'
        
        # Plot signal
        ax_counts.plot(wave_nm, signal, color=color, lw=1.2, label=f'Signal (Arm {arm_idx})')
        # Plot total noise
        ax_counts.plot(wave_nm, noise, color=color, lw=1.0, linestyle=':', alpha=0.8, label=f'Total Noise')
        
    ax_counts.set_ylabel('Total Counts [e-/pixel]',fontsize=25)
    ax_counts.set_xlabel('Wavelength [nm]',fontsize=25)
    ax_counts.set_yscale('log')
    # Use unique labels for the legend
    handles, labels = ax_counts.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_counts.legend(by_label.values(), by_label.keys(), loc='upper left', frameon=False, ncol=3, fontsize=10)
    
    # Limit x axis
    ax_counts.set_xlim(350, 950)
    
    output_png = output_dir / "etc_demo_plot.png"
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_png}")

if __name__ == "__main__":
    generate_demo_plots(model_path = "galaxy/elliptical_001.fits", z = 0.3, target_mag = 20, t_exp = 900, n_exp = 1)
