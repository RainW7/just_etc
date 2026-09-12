
from just_etc.resources import DATA_DIR
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add local logic to path
from just_etc import JUSTExposureTimeCalculator, load_template, normalize_to_mag

def plot_fiber_snr_comparison():
    base_dir = Path.cwd()
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Load template and normalize to r = 20 AB mag
    tpl_path = DATA_DIR / "templates" / "galaxy" / "elliptical_001.fits"
    wave_aa, flux_flam = load_template(tpl_path)
    flux_norm, _ = normalize_to_mag(wave_aa, flux_flam, target_mag=20.0, band='r')
    
    # Initialize ETC
    etc = JUSTExposureTimeCalculator(calc_mode='fast')
    
    EFL_val = etc._spectro.EFL[0]
    ARCSEC_PER_URAD = 0.206264806247097
    
    # Define fiber parameters
    r1_arcsec = 1.36 / 2.0
    r2_arcsec = 1.5 / 2.0
    fiber_ent_rad1 = (r1_arcsec / ARCSEC_PER_URAD) * EFL_val
    fiber_ent_rad2 = (r2_arcsec / ARCSEC_PER_URAD) * EFL_val
    
    # Define observation scenarios
    t_exp = 1200.0
    n_exp = 3 # Total 1 hour
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax_pt = axes[0]
    ax_ext = axes[1]
    
    def get_snr_curves(r_eff, fiber_radius_val):
        etc.set_obs_conditions(r_eff=r_eff, seeing_fwhm_800=0.8)
        etc._spectro.fiber_ent_rad = fiber_radius_val
        result = etc.compute_snr(wave_aa, flux_norm, t_exp, n_exp)
        
        all_wave = []
        all_snr = []
        for res in result:
            all_wave.append(res['wave_nm'])
            all_snr.append(res['snr'])
            
        return np.concatenate(all_wave), np.concatenate(all_snr)

    print("Computing SNR for point source (r_eff = 0.0) ...")
    w1_pt, snr1_pt = get_snr_curves(0.0, fiber_ent_rad1)
    w2_pt, snr2_pt = get_snr_curves(0.0, fiber_ent_rad2)
    
    print("Computing SNR for extended source (r_eff = 0.5\") ...")
    w1_ext, snr1_ext = get_snr_curves(0.5, fiber_ent_rad1)
    w2_ext, snr2_ext = get_snr_curves(0.5, fiber_ent_rad2)
    
    # Apply slight smoothing for visualization (pixel-to-pixel noise variation can be jagged)
    from scipy.ndimage import gaussian_filter1d
    smooth_sigma = 5
    
    snr1_pt_s = gaussian_filter1d(snr1_pt, smooth_sigma)
    snr2_pt_s = gaussian_filter1d(snr2_pt, smooth_sigma)
    snr1_ext_s = gaussian_filter1d(snr1_ext, smooth_sigma)
    snr2_ext_s = gaussian_filter1d(snr2_ext, smooth_sigma)
    
    # Plot Point Source
    ax_pt.plot(w1_pt, snr1_pt_s, lw=1.5, color='blue', label='1.36" Fiber')
    ax_pt.plot(w2_pt, snr2_pt_s, lw=1.5, color='red', label='1.5" Fiber')
    ax_pt.set_title('Continuum SNR vs Wavelength — Point Source (r_eff = 0.0")\n1 Hour Exposure (3x20m), Elliptical Galaxy (r=20 AB mag), Seeing=0.8"')
    ax_pt.set_ylabel('Signal-to-Noise Ratio (per pixel)')
    ax_pt.grid(True, linestyle=':', alpha=0.7)
    ax_pt.legend()
    
    # Plot Extended Source
    ax_ext.plot(w1_ext, snr1_ext_s, lw=1.5, color='blue', label='1.36" Fiber')
    ax_ext.plot(w2_ext, snr2_ext_s, lw=1.5, color='red', label='1.5" Fiber')
    ax_ext.set_title('Continuum SNR vs Wavelength — Extended Source (r_eff = 0.5")')
    ax_ext.set_xlabel('Wavelength [nm]')
    ax_ext.set_ylabel('Signal-to-Noise Ratio (per pixel)')
    ax_ext.grid(True, linestyle=':', alpha=0.7)
    ax_ext.legend()
    
    plt.tight_layout()
    output_png = output_dir / "fiber_snr_comparison.png"
    plt.savefig(output_png, dpi=300)
    plt.close()
    
    print(f"\nAnalysis plot generated: {output_png}")
    
    # Calculate representative stats (median over the V-band region, ~500-600 nm)
    mask = (w1_pt > 500) & (w1_pt < 600)
    print("\n--- Median SNR Comparison (500 - 600 nm) ---")
    print("Point Source:")
    print(f"  1.36\" Fiber: {np.median(snr1_pt[mask]):.3f}")
    print(f"  1.5\" Fiber: {np.median(snr2_pt[mask]):.3f}")
    pt_ratio = np.median(snr2_pt[mask]) / np.median(snr1_pt[mask])
    print(f"  Relative SNR: {pt_ratio:.3f}x ({(pt_ratio-1)*100:.1f}%)")
    
    print("Extended Source (r_eff=0.5\"):")
    print(f"  1.36\" Fiber: {np.median(snr1_ext[mask]):.3f}")
    print(f"  1.5\" Fiber: {np.median(snr2_ext[mask]):.3f}")
    ext_ratio = np.median(snr2_ext[mask]) / np.median(snr1_ext[mask])
    print(f"  Relative SNR: {ext_ratio:.3f}x ({(ext_ratio-1)*100:.1f}%)")

if __name__ == '__main__':
    plot_fiber_snr_comparison()
