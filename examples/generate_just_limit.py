
from just_etc.resources import DATA_DIR
import os
import sys
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch
from pathlib import Path

# ETC_py_v1 — all modules co-located in this directory
V1_DIR = Path.cwd()

from just_etc import ETC_py_optimized as ETC_py

# Configuration
OUTPUT_DIR = V1_DIR / "plot_data"
OUTPUT_DIR.mkdir(exist_ok=True)
SPEC_FILE = DATA_DIR / "spec.dat"

def run_etc_instance(mode, t_exp, n_exp, filename_prefix, 
                    do_continuum=False, do_line=False):
    """
    Run ETC with specific parameters.
    """
    # Set Mode
    if hasattr(ETC_py, 'set_calculation_mode'):
        ETC_py.set_calculation_mode(mode)
    
    # Define Inputs
    inputs = [
        str(SPEC_FILE),          # Config file
        "10003",                 # SkyType (New Moon)
        "0.8",                   # Seeing
        "45.0",                  # Zenith Angle
        "0.03",                  # EBV
        "0.675",                 # Field Angle
        "0.03",                  # Decenter
        str(t_exp),              # Time per exposure
        str(int(n_exp)),         # Number of exposures
        "0.01",                  # Systematics (1%)
        "0.02",                  # Diffuse Stray (2%)
        "0",                     # Reuse Noise?
        str(OUTPUT_DIR / f"{filename_prefix}_noise.txt"), # Noise File
        "-",                     # ELG File (Skip)
    ]
    
    # Single Line Inputs
    if do_line:
        inputs.append(str(OUTPUT_DIR / f"{filename_prefix}_line_snr.txt"))
        inputs.append("1.0e-17") # Flux
        inputs.append("70.0")    # Sigma
    else:
        inputs.append("-")
        inputs.append("1.0e-17") # Dummy
        inputs.append("70.0")    # Dummy

    # Continuum Inputs
    if do_continuum:
        inputs.append(str(OUTPUT_DIR / f"{filename_prefix}_cont_snr.txt"))
    else:
        inputs.append("-")
        
    # OII Inputs
    inputs.append("-") # Skip OII catalog
    
    # Mag File
    inputs.append("-") # Skip Mag file

    # Input iterator
    input_iter = iter(inputs)
    
    def mocked_input(prompt=""):
        try:
            val = next(input_iter)
            return val
        except StopIteration:
            return ""

    print(f"--- Running ETC: {filename_prefix} (Exp={t_exp}s x {n_exp}) ---")
    with patch('builtins.input', side_effect=mocked_input):
        ETC_py.main()
    print("--- Done ---\n")

def generate_plots():
    # Style configuration
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'legend.frameon': False
    })

    # --- Run 1: Continuum (8 x 450s) ---
    run_etc_instance(mode='accurate', t_exp=450.0, n_exp=8, 
                     filename_prefix="run1_cont", do_continuum=True)
    
    # --- Run 2: Single Line (2 x 450s = 15 min) ---
    run_etc_instance(mode='accurate', t_exp=450.0, n_exp=2, 
                     filename_prefix="run2_line", do_line=True)

    # Setup 1x2 Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Colors for Arms (Blue, Red, Orange)
    arm_colors = ['#1f77b4', '#d62728', '#ff7f0e']

    # --- LEFT PLOT: Continuum Limiting Magnitude ---
    cont_file = OUTPUT_DIR / "run1_cont_cont_snr.txt"
    if cont_file.exists():
        data = np.loadtxt(cont_file)
        mask = data[:, 3] > 0
        
        arms = np.unique(data[mask, 0])
        print(f"Detected Arms in Continuum Data: {arms}")
        
        arm_stats = []
        
        for arm in arms:
            arm_mask = (data[:, 0] == arm) & (data[:, 3] > 0)
            waves = data[arm_mask, 2]
            snrs = data[arm_mask, 3]
            
            # 3-pixel binning correction
            snrs_binned = snrs * np.sqrt(3.0)
            
            # Limiting Mag (SNR=5)
            # Avoid log10(0)
            valid_snr = snrs_binned > 1e-9
            if not np.any(valid_snr):
                print(f"Arm {arm}: No valid SNR.")
                continue
                
            lim_mags = 22.5 + 2.5 * np.log10(snrs_binned[valid_snr] / 5.0)
            waves = waves[valid_snr]
            
            # Edge Trimming (2%)
            n_pts = len(waves)
            crop = int(n_pts * 0.02)
            if n_pts > 3 * crop: # Ensure enough points
                waves = waves[crop:-crop]
                lim_mags = lim_mags[crop:-crop]
            
            if len(waves) == 0:
                print(f"Arm {arm}: Empty after trim.")
                continue

            # Store arm stats for annotation
            arm_stats.append(f"Arm {int(arm)}: Mean={np.nanmean(lim_mags):.2f}, Best={np.nanmax(lim_mags):.2f}")

            color = arm_colors[int(arm)] if int(arm) < len(arm_colors) else 'k'
            ax1.plot(waves, lim_mags, color=color, lw=1.0, alpha=0.9)
            
            # 3 Segments per Arm Strategy
            w_min, w_max = waves[0], waves[-1]
            total_width = w_max - w_min
            seg_width = total_width / 3.0
            
            for i_seg in range(3):
                w_start = w_min + i_seg * seg_width
                w_end = w_start + seg_width
                # Ensure last segment reaches end
                if i_seg == 2: w_end = w_max + 0.001

                win_mask = (waves >= w_start) & (waves < w_end)
                if not np.any(win_mask): continue
                
                win_vals = lim_mags[win_mask]
                win_ws = waves[win_mask]
                
                # --- ROBUST MEAN CALCULATION ---
                robust_mask = np.ones(len(win_vals), dtype=bool)
                
                if len(win_vals) > 5:
                    best_in_win = np.nanpercentile(win_vals, 95)
                    # Keep values deeper (larger) than best - 0.4 (Strict filter to trim edges)
                    robust_mask = win_vals > (best_in_win - 0.4)
                    
                    # USER REQUEST: Preserve blue end of Arm 0
                    if int(arm) == 0 and i_seg == 0:
                        robust_mask[:] = True
                    
                    if np.sum(robust_mask) == 0:
                        # Fallback if aggressive filter removes all
                        robust_mask = np.ones(len(win_vals), dtype=bool)

                mean_val = np.nanmean(win_vals[robust_mask])
                
                # Determine actual width based on used data
                used_ws = win_ws[robust_mask]
                bar_w_start = np.min(used_ws)
                bar_w_end = np.max(used_ws)
                
                # Use local envelope (percentile) around the center
                best_wave = (w_start + w_end) / 2.0
                
                # Local window +/- 10 nm around center (for marker Y-pos)
                local_mask = (waves >= best_wave - 10.0) & (waves <= best_wave + 10.0)
                if np.any(local_mask):
                    local_vals = lim_mags[local_mask]
                    best_val = np.nanpercentile(local_vals, 95)
                else:
                    sort_idx = np.argsort(waves)
                    best_val = np.interp(best_wave, waves[sort_idx], lim_mags[sort_idx])

                # Gray bar (Using ACTUAL used range)
                ax1.fill_between([bar_w_start, bar_w_end], 
                                [mean_val-0.05, mean_val-0.05], 
                                [mean_val+0.05, mean_val+0.05], 
                                color='gray', alpha=0.5, edgecolor=None)
                
                # Open circle
                ax1.plot(best_wave, best_val, 'o', markerfacecolor='white', 
                        markeredgecolor='black', markersize=6, markeredgewidth=1.2, zorder=10)

        # Annotate Arm Stats
        stats_text = "\n".join(arm_stats)
        ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Limiting magnitude for continuum [AB mag]')
        ax1.set_title('3600 sec. exposure, S/N=5, 3 pix binning')
        ax1.set_ylim(24, 16) 
        ax1.set_xlim(350, 1000)
        ax1.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)

    # --- RIGHT PLOT: Single Line Limiting Flux ---
    line_file = OUTPUT_DIR / "run2_line_line_snr.txt"
    if line_file.exists():
        data = np.loadtxt(line_file)
        waves = data[:, 0]
        
        flux_stats = []
        
        # Conversion constant for Mag
        OFFSET = -21.25
        def flux_to_mag_val(f):
            if f <= 0: return 99.99
            return -2.5 * np.log10(f) + OFFSET

        # Disable top ticks for primary axis to avoid clutter with Redshift axis
        ax2.tick_params(top=False)

        for i_arm in range(3):
            if 3 + i_arm < data.shape[1]:
                snr_arm = data[:, 3 + i_arm]
                valid_mask = snr_arm > 1e-9
                
                if np.any(valid_mask):
                    w_arm = waves[valid_mask]
                    s_arm = snr_arm[valid_mask]
                    
                    lim_flux = 1.0e-17 * (6.0 / s_arm)
                    
                    n_pts = len(w_arm)
                    crop = int(n_pts * 0.02)
                    if n_pts > 3 * crop:
                        w_arm = w_arm[crop:-crop]
                        lim_flux = lim_flux[crop:-crop]
                    
                    if len(w_arm) == 0:
                        continue

                    # Store stats with Mag
                    mean_f = np.nanmean(lim_flux)
                    best_f = np.nanmin(lim_flux)
                    mean_m = flux_to_mag_val(mean_f)
                    best_m = flux_to_mag_val(best_f)
                    
                    flux_stats.append(f"Arm {i_arm}: Mean={mean_f:.2e} (~{mean_m:.2f} mag), Best={best_f:.2e} (~{best_m:.2f} mag)")
                    
                    ax2.plot(w_arm, lim_flux, color=arm_colors[i_arm], lw=1.0, alpha=0.9)
                    
                    # 3 Segments per Arm Strategy
                    w_min, w_max = w_arm[0], w_arm[-1]
                    total_width = w_max - w_min
                    seg_width = total_width / 3.0
                    
                    for i_seg in range(3):
                        w_start = w_min + i_seg * seg_width
                        w_end = w_start + seg_width
                        if i_seg == 2: w_end = w_max + 0.001
                        
                        win_mask = (w_arm >= w_start) & (w_arm < w_end)
                        if not np.any(win_mask): continue
                        
                        win_vals = lim_flux[win_mask]
                        win_ws = w_arm[win_mask]
                        
                        # --- ROBUST MEAN CALCULATION (Flux) ---
                        robust_mask = np.ones(len(win_vals), dtype=bool)

                        if len(win_vals) > 5:
                            best_in_win = np.nanpercentile(win_vals, 5) # Lowest flux
                            # Keep values within factor of 1.5 of the best (Stricter filter)
                            robust_mask = win_vals < (best_in_win * 1.5)
                            
                            # USER REQUEST: Preserve blue end of Arm 0
                            if i_arm == 0 and i_seg == 0:
                                robust_mask[:] = True
                            
                            if np.sum(robust_mask) == 0:
                                robust_mask = np.ones(len(win_vals), dtype=bool)

                        mean_val = np.nanmean(win_vals[robust_mask])
                        
                        # Determine actual width
                        used_ws = win_ws[robust_mask]
                        bar_w_start = np.min(used_ws)
                        bar_w_end = np.max(used_ws)
                        
                        # Use local envelope (percentile) around the center
                        best_wave = (w_start + w_end) / 2.0
                        
                        # Local window +/- 10 nm around center
                        local_mask = (w_arm >= best_wave - 10.0) & (w_arm <= best_wave + 10.0)
                        if np.any(local_mask):
                            local_vals = lim_flux[local_mask]
                            best_val = np.nanpercentile(local_vals, 5)
                        else:
                            sort_idx = np.argsort(w_arm)
                            best_val = np.interp(best_wave, w_arm[sort_idx], lim_flux[sort_idx])
                        
                        ax2.fill_between([bar_w_start, bar_w_end], 
                                        [mean_val * 0.95, mean_val * 0.95], 
                                        [mean_val * 1.05, mean_val * 1.05],
                                        color='gray', alpha=0.5, edgecolor=None)
                        
                        ax2.plot(best_wave, best_val, 'o', markerfacecolor='white', 
                                markeredgecolor='black', markersize=6, markeredgewidth=1.2, zorder=10)
        
        # Annotate Flux Stats
        stats_text = "\n".join(flux_stats)
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Wavelength [nm]')
        ax2.set_ylabel(r'Limiting flux of a single emission line [erg/s/cm$^2$]')
        
        ax2.text(0.1, 0.1, '450 sec. $\\times$ 2 exposure, S/N=6', transform=ax2.transAxes, fontsize=14)
        
        ax2.set_yscale('log')
        ax2.set_ylim(1e-17, 8e-16) 
        ax2.set_xlim(350, 1000)
        
        ax2.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Redshift Axis ([OII] 372.7nm)
        lambda_oii = 372.7
        def z_to_wave(z): return lambda_oii * (1 + z)
        def wave_to_z(w): return (w / lambda_oii) - 1
            
        secax = ax2.secondary_xaxis('top', functions=(wave_to_z, z_to_wave))
        secax.set_xlabel('Redshift')

        # --- Secondary Y-Axis: Approx AB Magnitude ---
        # Conversion assumes reference wavelength 800nm and sigma=70km/s
        # Delta_nu = nu * (sigma/c) = (c/lambda) * (sigma/c) = sigma / lambda
        # sigma = 7e6 cm/s, lambda = 8e-5 cm -> Delta_nu ~ 8.75e10 Hz
        # m = -2.5 log(F / Delta_nu) - 48.60
        # m = -2.5 log(F) + 2.5 log(Delta_nu) - 48.60
        # Constant C = 2.5 * log10(8.75e10) - 48.60 = 27.35 - 48.60 = -21.25
        # m = -2.5 log10(F) - 21.25
        
        # OFFSET defined earlier (-21.25)
        def flux_to_mag(f):
             # Handle 0 or negative flux for log
             f = np.ma.masked_less_equal(f, 0)
             return -2.5 * np.log10(f) + OFFSET

        def mag_to_flux(m):
             return 10**((m - OFFSET) / -2.5)

        import matplotlib.ticker as ticker
        
        secax_y = ax2.secondary_yaxis('right', functions=(flux_to_mag, mag_to_flux))
        secax_y.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        secax_y.set_ylabel('Approx. AB Mag (70km/s@800nm), zp=21.25')
        # Force inverted ticks (higher mag = lower flux) usually happens naturally with log flux
        # But we need to check visual orientation. 
        # Flux 1e-17 -> Mag 21.25. Flux 8e-16 -> Mag ~16.5.
        # Log axis goes Low->High (1e-17 -> 1e-16). So Mag axis will go 21 -> 16 (Inverted).
        # This matches standard mag conventions.

        # Ensure primary top ticks are OFF (to avoid clash with Redshift axis)
        # MUST disable 'both' (major and minor) because minor ticks are globally enabled
        ax2.tick_params(axis='x', which='both', top=False)
        ax2.tick_params(axis='x', which='both', labeltop=False) # Redundant safety

    plt.tight_layout()
    fig.suptitle('JUST (trimmed throughput wavelength range)', fontsize=20, y=1.05)
    out_png = OUTPUT_DIR / "limiting_mag_just_trimmed.png"
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved {out_png}")
    plt.close(fig)
    

if __name__ == "__main__":
    generate_plots()
