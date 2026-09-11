import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the local directory to sys.path so we can import ETC modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ETC_py_optimized as ETC_py
from just_etc_api import JUSTExposureTimeCalculator, load_template, normalize_to_mag

# Global plot styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

def plot_ideal_efficiency(output_dir):
    """
    Part 1: Plot the total end-to-end efficiency WITHOUT considering fiber geometric loss.
    (i.e., assuming 100% of the light enters the fiber).
    """
    print("\n--- Part 1: Calculating Ideal System Efficiency (No Fiber Loss) ---")
    etc = JUSTExposureTimeCalculator(calc_mode='fast')
    obs = etc._make_obs()
    sp = etc._spectro
    fa = etc._obs_params['field_angle']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for ia in range(sp.N_arms):
        Npix = sp.npix[ia]
        wave_nm = sp.lmin[ia] + sp.dl[ia] * (np.arange(Npix) + 0.5)
        
        eff_ext = np.zeros(Npix)
        eff_atm = np.zeros(Npix)
        eff_vig = np.zeros(Npix)
        eff_inst = np.zeros(Npix)
        eff_trace = np.zeros(Npix)
        
        for ipix in range(Npix):
            lam = wave_nm[ipix]
            # 1. Extinction
            eff_ext[ipix] = 10.0 ** (-0.4 * ETC_py.gsGalactic_Alambda__EBV(lam) * obs.EBV)
            # 2. Atmosphere
            eff_atm[ipix] = ETC_py.gsAtmTrans(obs, lam, 0x0)
            # 3. Vignetting
            rfov = sp.rfov
            i = int(np.floor(4 * fa / rfov))
            if i < 0: Vig = sp.vignette[0]
            elif i >= 4: Vig = sp.vignette[4]
            else:
                frac = 4 * fa / rfov - i
                Vig = sp.vignette[i] + (sp.vignette[i+1] - sp.vignette[i]) * frac
            eff_vig[ipix] = Vig
            # 4. Instrument Throughput
            imin = sp.istart[ia]
            imax = sp.istart[ia + 1]
            if lam <= sp.l[imin]: Thr = sp.T[imin]
            elif lam >= sp.l[imax - 1]: Thr = sp.T[imax - 1]
            else:
                ti = imin
                while lam > sp.l[ti + 1]: ti += 1
                fr = (lam - sp.l[ti]) / (sp.l[ti+1] - sp.l[ti])
                Thr = sp.T[ti] + (sp.T[ti+1] - sp.T[ti]) * fr
            eff_inst[ipix] = Thr
            # 5. Trace Fraction
            eff_trace[ipix] = ETC_py.gsFracTrace(sp, obs, ia, lam, 0)
        
        # KEY DIFFERENCE: eff_geo is forced to 1.0
        cum_eff = eff_ext * eff_atm * eff_vig * eff_inst * eff_trace
        
        ax.plot(wave_nm, cum_eff, color=colors[ia], lw=2, label=f"Arm {ia}")
        ax.fill_between(wave_nm, 0, cum_eff, color=colors[ia], alpha=0.2)

    ax.set_xlabel("Wavelength [nm]",fontsize=20)
    ax.set_ylabel("Ideal Cumulative Efficiency [0-1]",fontsize=20)
    ax.set_title("Total System Efficiency (Assuming 100% Fiber Throughput)",fontsize=20)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 0.6)
    ax.set_xlim(350, 950)
    
    out_path = output_dir / "ideal_efficiency_no_fiber_loss.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ideal efficiency plot to: {out_path}")


def calculate_fiber_loss(output_dir):
    """
    Part 2: Estimate fiber optical loss for point sources (0") vs extended sources (0.3", 0.8", 1.2").
    Calculated at a typical median seeing of 0.8".
    """
    print("\n--- Part 2: Fiber Loss Estimation (Point vs Extended Sources) ---")
    etc = JUSTExposureTimeCalculator(calc_mode='fast')
    etc.set_obs_conditions(seeing_fwhm_800=0.8) # Fixed median seeing
    obs = etc._make_obs()
    sp = etc._spectro
    de = etc._obs_params['decenter']
    fa = etc._obs_params['field_angle']
    
    lam = 600.0 # Calculate at 600 nm for a representative value
    
    reff_values = [0, 0.3, 0.8, 1.2]
    labels = ["Point Source (0\")", "Extended (0.3\")", "Extended (0.8\")", "Extended (1.2\")"]
    
    print(f"Calculating Fiber Geometric Throughput at 600 nm (Seeing = 0.8\"):")
    
    throughputs = []
    for r_eff in reff_values:
        eff = ETC_py.gsGeometricThroughput(sp, obs, lam, r_eff, de, fa, 0x0)
        throughputs.append(eff)
        print(f"  {r_eff} arcsec Reff : {eff*100:.2f}% (Loss: {(1-eff)*100:.2f}%)")
        
    # Plotting the results as a bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, [eff * 100 for eff in throughputs], color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'])
    ax.set_ylabel("Fiber Geometric Efficiency",fontsize=20)
    ax.set_title("Fiber Geometric Throughput (Seeing = 0.8\" @ 600 nm)",fontsize=20)
    ax.set_ylim(0, 100)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=20)
                    
    out_path = output_dir / "fiber_loss_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved fiber loss bar chart to: {out_path}")


def find_limiting_magnitude(wave_aa_z, flux_flam, etc, t_exp, target_snr, ref_wave_range=(750.0, 920.0), band='z', mag_min=18.0, mag_max=26.0, tol=0.05):
    """
    Binary search to find the AB magnitude that yields exactly the target_snr.
    """
    mag_low = mag_min
    mag_high = mag_max
    
    best_mag = (mag_low + mag_high) / 2.0
    
    for _ in range(15):  # Max 15 iterations is plenty for magnitude binary search
        test_mag = (mag_low + mag_high) / 2.0
        
        flux_norm, _ = normalize_to_mag(wave_aa_z, flux_flam, target_mag=test_mag, band=band)
        results = etc.compute_snr(wave_aa_z, flux_norm, t_exp=t_exp, n_exp=1)
        
        # Find median SNR over the specified ref_wave_range
        snr_list = []
        for arm_res in results:
            wn = arm_res['wave_nm']
            snr_arr = arm_res['snr']
            # Find pixels within the wavelength range
            mask = (wn >= ref_wave_range[0]) & (wn <= ref_wave_range[1])
            if np.any(mask):
                # Smooth slightly to avoid noise spikes heavily biasing the median
                snr_smooth = np.convolve(snr_arr, np.ones(11)/11, mode='same')
                snr_list.extend(snr_smooth[mask])
                
        if len(snr_list) > 0:
            snr_at_ref = float(np.median(snr_list))
        else:
            snr_at_ref = 0.0
        if abs(snr_at_ref - target_snr) / target_snr < tol:
            return test_mag, results
            
        if snr_at_ref > target_snr:
            mag_low = test_mag # Object is too bright, make it fainter
        else:
            mag_high = test_mag # Object is too faint, make it brighter
            
        best_mag = test_mag
        
    # Recalculate one last time to return the results array for best_mag
    flux_norm, _ = normalize_to_mag(wave_aa_z, flux_flam, target_mag=best_mag, band=band)
    results = etc.compute_snr(wave_aa_z, flux_norm, t_exp=t_exp, n_exp=1)
    
    return best_mag, results


def simulate_snr_spectra(output_dir):
    """
    Part 3: Show simulated spectra for 15m, 30m, 60m reaching continuum SNR=3 or 5.
    """
    print("\n--- Part 3: Limiting Magnitude Solver & Simulated Spectra ---")
    base_dir = Path(__file__).resolve().parent
    template_path = base_dir / "templates" / "galaxy" / "elliptical_001.fits"
    
    z = 0.3
    wave_aa, flux_flam = load_template(template_path)
    wave_aa_z = wave_aa * (1 + z)
    
    # Target configurations
    t_exp_list = [900, 1800, 3600] # 15 min, 30 min, 60 min
    t_labels = ["15 min", "30 min", "60 min"]
    target_snrs = [3, 5]
    ref_band_range = (750.0, 920.0) # r-band wavelength range in nm
    
    etc = JUSTExposureTimeCalculator(calc_mode='fast')
    # Set to a typical extended source
    etc.set_obs_conditions(seeing_fwhm_800=0.8, r_eff=0.3)
    
    fig, axes = plt.subplots(len(target_snrs), len(t_exp_list), figsize=(18, 10), sharex=True)
    
    # Colors for arms
    arm_colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for row_idx, snr_target in enumerate(target_snrs):
        for col_idx, t_exp in enumerate(t_exp_list):
            print(f"Solving for t_exp = {t_exp}s, Target SNR = {snr_target}...")
            lim_mag, results = find_limiting_magnitude(wave_aa_z, flux_flam, etc, t_exp, snr_target, ref_wave_range=ref_band_range)
            print(f"  -> Limiting r-band Mag: {lim_mag:.2f}")
            
            ax = axes[row_idx, col_idx]
            
            # Re-normalize just to plot the input spectrum scaled down to visual limits
            flux_norm, _ = normalize_to_mag(wave_aa_z, flux_flam, target_mag=lim_mag, band='r')
            
            # Plot the resulting SNR curve
            for arm_res in results:
                ia = arm_res['arm']
                wn = arm_res['wave_nm']
                snr = arm_res['snr']
                
                # Raw and smoothed SNR
                snr_smooth = np.convolve(snr, np.ones(11)/11, mode='same')
                
                ax.plot(wn, snr, color=arm_colors[ia], alpha=0.3, lw=0.5)
                ax.plot(wn, snr_smooth, color=arm_colors[ia], lw=1.5, label=f"Arm {ia}")
            
            ax.axhline(snr_target, color='gray', linestyle='--', alpha=0.8)
            ax.axvline(ref_band_range[0], color='purple', linestyle=':', alpha=0.5)
            ax.axvline(ref_band_range[1], color='purple', linestyle=':', alpha=0.5)
            
            # Formatting subplot titles
            title_str = ""
            if row_idx == 0:
                title_str += f"{t_labels[col_idx]} Exposure\n"
            title_str += f"$z_{{lim}}$ = {lim_mag:.2f} ({ref_band_range[0]:.0f}-{ref_band_range[1]:.0f}nm)"
            ax.set_title(title_str, fontsize=25, fontweight='bold')
            
            if col_idx == 0:
                ax.set_ylabel("Total S/N per pixel", fontsize=25)
            if row_idx == len(target_snrs) - 1:
                ax.set_xlabel("Wavelength [nm]", fontsize=25)
                
            # Row labels on the right side
            if col_idx == len(t_exp_list) - 1:
                ax2 = ax.twinx()
                ax2.set_ylabel(f"Target SNR = {snr_target}", fontsize=25, fontweight='bold', rotation=270, labelpad=30)
                ax2.set_yticks([])
                
            ax.set_xlim(350, 950)
            ax.set_ylim(0, 15) # Cap Y axis for better view
            
            # Legend only on first panel
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper right', frameon=False)

    plt.tight_layout()
    out_path = output_dir / "simulated_limiting_mag_spectra.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved simulated spectra grid to: {out_path}")

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    plot_ideal_efficiency(output_dir)
    calculate_fiber_loss(output_dir)
    simulate_snr_spectra(output_dir)
