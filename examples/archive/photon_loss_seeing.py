
from just_etc.resources import DATA_DIR
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from just_etc import ETC_py_optimized as ETC_py
from just_etc import JUSTExposureTimeCalculator

def analyze_seeing_photon_loss():
    base_dir = Path.cwd()
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    print("Initializing ETC...")
    etc = JUSTExposureTimeCalculator(calc_mode='fast')
    sp = etc._spectro
    fa = etc._obs_params['field_angle']
    de = etc._obs_params['decenter']
    
    # Point source
    r_eff = 0.0 
    
    seeing_values = [0.6, 0.8, 1.2, 1.5]
    colors_seeing = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    ax_geo, ax_cum = axes
    
    print("Calculating efficiencies for different seeing conditions...")
    
    for seeing, color in zip(seeing_values, colors_seeing):
        print(f" -> Seeing = {seeing}\"")
        
        # Update observation parameters with the new seeing value
        etc.set_obs_conditions(seeing_fwhm_800=seeing)
        obs = etc._make_obs()
        
        # Arrays to accumulate the combined spectrum across all arms for plotting
        all_wave = []
        all_eff_geo = []
        all_cum_eff = []
        
        for ia in range(sp.N_arms):
            Npix = sp.npix[ia]
            lmin = sp.lmin[ia]
            dl = sp.dl[ia]
            wave_nm = lmin + dl * (np.arange(Npix) + 0.5)
            
            eff_ext = np.zeros(Npix)
            eff_atm = np.zeros(Npix)
            eff_vig = np.zeros(Npix)    # Vignetting
            eff_inst = np.zeros(Npix)   # Instrument Transmission
            eff_geo = np.zeros(Npix)    # Fiber injection geometric throughput
            eff_trace = np.zeros(Npix)  # Fraction of trace inside extraction window
            
            for ipix in range(Npix):
                lam = wave_nm[ipix]
                
                # 1. Galactic Extinction
                eff_ext[ipix] = 10.0 ** (-0.4 * ETC_py.gsGalactic_Alambda__EBV(lam) * obs.EBV)
                
                # 2. Atmospheric Transmission
                eff_atm[ipix] = ETC_py.gsAtmTrans(obs, lam, 0x0)
                
                # 3. Vignetting
                rfov = sp.rfov
                i = int(np.floor(4 * fa / rfov))
                if i < 0:
                    Vig = sp.vignette[0]
                elif i >= 4:
                    Vig = sp.vignette[4]
                else:
                    frac = 4 * fa / rfov - i
                    Vig = sp.vignette[i] + (sp.vignette[i+1] - sp.vignette[i]) * frac
                eff_vig[ipix] = Vig
                
                # 4. Instrument Throughput
                imin = sp.istart[ia]
                imax = sp.istart[ia + 1]
                if lam <= sp.l[imin]:
                    Thr = sp.T[imin]
                elif lam >= sp.l[imax - 1]:
                    Thr = sp.T[imax - 1]
                else:
                    ti = imin
                    while lam > sp.l[ti + 1]:
                        ti += 1
                    fr = (lam - sp.l[ti]) / (sp.l[ti+1] - sp.l[ti])
                    Thr = sp.T[ti] + (sp.T[ti+1] - sp.T[ti]) * fr
                eff_inst[ipix] = Thr
                
                # 5. Geometrical (Fiber Injection)
                eff_geo[ipix] = ETC_py.gsGeometricThroughput(sp, obs, lam, r_eff, de, fa, 0x0)
                
                # 6. Trace Fraction
                eff_trace[ipix] = ETC_py.gsFracTrace(sp, obs, ia, lam, 0)
            
            cum_eff = eff_ext * eff_atm * eff_vig * eff_inst * eff_geo * eff_trace
            
            all_wave.append(wave_nm)
            all_eff_geo.append(eff_geo)
            all_cum_eff.append(cum_eff)
            
        # Combine arms for a continuous plot line (or plot each arm separately with same color)
        # It's cleaner to plot each arm with the same color but no label to avoid legend clutter,
        # and only label the first arm's segment.
        for ia in range(sp.N_arms):
            label = f'Seeing {seeing}"' if ia == 0 else ""
            ax_geo.plot(all_wave[ia], all_eff_geo[ia], color=color, lw=2, label=label)
            ax_cum.plot(all_wave[ia], all_cum_eff[ia], color=color, lw=2, label=label)

    # Styling for Geometric Throughput Plot
    ax_geo.set_ylabel("Fiber Injection Geo. Efficiency",fontsize=25)
    ax_geo.set_title("Impact of Seeing on Fiber Geometric Efficiency (Point Source)",fontsize=20)
    ax_geo.grid(True, linestyle=':', alpha=0.7)
    ax_geo.legend(loc='lower right', frameon=False)
    ax_geo.set_ylim(0, 1.05)
    
    # Styling for Cumulative Efficiency Plot
    ax_cum.set_xlabel("Wavelength [nm]",fontsize=25)
    ax_cum.set_ylabel("Total End-to-End Efficiency",fontsize=25)
    ax_cum.set_title("Impact of Seeing on Total Photon Collection Efficiency",fontsize=20)
    ax_cum.grid(True, linestyle=':', alpha=0.7)
    ax_cum.legend(loc='upper right', frameon=False)
    ax_cum.set_xlim(350, 950)
    # Total efficiency rarely exceeds 0.5 for typical spectrographs
    ax_cum.set_ylim(0, 0.4)
    
    plt.tight_layout()
    output_png = output_dir / "photon_loss_seeing_cascade.png"
    plt.savefig(output_png, dpi=300)
    plt.close()
    
    print(f"Analysis plot saved to {output_png}")

if __name__ == '__main__':
    analyze_seeing_photon_loss()
