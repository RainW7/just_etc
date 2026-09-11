
from just_etc.resources import DATA_DIR
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from just_etc import ETC_py_optimized as ETC_py
from just_etc import JUSTExposureTimeCalculator

def analyze_photon_loss():
    base_dir = Path.cwd()
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    print("Initializing ETC...")
    etc = JUSTExposureTimeCalculator(calc_mode='fast')
    obs = etc._make_obs()
    sp = etc._spectro
    fa = etc._obs_params['field_angle']
    de = etc._obs_params['decenter']
    
    # Let's assume a dummy point source with r_eff = 0
    # To study purely the optical efficiency, we use point source centered on fiber.
    r_eff = 0.8
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 14), sharex=True)
    ax_eff, ax_cum = axes
    
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    labels_plotted = False
    
    print("Calculating efficiencies per arm...")
    for ia in range(sp.N_arms):
        Npix = sp.npix[ia]
        lmin = sp.lmin[ia]
        dl = sp.dl[ia]
        wave_nm = lmin + dl * (np.arange(Npix) + 0.5)
        
        eff_ext = np.zeros(Npix)
        eff_atm = np.zeros(Npix)
        eff_vig = np.zeros(Npix)    # Vignetting
        eff_inst = np.zeros(Npix)   # Instrument/Telescope Transmission (Thr in gsAeff)
        eff_geo = np.zeros(Npix)    # Fiber injection geometric throughput
        eff_trace = np.zeros(Npix)  # Fraction of trace inside extraction window
        
        for ipix in range(Npix):
            lam = wave_nm[ipix]
            
            # 1. Galactic Extinction
            eff_ext[ipix] = 10.0 ** (-0.4 * ETC_py.gsGalactic_Alambda__EBV(lam) * obs.EBV)
            
            # 2. Atmospheric Transmission
            eff_atm[ipix] = ETC_py.gsAtmTrans(obs, lam, 0x0)
            
            # 3. Vignetting & 4. Instrument Throughput
            # Extracted from gsAeff_fast logic
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
            
        color = colors[ia]
        label_suffix = "" if labels_plotted else ""
        
        # Plot Individual Efficiencies
        ax_eff.plot(wave_nm, eff_ext, color='black', linewidth = 3, linestyle='-', alpha=0.3, label="Galactic Extinction" if not labels_plotted else "")
        ax_eff.plot(wave_nm, eff_atm, color='purple', linewidth = 3, linestyle='--', label="Atmospheric Transmission" if not labels_plotted else "")
        ax_eff.plot(wave_nm, eff_vig, color='cyan', linewidth = 3, linestyle='-.', label="Telescope Vignetting" if not labels_plotted else "")
        ax_eff.plot(wave_nm, eff_geo, color='orange', linewidth = 3, linestyle='-', label=f"Fiber Injection (Geo)" if not labels_plotted else "")
        ax_eff.plot(wave_nm, eff_trace, color='brown', linewidth = 3, linestyle=':', label="CCD Extraction Trace" if not labels_plotted else "")
        
        ax_eff.plot(wave_nm, eff_inst, color=color, lw=3, label=f"Instrument Throughput (Arm {ia})")
        
        # Plot cumulative efficiency
        cum_eff = eff_ext * eff_atm * eff_vig * eff_inst * eff_geo * eff_trace
        ax_cum.plot(wave_nm, cum_eff, color=color, lw=3, label=f"Total System Efficiency (Arm {ia})")
        
        # To show the progressive drop
        # Base starts at 1.0
        # drop 1: ext
        # drop 2: ext * atm ... etc
        ax_cum.fill_between(wave_nm, 0, cum_eff, color=color, alpha=0.2)
        
        labels_plotted = True

    ax_eff.set_ylabel("Individual Efficiency / Transmission [0-1]",fontsize=25)
    ax_eff.set_title("Photon Loss Breakdown: Individual Components",fontsize=25)
    ax_eff.grid(True, linestyle=':', alpha=0.7)
    ax_eff.legend(loc='lower center', ncol=3, fontsize=14)
    ax_eff.set_ylim(0, 1.05)
    
    ax_cum.set_xlabel("Wavelength [nm]",fontsize=25)
    ax_cum.set_ylabel("Cumulative Total Efficiency [0-1]",fontsize=25)
    ax_cum.set_title("Total End-to-End Photon Collection Efficiency",fontsize=25)
    ax_cum.grid(True, linestyle=':', alpha=0.7)
    ax_cum.legend(loc='lower center', fontsize=20)
    ax_cum.set_ylim(0, 0.3)
    
    plt.tight_layout()
    output_png = output_dir / "photon_loss_cascade.png"
    plt.savefig(output_png, dpi=300)
    plt.close()
    
    print(f"Analysis plot saved to {output_png}")

if __name__ == '__main__':
    analyze_photon_loss()
