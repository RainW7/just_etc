"""
simulate_mock_spectrum.py
=========================
Simulates realistic 1D observed spectra for different galaxy types using the JUST
Exposure Time Calculator (ETC) API (`just_etc_api.py`).

Demonstrates 4 galaxy target scenarios:
1. Starburst / Emission Line Galaxy (ELG) at r = 21.0 AB mag (1h exposure)
2. Quiescent Elliptical Galaxy at r = 20.0 AB mag (30 min exposure)
3. Star-Forming Dwarf Irregular (dIrr/BCD, M_* = 10^8 M_sun, z = 0.015, 1h exposure)
4. Quiescent Dwarf Spheroidal (dSph/dE, M_* = 10^8 M_sun, z = 0.015, 1h exposure)

Outputs:
- Visualization plot saved to `output/mock_spectra/ssp_dwarf_mock_spectrum_demo.png`
"""

from just_etc.resources import DATA_DIR

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_V1_DIR = Path.cwd()

from just_etc import JUSTExposureTimeCalculator, load_template, normalize_to_mag
from just_etc.dwarf_ssp_model import build_dwarf_ssp_spectrum

# Global plot styling for publication-quality figures
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

# Pixels per resolution element for JUST spectrograph (3.0 pixels / res element for FWHM = 1.5 Å, step = 0.5 Å)
PIX_PER_RES = 3.0
SQRT_PIX_PER_RES = np.sqrt(PIX_PER_RES)


def run_simulation_demo():
    print("=" * 70)
    print("JUST Spectrograph 1D Mock Spectrum Simulation Demo (SNR/res)")
    print("=" * 70)

    output_dir = _V1_DIR / "output" / "mock_spectra"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_save_path = output_dir / "ssp_dwarf_mock_spectrum_demo.png"

    # Initialize JUST ETC (fast mode for interactive demo)
    etc = JUSTExposureTimeCalculator(calc_mode='fast')
    etc.set_obs_conditions(seeing_fwhm_800=0.8, zenith_angle=45.0)

    fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)
    arm_colors = ['#1f77b4', '#2ca02c', '#d62728']
    arm_names = ['Blue (Arm 0)', 'Green (Arm 1)', 'Red (Arm 2)']

    box_props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='#cccccc')

    # ------------------------------------------------------------------
    # Target 1: Starburst / Emission Line Galaxy (r = 21.0 mag, 1 hour exp)
    # ------------------------------------------------------------------
    print("\n[1/4] Simulating Starburst ELG (r = 21.0 mag, 1 hour exp)...")
    sb_tpl = DATA_DIR / "templates" / "starburst" / "sb4_kinney_fuv_001.fits"
    w_sb, f_sb = load_template(sb_tpl)
    f_sb_norm, _ = normalize_to_mag(w_sb, f_sb, target_mag=21.0, band='r')

    mock_sb = etc.simulate_mock_observation(w_sb, f_sb_norm, t_exp=900.0, n_exp=4, seed=42)
    snr_sb_pix = [np.nanmedian(arm['snr']) for arm in mock_sb]
    snr_sb_res = [s * SQRT_PIX_PER_RES for s in snr_sb_pix]

    ax1 = axes[0]
    ax1.plot(w_sb, f_sb_norm, color='black', lw=1.2, alpha=0.85, label='Intrinsic Spectrum (r=21.0 mag)')
    for arm in mock_sb:
        ia = arm['arm']
        ax1.plot(arm['wave_aa'], arm['flux_mock'], color=arm_colors[ia], lw=0.55, alpha=0.75,
                 label=f'JUST Observed (Arm {ia}, 1h, SNR_res={snr_sb_res[ia]:.1f})')
    ax1.set_ylabel(r'$F_\lambda$ [cgs]', fontsize=11)
    ax1.set_title('1. Starburst / Emission Line Galaxy (ELG) at r = 21.0 AB mag (1h Exposure: 4 x 900s)', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-18, 5e-16)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8.5)

    info_text1 = (
        r"$\mathbf{Target\ & \ Observational\ Parameters:}$" + "\n" +
        r"$m_r = 21.0\ \mathrm{mag}\ (\mathrm{AB}),\ \mathrm{Exp} = 4\times 900\mathrm{s}\ (1.0\mathrm{h})$" + "\n" +
        r"$\mathbf{JUST\ Performance\ (SNR/res):}$" + "\n" +
        f"Arm 0 (Blue):   SNR/res = {snr_sb_res[0]:.2f}  (SNR/pix = {snr_sb_pix[0]:.2f})\n" +
        f"Arm 1 (Green): SNR/res = {snr_sb_res[1]:.2f}  (SNR/pix = {snr_sb_pix[1]:.2f})\n" +
        f"Arm 2 (Red):     SNR/res = {snr_sb_res[2]:.2f}  (SNR/pix = {snr_sb_pix[2]:.2f})"
    )
    ax1.text(0.02, 0.93, info_text1, transform=ax1.transAxes, fontsize=8.5, verticalalignment='top', bbox=box_props)

    # ------------------------------------------------------------------
    # Target 2: Quiescent Elliptical Galaxy (r = 20.0 mag, 30 min exp)
    # ------------------------------------------------------------------
    print("\n[2/4] Simulating Quiescent Elliptical (r = 20.0 mag, 30 min exp)...")
    ell_tpl = DATA_DIR / "templates" / "galaxy" / "elliptical_001.fits"
    w_ell, f_ell = load_template(ell_tpl)
    f_ell_norm, _ = normalize_to_mag(w_ell, f_ell, target_mag=20.0, band='r')

    mock_ell = etc.simulate_mock_observation(w_ell, f_ell_norm, t_exp=450.0, n_exp=4, seed=123)
    snr_ell_pix = [np.nanmedian(arm['snr']) for arm in mock_ell]
    snr_ell_res = [s * SQRT_PIX_PER_RES for s in snr_ell_pix]

    ax2 = axes[1]
    ax2.plot(w_ell, f_ell_norm, color='black', lw=1.2, alpha=0.85, label='Intrinsic Spectrum (r=20.0 mag)')
    for arm in mock_ell:
        ia = arm['arm']
        ax2.plot(arm['wave_aa'], arm['flux_mock'], color=arm_colors[ia], lw=0.55, alpha=0.75,
                 label=f'JUST Observed (Arm {ia}, 30m, SNR_res={snr_ell_res[ia]:.1f})')
    ax2.set_ylabel(r'$F_\lambda$ [cgs]', fontsize=11)
    ax2.set_title('2. Quiescent Elliptical Galaxy at r = 20.0 AB mag (30 min Exposure: 4 x 450s)', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-18, 5e-16)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8.5)

    info_text2 = (
        r"$\mathbf{Target\ & \ Observational\ Parameters:}$" + "\n" +
        r"$m_r = 20.0\ \mathrm{mag}\ (\mathrm{AB}),\ \mathrm{Exp} = 4\times 450\mathrm{s}\ (0.5\mathrm{h})$" + "\n" +
        r"$\mathbf{JUST\ Performance\ (SNR/res):}$" + "\n" +
        f"Arm 0 (Blue):   SNR/res = {snr_ell_res[0]:.2f}  (SNR/pix = {snr_ell_pix[0]:.2f})\n" +
        f"Arm 1 (Green): SNR/res = {snr_ell_res[1]:.2f}  (SNR/pix = {snr_ell_pix[1]:.2f})\n" +
        f"Arm 2 (Red):     SNR/res = {snr_ell_res[2]:.2f}  (SNR/pix = {snr_ell_pix[2]:.2f})"
    )
    ax2.text(0.02, 0.93, info_text2, transform=ax2.transAxes, fontsize=8.5, verticalalignment='top', bbox=box_props)

    # ------------------------------------------------------------------
    # Target 3: Stellar Mass M_* = 10^8 M_sun Star-Forming Dwarf (dIrr/BCD)
    # ------------------------------------------------------------------
    print("\n[3/4] Simulating Stellar-Mass Driven dIrr (M_* = 10^8 M_sun, z = 0.015)...")
    w_dirr, f_dirr, meta_dirr = build_dwarf_ssp_spectrum(
        dwarf_type='dIrr',
        stellar_mass=1e8,
        redshift=0.015,
        ssfr=3e-10,
        vel_disp_kms=25.0
    )

    etc_dwarf = JUSTExposureTimeCalculator(calc_mode='fast')
    etc_dwarf.set_obs_conditions(r_eff=0.8, seeing_fwhm_800=0.8)

    mock_dirr = etc_dwarf.simulate_mock_observation(w_dirr, f_dirr, t_exp=900.0, n_exp=4, seed=999)
    snr_dirr_pix = [np.nanmedian(arm['snr']) for arm in mock_dirr]
    snr_dirr_res = [s * SQRT_PIX_PER_RES for s in snr_dirr_pix]

    ax3 = axes[2]
    lbl_dirr = f"SSP Intrinsic (M_*=10^8 M_⊙, SFR={meta_dirr['sfr_Msun_yr']:.3f}M_⊙/yr, m_r={meta_dirr['apparent_mag_r']:.2f})"
    ax3.plot(w_dirr, f_dirr, color='black', lw=1.2, alpha=0.85, label=lbl_dirr)
    for arm in mock_dirr:
        ia = arm['arm']
        ax3.plot(arm['wave_aa'], arm['flux_mock'], color=arm_colors[ia], lw=0.55, alpha=0.75,
                 label=f'JUST Observed (Arm {ia}, 1h, SNR_res={snr_dirr_res[ia]:.1f})')
    ax3.set_ylabel(r'$F_\lambda$ [cgs]', fontsize=11)
    ttl_dirr = f"3. Star-Forming Dwarf Irregular (M_* = 10^8 M_⊙, z = 0.015, D_L = {meta_dirr['distance_mpc']:.1f} Mpc, m_r = {meta_dirr['apparent_mag_r']:.2f} mag, 1h)"
    ax3.set_title(ttl_dirr, fontsize=12)
    ax3.set_yscale('log')
    ax3.set_ylim(1e-18, 5e-15)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=8.5)

    info_text3 = (
        r"$\mathbf{Astrophysical\ Parameters:}$" + "\n" +
        f"Stellar Mass:  M_* = {meta_dirr['stellar_mass_Msun']:.1e} M_sun\n" +
        f"Redshift & D_L:  z = {meta_dirr['redshift']}, D_L = {meta_dirr['distance_mpc']:.1f} Mpc\n" +
        f"Mag & SFR:      m_r = {meta_dirr['apparent_mag_r']:.2f} mag, SFR = {meta_dirr['sfr_Msun_yr']:.3f} M_sun/yr\n" +
        r"$\mathbf{JUST\ Performance\ (1.0h\ Exp,\ r_{eff}=0.8''):}$" + "\n" +
        f"Arm 0 (Blue):   SNR/res = {snr_dirr_res[0]:.2f}  (SNR/pix = {snr_dirr_pix[0]:.2f})\n" +
        f"Arm 1 (Green): SNR/res = {snr_dirr_res[1]:.2f}  (SNR/pix = {snr_dirr_pix[1]:.2f})\n" +
        f"Arm 2 (Red):     SNR/res = {snr_dirr_res[2]:.2f}  (SNR/pix = {snr_dirr_pix[2]:.2f})"
    )
    ax3.text(0.02, 0.93, info_text3, transform=ax3.transAxes, fontsize=8.5, verticalalignment='top', bbox=box_props)

    # ------------------------------------------------------------------
    # Target 4: Stellar Mass M_* = 10^8 M_sun Quiescent Dwarf (dSph)
    # ------------------------------------------------------------------
    print("\n[4/4] Simulating Stellar-Mass Driven dSph (M_* = 10^8 M_sun, z = 0.015)...")
    w_dsph, f_dsph, meta_dsph = build_dwarf_ssp_spectrum(
        dwarf_type='dSph',
        stellar_mass=1e8,
        redshift=0.015,
        vel_disp_kms=15.0
    )

    etc_dsph = JUSTExposureTimeCalculator(calc_mode='fast')
    etc_dsph.set_obs_conditions(r_eff=1.0, seeing_fwhm_800=0.8)

    mock_dsph = etc_dsph.simulate_mock_observation(w_dsph, f_dsph, t_exp=900.0, n_exp=4, seed=777)
    snr_dsph_pix = [np.nanmedian(arm['snr']) for arm in mock_dsph]
    snr_dsph_res = [s * SQRT_PIX_PER_RES for s in snr_dsph_pix]

    ax4 = axes[3]
    lbl_dsph = f"SSP Intrinsic (M_*=10^8 M_⊙, BC03 12Gyr, m_r={meta_dsph['apparent_mag_r']:.2f})"
    ax4.plot(w_dsph, f_dsph, color='black', lw=1.2, alpha=0.85, label=lbl_dsph)
    for arm in mock_dsph:
        ia = arm['arm']
        ax4.plot(arm['wave_aa'], arm['flux_mock'], color=arm_colors[ia], lw=0.55, alpha=0.75,
                 label=f'JUST Observed (Arm {ia}, 1h, SNR_res={snr_dsph_res[ia]:.1f})')
    ax4.set_xlabel('Wavelength [Å]', fontsize=12)
    ax4.set_ylabel(r'$F_\lambda$ [cgs]', fontsize=11)
    ttl_dsph = f"4. Quiescent Dwarf Spheroidal (M_* = 10^8 M_⊙, z = 0.015, (M/L)_r = {meta_dsph['ml_ratio_r']:.1f}, m_r = {meta_dsph['apparent_mag_r']:.2f} mag, 1h)"
    ax4.set_title(ttl_dsph, fontsize=12)
    ax4.set_xlim(3650, 9250)
    ax4.set_yscale('log')
    ax4.set_ylim(5e-19, 1e-16)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper right', fontsize=8.5)

    info_text4 = (
        r"$\mathbf{Astrophysical\ Parameters:}$" + "\n" +
        f"Stellar Mass:  M_* = {meta_dsph['stellar_mass_Msun']:.1e} M_sun\n" +
        f"Redshift & D_L:  z = {meta_dsph['redshift']}, D_L = {meta_dsph['distance_mpc']:.1f} Mpc\n" +
        f"Mag & (M/L)_r:   m_r = {meta_dsph['apparent_mag_r']:.2f} mag, (M/L)_r = {meta_dsph['ml_ratio_r']:.1f}\n" +
        r"$\mathbf{JUST\ Performance\ (1.0h\ Exp,\ r_{eff}=1.0''):}$" + "\n" +
        f"Arm 0 (Blue):   SNR/res = {snr_dsph_res[0]:.2f}  (SNR/pix = {snr_dsph_pix[0]:.2f})\n" +
        f"Arm 1 (Green): SNR/res = {snr_dsph_res[1]:.2f}  (SNR/pix = {snr_dsph_pix[1]:.2f})\n" +
        f"Arm 2 (Red):     SNR/res = {snr_dsph_res[2]:.2f}  (SNR/pix = {snr_dsph_pix[2]:.2f})"
    )
    ax4.text(0.02, 0.93, info_text4, transform=ax4.transAxes, fontsize=8.5, verticalalignment='top', bbox=box_props)

    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Simulation plot updated and saved to:\n   {plot_save_path}")


if __name__ == '__main__':
    run_simulation_demo()
