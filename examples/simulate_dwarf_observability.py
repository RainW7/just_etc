
from just_etc.resources import DATA_DIR
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from just_etc import ETC_py_optimized as ETC_py
from just_etc import JUSTExposureTimeCalculator, load_template, normalize_to_mag

# Global plot styling for publication-quality figures
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

def convert_surface_brightness_to_mag(mu_eff, r_eff):
    """
    Convert the average surface brightness within the effective radius (half-light radius)
    to the integrated total AB magnitude for an exponential disk profile.
    
    Parameters
    ----------
    mu_eff : float
        Average surface brightness within r_eff (mag/arcsec^2).
    r_eff : float
        Effective radius (half-light radius) in arcseconds.
        
    Returns
    -------
    m_tot : float
        Integrated total AB magnitude of the galaxy.
    """
    # The average intensity within the half-light radius is defined as:
    # I_mean = (0.5 * L_tot) / (pi * r_eff^2)
    # Correspondingly: mu_eff = m_tot + 2.5 * log10(2 * pi * r_eff^2)
    m_tot = mu_eff - 2.5 * np.log10(2.0 * np.pi * (r_eff ** 2))
    return m_tot

def add_emission_lines(wave_aa, flux_flam, lines_dict, vel_disp_kms=20.0):
    """
    Add Gaussian emission lines to a continuum spectrum.
    Useful for simulating Dwarf Irregulars (dIrr) or Blue Compact Dwarfs (BCDs).
    
    Parameters
    ----------
    wave_aa : ndarray
        Wavelength array in Angstroms.
    flux_flam : ndarray
        Flux array in erg/s/cm^2/Angstrom.
    lines_dict : dict
        Dictionary of {rest_wavelength_aa: total_line_flux_cgs}
        where total_line_flux_cgs is in erg/s/cm^2.
    vel_disp_kms : float
        Velocity dispersion in km/s (typically 10-30 km/s for dwarf galaxies).
    """
    c_kms = 299792.458 # Speed of light in km/s
    flux_with_lines = np.copy(flux_flam)
    
    for lam0, line_flux in lines_dict.items():
        # Sigma in wavelength space: sigma_lam = lam0 * (sigma_v / c)
        sigma_lam = lam0 * (vel_disp_kms / c_kms)
        # Compute Gaussian profile
        gaussian = (line_flux / (np.sqrt(2.0 * np.pi) * sigma_lam)) * np.exp(-0.5 * ((wave_aa - lam0) / sigma_lam)**2)
        flux_with_lines += gaussian
        
    return flux_with_lines

def analyze_fiber_throughput(output_dir):
    """
    Plot the fiber geometric throughput at 600 nm as a function of r_eff (0.1" to 5.0")
    under different atmospheric seeing conditions.
    """
    print("\n--- Running Fiber Throughput Analysis for Dwarf Galaxies ---")
    etc = JUSTExposureTimeCalculator(calc_mode='fast')
    sp = etc._spectro
    obs = etc._make_obs()
    fa = etc._obs_params['field_angle']
    de = etc._obs_params['decenter']
    
    reff_array = np.linspace(0.1, 5.0, 100)
    seeing_values = [0.6, 0.8, 1.2, 1.5]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for seeing, color in zip(seeing_values, colors):
        obs.seeing_fwhm_800 = seeing
        eff_list = []
        for r in reff_array:
            eff = ETC_py.gsGeometricThroughput(sp, obs, 600.0, r, de, fa, 0x0)
            eff_list.append(eff)
        ax.plot(reff_array, eff_list, lw=2, color=color, label=f'Seeing = {seeing}"')
        
    ax.axvline(0.68, color='purple', linestyle='--', alpha=0.7, label='Fiber Radius (0.68")')
    ax.set_xlabel('Half-Light Radius r_eff [arcsec]', fontsize=14)
    ax.set_ylabel('Fiber Geometric Throughput at 600 nm', fontsize=14)
    ax.set_title('JUST Fiber Aperture Loss for Dwarf Galaxies (Exponential Profile)', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=11)
    
    out_path = output_dir / "dwarf_fiber_throughput.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved throughput plot to: {out_path}")

def estimate_stellar_mass(m_tot, redshift, ml_ratio, H0=70.0):
    """
    Estimate stellar mass of the dwarf galaxy using standard cosmology.
    """
    c_kms = 299792.458
    d_L_Mpc = (c_kms * redshift) / H0
    # Distance modulus
    dist_mod = 5.0 * np.log10(d_L_Mpc * 1e6 / 10.0)
    M_r = m_tot - dist_mod
    M_sun_r = 4.65
    L_r = 10.0 ** (-0.4 * (M_r - M_sun_r))
    stellar_mass = ml_ratio * L_r
    return stellar_mass

def simulate_quiescent_dwarf(etc, output_dir, redshift=0.01, mu_r=24.0, r_eff=2.0, t_exp=3600.0, n_exp=3):
    """
    Simulate a Dwarf Elliptical / Spheroidal (dE / dSph) galaxy.
    Uses an old stellar population template.
    """
    print(f"\n--- Simulating Quiescent Dwarf (dE/dSph) ---")
    print(f"  Redshift = {redshift}")
    print(f"  r_eff = {r_eff} arcsec")
    print(f"  Average surface brightness <mu_r> = {mu_r} mag/arcsec^2")
    
    # 1. Convert SB to integrated total magnitude
    m_tot = convert_surface_brightness_to_mag(mu_r, r_eff)
    print(f"  -> Corresponding total r-band magnitude: {m_tot:.2f} AB mag")
    
    # Estimate stellar mass (assuming M/L_r = 3.0 for old stellar population)
    ml_ratio = 3.0
    stellar_mass = estimate_stellar_mass(m_tot, redshift, ml_ratio)
    print(f"  -> Estimated stellar mass: {stellar_mass:.2e} M_sun (with M/L_r = {ml_ratio})")
    
    # 2. Load templates (using elliptical / bc03 old population)
    parent_dir = Path.cwd()
    template_path = DATA_DIR / "templates" / "bc03" / "tau06_z02_12000_000_001.fits"
    wave_aa, flux_flam = load_template(template_path)
    
    # 3. Apply redshift and normalize
    wave_z = wave_aa * (1.0 + redshift)
    flux_norm, _ = normalize_to_mag(wave_z, flux_flam, target_mag=m_tot, band='r')
    
    # 4. Run ETC
    etc.set_obs_conditions(seeing_fwhm_800=0.8, r_eff=r_eff)
    results = etc.compute_snr(wave_z, flux_norm, t_exp=t_exp, n_exp=n_exp)
    
    # 5. Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    arm_colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for arm_res in results:
        ia = arm_res['arm']
        wn = arm_res['wave_nm']
        snr = arm_res['snr']
        
        # Smooth for visualization
        snr_smooth = np.convolve(snr, np.ones(11)/11, mode='same')
        
        ax.plot(wn, snr, color=arm_colors[ia], alpha=0.2, lw=0.5)
        ax.plot(wn, snr_smooth, color=arm_colors[ia], lw=2, label=f'Arm {ia}')
        
    ax.axhline(3.0, color='gray', linestyle='--', alpha=0.7, label='S/N = 3 (Redshift Limit)')
    ax.axhline(5.0, color='gray', linestyle=':', alpha=0.7, label='S/N = 5 (Kinematics Limit)')
    
    ax.set_xlabel('Wavelength [nm]', fontsize=14)
    ax.set_ylabel('Total S/N per pixel', fontsize=14)
    ax.set_title(f'dE/dSph Dwarf Galaxy Observability\nM_*={stellar_mass:.2e} M_sun (M/L_r={ml_ratio}), <mu_r>_eff={mu_r} mag/arcsec^2, r_eff={r_eff}", Exp={n_exp}x{t_exp/60:.0f} min', fontsize=11, fontweight='bold')
    ax.set_xlim(350, 950)
    ax.set_ylim(0, max(10, ax.get_ylim()[1]))
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=10, loc='upper right')
    
    out_path = output_dir / "quiescent_dwarf_snr.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved quiescent dwarf simulation plot to: {out_path}")

def simulate_starforming_dwarf(etc, output_dir, redshift=0.02, mu_r=23.5, r_eff=1.5, t_exp=1800.0, n_exp=2):
    """
    Simulate a Dwarf Irregular / Blue Compact Dwarf (dIrr / BCD).
    Uses a young starburst template and adds mock emission lines (H-alpha, [OIII], etc.).
    """
    print(f"\n--- Simulating Star-Forming Dwarf (dIrr/BCD) ---")
    print(f"  Redshift = {redshift}")
    print(f"  r_eff = {r_eff} arcsec")
    print(f"  Average surface brightness <mu_r> = {mu_r} mag/arcsec^2")
    
    # 1. Convert SB to integrated total magnitude
    m_tot = convert_surface_brightness_to_mag(mu_r, r_eff)
    print(f"  -> Corresponding total r-band magnitude: {m_tot:.2f} AB mag")
    
    # Estimate stellar mass (assuming M/L_r = 0.5 for young starburst population)
    ml_ratio = 0.5
    stellar_mass = estimate_stellar_mass(m_tot, redshift, ml_ratio)
    print(f"  -> Estimated stellar mass: {stellar_mass:.2e} M_sun (with M/L_r = {ml_ratio})")
    
    # 2. Load templates (using a starburst galaxy template)
    parent_dir = Path.cwd()
    template_path = DATA_DIR / "templates" / "galaxy" / "sb2_b2004a_001.fits"
    wave_aa, flux_flam = load_template(template_path)
    
    # 3. Add emission lines in rest frame
    mock_lines = {
        3727.0: 1.5e-14,  # [OII]
        4861.0: 1.0e-14,  # H-beta
        4959.0: 1.0e-14,  # [OIII]
        5007.0: 3.0e-14,  # [OIII]
        6563.0: 2.86e-14, # H-alpha
        6584.0: 0.5e-14,  # [NII]
        6717.0: 0.4e-14,  # [SII]
        6731.0: 0.3e-14,  # [SII]
    }
    
    flux_with_lines = add_emission_lines(wave_aa, flux_flam, mock_lines, vel_disp_kms=25.0)
    
    # 4. Apply redshift and normalize
    wave_z = wave_aa * (1.0 + redshift)
    flux_norm, _ = normalize_to_mag(wave_z, flux_with_lines, target_mag=m_tot, band='r')
    
    # 5. Run ETC
    etc.set_obs_conditions(seeing_fwhm_800=0.8, r_eff=r_eff)
    results = etc.compute_snr(wave_z, flux_norm, t_exp=t_exp, n_exp=n_exp)
    
    # 6. Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    arm_colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for arm_res in results:
        ia = arm_res['arm']
        wn = arm_res['wave_nm']
        snr = arm_res['snr']
        
        # Plot both raw SNR (to show emission line spikes) and smoothed SNR
        snr_smooth = np.convolve(snr, np.ones(5)/5, mode='same')
        
        # Raw SNR plot in thin lines to capture emission line spikes
        ax.plot(wn, snr, color=arm_colors[ia], alpha=0.8, lw=0.8, label=f'Arm {ia}' if ia==0 else "")
        
    ax.axhline(5.0, color='gray', linestyle='--', alpha=0.7, label='S/N = 5 Limit')
    
    # Mark redshifted emission line locations in the plot
    rest_lines_to_label = {
        "[OII] 372.7": 372.7,
        "H-beta": 486.1,
        "[OIII] 500.7": 500.7,
        "H-alpha": 656.3
    }
    for name, lam_rest in rest_lines_to_label.items():
        lam_obs = lam_rest * (1.0 + redshift)
        ax.axvline(lam_obs, color='purple', linestyle=':', alpha=0.5)
        ax.text(lam_obs + 2, ax.get_ylim()[1]*0.8, name, rotation=90, color='purple', fontsize=9)
        
    ax.set_xlabel('Wavelength [nm]', fontsize=14)
    ax.set_ylabel('Total S/N per pixel', fontsize=14)
    ax.set_title(f'dIrr/BCD Star-Forming Dwarf Galaxy Observability\nM_*={stellar_mass:.2e} M_sun (M/L_r={ml_ratio}), <mu_r>_eff={mu_r} mag/arcsec^2, r_eff={r_eff}", Exp={n_exp}x{t_exp/60:.0f} min', fontsize=11, fontweight='bold')
    ax.set_xlim(350, 950)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=10, loc='upper right')
    
    out_path = output_dir / "starforming_dwarf_snr.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved star-forming dwarf simulation plot to: {out_path}")

if __name__ == '__main__':
    # Define outputs
    base_dir = Path.cwd()
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize the Exposure Time Calculator in fast mode
    etc = JUSTExposureTimeCalculator(calc_mode='fast')
    
    # Run the three analyses
    analyze_fiber_throughput(output_dir)
    simulate_quiescent_dwarf(etc, output_dir, redshift=0.01, mu_r=23.0, r_eff=1.5, t_exp=3600.0, n_exp=3)
    simulate_starforming_dwarf(etc, output_dir, redshift=0.015, mu_r=23.0, r_eff=1.5, t_exp=1800.0, n_exp=2)
    
    print("\n✅ All dwarf galaxy simulations and plots completed successfully!")
