"""
dwarf_ssp_model.py
==================
Astrophysically accurate Dwarf Galaxy Spectrum Generator based on SSP (Stellar Population Synthesis)
continuum models, stellar mass scaling (M_* / M_sun), mass-to-light ratios (M/L)_r, Kennicutt (1998) 
SFR-to-Halpha relations, and low-metallicity H II region nebular emission line physics.

Supported Dwarf Galaxy Types:
1. 'dIrr' / 'starburst_dwarf': Star-forming Dwarf Irregular / Blue Compact Dwarf (BCD).
   - Continuum: Low-metallicity Young SSP (5-25 Myr) or Magellanic Irregular (Im).
   - SFR: Derived from sSFR (Specific Star Formation Rate) * M_*.
   - Nebular Lines: Kennicutt H-alpha flux + high excitation [OIII]/Hb ~ 4.5, low [NII]/Ha ~ 0.08.
   - Kinematics: Narrow lines (sigma_v ~ 20-30 km/s).
2. 'dSph' / 'quiescent_dwarf': Quiescent Dwarf Elliptical / Dwarf Spheroidal.
   - Continuum: Old metal-poor SSP (10-12 Gyr).
   - Kinematics: Narrow stellar absorption lines (sigma_v ~ 10-15 km/s), no gas emission lines.
3. 'post_starburst_dwarf' / 'ea_dwarf': Post-Starburst (E+A) Dwarf Galaxy.
   - Continuum: Intermediate-age SSP (100-500 Myr A-type star dominated).
   - Features: Deep Balmer absorption lines (Hd, Hg, Hb), no gas emission lines.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .resources import DATA_DIR

_V1_DIR = DATA_DIR

from .just_etc_api import load_template, normalize_to_mag

# Solar absolute AB magnitude in r-band
M_SUN_R = 4.65
# Speed of light in km/s
C_KMS = 299792.458
# Hubble constant H0 [km/s/Mpc]
H0 = 70.0


def get_luminosity_distance_mpc(z):
    """
    Computes luminosity distance D_L in Mpc for small redshifts (z < 0.2).
    """
    if z <= 0.0:
        return 10.0  # 10 Mpc default
    return (C_KMS * z / H0) * (1.0 + 0.5 * (1.0 - 0.3) * z)


def apply_kinematic_broadening(wave_aa, flux, vel_disp_kms):
    """
    Apply Gaussian kinematic velocity dispersion broadening to a spectrum.
    """
    if vel_disp_kms <= 0.0:
        return flux.copy()

    dw = np.median(np.diff(wave_aa))
    w_med = np.median(wave_aa)
    sigma_pixels = (w_med * (vel_disp_kms / C_KMS)) / dw

    if sigma_pixels < 0.1:
        return flux.copy()

    return gaussian_filter1d(flux, sigma=sigma_pixels)


def build_dwarf_ssp_spectrum(dwarf_type='dIrr',
                            stellar_mass=None,
                            redshift=0.015,
                            distance_mpc=None,
                            ml_ratio=None,
                            ssfr=3e-10,
                            target_mag=None,
                            band='r',
                            vel_disp_kms=25.0,
                            wave_grid=None):
    """
    Build a physically complete, stellar-mass or magnitude driven SSP dwarf galaxy spectrum.

    Parameters
    ----------
    dwarf_type : str
        'dIrr' / 'starburst_dwarf' : Star-forming Dwarf Irregular / BCD
        'dSph' / 'quiescent_dwarf' : Quiescent Dwarf Elliptical / Dwarf Spheroidal
        'ea_dwarf' / 'post_starburst' : Post-Starburst E+A Dwarf
    stellar_mass : float, optional
        Total stellar mass M_* in solar masses [M_sun] (e.g. 1e7, 1e8, 1e9).
        If provided, apparent magnitude m_r is derived from M_*, (M/L)_r, and z.
    redshift : float
        Cosmological redshift z.
    distance_mpc : float, optional
        Luminosity distance in Mpc. Overrides redshift calculation if specified.
    ml_ratio : float, optional
        Mass-to-light ratio (M/L)_r in r-band [M_sun / L_sun,r].
        Defaults: dIrr ~ 0.5, dSph ~ 2.0, ea_dwarf ~ 0.8.
    ssfr : float
        Specific Star Formation Rate sSFR [yr⁻¹] (default 3e-10 yr⁻¹ for dIrr).
    target_mag : float, optional
        Direct apparent AB magnitude for normalization. Used if stellar_mass is None.
    band : str
        Photometric band for magnitude normalization ('r', 'g', 'z', etc.).
    vel_disp_kms : float
        Internal velocity dispersion [km/s].
    wave_grid : ndarray, optional
        Custom wavelength grid [Å]. Defaults to 3500 - 9850 Å (step 0.5 Å).

    Returns
    -------
    wave_out : ndarray [Å]
    flux_out : ndarray [erg/s/cm²/Å]
    metadata : dict
        Calculated astrophysical metadata dictionary containing:
        - stellar_mass_Msun
        - sfr_Msun_yr
        - apparent_mag_r
        - absolute_mag_r
        - ml_ratio_r
        - distance_mpc
        - distance_modulus
        - vel_disp_kms
        - line_fluxes_cgs
    """
    tpl_dir = _V1_DIR / 'templates'

    if wave_grid is None:
        wave_grid = np.arange(3500.0, 9850.0, 0.5)

    # 1. Luminosity Distance & Distance Modulus
    if distance_mpc is None:
        dL_mpc = get_luminosity_distance_mpc(redshift)
    else:
        dL_mpc = distance_mpc

    dist_mod = 5.0 * np.log10(max(dL_mpc, 1e-3) * 1e6 / 10.0)
    dL_cm = dL_mpc * 3.08567758149137e24

    # Default M/L ratios if not specified
    if ml_ratio is None:
        if dwarf_type in ('dIrr', 'starburst_dwarf', 'bcd'):
            ml_ratio = 0.45
        elif dwarf_type in ('dSph', 'quiescent_dwarf', 'dE'):
            ml_ratio = 2.00
        else:
            ml_ratio = 0.75

    # 2. Derive Apparent Magnitude m_r and Absolute Magnitude M_r from Stellar Mass M_*
    sfr_val = 0.0
    if stellar_mass is not None:
        # L_r [L_sun,r] = M_* / (M/L)_r
        L_r = stellar_mass / ml_ratio
        M_r = M_SUN_R - 2.5 * np.log10(L_r)
        m_r_calc = M_r + dist_mod
        app_mag = m_r_calc
    else:
        if target_mag is not None:
            app_mag = target_mag
            M_r = app_mag - dist_mod
            L_r = 10.0 ** (0.4 * (M_SUN_R - M_r))
            stellar_mass = L_r * ml_ratio
        else:
            # Fallback default: M_* = 1e8 M_sun
            stellar_mass = 1e8
            L_r = stellar_mass / ml_ratio
            M_r = M_SUN_R - 2.5 * np.log10(L_r)
            app_mag = M_r + dist_mod

    metadata = dict(
        dwarf_type=dwarf_type,
        stellar_mass_Msun=stellar_mass,
        sfr_Msun_yr=0.0,
        apparent_mag_r=app_mag,
        absolute_mag_r=M_r,
        ml_ratio_r=ml_ratio,
        distance_mpc=dL_mpc,
        distance_modulus=dist_mod,
        redshift=redshift,
        vel_disp_kms=vel_disp_kms,
        line_fluxes_cgs={},
    )

    # 3. Construct Stellar Continuum + Nebular Lines according to dwarf_type
    if dwarf_type in ('dIrr', 'starburst_dwarf', 'bcd'):
        # -------------------------------------------------------------
        # 1. Star-Forming Dwarf (dIrr / BCD)
        # Continuum: Low-Z Young SSP (5-25 Myr) / Magellanic Irregular
        # -------------------------------------------------------------
        tpl_file = tpl_dir / 'galaxy' / 'im_cb2004a_001.fits'
        w_raw, f_raw = load_template(tpl_file)

        w_rest = wave_grid / (1.0 + redshift)
        f_cont = np.interp(w_rest, w_raw, f_raw, left=0.0, right=0.0)

        # Normalize continuum to apparent r-band magnitude
        f_cont_norm, _ = normalize_to_mag(wave_grid, f_cont, target_mag=app_mag, band=band)
        f_cont_broad = apply_kinematic_broadening(wave_grid, f_cont_norm, vel_disp_kms)

        # Calculate Star Formation Rate (SFR = M_* * sSFR)
        sfr_val = stellar_mass * ssfr
        metadata['sfr_Msun_yr'] = sfr_val

        # Kennicutt (1998) H-alpha luminosity: L(Ha) [erg/s] = SFR [M_sun/yr] / 7.9e-42
        L_ha = sfr_val / 7.9e-42
        F_ha = L_ha / (4.0 * np.pi * dL_cm ** 2) if dL_cm > 0 else 1e-16

        # H-beta flux (Case B recombination ratio Ha/Hb = 2.86)
        F_hb = F_ha / 2.86

        # Low-metallicity H II region emission line ratios
        line_fluxes = {
            3727.1: F_hb * 2.50,         # [OII] 3727
            3868.8: F_hb * 0.40,         # [NeIII] 3869
            4101.7: F_hb * 0.26,         # H-delta
            4340.5: F_hb * 0.47,         # H-gamma
            4861.3: F_hb * 1.00,         # H-beta
            4958.9: F_hb * 1.50,         # [OIII] 4959
            5006.8: F_hb * 4.50,         # [OIII] 5007 (High excitation)
            6548.0: F_ha * 0.026,        # [NII] 6548
            6562.8: F_ha,                # H-alpha
            6583.5: F_ha * 0.08,         # [NII] 6584 (Weak [NII] in metal-poor dwarf)
            6716.4: F_ha * 0.12,         # [SII] 6717
            6730.8: F_ha * 0.09,         # [SII] 6731
        }
        metadata['line_fluxes_cgs'] = line_fluxes

        # Superimpose Gaussian emission lines
        dw = np.median(np.diff(wave_grid))
        f_total = f_cont_broad.copy()

        for line_w_rest, flux_line in line_fluxes.items():
            w0 = line_w_rest * (1.0 + redshift)
            if wave_grid[0] <= w0 <= wave_grid[-1]:
                sigma_aa = w0 * (vel_disp_kms / C_KMS)
                sigma_aa = max(sigma_aa, dw * 0.5)
                gauss = (flux_line / (np.sqrt(2.0 * np.pi) * sigma_aa)) * np.exp(-0.5 * ((wave_grid - w0) / sigma_aa) ** 2)
                f_total += gauss

        return wave_grid, f_total, metadata

    elif dwarf_type in ('dSph', 'quiescent_dwarf', 'dE'):
        # -------------------------------------------------------------
        # 2. Quiescent Dwarf (dSph / dE)
        # Continuum: Old metal-poor BC03 SSP (12 Gyr)
        # -------------------------------------------------------------
        tpl_file = tpl_dir / 'bc03' / 'tau06_z02_12000_000_001.fits'
        if not tpl_file.exists():
            tpl_file = tpl_dir / 'galaxy' / 'elliptical_001.fits'

        w_raw, f_raw = load_template(tpl_file)

        w_rest = wave_grid / (1.0 + redshift)
        f_cont = np.interp(w_rest, w_raw, f_raw, left=0.0, right=0.0)

        f_cont_norm, _ = normalize_to_mag(wave_grid, f_cont, target_mag=app_mag, band=band)
        f_total = apply_kinematic_broadening(wave_grid, f_cont_norm, vel_disp_kms)

        return wave_grid, f_total, metadata

    elif dwarf_type in ('ea_dwarf', 'post_starburst'):
        # -------------------------------------------------------------
        # 3. Post-Starburst (E+A) Dwarf Galaxy
        # Continuum: 25-100 Myr SSP with deep A-star Balmer absorption
        # -------------------------------------------------------------
        tpl_file = tpl_dir / 'galaxy' / 'ssp_25myr_z008_001.fits'
        w_raw, f_raw = load_template(tpl_file)

        w_rest = wave_grid / (1.0 + redshift)
        f_cont = np.interp(w_rest, w_raw, f_raw, left=0.0, right=0.0)

        f_cont_norm, _ = normalize_to_mag(wave_grid, f_cont, target_mag=app_mag, band=band)
        f_total = apply_kinematic_broadening(wave_grid, f_cont_norm, vel_disp_kms)

        return wave_grid, f_total, metadata

    else:
        raise ValueError(f"Unknown dwarf_type: {dwarf_type}. Choose from 'dIrr', 'dSph', or 'ea_dwarf'.")
