
from just_etc.resources import DATA_DIR
import sys
from pathlib import Path
import numpy as np
import math
import warnings

# Temporarily suppress warnings for clean output
warnings.filterwarnings('ignore')

# ETC_py_v1 — all modules co-located in this directory
current_dir = Path.cwd()

from just_etc import ETC_py_optimized as ETC_py

def configure_just_setup():
    """Initializes and returns the spectrograph and observation configurations."""
    # 1. Setup Spectrograph
    config_file = str(DATA_DIR / "spec.dat")
    spectro = ETC_py.SpectroAttrib()
    ETC_py.gsReadSpectrographConfig(config_file, spectro)
    spectro.sysfrac = 0.00  # Error floor
    spectro.diffuse_stray = 0.0

    # 2. Setup Observation Conditions
    obs = ETC_py.ObsAttrib()
    obs.skytype = int("10003", 16)
    obs.seeing_fwhm_800 = 0.8
    obs.zenithangle = 45.0
    obs.EBV = 0.03
    obs.lunarZA = 135.0
    obs.lunarangle = 90.0
    obs.lunarphase = 0.25

    return spectro, obs

def normalize_spectrum_to_mag(wave_in_a, flux_in_1e17, target_mag, band='r'):
    """
    Normalizes a spectrum to a given AB magnitude in a specified band.
    
    Parameters:
    -----------
    wave_in_a : ndarray
        1D array of wavelength in Angstroms
    flux_in_1e17 : ndarray
        1D array of flux in 1e-17 erg/s/cm2/A
    target_mag : float
        Target AB magnitude
    band : str
        The photometric band to use for normalization. 
        Supported: 'u', 'g', 'r', 'i', 'z', 'V', 'arm1', 'arm2', 'arm3'.
        
    Returns:
    --------
    normalized_flux : ndarray
        The scaled flux array in 1e-17 erg/s/cm2/A.
    """
    # Effective wavelengths in Angstroms for standard broadband filters
    eff_waves = {
        'u': 3543.0,
        'g': 4770.0,
        'r': 6231.0,
        'i': 7625.0,
        'z': 9134.0,
        'V': 5510.0,
        'arm1': 5050.0,
        'arm2': 6670.0,
        'arm3': 8610.0,
    }
    
    if band not in eff_waves:
        raise ValueError(f"Band '{band}' not supported. Choose from {list(eff_waves.keys())}")
        
    eff_wave = eff_waves[band]
    
    # Target flux in erg/s/cm2/Hz for given AB magnitude
    # Formula: m_AB = -2.5 * log10(F_nu) - 48.60 
    target_f_nu = 3.631e-20 * 10 ** (-0.4 * target_mag)
    
    # Speed of light in Angstrom/s
    c_a_s = 2.99792458e18
    
    # Target flux in erg/s/cm2/A: F_lambda = F_nu * c / lambda^2
    target_f_lambda = target_f_nu * c_a_s / (eff_wave ** 2)
    
    # Convert to the internal 1e-17 erg/s/cm2/A units
    target_f_lambda_1e17 = target_f_lambda * 1e17
    
    # Average flux in a closely matched window around the effective wavelength
    # This mitigates matching precisely onto a deep absorption line.
    mask = (wave_in_a >= eff_wave - 500.0) & (wave_in_a <= eff_wave + 500.0)
    if not np.any(mask):
        current_f_lambda_1e17 = np.interp(eff_wave, wave_in_a, flux_in_1e17)
    else:
        current_f_lambda_1e17 = np.mean(flux_in_1e17[mask])
        
    if current_f_lambda_1e17 <= 0:
        raise ValueError(f"Template flux is zero or negative around {eff_wave} A. Cannot normalize.")
        
    # Scale uniformly
    scale_factor = target_f_lambda_1e17 / current_f_lambda_1e17
    
    return flux_in_1e17 * scale_factor

def compute_stellar_snr(wave_in_a, flux_in_1e17, exp_time=450.0, n_exp=8):
    """
    Computes the JUST ETC simulated Signal-to-Noise Ratio for a given stellar spectrum.
    
    Parameters:
    -----------
    wave_in_a : ndarray
        1D array of wavelength in Angstroms
    flux_in_1e17 : ndarray
        1D array of flux in 1e-17 erg/s/cm2/A
    exp_time : float
        Exposure time in seconds per frame (default: 450.0)
    n_exp : int
        Number of exposures (default: 8)
        
    Returns:
    --------
    out_wave_a : list of ndarray
        Wavelength arrays in Angstroms for each spectrograph arm.
    out_snr : list of ndarray
        SNR arrays corresponding to the wavelength arrays for each arm.
    """
    spectro, obs = configure_just_setup()
    
    # Enable optimized mode for ETC speedup
    if hasattr(ETC_py, 'set_calculation_mode'):
        ETC_py.set_calculation_mode('fast')
        
    if not ETC_py.MODEL_DATA_AVAILABLE:
        print("WARNING: Gemini/modeldata.py is missing. ETC calculations might be incomplete.")

    wave_in_nm = wave_in_a / 10.0
    fieldang = 0.675
    decent = 0.03

    spNoise = []
    spSample = []
    
    # Calculate ETC intrinsic noise components directly
    for ia in range(spectro.N_arms):
        noise, sky, sample = ETC_py.gsGetNoise(
            spectro, obs, ia, fieldang, exp_time, 0x0
        )
        spNoise.append(noise)
        spSample.append(sample)

    out_wave_a = []
    out_snr = []
    
    c_nm_s = 2.99792458e17 # Speed of light in nm/s
    
    for ia in range(spectro.N_arms):
        Npix = spectro.npix[ia]
        dl = spectro.dl[ia]
        lmin = spectro.lmin[ia]
        
        # Arm native wavegrid in nm
        arm_wave_nm = lmin + dl * (np.arange(Npix) + 0.5)
        # Store output wavegrid in Angstroms
        out_wave_a.append(arm_wave_nm * 10.0)
        
        arm_snr = np.zeros(Npix)
        
        # Interpolate the flux (1e-17 erg/s/cm2/A) onto the native arm grid
        flux_interp = np.interp(arm_wave_nm, wave_in_nm, flux_in_1e17, left=0, right=0)
        
        # Convert flux_lambda (in erg/s/cm2/nm) to flux_nu (in erg/s/cm2/Hz)
        # 1e-17 erg/s/cm2/A = 1e-16 erg/s/cm2/nm
        F_lambda_nm = flux_interp * 1e-16
        src_cont = F_lambda_nm * (arm_wave_nm**2) / c_nm_s
        
        for ipix in range(Npix):
            if src_cont[ipix] <= 0:
                continue
                
            lambda_nm = arm_wave_nm[ipix]
            
            # Atmospheric Transmission Lookups
            trans = ETC_py.gsAtmTrans(obs, lambda_nm, 0x0)
            
            # Counts conversion (per pixel per exposure)
            counts = (src_cont[ipix] * trans *
                     10 ** (-0.4 * ETC_py.gsGalactic_Alambda__EBV(lambda_nm) * obs.EBV) *
                     ETC_py.gsGeometricThroughput(spectro, obs, lambda_nm, 0.0, decent, fieldang, 0x0) *
                     ETC_py.gsFracTrace(spectro, obs, ia, lambda_nm, 0) *
                     ETC_py.PHOTONS_PER_ERG_1NM * lambda_nm * exp_time *
                     ETC_py.gsAeff(spectro, obs, ia, lambda_nm, fieldang) * 1e4)
                     
            counts *= c_nm_s * dl / (lambda_nm * lambda_nm)
            
            sample_factor = spSample[ia]
            noise_var = spNoise[ia][ipix]
            
            if sample_factor * counts + noise_var > 0:
                snr = counts / math.sqrt(sample_factor * counts + noise_var)
            else:
                snr = 0.0
                
            # Combine N exposures
            arm_snr[ipix] = snr * math.sqrt(n_exp)
            
        out_snr.append(arm_snr)
        
    return out_wave_a, out_snr

if __name__ == "__main__":
    test_w = np.linspace(3500, 10000, 1000)
    test_f = np.ones(1000) * 10.0 # 10 x 1e-17
    out_w, out_s = compute_stellar_snr(test_w, test_f)
    print("Test computation finished successfully.")
