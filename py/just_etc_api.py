"""
just_etc_api.py — JUST Exposure Time Calculator, User-Facing API
=================================================================
ETC_py_v1: high-level wrapper around ETC_py_optimized.py

Primary classes / functions
---------------------------
JUSTExposureTimeCalculator
    .set_obs_conditions(...)        — change observing conditions
    .compute_snr(wave_a, flux_flam, t_exp, n_exp)
        Given a spectrum (wavelength in Å, flux in FLAM) and exposure
        parameters, return per-pixel (wave, SNR, signal_counts,
        noise_variance) for each spectrograph arm.
    .solve_exposure_time(wave_a, flux_flam, target_snr, ref_wave_nm,
                         n_exp, ...)
        Given a spectrum and target SNR at a reference wavelength,
        return the required single-exposure time t_exp.

load_template(filepath)
    Load a non-stellar FITS BinTable template → (wave_Å, flux_FLAM).

normalize_to_mag(wave_a, flux_flam, target_mag, band)
    Normalise a spectrum to a given AB magnitude in a photometric band.
"""

import sys
import math
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

# ---------------------------------------------------------------------------
# Ensure local ETC engine is on the path
# ---------------------------------------------------------------------------
_V1_DIR = Path(__file__).resolve().parent
if str(_V1_DIR) not in sys.path:
    sys.path.insert(0, str(_V1_DIR))

import ETC_py_optimized as ETC_py

# ---------------------------------------------------------------------------
# Speed of light constants
# ---------------------------------------------------------------------------
_C_NM_S  = 2.99792458e17   # nm / s
_C_AA_S  = 2.99792458e18   # Å  / s

# ---------------------------------------------------------------------------
# Band definitions for magnitude normalisation
# ---------------------------------------------------------------------------
_BAND_EFF_WAVE_AA = {
    # SDSS / Johnson broad-bands (Å)
    'u':  3543.0,
    'g':  4770.0,
    'r':  6231.0,
    'i':  7625.0,
    'z':  9134.0,
    'V':  5510.0,
    'B':  4400.0,
    'R':  6500.0,
    'I':  8060.0,
    # JUST spectrograph arm centres (approximate)
    'arm0': 4675.0,   # blue  365–570 nm
    'arm1': 6425.0,   # green 540–745 nm
    'arm2': 8225.0,   # red   720–925 nm
}

# Approximate FWHM window (Å) used when computing in-band average flux
_BAND_WINDOW_AA = {
    'u': 600, 'g': 1100, 'r': 1200, 'i': 1200, 'z': 1200,
    'V': 800, 'B': 800, 'R': 1500, 'I': 1500,
    'arm0': 2000, 'arm1': 2000, 'arm2': 2000,
}


# ===========================================================================
# Template loading
# ===========================================================================

def load_template(filepath):
    """
    Load a non-stellar galaxy spectral template in FITS BinTable format.

    The templates in ETC_py_v1/templates/ follow the format produced by the
    LePhare / ETC_CIGALE template library:
      - HDU 0 : empty PrimaryHDU
      - HDU 1 : BinTableHDU with columns WAVELENGTH [Å] and FLUX/flux [FLAM,
                i.e. erg/s/cm²/Å, normalised internally by the library].

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    wave_aa : ndarray
        Wavelength array in Angstroms.
    flux_flam : ndarray
        Flux array in erg/s/cm²/Å  (FLAM).  The absolute normalisation is
        arbitrary — use normalize_to_mag() to set the desired flux level.
    """
    fpath = Path(filepath)
    if not fpath.exists():
        raise FileNotFoundError(f"Template not found: {fpath}")

    with fits.open(fpath) as hdul:
        # Find the first BinTableHDU
        tbl = None
        for hdu in hdul:
            if isinstance(hdu, fits.BinTableHDU):
                tbl = hdu
                break
        if tbl is None:
            raise ValueError(f"No BinTableHDU found in {fpath}")

        cols = [c.name.upper() for c in tbl.columns]
        # wavelength column
        wave_col = next((c for c in tbl.columns if c.name.upper() in ('WAVELENGTH', 'WAVE', 'LAMBDA')), None)
        flux_col = next((c for c in tbl.columns if c.name.upper() in ('FLUX', 'FLAM', 'FFLUX')), None)

        if wave_col is None or flux_col is None:
            raise ValueError(
                f"Cannot identify WAVELENGTH / FLUX columns in {fpath}. "
                f"Found: {[c.name for c in tbl.columns]}"
            )

        wave_aa  = np.asarray(tbl.data[wave_col.name], dtype=np.float64)
        flux_flam = np.asarray(tbl.data[flux_col.name], dtype=np.float64)

    # Ensure ascending wavelength order
    sort_idx = np.argsort(wave_aa)
    wave_aa   = wave_aa[sort_idx]
    flux_flam = flux_flam[sort_idx]

    return wave_aa, flux_flam


def list_templates(template_dir=None):
    """
    Return a dict  { 'category/filename': Path }  of available FITS templates.

    Parameters
    ----------
    template_dir : Path or None
        Defaults to ETC_py_v1/templates/.
    """
    if template_dir is None:
        template_dir = _V1_DIR / "templates"
    template_dir = Path(template_dir)
    found = {}
    for fits_file in sorted(template_dir.rglob("*.fits")):
        key = str(fits_file.relative_to(template_dir))
        found[key] = fits_file
    return found


# ===========================================================================
# Flux normalisation
# ===========================================================================

def normalize_to_mag(wave_aa, flux_flam, target_mag, band='r'):
    """
    Normalise a spectrum so that its AB magnitude in *band* equals *target_mag*.

    The normalised spectrum can be passed directly to
    JUSTExposureTimeCalculator.compute_snr().

    Parameters
    ----------
    wave_aa : ndarray
        Wavelength in Angstroms.
    flux_flam : ndarray
        Input flux in erg/s/cm²/Å (FLAM).  Absolute normalisation is arbitrary.
    target_mag : float
        Desired AB magnitude in *band*.
    band : str
        Photometric band name.  Supported:
        'u', 'g', 'r', 'i', 'z', 'V', 'B', 'R', 'I', 'arm0', 'arm1', 'arm2'.

    Returns
    -------
    flux_norm : ndarray
        Scaled flux in erg/s/cm²/Å.
    scale_factor : float
        Multiplicative scale factor applied.
    """
    band = band.lower() if band.lower() in _BAND_EFF_WAVE_AA else band
    if band not in _BAND_EFF_WAVE_AA:
        raise ValueError(
            f"Band '{band}' not recognised. Choose from {list(_BAND_EFF_WAVE_AA.keys())}"
        )

    eff_wave_aa = _BAND_EFF_WAVE_AA[band]
    half_window = _BAND_WINDOW_AA.get(band, 1000) / 2.0

    # Target flux density in erg/s/cm²/Hz (AB system)
    target_f_nu = 3.631e-20 * 10.0 ** (-0.4 * target_mag)

    # Convert to FLAM at the effective wavelength: F_λ = F_ν * c / λ²
    target_f_lambda = target_f_nu * _C_AA_S / (eff_wave_aa ** 2)   # erg/s/cm²/Å

    # Current mean FLAM in the band window
    mask = (wave_aa >= eff_wave_aa - half_window) & (wave_aa <= eff_wave_aa + half_window)
    if not np.any(mask):
        # Fall back to simple interpolation if the window is not covered
        current_f_lambda = float(np.interp(eff_wave_aa, wave_aa, flux_flam))
    else:
        current_f_lambda = float(np.mean(flux_flam[mask]))

    if current_f_lambda <= 0:
        raise ValueError(
            f"Template flux is zero or negative around {eff_wave_aa:.1f} Å. "
            "Cannot normalise."
        )

    scale_factor = target_f_lambda / current_f_lambda
    return flux_flam * scale_factor, scale_factor


# ===========================================================================
# Main ETC class
# ===========================================================================

class JUSTExposureTimeCalculator:
    """
    High-level wrapper around ETC_py_optimized for the JUST spectrograph.

    Typical usage
    -------------
    >>> etc = JUSTExposureTimeCalculator()
    >>> wave, flux = load_template('galaxy/elliptical_001.fits')
    >>> wave, flux = normalize_to_mag(wave, flux, target_mag=20.0, band='r')
    >>> result = etc.compute_snr(wave, flux, t_exp=900.0, n_exp=4)
    >>> for arm in result:
    ...     print(arm['arm'], arm['wave_nm'], arm['snr'])
    """

    # Default JUST observing conditions
    _DEFAULTS = dict(
        skytype_hex='10003',
        seeing_fwhm_800=0.8,    # arcsec FWHM at 800 nm
        zenith_angle=45.0,      # degrees
        ebv=0.03,               # E(B-V)
        field_angle=0.675,      # degrees (typical field position)
        decenter=0.03,          # arcsec (fiber mis-centring)
        lunar_za=135.0,         # Moon zenith angle (>90  →  below horizon)
        lunar_angle=90.0,       # Moon–target separation (degrees)
        lunar_phase=0.25,       # Moon phase (0=new, 0.5=full, 1=new)
        sysfrac=0.01,           # Systematic sky-subtraction floor (rms/pixel)
        diffuse_stray=0.02,     # Diffuse stray light fraction
        r_eff=0.0,              # Galaxy effective radius (arcsec, 0 for point source)
    )

    def __init__(self, spec_file=None, calc_mode='accurate'):
        """
        Parameters
        ----------
        spec_file : str or Path, optional
            Path to the JUST spectrograph configuration file.
            Defaults to ETC_py_v1/spec.dat.
        calc_mode : str
            ETC calculation precision mode: 'fast', 'balanced', or 'accurate'.
        """
        if spec_file is None:
            spec_file = _V1_DIR / 'spec.dat'
        self.spec_file = Path(spec_file)
        if not self.spec_file.exists():
            raise FileNotFoundError(f"Spectrograph config not found: {self.spec_file}")

        self.calc_mode = calc_mode
        self._obs_params = dict(self._DEFAULTS)  # mutable copy

        # Load spectrograph configuration once
        self._spectro = ETC_py.SpectroAttrib()
        ETC_py.gsReadSpectrographConfig(str(self.spec_file), self._spectro)

        # Check for atmospheric data tables
        if not ETC_py.MODEL_DATA_AVAILABLE:
            warnings.warn(
                "ETC_py_optimized: modeldata.py not found — sky emission and "
                "atmospheric transmission tables are unavailable. "
                "Results may be incorrect.",
                RuntimeWarning
            )

    # ------------------------------------------------------------------
    # Public: configure observing conditions
    # ------------------------------------------------------------------

    def set_obs_conditions(self, **kwargs):
        """
        Update observing conditions.  Accepts any keyword from _DEFAULTS:
            seeing_fwhm_800, zenith_angle, ebv, field_angle, decenter,
            lunar_za, lunar_angle, lunar_phase, sysfrac, diffuse_stray,
            skytype_hex.
        """
        unknown = set(kwargs) - set(self._DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown observing parameters: {unknown}")
        self._obs_params.update(kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_obs(self):
        """Build an ObsAttrib from current _obs_params."""
        obs = ETC_py.ObsAttrib()
        obs.skytype       = int(self._obs_params['skytype_hex'], 16)
        obs.seeing_fwhm_800 = self._obs_params['seeing_fwhm_800']
        obs.zenithangle   = self._obs_params['zenith_angle']
        obs.EBV           = self._obs_params['ebv']
        obs.lunarZA       = self._obs_params['lunar_za']
        obs.lunarangle    = self._obs_params['lunar_angle']
        obs.lunarphase    = self._obs_params['lunar_phase']
        return obs

    def _configure_spectro(self, n_exp):
        """
        Apply systematics to the spectrograph object.

        sysfrac scales with sqrt(n_exp) because it represents a correlated
        floor that does not average down (as in the original C code).
        """
        sp = self._spectro
        sp.sysfrac       = self._obs_params['sysfrac'] * math.sqrt(n_exp)
        sp.diffuse_stray = self._obs_params['diffuse_stray']
        return sp

    def _set_mode(self):
        if hasattr(ETC_py, 'set_calculation_mode'):
            ETC_py.set_calculation_mode(self.calc_mode)

    def _arm_noise(self, obs, t_exp):
        """
        Pre-compute the noise vector for every arm.

        Returns
        -------
        list of (noise_vec, sample_factor) per arm.
        """
        sp = self._spectro
        fa = self._obs_params['field_angle']
        results = []
        for ia in range(sp.N_arms):
            noise, _sky, sf = ETC_py.gsGetNoise(sp, obs, ia, fa, t_exp, 0x0)
            results.append((noise, sf))
        return results

    @staticmethod
    def _flam_to_fnu_per_pixel(wave_nm, flux_flam_aa, dl_nm):
        """
        Convert F_λ  [erg/s/cm²/Å]  →  F_ν [erg/s/cm²/Hz] per l-pixel.

        F_ν = F_λ · λ² / c                   (λ in cm, c in cm/s)
        Then multiply by (Δλ_pixel / c) to get per-pixel.

        Here we work entirely in nm so:
            F_ν = F_λ [erg/s/cm²/Å] × 10 [Å/nm] × λ_nm² / c_nm_s

        Parameters
        ----------
        wave_nm : float
            Wavelength [nm].
        flux_flam_aa : float
            Flux density [erg/s/cm²/Å].
        dl_nm : float
            Pixel width [nm].

        Returns
        -------
        float  : F_ν per pixel [erg/s/cm²/Hz · (–)]  (same units expected by the ETC engine).
        """
        # F_lambda_nm = flux in erg/s/cm2/nm (1 Å = 0.1 nm, so *10)
        f_lambda_nm = flux_flam_aa * 10.0
        # F_nu = F_lambda * lambda^2 / c   (both in nm/nm-units system)
        f_nu = f_lambda_nm * wave_nm ** 2 / _C_NM_S   # erg/s/cm2/Hz
        # Multiply by c/lambda^2 * dl gives counts per pixel — but the ETC
        # engine expects F_nu (per Hz) and handles the dl conversion internally.
        return f_nu

    # ------------------------------------------------------------------
    # Public: SNR computation
    # ------------------------------------------------------------------

    def compute_snr(self, wave_aa, flux_flam, t_exp, n_exp=1):
        """
        Compute continuum SNR for an arbitrary input spectrum.

        Parameters
        ----------
        wave_aa : array-like
            Wavelength in Angstroms.
        flux_flam : array-like
            Flux density in erg/s/cm²/Å (FLAM).  Use normalize_to_mag() to
            set the correct brightness before calling this method.
        t_exp : float
            Single-exposure time in seconds.
        n_exp : int
            Number of co-added exposures.

        Returns
        -------
        list of dict, one per spectrograph arm, each containing:
            arm         : int    — arm index (0, 1, 2)
            wave_nm     : ndarray — wavelength grid [nm]
            wave_aa     : ndarray — wavelength grid [Å]
            snr         : ndarray — SNR per pixel (combined n_exp exposures)
            signal      : ndarray — source photon counts [e⁻/pixel/exposure]
            noise_var   : ndarray — total noise variance [e⁻²/pixel/exposure]
            snr_1exp    : ndarray — SNR per single exposure
        """
        self._set_mode()
        wave_aa  = np.asarray(wave_aa,  dtype=np.float64)
        flux_flam = np.asarray(flux_flam, dtype=np.float64)
        wave_nm   = wave_aa / 10.0

        obs = self._make_obs()
        sp  = self._configure_spectro(n_exp)
        fa  = self._obs_params['field_angle']
        de  = self._obs_params['decenter']

        noise_list = self._arm_noise(obs, t_exp)

        results = []
        for ia in range(sp.N_arms):
            Npix   = sp.npix[ia]
            lmin   = sp.lmin[ia]   # nm
            dl     = sp.dl[ia]     # nm/pixel
            noise_vec, sf = noise_list[ia]

            arm_wave_nm = lmin + dl * (np.arange(Npix) + 0.5)

            # Interpolate the input spectrum onto the arm's native wavelength grid
            flux_interp_aa = np.interp(arm_wave_nm, wave_nm, flux_flam, left=0.0, right=0.0)

            # Convert FLAM [erg/s/cm²/Å] → F_nu [erg/s/cm²/Hz]
            # F_nu = F_lam (erg/s/cm²/nm) × λ_nm² / c_nm_s
            #   where F_lam_nm = F_lam_aa × 10
            f_nu = flux_interp_aa * 10.0 * arm_wave_nm ** 2 / _C_NM_S

            snr_1exp  = np.zeros(Npix)
            signal    = np.zeros(Npix)
            noise_out = np.zeros(Npix)

            for ipix in range(Npix):
                if f_nu[ipix] <= 0.0:
                    continue
                lambda_nm = arm_wave_nm[ipix]

                # Atmospheric transmission at this pixel
                atm = ETC_py.gsAtmTrans(obs, lambda_nm, 0x0)

                # Galactic extinction
                ext = 10.0 ** (-0.4 * ETC_py.gsGalactic_Alambda__EBV(lambda_nm) * obs.EBV)

                # Geometric throughput
                re  = self._obs_params.get('r_eff', 0.0)
                geo = ETC_py.gsGeometricThroughput(sp, obs, lambda_nm, re, de, fa, 0x0)

                # Fraction of light in spectral trace
                frac_trace = ETC_py.gsFracTrace(sp, obs, ia, lambda_nm, 0)

                # Effective collecting area [cm²]
                aeff = ETC_py.gsAeff(sp, obs, ia, lambda_nm, fa) * 1e4  # m² → cm²

                # Source counts  [e⁻ / pixel / exposure]
                # counts = F_nu [erg/s/cm2/Hz] × atm × ext × geo × frac_trace
                #          × PHOTONS_PER_ERG_1NM × λ [nm] × t_exp [s]
                #          × Aeff [cm²]
                #          × c/λ² × dl   (conversion Hz⁻¹ → pixel)
                counts = (f_nu[ipix] * atm * ext * geo * frac_trace
                          * ETC_py.PHOTONS_PER_ERG_1NM * lambda_nm * t_exp
                          * aeff
                          * (_C_NM_S * dl / (lambda_nm * lambda_nm)))

                noise_var = sf * counts + noise_vec[ipix]

                if noise_var > 0:
                    snr_1exp[ipix] = counts / math.sqrt(noise_var)

                signal[ipix]    = counts
                noise_out[ipix] = noise_var

            results.append(dict(
                arm       = ia,
                wave_nm   = arm_wave_nm,
                wave_aa   = arm_wave_nm * 10.0,
                snr_1exp  = snr_1exp,
                snr       = snr_1exp * math.sqrt(n_exp),
                signal    = signal,
                noise_var = noise_out,
            ))

        return results

    # ------------------------------------------------------------------
    # Public: simulate mock observation (intrinsic vs noisy observed spectrum)
    # ------------------------------------------------------------------

    def simulate_mock_observation(self, wave_aa, flux_flam, t_exp, n_exp=1, seed=None):
        """
        Simulate a mock 1D observed spectrum for JUST based on physical noise modeling.

        Parameters
        ----------
        wave_aa : array-like
            Intrinsic input wavelength in Angstroms.
        flux_flam : array-like
            Intrinsic input flux density in erg/s/cm²/Å (FLAM).
        t_exp : float
            Single-exposure time in seconds.
        n_exp : int
            Number of co-added exposures.
        seed : int or None, optional
            Random seed for reproducible noise realization.

        Returns
        -------
        list of dict, one per spectrograph arm, each containing:
            arm            : int — arm index (0, 1, 2)
            wave_aa        : ndarray — wavelength grid [Å]
            flux_intrinsic : ndarray — input flux interpolated onto arm grid [erg/s/cm²/Å]
            flux_mock      : ndarray — simulated noisy observed spectrum [erg/s/cm²/Å]
            signal_e       : ndarray — total co-added source photon counts [e⁻]
            noise_e        : ndarray — total noise standard deviation [e⁻]
            snr            : ndarray — per-pixel SNR
        """
        results_snr = self.compute_snr(wave_aa, flux_flam, t_exp=t_exp, n_exp=n_exp)
        rng = np.random.default_rng(seed)

        mock_results = []
        for arm in results_snr:
            ia      = arm['arm']
            w_arm   = arm['wave_aa']
            snr     = arm['snr']
            sig_exp = arm['signal']     # e⁻ per pixel per exposure
            var_exp = arm['noise_var']  # e⁻² per pixel per exposure

            # Total counts and noise variance over n_exp co-added exposures
            sig_tot = sig_exp * n_exp
            var_tot = var_exp * n_exp
            std_tot = np.sqrt(np.maximum(var_tot, 0.0))

            # Interpolate intrinsic flux density onto arm wavelength grid
            f_int = np.interp(w_arm, wave_aa, flux_flam, left=0.0, right=0.0)

            # Draw random Gaussian noise realization in electron space (symmetric unbiased noise)
            noise_e_rand = rng.normal(0.0, std_tot)
            obs_e = sig_tot + noise_e_rand

            # Convert noisy electrons back to flux density [erg/s/cm²/Å]
            mask = sig_tot > 0.0
            f_mock = np.zeros_like(f_int)
            f_mock[mask] = f_int[mask] * (obs_e[mask] / sig_tot[mask])

            mock_results.append(dict(
                arm            = ia,
                wave_aa        = w_arm,
                flux_intrinsic = f_int,
                flux_mock      = f_mock,
                signal_e       = sig_tot,
                noise_e        = std_tot,
                snr            = snr,
            ))

        return mock_results

    # ------------------------------------------------------------------
    # Public: solve for exposure time
    # ------------------------------------------------------------------

    def solve_exposure_time(self, wave_aa, flux_flam, target_snr,
                            ref_wave_nm, n_exp=1,
                            t_exp_init=600.0,
                            t_exp_min=10.0, t_exp_max=86400.0,
                            rtol=0.01, max_iter=12):
        """
        Find the single-exposure time *t_exp* such that the continuum SNR
        at *ref_wave_nm* (after *n_exp* exposures) equals *target_snr*.

        Algorithm
        ---------
        1. Compute SNR(t_init).
        2. Use SNR ∝ √t approximation to refine the initial estimate.
        3. Iterate with the Illinois secant/bisection hybrid until
           |SNR - target_snr| / target_snr < rtol  or max_iter reached.

        Parameters
        ----------
        wave_aa : array-like
            Wavelength in Angstroms.
        flux_flam : array-like
            Flux in erg/s/cm²/Å.
        target_snr : float
            Desired SNR (after n_exp exposures).
        ref_wave_nm : float
            Reference wavelength in nm at which the SNR condition is applied.
            Must lie within the wavelength range of at least one spectrograph arm.
        n_exp : int
            Number of co-added exposures.
        t_exp_init : float
            Initial guess for single-exposure time [s].
        t_exp_min, t_exp_max : float
            Search bounds [s].
        rtol : float
            Relative tolerance on SNR convergence (default 1 %).
        max_iter : int
            Maximum number of ETC evaluations.

        Returns
        -------
        dict with keys:
            t_exp       : float — solved single-exposure time [s]
            t_total     : float — total integration time = t_exp × n_exp [s]
            snr_achieved: float — SNR at ref_wave_nm at convergence
            n_iter      : int   — number of iterations used
            converged   : bool
            arm         : int   — arm index where ref_wave_nm was found
        """
        self._set_mode()

        def _snr_at_t(t):
            """Return SNR at ref_wave_nm for a given t_exp."""
            r = self.compute_snr(wave_aa, flux_flam, t_exp=t, n_exp=n_exp)
            for arm_res in r:
                wn = arm_res['wave_nm']
                if wn[0] <= ref_wave_nm <= wn[-1]:
                    snr_arr = arm_res['snr']
                    arm_idx = arm_res['arm']
                    # Interpolate SNR at the exact reference wavelength
                    snr_val = float(np.interp(ref_wave_nm, wn, snr_arr))
                    return snr_val, arm_idx
            raise ValueError(
                f"ref_wave_nm={ref_wave_nm:.1f} nm is not covered by any "
                f"JUST spectrograph arm (arms cover "
                f"{self._spectro.lmin[0]:.0f}–{self._spectro.lmax[self._spectro.N_arms-1]:.0f} nm)."
            )

        # --- Step 1: initial evaluation ---
        snr0, arm_found = _snr_at_t(t_exp_init)
        n_iter = 1

        if snr0 <= 0:
            raise RuntimeError(
                f"SNR is zero at ref_wave_nm={ref_wave_nm:.1f} nm with "
                f"t_exp={t_exp_init:.1f} s. Check that the spectrum has "
                "non-zero flux at that wavelength."
            )

        # --- Step 2: √t scaling estimate ---
        t_guess = t_exp_init * (target_snr / snr0) ** 2
        t_guess = float(np.clip(t_guess, t_exp_min, t_exp_max))

        snr1, _ = _snr_at_t(t_guess)
        n_iter += 1

        # Check convergence already
        if abs(snr1 - target_snr) / target_snr < rtol:
            return dict(t_exp=t_guess, t_total=t_guess * n_exp,
                        snr_achieved=snr1, n_iter=n_iter,
                        converged=True, arm=arm_found)

        # --- Step 3: Illinois false-position / bisection hybrid ---
        # We need to bracket the root  f(t) = SNR(t) - target_snr = 0.
        # SNR increases monotonically with t, so:
        #   f(t_lo) < 0  <=>  SNR(t_lo) < target_snr  → t_lo is too short
        #   f(t_hi) > 0  <=>  SNR(t_hi) > target_snr  → t_hi is too long

        # Build an initial bracket from the two points we already have
        pts = sorted([(t_exp_init, snr0), (t_guess, snr1)], key=lambda x: x[0])
        (t_lo, s_lo), (t_hi, s_hi) = pts

        # Expand bracket if both points are on the same side
        if s_lo > target_snr and s_hi > target_snr:
            # Both too fast: shrink toward t_min
            t_lo, s_lo = t_exp_min, _snr_at_t(t_exp_min)[0]; n_iter += 1
        elif s_lo < target_snr and s_hi < target_snr:
            # Both too slow: expand toward t_max
            t_hi, s_hi = t_exp_max, _snr_at_t(t_exp_max)[0]; n_iter += 1

        converged = False
        for _ in range(max_iter - n_iter):
            if n_iter >= max_iter:
                break
            if abs(s_hi - target_snr) / target_snr < rtol:
                t_guess, snr1 = t_hi, s_hi;  converged = True;  break
            if abs(s_lo - target_snr) / target_snr < rtol:
                t_guess, snr1 = t_lo, s_lo;  converged = True;  break

            # Illinois secant step
            ds = s_hi - s_lo
            if abs(ds) < 1e-12:
                t_new = 0.5 * (t_lo + t_hi)
            else:
                t_new = t_lo + (target_snr - s_lo) * (t_hi - t_lo) / ds
            t_new = float(np.clip(t_new, t_exp_min, t_exp_max))

            snr_new, _ = _snr_at_t(t_new)
            n_iter += 1

            if snr_new < target_snr:
                t_lo, s_lo = t_new, snr_new
            else:
                t_hi, s_hi = t_new, snr_new

            t_guess, snr1 = t_new, snr_new
            if abs(snr_new - target_snr) / target_snr < rtol:
                converged = True;  break

        return dict(t_exp=t_guess, t_total=t_guess * n_exp,
                    snr_achieved=snr1, n_iter=n_iter,
                    converged=converged, arm=arm_found)


# ===========================================================================
# Quick self-test
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("JUST ETC API -- Self-test")
    print("=" * 60)

    # --- 1. List available templates ---
    templates = list_templates()
    print(f"\nFound {len(templates)} templates:")
    for k in list(templates.keys())[:6]:
        print(f"  {k}")
    if len(templates) > 6:
        print(f"  ... and {len(templates)-6} more.")

    # --- 2. Load and normalise an elliptical galaxy template ---
    tpl_path = _V1_DIR / "templates" / "galaxy" / "elliptical_001.fits"
    if tpl_path.exists():
        wave, flux = load_template(tpl_path)
        print(f"\nLoaded elliptical template: {len(wave)} points, "
              f"λ=[{wave[0]:.0f}, {wave[-1]:.0f}] Å")

        flux_norm, sf = normalize_to_mag(wave, flux, target_mag=20.0, band='r')
        print(f"Normalised to r=20 AB mag (scale factor = {sf:.4e})")

        # --- 3. Compute SNR ---
        print("\nComputing SNR (t_exp=900s, n_exp=4, 'fast' mode)...")
        etc = JUSTExposureTimeCalculator(calc_mode='fast')
        result = etc.compute_snr(wave, flux_norm, t_exp=900.0, n_exp=4)

        for arm in result:
            ia   = arm['arm']
            snr  = arm['snr']
            wn   = arm['wave_nm']
            mask = snr > 0
            if np.any(mask):
                msnr = np.nanmedian(snr[mask])
                print(f"  Arm {ia} ({wn[0]:.0f}–{wn[-1]:.0f} nm): "
                      f"median SNR = {msnr:.2f}")
            else:
                print(f"  Arm {ia}: no valid SNR pixels")

        # --- 4. Solve for t_exp to reach SNR=10 at 600 nm ---
        print("\nSolving for t_exp to reach SNR=10 at 600 nm (n_exp=4)...")
        sol = etc.solve_exposure_time(wave, flux_norm,
                                      target_snr=10.0, ref_wave_nm=600.0,
                                      n_exp=4, t_exp_init=900.0)
        print(f"  t_exp  = {sol['t_exp']:.1f} s  "
              f"(total = {sol['t_total']/60:.1f} min)")
        print(f"  SNR achieved = {sol['snr_achieved']:.3f}")
        print(f"  Converged: {sol['converged']}  ({sol['n_iter']} iterations)")
    else:
        print(f"\nTemplate not found at {tpl_path} — skipping compute_snr test.")

    print("\nSelf-test complete.")
