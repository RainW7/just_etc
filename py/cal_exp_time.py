"""
cal_exp_time.py
================
CLI tool and script for calculating exposure times or SNR using JUST ETC.

Usage Examples:
  1. Default run (Demo solve target SNR = 8.0 at r=21.0 mag):
     python cal_exp_time.py

  2. Compute SNR for a given exposure time (e.g. 1800s for r=21.5 mag star):
     python cal_exp_time.py --mag 21.5 --band r --texp 1800 --target point

  3. Solve required exposure time for target SNR (e.g. SNR=5.0 for r=22.0 mag extended galaxy with reff=0.6"):
     python cal_exp_time.py --target-snr 5.0 --mag 22.0 --band r --target extended --reff 0.6
"""

import sys
import argparse
from pathlib import Path

_V1_DIR = Path(__file__).resolve().parent
if str(_V1_DIR) not in sys.path:
    sys.path.insert(0, str(_V1_DIR))

from just_etc_api import JUSTExposureTimeCalculator, load_template, normalize_to_mag


def main():
    parser = argparse.ArgumentParser(description="JUST Spectrograph Exposure Time & SNR Calculator")
    parser.add_argument("--mag", type=float, default=21.0, help="Target magnitude (default: 21.0)")
    parser.add_argument("--band", type=str, default="r", help="Photometric band (default: 'r')")
    parser.add_argument("--target", type=str, choices=["point", "extended"], default="extended", help="Target type: 'point' or 'extended' (default: 'extended')")
    parser.add_argument("--reff", type=float, default=0.5, help="Effective half-light radius in arcsec for extended target (default: 0.5)")
    parser.add_argument("--template", type=str, default="starburst/sb4_kinney_fuv_001.fits", help="Template SED path or category/filename (default: 'starburst/sb4_kinney_fuv_001.fits')")
    parser.add_argument("--seeing", type=float, default=1.0, help="Seeing FWHM at 800nm in arcsec (default: 1.0)")
    parser.add_argument("--nexp", type=int, default=4, help="Number of split exposures (default: 4)")
    parser.add_argument("--mode", type=str, choices=["fast", "balanced", "accurate"], default="fast", help="ETC calculation mode (default: 'fast')")

    # Calculation mode switches
    parser.add_argument("--texp", type=float, default=None, help="Single exposure time in seconds (if specified, calculates SNR)")
    parser.add_argument("--target-snr", type=float, default=None, help="Target SNR to solve exposure time for (e.g. 5.0 or 8.0)")
    parser.add_argument("--ref-wave", type=float, default=600.0, help="Reference wavelength in nm for exposure solver (default: 600.0)")

    args = parser.parse_args()

    # 1. Initialize ETC engine
    etc = JUSTExposureTimeCalculator(calc_mode=args.mode)
    r_eff_val = args.reff if args.target == "extended" else 0.0
    etc.set_obs_conditions(seeing_fwhm_800=args.seeing, r_eff=r_eff_val)

    # 2. Resolve & Load SED template
    template_path = Path(args.template)
    if not template_path.is_absolute():
        template_path = _V1_DIR / "templates" / args.template
        if not template_path.exists() and not args.template.startswith("templates/"):
            template_path = _V1_DIR / "templates" / args.template

    if not template_path.exists():
        print(f"❌ Error: Template file not found: {template_path}")
        sys.exit(1)

    wave_aa, flux_flam = load_template(template_path)

    # 3. Normalize flux to target magnitude
    flux_norm, scale_factor = normalize_to_mag(wave_aa, flux_flam, target_mag=args.mag, band=args.band)
    print(f"✅ Loaded template: {template_path.name}")
    print(f"✅ Spectrum normalized to {args.band}={args.mag:.2f} AB mag (scale factor = {scale_factor:.4e})")

    # 4. Mode branch
    if args.texp is not None:
        # --- Mode A: Compute SNR given texp ---
        print(f"\n📊 Computing SNR for t_exp = {args.texp:.1f}s x {args.nexp} exposures (Total {args.texp * args.nexp:.1f}s)...")
        results = etc.compute_snr(
            wave_aa=wave_aa,
            flux_flam=flux_norm,
            t_exp=args.texp,
            n_exp=args.nexp
        )

        arm_names = {0: 'BLUE (Arm 0)', 1: 'RED (Arm 1)', 2: 'Z (Arm 2)'}
        print("-" * 55)
        print(f"🎯 SNR Calculation Results ({args.target.upper()} target, Seeing = {args.seeing}\"):")
        for arm_info in results:
            arm_idx = arm_info['arm']
            name = arm_names.get(arm_idx, f"Arm {arm_idx}")
            w_min = arm_info['wave_nm'][0]
            w_max = arm_info['wave_nm'][-1]
            snr_array = arm_info['snr']
            snr_mean = float(np.mean(snr_array))
            snr_median = float(np.median(snr_array))
            print(f"🔹 {name} ({w_min:.1f} - {w_max:.1f} nm): Mean SNR = {snr_mean:.2f}, Median SNR = {snr_median:.2f}")
        print("-" * 55)

    else:
        # --- Mode B: Solve Exposure Time given target_snr ---
        target_snr = args.target_snr if args.target_snr is not None else 8.0
        print(f"\n⏱️ Solving exposure time to reach SNR = {target_snr:.1f} at {args.ref_wave:.1f} nm ({args.nexp} exposures)...")

        solution = etc.solve_exposure_time(
            wave_aa=wave_aa,
            flux_flam=flux_norm,
            target_snr=target_snr,
            ref_wave_nm=args.ref_wave,
            n_exp=args.nexp,
            t_exp_init=600.0
        )

        print("-" * 55)
        print(f"🎯 Calculation Complete (Converged in {solution['n_iter']} iterations):")
        print(f"🔹 Recommended Single Exp Time: {solution['t_exp']:.1f} sec")
        print(f"🔹 Exposure Count              : {args.nexp}")
        print(f"🔹 Total Observation Time       : {solution['t_total']/3600:.2f} hours ({solution['t_total']:.1f} sec)")
        print(f"🔹 Achieved SNR                 : {solution['snr_achieved']:.3f} (at {args.ref_wave:.1f} nm in Arm {solution['arm']})")
        print("-" * 55)


if __name__ == "__main__":
    import numpy as np
    main()
