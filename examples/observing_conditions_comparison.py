"""Compare the effects of seeing and source size on JUST ETC predictions."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from just_etc import JUSTExposureTimeCalculator, load_template, normalize_to_mag

COLORS = ("#1f77b4", "#2ca02c", "#ff7f0e", "#d62728")
PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
}


def _smooth(values, width=11):
    kernel = np.ones(width) / width
    return np.convolve(np.pad(values, width // 2, mode="edge"), kernel, mode="valid")[: len(values)]


def generate_conditions_plot(
    template="galaxy/sb2_b2004a_001.fits",
    redshift=0.6,
    magnitude=20.5,
    exposure_time=2500.0,
    exposures=1,
    output=Path("output/etc_observing_conditions.png"),
):
    """Save a compact comparison of seeing and effective-radius scenarios."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wave, flux = load_template(template)
    wave = wave * (1 + redshift)
    flux, _ = normalize_to_mag(wave, flux, target_mag=magnitude, band="r")
    calculator = JUSTExposureTimeCalculator(calc_mode="fast")

    scenarios = (
        (
            "Atmospheric seeing",
            [(0.6, '0.6″'), (0.8, '0.8″'), (1.2, '1.2″'), (1.5, '1.5″')],
        ),
        ("Source effective radius", [(0.0, 'Point source'), (0.3, 'Compact · 0.3″'),
                                     (1.0, 'Extended · 1.0″'), (2.0, 'Extended · 2.0″')]),
    )

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        for index, (title, values) in enumerate(scenarios):
            ax = axes[index]
            for color, (value, label) in zip(COLORS, values):
                conditions = {"seeing_fwhm_800": value, "r_eff": 0.0} if index == 0 else {
                    "seeing_fwhm_800": 0.8, "r_eff": value
                }
                calculator.set_obs_conditions(**conditions)
                results = calculator.compute_snr(
                    wave, flux, t_exp=exposure_time, n_exp=exposures
                )
                for arm_result in results:
                    arm = arm_result["arm"]
                    ax.plot(arm_result["wave_nm"], _smooth(arm_result["snr"]),
                            color=color, lw=1.6, label=label if arm == 0 else None)

            ax.axhline(5, color="gray", ls="--", lw=1.0, alpha=0.8, label="S/N = 5")
            subtitle = "Point source" if index == 0 else 'Seeing = 0.8″'
            ax.set_title(f"{title} ({subtitle})", fontsize=13)
            ax.set_ylabel("S/N per pixel")
            ax.set_ylim(bottom=0)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(ncol=5, loc="upper right", fontsize=9)

        axes[-1].set_xlabel("Observed wavelength [nm]")
        axes[-1].set_xlim(350, 950)
        fig.suptitle(
            f"Sensitivity to Observing Conditions (r = {magnitude:g} AB mag, z = {redshift:g}, "
            f"{exposures} × {exposure_time:g} s)",
            fontsize=15,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved observing-condition comparison to {output.resolve()}")
    return output


def _parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="galaxy/sb2_b2004a_001.fits")
    parser.add_argument("--redshift", type=float, default=0.6)
    parser.add_argument("--magnitude", type=float, default=20.5)
    parser.add_argument("--exposure-time", type=float, default=2500.0)
    parser.add_argument("--exposures", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("output/etc_observing_conditions.png"))
    return parser.parse_args()


if __name__ == "__main__":
    generate_conditions_plot(**vars(_parse_args()))
