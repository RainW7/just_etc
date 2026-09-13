"""Create the primary JUST ETC spectrum and signal-to-noise demonstration."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from just_etc import JUSTExposureTimeCalculator, load_template, normalize_to_mag

ARM_COLORS = ("#1f77b4", "#2ca02c", "#d62728")
PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
}
EMISSION_LINES = {
    r"Ly$\alpha$": 1215.67,
    r"[O II]": 3727.0,
    r"H$\beta$": 4861.3,
    r"[O III]": 5006.8,
    r"H$\alpha$": 6562.8,
}


def _smooth(values, window=9):
    """Return a centered moving average without depressing either edge."""
    window = min(window, len(values))
    if window < 2:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _style_axes(axes):
    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.6)


def generate_demo_plot(
    template="galaxy/elliptical_001.fits",
    redshift=0.3,
    magnitude=20.0,
    band="r",
    exposure_time=900.0,
    exposures=1,
    output=Path("output/etc_demo_plot.png"),
):
    """Run a representative ETC calculation and save a three-panel overview."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    wave_rest, flux = load_template(template)
    wave_observed = wave_rest * (1.0 + redshift)
    flux, _ = normalize_to_mag(wave_observed, flux, target_mag=magnitude, band=band)

    calculator = JUSTExposureTimeCalculator(calc_mode="fast")
    results = calculator.compute_snr(
        wave_observed, flux, t_exp=exposure_time, n_exp=exposures
    )

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            3, 1, figsize=(11, 10), sharex=True,
            gridspec_kw={"height_ratios": (1.1, 1, 1), "hspace": 0.08},
        )
        spectrum_ax, snr_ax, counts_ax = axes
        _style_axes(axes)

        spectrum_ax.plot(wave_observed / 10.0, flux, color="black", lw=1.2, alpha=0.85)
        spectrum_ax.set_yscale("log")
        spectrum_ax.set_ylabel(
            "Flux density\n" + r"[erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]"
        )
        spectrum_ax.set_title(
            f"JUST ETC Observation Preview ({band} = {magnitude:g} AB mag, z = {redshift:g})",
            fontsize=13, pad=10,
        )
        spectrum_ax.text(
            0.99, 0.97, f"{exposures} × {exposure_time:g} s  |  {template}",
            transform=spectrum_ax.transAxes, ha="right", va="top",
            color="dimgray", fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "alpha": 0.88, "edgecolor": "#cccccc"},
        )

        for name, rest_wave in EMISSION_LINES.items():
            observed_nm = rest_wave * (1 + redshift) / 10.0
            if 350 <= observed_nm <= 950:
                for ax in axes:
                    ax.axvline(observed_nm, color="gray", ls=":", lw=1.0, alpha=0.7, zorder=0)
                spectrum_ax.annotate(
                    name, xy=(observed_nm, 0.04), xycoords=("data", "axes fraction"),
                    rotation=90, ha="right", va="bottom", color="dimgray", fontsize=9,
                )

        for arm_result in results:
            arm = arm_result["arm"]
            color = ARM_COLORS[arm % len(ARM_COLORS)]
            wavelength = arm_result["wave_nm"]
            snr = arm_result["snr"]
            snr_ax.plot(wavelength, snr, color=color, alpha=0.16, lw=0.6)
            snr_ax.plot(wavelength, _smooth(snr), color=color, lw=1.8, label=f"Arm {arm}")

            total_signal = arm_result["signal"] * exposures
            total_noise = np.sqrt(arm_result["noise_var"] * exposures)
            counts_ax.plot(
                wavelength, total_signal, color=color, lw=1.5,
                label=f"Arm {arm} signal",
            )
            counts_ax.plot(wavelength, total_noise, color=color, lw=1.1, ls="--", alpha=0.85)

        snr_ax.axhline(5, color="gray", ls="--", lw=1.2, alpha=0.8, label="S/N = 5")
        snr_ax.set_ylabel("S/N per pixel")
        snr_ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        snr_ax.legend(ncol=4, loc="upper right", fontsize=9)
        counts_ax.set_yscale("log")
        counts_ax.set_ylabel(r"Electrons pixel$^{-1}$")
        counts_ax.set_xlabel("Observed wavelength [nm]")
        counts_ax.set_xlim(350, 950)
        counts_ax.legend(ncol=3, loc="upper left", fontsize=9)
        counts_ax.text(0.995, 0.06, "Dashed curves: total noise", transform=counts_ax.transAxes,
                       ha="right", color="dimgray", fontsize=9)

        fig.align_ylabels(axes)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved ETC overview to {output.resolve()}")
    return output


def _parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="starburst/sb3_kinney_fuv_001.fits")
    parser.add_argument("--redshift", type=float, default=0.3)
    parser.add_argument("--magnitude", type=float, default=20.0)
    parser.add_argument("--band", default="r")
    parser.add_argument("--exposure-time", type=float, default=900.0)
    parser.add_argument("--exposures", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("output/etc_demo_plot.png"))
    return parser.parse_args()


if __name__ == "__main__":
    generate_demo_plot(**vars(_parse_args()))
