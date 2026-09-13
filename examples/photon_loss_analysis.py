"""Visualize the wavelength-dependent JUST photon budget."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from just_etc import ETC_py_optimized as ETC_py
from just_etc import JUSTExposureTimeCalculator

ARM_COLORS = ("#1f77b4", "#2ca02c", "#d62728")
COMPONENT_STYLES = {
    "Galactic extinction": ("black", "-"),
    "Atmosphere": ("purple", "--"),
    "Vignetting": ("cyan", "-."),
    "Fiber injection": ("orange", "-"),
    "Trace extraction": ("brown", ":"),
}

PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
}


def _interpolate_throughput(spectrograph, arm, wavelength):
    start, stop = spectrograph.istart[arm], spectrograph.istart[arm + 1]
    return np.interp(
        wavelength,
        spectrograph.l[start:stop],
        spectrograph.T[start:stop],
    )


def generate_photon_budget(
    seeing=0.8,
    effective_radius=0.0,
    output=Path("output/photon_budget.png"),
):
    """Calculate each throughput term and save an end-to-end efficiency plot."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    calculator = JUSTExposureTimeCalculator(calc_mode="fast")
    calculator.set_obs_conditions(seeing_fwhm_800=seeing, r_eff=effective_radius)
    observation = calculator._make_obs()
    spectrograph = calculator._spectro
    field_angle = calculator._obs_params["field_angle"]
    decenter = calculator._obs_params["decenter"]

    with plt.rc_context(PLOT_STYLE):
        fig, (component_ax, total_ax) = plt.subplots(
            2, 1, figsize=(11, 9), sharex=True
        )

        for arm in range(spectrograph.N_arms):
            wavelength = spectrograph.lmin[arm] + spectrograph.dl[arm] * (
                np.arange(spectrograph.npix[arm]) + 0.5
            )
            vignetting = np.interp(
                field_angle,
                np.linspace(0, spectrograph.rfov, 5),
                spectrograph.vignette,
            )
            components = {
                "Galactic extinction": np.array([
                    10 ** (-0.4 * ETC_py.gsGalactic_Alambda__EBV(w) * observation.EBV)
                    for w in wavelength
                ]),
                "Atmosphere": np.array([ETC_py.gsAtmTrans(observation, w, 0) for w in wavelength]),
                "Vignetting": np.full_like(wavelength, vignetting),
                "Fiber injection": np.array([
                    ETC_py.gsGeometricThroughput(
                        spectrograph, observation, w, effective_radius,
                        decenter, field_angle, 0,
                    ) for w in wavelength
                ]),
                "Trace extraction": np.array([
                    ETC_py.gsFracTrace(spectrograph, observation, arm, w, 0)
                    for w in wavelength
                ]),
            }
            instrument = _interpolate_throughput(spectrograph, arm, wavelength)

            for name, values in components.items():
                color, linestyle = COMPONENT_STYLES[name]
                component_ax.plot(
                    wavelength, values, color=color, ls=linestyle, lw=2.0,
                    label=name if arm == 0 else None,
                )
            component_ax.plot(
                wavelength, instrument, color=ARM_COLORS[arm], lw=2.2,
                label=f"Instrument throughput (Arm {arm})",
            )

            total = instrument.copy()
            for values in components.values():
                total *= values
            total_ax.plot(wavelength, total, color=ARM_COLORS[arm], lw=2.2,
                          label=f"Total system efficiency (Arm {arm})")
            total_ax.fill_between(wavelength, total, color=ARM_COLORS[arm], alpha=0.2)

        for ax in (component_ax, total_ax):
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.set_ylim(0, 0.4)
        component_ax.set_ylim(0, 1.05)
        component_ax.set_ylabel("Transmission")
        component_ax.set_title("Photon Loss Breakdown: Individual Components", fontsize=13)
        component_ax.legend(ncol=3, loc="lower center", fontsize=9)
        total_ax.set_ylabel("End-to-end efficiency")
        total_ax.set_xlabel("Wavelength [nm]")
        total_ax.set_xlim(350, 950)
        total_ax.set_title("Total End-to-End Photon Collection Efficiency", fontsize=13)
        total_ax.legend(ncol=3, loc="upper right", fontsize=9)
        fig.suptitle(
            f"JUST Photon Budget (Seeing = {seeing:g}″, "
            f"Effective Radius = {effective_radius:g}″)",
            fontsize=15,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved photon budget to {output.resolve()}")
    return output


def _parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--seeing", type=float, default=0.8)
    parser.add_argument("--effective-radius", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("output/photon_budget.png"))
    return parser.parse_args()


if __name__ == "__main__":
    generate_photon_budget(**vars(_parse_args()))
