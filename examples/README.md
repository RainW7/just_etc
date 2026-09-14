# JUST ETC showcase

This directory contains four polished visualization examples plus two FITS-generation
examples. Together they cover the main ETC story without presenting every research
scratch script as a supported entry point.

| Example | What it demonstrates | Default output |
| --- | --- | --- |
| `plot_demo.py` | Template loading, AB normalization, multi-arm S/N, signal and noise | `output/etc_demo_plot.png` |
| `observing_conditions_comparison.py` | Sensitivity to atmospheric seeing and target angular size | `output/etc_observing_conditions.png` |
| `photon_loss_analysis.py` | Instrument, atmosphere, fiber, and extraction losses | `output/photon_budget.png` |
| `show_mock_spectra.py` | Noisy JUST spectra for four representative galaxy targets | `output/mock_spectra/ssp_dwarf_mock_spectrum_demo.png` |
| `simulate_mock_spectrum.py` | Generate a Redrock-compatible noisy JUST FITS spectrum from a template | `output/just_mock_spectrum.fits` |
| `simulate_justspecsim_spectrum.py` | Generate a BGS galaxy with `just_specsim`, observe it with JUST ETC, and write Redrock FITS plus a diagnostic plot | `output/justspecsim_redrock.fits`, `output/justspecsim_redrock_spectrum.png` |

Install `just_etc`, then run the examples from any writable directory:

```sh
python /path/to/just_etc/examples/plot_demo.py
python /path/to/just_etc/examples/observing_conditions_comparison.py
python /path/to/just_etc/examples/photon_loss_analysis.py
python /path/to/just_etc/examples/show_mock_spectra.py
python /path/to/just_etc/examples/simulate_mock_spectrum.py
python /path/to/just_etc/examples/simulate_justspecsim_spectrum.py \
    --just-specsim-root /path/to/just_specsim
```

The three compact plotting programs and both FITS generators support `--help` and accept
an `--output` path. `show_mock_spectra.py` retains its four fixed science cases. The
`just_specsim` bridge expects that checkout's large BGS basis file under
`example/DESI_templates/`; its `--just-specsim-root` default finds a sibling checkout
named `just_specsim`. All programs create missing output folders automatically.

## Customize the `just_specsim` bridge

Pass options after the script name to override individual defaults; unspecified options
keep their defaults. For example, from the `just_etc` repository root:

```sh
python examples/simulate_justspecsim_spectrum.py \
    --just-specsim-root /path/to/just_specsim \
    --redshift 0.2 \
    --absolute-magnitude-r -21 \
    --color-gr 0.7 \
    --apparent-magnitude 20.0 \
    --exposure-time 900 \
    --exposures 4 \
    --seeing 1.0 \
    --noise-seed 42 \
    --output output/z02_r20.fits \
    --plot-output output/z02_r20.png
```

The main options are:

| Purpose | Options | CLI defaults |
| --- | --- | --- |
| Galaxy template and redshift | `--redshift`, `--absolute-magnitude-r`, `--color-gr`, `--template-seed` | `0.1`, `-21`, `0.5`, `1` |
| Final apparent brightness | `--apparent-magnitude` | `19.5` AB in DECam r |
| Exposure and observing conditions | `--exposure-time`, `--exposures`, `--seeing`, `--zenith-angle`, `--effective-radius`, `--ebv` | `180 s`, `1`, `0.8 arcsec`, `45 deg`, `0 arcsec`, `0.03` |
| Reproducibility and ETC accuracy | `--noise-seed`, `--calc-mode` | `42`, `fast` |
| Target metadata | `--target-id`, `--ra`, `--dec` | `10001`, `150 deg`, `2 deg` |
| Files and display | `--output`, `--plot-output`, `--show-plot` | FITS at `output/justspecsim_redrock.fits`; PNG uses the FITS stem plus `_spectrum.png`; do not show a window |

`--absolute-magnitude-r` and `--color-gr` choose the nearest available `just_specsim`
BGS template; they do not calculate a new spectrum at arbitrary exact values.
`--apparent-magnitude` rescales that template to the requested final DECam r magnitude.
`--exposure-time` is the time for each exposure, so total integration time is
`exposure-time × exposures`. The two FITS-generation scripts both use `--output`; the
template-based `simulate_mock_spectrum.py` instead uses `--magnitude` and `--band` for
its brightness normalization.

To see every option for the current checkout, run:

```sh
python examples/simulate_justspecsim_spectrum.py --help
```

For repeated comparisons, give each run a distinct FITS and PNG filename. Otherwise the
same output path is overwritten. Relative output paths are resolved from the directory
where the command is run. Add `--show-plot` to save the PNG and open it in an interactive
Matplotlib window.

The same simulation can be called from Python. Run this from the `examples/` directory
or add it to `sys.path` first:

```python
from simulate_justspecsim_spectrum import simulate_justspecsim_spectrum

simulate_justspecsim_spectrum(
    redshift=0.2,
    absolute_magnitude_r=-21,
    color_gr=0.7,
    apparent_magnitude=20.0,
    exposure_time=900,
    exposures=4,
    output="output/z02_r20.fits",
    plot_output="output/z02_r20.png",
)
```

The keyword arguments in Python use underscores (for example, `exposure_time`); their
command-line equivalents use hyphens (for example, `--exposure-time`).

`just_specsim.SpectrumMaker` already returns an observer-frame spectrum normalized in
units of 1e-17 erg s^-1 cm^-2 Angstrom^-1. The bridge therefore does not redshift or
normalize it a second time: it multiplies by 1e-17 to obtain the physical FLAM required
by `JUSTExposureTimeCalculator`. Use `--apparent-magnitude` only when an explicit final
DECam r magnitude is desired.

The bridge must run in one Python >=3.10 environment containing the runtime dependencies
of both packages; the legacy Python 3.6 `base` environment cannot import the current
`just_specsim`. For a source-tree run in the configured `desi` environment, use:

```sh
cd /Users/rain/just_etc
PYTHONPATH=/Users/rain/just_etc/py \
    /Users/rain/miniconda3/envs/desi/bin/python \
    examples/simulate_justspecsim_spectrum.py \
    --just-specsim-root /Users/rain/just_specsim \
    --redshift 0.1 --absolute-magnitude-r -21 --color-gr 0.7 \
    --exposure-time 900 --exposures 4 \
    --output output/justspecsim_redrock.fits \
    --plot-output output/justspecsim_redrock_spectrum.png
```

The three-panel plot shows the full noiseless observer-frame `just_specsim` spectrum,
the three noisy JUST arms with their corresponding noiseless references, and the
combined per-pixel S/N of each arm. Add `--show-plot` when running in an interactive
desktop session to display the Matplotlib window as well as saving the PNG. For
readability, only the flux panel masks arm-edge pixels with modeled S/N < 0.5; the S/N
panel shows the full arm coverage, and the Redrock FITS arrays remain unchanged.

## Archived research scripts

Earlier exploratory, batch-processing, and specialized plotting programs are preserved
under [`archive/`](archive/README.md). They are retained as implementation references,
not as the recommended showcase, and may require external datasets or optional software.
