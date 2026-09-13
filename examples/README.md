# JUST ETC showcase

This directory contains four polished visualization examples plus one FITS-generation
example. Together they cover the main ETC story without presenting every research
scratch script as a supported entry point.

| Example | What it demonstrates | Default output |
| --- | --- | --- |
| `plot_demo.py` | Template loading, AB normalization, multi-arm S/N, signal and noise | `output/etc_demo_plot.png` |
| `observing_conditions_comparison.py` | Sensitivity to atmospheric seeing and target angular size | `output/etc_observing_conditions.png` |
| `photon_loss_analysis.py` | Instrument, atmosphere, fiber, and extraction losses | `output/photon_budget.png` |
| `show_mock_spectra.py` | Noisy JUST spectra for four representative galaxy targets | `output/mock_spectra/ssp_dwarf_mock_spectrum_demo.png` |
| `simulate_mock_spectrum.py` | Generate a Redrock-compatible noisy JUST FITS spectrum from a template | `output/just_mock_spectrum.fits` |

Install `just_etc`, then run the examples from any writable directory:

```sh
python /path/to/just_etc/examples/plot_demo.py
python /path/to/just_etc/examples/observing_conditions_comparison.py
python /path/to/just_etc/examples/photon_loss_analysis.py
python /path/to/just_etc/examples/show_mock_spectra.py
python /path/to/just_etc/examples/simulate_mock_spectrum.py
```

The three compact plotting programs and the FITS generator support `--help` and accept
an `--output` path. `show_mock_spectra.py` retains its four fixed science cases. All
programs use package-managed template paths and create missing output folders
automatically.

## Archived research scripts

Earlier exploratory, batch-processing, and specialized plotting programs are preserved
under [`archive/`](archive/README.md). They are retained as implementation references,
not as the recommended showcase, and may require external datasets or optional software.
