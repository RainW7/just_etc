# JUST ETC showcase

This directory intentionally keeps only three polished, runnable plots. Together they
cover the main ETC story without presenting every research scratch script as a supported
entry point.

| Example | What it demonstrates | Default output |
| --- | --- | --- |
| `plot_demo.py` | Template loading, AB normalization, multi-arm S/N, signal and noise | `output/etc_demo_plot.png` |
| `plot_demo_extended.py` | Sensitivity to atmospheric seeing and target angular size | `output/etc_observing_conditions.png` |
| `photon_loss_analysis.py` | Instrument, atmosphere, fiber, and extraction losses | `output/photon_budget.png` |

Install `just_etc`, then run the examples from any writable directory:

```sh
python /path/to/just_etc/examples/plot_demo.py
python /path/to/just_etc/examples/plot_demo_extended.py
python /path/to/just_etc/examples/photon_loss_analysis.py
```

Each program supports `--help` and accepts an `--output` path. The first two also expose
the target and exposure parameters, so they can be reused without editing source code.
The plotting programs use package-managed template paths and create missing output
folders automatically.

## Archived research scripts

Earlier exploratory, batch-processing, and specialized plotting programs are preserved
under [`archive/`](archive/README.md). They are retained as implementation references,
not as the recommended showcase, and may require external datasets or optional software.
