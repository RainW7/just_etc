# Research examples

Install `just_etc` first. Run scripts from a writable working directory, for example:

```sh
python /path/to/just_etc/examples/plot_demo.py
```

Inputs bundled with the package are found independently of the working directory.
Generated outputs go below the working directory, usually `output/`, `limit_mag/`
or `compare/`. Scripts retain the scientific settings from the source version;
inspect them before running expensive simulations.

| Workflows | Scripts |
| --- | --- |
| Basic plots | `plot_demo.py`, `plot_demo_extended.py` |
| Fiber coupling and photon losses | `plot_fiber_*.py`, `photon_loss_*.py` |
| Limiting magnitude and survey simulations | `generate_just_limit.py`, `run_survey_simulations.py` |
| Mock and dwarf spectra | `simulate_mock_spectrum.py`, `simulate_dwarf_observability.py` |
| Low-level presets and exploratory calculation | `run_etc_preset.py`, `calculate_star_snr.py`, `scratch_geo.py` |
| Throughput FITS export | `generate_just_specdat_fits.py` |
| Batch FITS simulation | `process_bgs_10k_redrock.py` |
| Optional DESI/Redrock workflows | `convert_input_fits_to_redrock.py`, `generate_redrock_just_fits.py` |

The Redrock workflows require `desispec`, `redrock` and their template/configuration
data. Use your own `RR_TEMPLATE_DIR` if required by that installation. Large input
spectral libraries and historical comparison datasets are not included. For the
batch simulator, `--input` and `--output` are required. The PBS file under `bin/`
is a configurable example, not a preconfigured job for any particular cluster.

Core tests do not execute full survey or Redrock workflows. Historical comparison
reports live under `doc/validation`; their referenced external datasets must be
provided separately when reproducing the comparisons.
