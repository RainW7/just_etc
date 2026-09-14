# Archived examples

These scripts are historical or specialized research workflows moved out of the main
showcase. They are preserved unchanged for reference and reproducibility; they are not
maintained as the primary getting-started examples.

- **Fiber and photon-loss studies:** `plot_fiber_*.py`, `photon_loss_seeing.py`
- **Magnitude and survey simulations:** `generate_just_limit.py`, `run_survey_simulations.py`
- **Dwarf spectra:** `simulate_dwarf_observability.py`
- **Low-level experiments:** `run_etc_preset.py`, `calculate_star_snr.py`, `scratch_geo.py`
- **FITS and Redrock workflows:** `generate_just_specdat_fits.py`,
  `convert_input_fits_to_redrock.py`, `generate_redrock_just_fits.py`, and
  `process_bgs_10k_redrock.py`

For a Python API that accepts wavelength and flux arrays directly, use
`from just_etc import convet_to_redrock_format`. The archived
`process_bgs_10k_redrock.py` command remains as a wrapper for its legacy input
FITS layout and delegates the ETC simulation and Redrock output to this API.

Some workflows require large inputs that are not distributed with the package. Redrock
scripts additionally require `desispec`, `redrock`, and their template/configuration data.
Run archived programs only after reviewing their paths and scientific assumptions.
