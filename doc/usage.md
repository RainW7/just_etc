# API and scientific conventions

## Spectra and instrument data

```python
from just_etc import (
    JUSTExposureTimeCalculator, load_template, list_templates, normalize_to_mag,
)

templates = list_templates()  # category/filename -> installed Path
wave, flux = load_template("starburst/sb4_kinney_fuv_001.fits")
flux, scale = normalize_to_mag(wave, flux, target_mag=21.5, band="r")
etc = JUSTExposureTimeCalculator(calc_mode="fast")
```

`load_template` accepts an existing local file (absolute or relative), a bundled
`category/filename`, or `templates/category/filename`. Existing local paths take
precedence. It loads the first FITS binary or ASCII table with wavelength and flux columns,
then sorts wavelengths in ascending order. Supply wavelength in Å and flux density
in erg/s/cm²/Å; the loader does not perform arbitrary FITS unit conversion.

The instrument default is bundled `data/spec.dat`. Override it explicitly with
`JUSTExposureTimeCalculator(spec_file="/path/to/spec.dat")`.

## Observing conditions

```python
etc.set_obs_conditions(
    seeing_fwhm_800=0.8,  # arcsec at 800 nm
    zenith_angle=45.0,   # degrees
    ebv=0.03,
    r_eff=0.5,           # arcsec; 0 for a spatial point source
    decenter=0.03,       # arcsec
    lunar_za=135.0,      # degrees; moon below the horizon
)
```

Additional parameters and defaults are in `JUSTExposureTimeCalculator._DEFAULTS`.
The three calculation modes are `fast`, `balanced`, and `accurate`. The API default
is `accurate`, whereas the CLI default is `fast`. The engine uses global calculation
settings; avoid concurrently using different modes in threads. Batch examples use
separate processes.

## SNR and exposure time

```python
arms = etc.compute_snr(wave, flux, t_exp=900.0, n_exp=4)
solution = etc.solve_exposure_time(
    wave, flux, target_snr=8.0, ref_wave_nm=600.0, n_exp=4,
    t_exp_init=600.0,
)
print(solution["t_exp"], solution["t_total"], solution["snr_achieved"])
```

Each SNR result dictionary contains:

| Key | Meaning |
| --- | --- |
| `arm` | Arm index, 0–2 |
| `wave_nm`, `wave_aa` | Native pixel-center wavelength grids |
| `signal` | Source electrons per pixel per exposure |
| `noise_var` | Noise variance per pixel per exposure, including configured terms |
| `snr_1exp` | Single-exposure per-pixel SNR |
| `snr` | Combined per-pixel SNR for `n_exp` exposures |

Compute summaries with NumPy; `snr_mean` and `snr_median` are not result keys.
The solver reference wavelength is in nm. Check the achieved SNR and warnings when
the target cannot be reached within the configured time/iteration bounds.

## Mock observations

```python
mock = etc.simulate_mock_observation(wave, flux, t_exp=900, n_exp=4, seed=42)
```

Results include `wave_aa`, `flux_intrinsic`, `flux_mock`, `signal_e`, `noise_e` and
`snr`. Noise is drawn in electron space using the supplied seed; source-free pixels
are treated as in the original implementation.

## Model limitations retained from the source

This packaging migration does not recalibrate the physical model. Magnitude
normalization uses mean FLAM in approximate band windows and an effective
wavelength; it is not synthetic photometry integrated over a real response curve.
Point/extended source settings describe the spatial profile, independently of the
spectral template. Input wavelengths outside the SED range are assigned zero flux.
Consult the source and historical technical manuals for the adopted atmosphere,
sky, throughput and systematic-noise assumptions before using results in research.
