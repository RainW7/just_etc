> Historical pre-packaging document. Paths and examples may be obsolete.
> Use the repository README.rst and doc/usage.md for current instructions.

# JUST Exposure Time Calculator (JUST ETC v1.0)

This directory contains the standardized distribution package and codebase for the **JUST (Jiaotong University Spectroscopic Telescope, 4.4-meter aperture) Exposure Time Calculator (ETC)**, designed for standalone scientific research, server deployments, and batch PBS jobs on the SJTU Gravity HPC cluster.

---

## 1. Directory Structure & File Map

```text
just_etc/py/
├── just_etc_api.py               # [Core Interface] High-level Python API (Recommended entry point)
├── ETC_py_optimized.py           # [Core Engine] Vectorized raytracing & SNR physical engine with Numba JIT
├── modeldata.py                  # [Physical Database] Atmospheric transmission tables, UVES sky lines & OH airglow
├── spec.dat                      # [Hardware Specs] JUST telescope optics, fiber positioner & 3-arm spectrograph specs
├── JUST_ETC_User_Guide.md        # [Technical Manual] Comprehensive physical models, mathematical formulas & parameters
│
├── cal_exp_time.py               # [CLI Tool] Rapid command-line tool for SNR calculation & inverse exposure solver
├── generate_just_limit.py        # [Limiting Mag] Continuum & emission-line limiting magnitude generator
├── simulate_mock_spectrum.py     # [Mock Spectrum] Mock observed spectrum generator with instrument noise
├── dwarf_ssp_model.py            # [SSP Physics] Dwarf galaxy spectrum builder based on stellar mass & sSFR
├── calculate_star_snr.py         # [Point Source] Stellar SNR multi-band evaluator
├── run_etc_preset.py             # [Batch Runner] Non-interactive automated preset configuration runner
│
├── process_bgs_10k_redrock.py    # [10k Spectra Parallel] Parallel ETC simulator for 10,000 spectra -> Redrock FITS
│
├── templates/                    # [FITS SED Library] High-resolution spectral templates
│   ├── galaxy/                   # Elliptical, Spiral (sa, sbc, scd) templates
│   ├── starburst/                # Starburst templates (sb1 ~ sb6, Kinney et al.)
│   ├── bc03/                     # Bruzual & Charlot (2003) SSP evolutionary synthetic templates
│   └── IRgalaxy/                 # LIRG/ULIRG & Starburst merger templates (M82, Arp220, Mrk231, NGC6240)
│
├── convert_input_fits_to_redrock.py  # FITS library to DESI/Redrock standard multi-arm FITS converter
├── generate_just_specdat_fits.py     # Converts spec.dat throughput tables into standard FITS curves
├── generate_redrock_just_fits.py     # Generates mock observation FITS data for Redrock fitting
│
├── plot_demo.py                  # [Demo] Basic SNR & spectral visualization script
├── plot_demo_extended.py         # [Demo] Multi-condition (seeing / magnitude) comparison plot
├── plot_fiber_comparison.py      # [Analysis] Fiber throughput vs target morphological size
├── plot_fiber_injection.py       # [Analysis] Fiber geometric injection efficiency & decenter error
├── plot_fiber_snr_comparison.py  # [Analysis] Fiber aperture vs galaxy shape impact on SNR
├── photon_loss_analysis.py       # [Analysis] End-to-end photon loss cascade analysis
├── photon_loss_seeing.py         # [Analysis] Atmospheric seeing impact on 3-arm fiber coupling
├── run_survey_simulations.py     # [Survey Sim] End-to-end spectroscopic survey efficiency simulator
│
└── requirements.txt              # Environment dependencies list (including Numba JIT)
```

---

## 2. Environment Setup & Installation

It is recommended to run in a dedicated Conda environment:

```bash
# 1. Activate your Conda environment
conda activate base

# 2. Install required core dependencies
pip install -r requirements.txt
```

Core dependencies:

- `numpy >= 1.20.0`: Numerical array and matrix operations
- `astropy >= 4.0`: FITS I/O, photometric systems, and astronomical constants
- `scipy >= 1.5.0`: Gaussian filtering, interpolation, and Bessel integrals
- `matplotlib >= 3.3.0`: Publication-quality plotting
- `numba >= 0.53.0`: **JIT Compiler** (accelerates low-level raytracing and numerical integrals)

*(Optional dependencies: `redrock` and `desispec` are only required if running DESI/Redrock redshift fitting tools on the server)*

---

## 3. Python API Complete Reference (`just_etc_api.py`)

In Python scripts or Jupyter Notebooks, `just_etc_api.py` provides the primary user-friendly interface.

### 3.1 Initialization & Observing Conditions

```python
from just_etc_api import JUSTExposureTimeCalculator, load_template, normalize_to_mag

# 1. Initialize calculator ('fast', 'balanced', or 'accurate' precision modes)
etc = JUSTExposureTimeCalculator(calc_mode='fast')

# 2. Configure observing conditions (supports any keyword argument)
etc.set_obs_conditions(
    seeing_fwhm_800=0.8,    # Seeing FWHM at 800 nm (arcsec, default: 0.8")
    zenith_angle=45.0,      # Target zenith angle (degrees, airmass X=1.414)
    ebv=0.03,               # Galactic foreground dust extinction E(B-V)
    decenter=0.03,          # Fiber positioning offset (arcsec, default: 0.03")
    r_eff=0.5,              # Target effective half-light radius (arcsec, 0.0 for point sources)
    lunar_za=135.0          # Moon zenith angle (>90° indicates moon below horizon / Dark Time)
)
```

### 3.2 Template Loading & Flux Normalization

```python
# 1. Load FITS template from templates/ directory
wave_aa, flux_flam = load_template('galaxy/elliptical_001.fits')

# 2. Normalize template flux to target AB magnitude (e.g. r = 21.5 mag)
# Supported bands: 'u', 'g', 'r', 'i', 'z', 'V', 'B'
flux_norm, scale_factor = normalize_to_mag(
    wave_aa=wave_aa,
    flux_flam=flux_flam,
    target_mag=21.5,
    band='r'
)
print(f"Template normalized. Scale factor = {scale_factor:.4e}")
```

### 3.3 Compute SNR Across All 3 Spectrograph Arms (`compute_snr`)

```python
# Compute continuum SNR for 4 exposures of 900s each (total 3600s)
results = etc.compute_snr(
    wave_aa=wave_aa,
    flux_flam=flux_norm,
    t_exp=900.0,
    n_exp=4
)

# Inspect results per arm
arm_names = {0: 'Blue (365-560 nm)', 1: 'Green/Red (540-745 nm)', 2: 'Z/NIR (720-925 nm)'}
for arm in results:
    ia = arm['arm']
    print(f"[{arm_names[ia]}] Mean SNR: {arm['snr_mean']:.2f}, Median SNR: {arm['snr_median']:.2f}")
    # Pixel-level arrays: wave_nm = arm['wave_nm'], snr = arm['snr']
```

### 3.4 Simulate Mock Observed Spectrum with Instrument Noise (`simulate_mock_observation`)

```python
# Generate realistic mock observation spectrum affected by throughput and noise
mock_obs = etc.simulate_mock_observation(
    wave_aa=wave_aa,
    flux_flam=flux_norm,
    t_exp=900.0,
    n_exp=4,
    seed=42  # Seed for reproducible noise realization
)

for arm in mock_obs:
    print(f"Arm {arm['arm']}:")
    print(f"  - True flux: arm['flux_intrinsic']")
    print(f"  - Noisy mock flux: arm['flux_mock']")
    print(f"  - Noise counts: arm['noise_e']")
```

### 3.5 Inverse Exposure Time Solver (`solve_exposure_time`)

```python
# Solve for required exposure time to reach SNR = 8.0 at 600 nm (in 4 split exposures)
solution = etc.solve_exposure_time(
    wave_aa=wave_aa,
    flux_flam=flux_norm,
    target_snr=8.0,
    ref_wave_nm=600.0,
    n_exp=4,
    t_exp_init=600.0
)

print(f"Recommended Single Exposure Time: {solution['t_exp']:.1f} sec")
print(f"Total Observation Time          : {solution['t_total']/3600:.2f} hours")
print(f"Achieved SNR                    : {solution['snr_achieved']:.3f} (Iterations: {solution['n_iter']})")
```

---

## 4. Command-Line Tools (CLI Usage)

### 4.1 `cal_exp_time.py`: Interactive and CLI Calculator

Supports full `argparse` command line flags:

```bash
cd /home/wenrun/ETC_py_v1

# Example 1: Compute SNR for a 21.5 mag point-source star with 1800s exposure
python cal_exp_time.py --mag 21.5 --band r --texp 1800 --target point

# Example 2: Solve required exposure time for an extended galaxy (r = 22.0 mag, Reff = 0.6") to reach SNR = 5.0
python cal_exp_time.py --target-snr 5.0 --mag 22.0 --band r --target extended --reff 0.6

# Example 3: Specify custom template and seeing condition (Seeing = 1.2")
python cal_exp_time.py --template galaxy/sbc_cww_001.fits --mag 20.5 --band g --seeing 1.2 --texp 900 --nexp 2

# Example 4: Run default demo
python cal_exp_time.py
```

### 4.2 `generate_just_limit.py`: Limiting Magnitude Curves

Generates continuum and emission line limiting magnitude relations across varying exposure times:

```bash
python generate_just_limit.py
# Outputs saved to limit_mag/ directory
```

### 4.3 `simulate_mock_spectrum.py`: SSP-Based Dwarf Galaxy Spectrum Simulation

Simulates physically motivated spectra using stellar mass $M_*$, sSFR, and Kennicutt (1998) emission line relations:

```bash
python simulate_mock_spectrum.py
# Outputs saved to output/mock_spectra/
```

---

## 5. Scientific Demonstration & Testing Scripts

| Script                                  | Purpose & Scientific Context                                 | Command                                        | Outputs                                 |
| :-------------------------------------- | :----------------------------------------------------------- | :--------------------------------------------- | :-------------------------------------- |
| `plot_demo.py`                          | Generates template spectra, 3-arm throughput, and SNR curves | `python plot_demo.py`                          | `output/etc_demo_plot.png`              |
| `plot_demo_extended.py`                 | Multi-magnitude ($r=18\sim 23$) and seeing SNR progression   | `python plot_demo_extended.py`                 | `output/etc_reff_impact.png`            |
| `photon_loss_analysis.py`               | Analyzes end-to-end photon loss breakdown                    | `python photon_loss_analysis.py`               | `output/photon_loss_cascade.png`        |
| `photon_loss_seeing.py`                 | Evaluates seeing degradation impact on B/R/Z fiber coupling  | `python photon_loss_seeing.py`                 | `output/photon_loss_seeing_cascade.png` |
| `plot_fiber_comparison.py`              | Compares point source vs extended source ($0.3'', 0.8'', 1.2''$) coupling | `python plot_fiber_comparison.py`              | `output/fiber_loss_comparison.png`      |
| `dwarf/simulate_dwarf_observability.py` | Observability and kinematics simulation for low surface brightness dwarfs | `python dwarf/simulate_dwarf_observability.py` | `dwarf/output/*.png`                    |

---

## 6. SJTU Gravity Cluster 10,000 Spectra Simulation & PBS (`qsub`) Submission

For the 10,000 BGS spectra library in `/home/yzgu/work/work-just/justspecsimu/milestone2026/spectrum_library/BGS_z01/input-spectra.fits`:

### 6.1 Method A: Submit PBS Batch Job (Recommended)

The job submission script `submit_bgs10k_etc.qsub` is preconfigured with 16-core multiprocessing and automated environment activation:

```bash
cd /home/wenrun/ETC_py_v1

# 1. Submit PBS job to cluster queue
qsub submit_bgs10k_etc.qsub

# 2. Check job status
qstat -u $USER

# 3. Monitor live log output
tail -f bgs10k.log
```

### 6.2 Method B: Interactive Multi-Core Execution on Compute Node

```bash
python process_bgs_10k_redrock.py \
    --input /home/yzgu/work/work-just/justspecsimu/milestone2026/spectrum_library/BGS_z01/input-spectra.fits \
    --output /home/wenrun/output/just_redrock_bgs_10k_obs.fits \
    --t_exp 900 \
    --n_exp 4 \
    --seeing 0.8 \
    --nproc 16
```

### 6.3 Output FITS Specification & Redrock Redshift Fitting

The generated `just_redrock_bgs_10k_obs.fits` conforms strictly to DESI / Redrock specifications:

- `B_WAVELENGTH`, `B_FLUX`, `B_IVAR`, `B_RESOLUTION`
- `R_WAVELENGTH`, `R_FLUX`, `R_IVAR`, `R_RESOLUTION`
- `Z_WAVELENGTH`, `Z_FLUX`, `Z_IVAR`, `Z_RESOLUTION`
- `FIBERMAP`, `SCORES`

Feed directly to `rrdesi` for redshift classification:

```bash
rrdesi --zbest /home/wenrun/output/redrock_zbest.fits /home/wenrun/output/just_redrock_bgs_10k_obs.fits
```

