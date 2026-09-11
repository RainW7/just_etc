> Historical pre-packaging document. Paths and examples may be obsolete.
> Use the repository README.rst and doc/usage.md for current instructions.

# JUST Exposure Time Calculator (ETC) Technical Reference & User Guide

> **Version**: v1.0 (Python Vectorized & JIT Engine)  
> **Project**: JUST (Jiaotong University Spectroscopic Telescope, 4.4-meter aperture)  
> **Environment**: `base` (Python >= 3.8, NumPy, SciPy, Astropy, Matplotlib, Numba)  
> **Purpose**: In-depth scientific and technical manual detailing the physical modeling framework, optical and instrumentation parameters, mathematical algorithms (fiber injection efficiency, atmospheric transmission, sky emission, CCD noise model, inverse exposure solver), and operational standards for the JUST ETC.

---

## 1. Telescope & Spectrograph Hardware Specifications

### 1.1 Telescope Optical System
The JUST telescope is a 4.4-meter optical/near-infrared telescope optimized for high-throughput wide-field multi-object spectroscopic surveys:
- **Primary Mirror Clear Aperture ($D_\text{tel}$)**: $4.4\text{ m}$
- **Secondary Mirror Central Obstruction Diameter ($D_\text{obs}$)**: $1.8\text{ m}$ (linear central obstruction ratio $\epsilon = 1.8 / 4.4 \approx 0.41$)
- **Effective Geometric Collecting Area ($A_\text{geom}$)**: $A_\text{geom} = \frac{\pi}{4}(4.4^2 - 1.8^2) \approx 12.66\text{ m}^2$
- **Effective Focal Length ($EFL$)**: $26.5\text{ m}$, corresponding to a primary focus operating speed of $f/5.5$
- **Field of View (FoV) Radius ($\theta_\text{FoV}$)**: $0.6^\circ$ (total FoV diameter $1.2^\circ$)
- **Focal Plane Plate Scale**:
  $$PS = \frac{180 \times 3600}{\pi \times 26500} \approx 7.784\text{ arcsec/mm} \quad (\approx 128.47\text{ }\mu\text{m/arcsec})$$

### 1.2 Multi-Object Fiber Positioner System
- **Physical Fiber Core Diameter ($d_\text{core}$)**: $175\text{ }\mu\text{m}$ (effective radius at focal surface $r_\text{fiber} = 87.5\text{ }\mu\text{m}$)
- **Angular Fiber Core Diameter on Sky**: $\theta_\text{fiber} = 175\text{ }\mu\text{m} \times 0.007784\text{ arcsec/}\mu\text{m} \approx 1.362''$ (angular radius $r_0 \approx 0.681''$)
- **Optical Spot Blur ($SPOT$)**: Optical RMS spot size across 5 radial field positions ($0.0^\circ, 0.15^\circ, 0.3^\circ, 0.45^\circ, 0.6^\circ$) are `19.1, 20.9, 24.4, 27.8, 31.3` $\mu\text{m}$.
- **Geometric Vignetting Factor ($VIGNET$)**: Vignetting transmission factors at the 5 field radii are `0.85, 0.84, 0.83, 0.81, 0.79`.

### 1.3 Three-Arm Medium-Resolution Spectrograph
The spectrograph splits incoming light into three independent optical arms (Arm 0 / Arm 1 / Arm 2) using dichroic beam splitters:

| Spectrograph Arm | Wavelength Coverage ($\lambda$) | Detector Pixels | Dispersion ($\Delta\lambda_\text{pix}$) | Spectral Resolution ($R = \lambda/\Delta\lambda$) | Trace Extraction Width | Readout Noise ($\sigma_\text{read}$) | Dark Current ($I_\text{dark}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Arm 0 (Blue)** | $365.0 \sim 560.0\text{ nm}$ | 4096 pixels | $\approx 0.050\text{ nm/pix}$ | $R \approx 2000 \sim 3000$ | 7 pixels | $4.0\,e^-\text{ RMS}$ | $0.010\,e^-/\text{pix/s}$ |
| **Arm 1 (Green/Red)** | $540.0 \sim 745.0\text{ nm}$ | 4096 pixels | $\approx 0.050\text{ nm/pix}$ | $R \approx 2500 \sim 3500$ | 7 pixels | $4.0\,e^-\text{ RMS}$ | $0.005\,e^-/\text{pix/s}$ |
| **Arm 2 (Z/NIR)** | $720.0 \sim 925.0\text{ nm}$ | 4096 pixels | $\approx 0.050\text{ nm/pix}$ | $R \approx 3000 \sim 4000$ | 7 pixels | $4.0\,e^-\text{ RMS}$ | $0.002\,e^-/\text{pix/s}$ |

---

## 2. Physical Algorithms & Mathematical Models

The JUST ETC simulates the complete photon transfer chain from extraterrestrial celestial emission through interstellar medium, Earth's atmosphere, telescope optics, fiber coupling, and dispersive spectrograph detection.

```
[Intrinsic Target Flux F_nu]
       │
       ▼  (Galactic Extinction A_lambda)
[Attenuated Spectrum]
       │
       ▼  (Atmospheric Extinction k(lambda) + Molecular Absorption)
[Ground-Level Spectrum] ──┐
       │                  │
       ▼                  ▼ (Night Sky Lines UVES/OH + Sky Continuum + Moonlight)
[Fiber Geometric Coupling Geo(r_eff, Seeing)] ◄── [Sky Background Radiation]
       │
       ▼  (Telescope Area A_eff + Spectrograph Throughput Thrput)
[CCD Photoelectrons S_source, B_sky]
       │
       ▼  (Poisson Noise + Dark Current + Readout Noise + Sky Subtraction Floor + Stray Light)
[Signal-to-Noise Ratio SNR / Inverse Exposure Time Solver]
```

### 2.1 Galactic Foreground Dust Extinction
Adopts the Cardelli, Clayton & Mathis (CCM 1989) / Fitzpatrick (1999) dust reddening law. For a given color excess $E(B-V)$ and wavelength $\lambda$, the extinction modulation factor is:
$$\eta_\text{ext}(\lambda) = 10^{-0.4 \cdot A_\lambda} = 10^{-0.4 \cdot R_V \cdot E(B-V) \cdot \left[ a(x) + b(x)/R_V \right]}$$
where $x = 1/\lambda\text{ }[\mu\text{m}^{-1}]$ and $R_V = 3.1$.

### 2.2 Atmospheric Transmission & Opacity Models
The ground-level received flux is modulated by two distinct atmospheric components:
1. **Continuous Extinction (Rayleigh, Aerosol, Ozone)**:
   $$\eta_\text{atm, cont}(\lambda) = 10^{-0.4 \cdot k(\lambda) \cdot X}$$
   where $X = \sec(\text{ZA})$ is the airmass, and $k(\lambda)$ accounts for Rayleigh scattering ($\propto \lambda^{-4}$), aerosol scattering ($\propto \lambda^{-1}$), and ozone Chappuis-band absorption.
2. **High-Resolution Molecular Absorption Tables**:
   Evaluates the Kitt Peak atmospheric transmission table `ATMTRANS_KP` ($500 \sim 1500\text{ nm}$ at $0.25\text{ \AA}$ resolution) and the Mauna Kea 3mm Precipitable Water Vapor (PWV) table `MKTRANS_3MM`.

### 2.3 Fiber Geometric Coupling Efficiency ($\eta_\text{geo}$)
Fiber geometric throughput accounts for the spatial convolution of **target intrinsic morphology (Point source or Sérsic Profile)**, **atmospheric seeing PSF**, **telescope optical spot blur**, and **fiber alignment offset (decenter)**:

#### A. Atmospheric Seeing Wavelength Dependence
Seeing FWHM follows Kolmogorov atmospheric turbulence scaling:
$$\text{FWHM}(\lambda) = \text{FWHM}_{800} \cdot \left( \frac{\lambda}{800\text{ nm}} \right)^{-0.2} \cdot X^{0.6}$$

#### B. 2D Bessel Convolution & Hankel Transform
For a source surface brightness profile $I(\theta)$ with half-light radius $r_\text{eff}$ and system PSF $P(\theta)$:
$$\eta_\text{geo} = \int_{\text{Fiber}} [I \otimes P](\mathbf{r}) \, d^2\mathbf{r} = 2\pi r_0 \int_0^\infty \tilde{I}(u) \tilde{P}(u) J_1(2\pi u r_0) J_0(2\pi u \Delta r) \, du$$
- $r_0$: Fiber angular radius ($0.681''$);
- $\Delta r$: Fiber centering offset (`decenter`, default $0.03''$);
- $J_0, J_1$: Bessel functions of the first kind;
- $\tilde{I}(u)$: Point source $\tilde{I}(u) \equiv 1$; Exponential disk ($n=1$) $\tilde{I}(u) = [1 + (2\pi u r_\text{scale})^2]^{-3/2}$; de Vaucouleurs elliptical ($n=4$) solved via polynomial Hankel transformation.

### 2.4 Night Sky Background Radiation Model
The night sky background $B_\text{sky}$ integrates three major physical components:
1. **Dark Sky Continuum**: In dark sky conditions, zenith surface brightness at $600\text{ nm}$ is $V \approx 21.55\text{ mag/arcsec}^2$.
2. **High-Resolution Night Sky Emission Lines**:
   - UV-Optical: ESO UVES database containing **2,816 resolved night sky emission lines** ($314 \sim 1042\text{ nm}$);
   - Near-Infrared: **698 resolved high-order OH vibrational-rotational airglow lines** (`OHDATA`).
3. **Krisciunas & Schaefer (1991) Physical Moonlight Model**:
   Dynamically computes scattered moonlight background as a function of lunar phase fraction $f_\text{moon}$, lunar zenith angle $\text{ZA}_\text{moon}$, and target-moon separation angle $\theta_\text{moon}$.

### 2.5 Detector Counts & Comprehensive Noise Model
For a single exposure time $t_\text{exp}$ (seconds) and $N_\text{exp}$ co-added exposures, the photoelectrons and noise variance per wavelength bin are:

#### A. Target Signal Photoelectrons ($S$)
$$S = F_\nu \cdot \eta_\text{atm}(\lambda) \cdot \eta_\text{ext}(\lambda) \cdot \eta_\text{geo}(\lambda) \cdot \eta_\text{trace} \cdot A_\text{eff}(\lambda) \cdot t_\text{exp} \cdot \left(\frac{c}{\lambda^2}\right) \Delta\lambda_\text{pix} \cdot \frac{\lambda}{hc}$$

#### B. Total Noise Variance ($\sigma^2$)
$$\sigma^2 = \sigma_\text{Poisson, source}^2 + \sigma_\text{Poisson, sky}^2 + \sigma_\text{sys, sky}^2 + \sigma_\text{stray}^2 + \sigma_\text{dark}^2 + \sigma_\text{readout}^2$$

- **Source & Sky Poisson Noise**: $\sigma_\text{Poisson, source}^2 = S$, $\sigma_\text{Poisson, sky}^2 = B_\text{sky}$;
- **Sky Subtraction Systematic Floor**: $\sigma_\text{sys, sky}^2 = (\text{sysfrac} \cdot B_\text{sky})^2$ (default $\text{sysfrac} = 0.01$, i.e. 1% residual floor);
- **Diffuse Stray Light Noise**: $\sigma_\text{stray}^2 = \text{diffuse\_stray} \cdot \langle B_\text{sky} \rangle \cdot t_\text{exp}$ (default 2% stray light fraction);
- **Dark Current Noise**: $\sigma_\text{dark}^2 = I_\text{dark} \cdot N_\text{pix, trace} \cdot t_\text{exp}$;
- **CCD Readout Noise**: $\sigma_\text{readout}^2 = N_\text{pix, trace} \cdot \sigma_\text{read}^2$ ($\sigma_\text{read} = 4\,e^-\text{ RMS}$, trace width $N_\text{pix, trace} = 7$ pixels).

#### C. Total Combined SNR
$$SNR_\text{total} = \sqrt{N_\text{exp}} \cdot \frac{S}{\sqrt{S + B_\text{sky} + (\text{sysfrac} \cdot B_\text{sky})^2 + \sigma_\text{stray}^2 + I_\text{dark} N_\text{trace} t_\text{exp} + N_\text{trace} \sigma_\text{read}^2}}$$

---

## 3. Numerical Optimization & Inverse Solver

### 3.1 Exposure Time Inverse Solver Algorithm
Given a target signal-to-noise ratio $SNR_\text{target}$, solving for $t_\text{exp}$ is a non-linear strictly monotonic root-finding problem.

#### Illinois Hybrid Secant / Bisection Solver Steps:
1. **Physics-Guided Initial Step Estimation**: Using initial guess $t_0 = 600\text{ s}$ and computing $SNR_0$, initial step scales as $t_1 \approx t_0 \cdot (SNR_\text{target} / SNR_0)^2$;
2. **Bracketing**: Identifies interval $[t_\text{lo}, t_\text{hi}]$ satisfying $SNR(t_\text{lo}) \le SNR_\text{target} \le SNR(t_\text{hi})$;
3. **Secant Step with Zero-Division Protection**:
   $$\Delta s = s_\text{hi} - s_\text{lo}$$
   $$t_\text{new} = \begin{cases} 0.5(t_\text{lo} + t_\text{hi}), & \text{if } |\Delta s| < 10^{-12} \\ t_\text{lo} + (SNR_\text{target} - s_\text{lo}) \frac{t_\text{hi} - t_\text{lo}}{\Delta s}, & \text{otherwise} \end{cases}$$
4. **Convergence Criterion**: Stops when relative error $|SNR(t_\text{new}) - SNR_\text{target}| / SNR_\text{target} < 0.01$ ($<1\%$), typically converging in **3 to 5 iterations**.

### 3.2 High-Throughput JIT Acceleration (`ETC_py_optimized.py`)
All inner raytracing loops, Bessel integrals, and multi-wavelength lookups are accelerated via Numba `@jit(nopython=True, fastmath=True)`, achieving single-spectrum calculation times of $< 2\text{ ms}$.

---

## 4. Parameter Reference & Sky Mask Encoding

### 4.1 Observing Conditions Configuration (`set_obs_conditions`)

| Parameter | Type | Default | Description & Recommended Values |
| :--- | :--- | :--- | :--- |
| `seeing_fwhm_800` | float | `0.8` | Atmospheric seeing FWHM at $800\text{ nm}$ (arcsec; Excellent: $0.6''$, Median: $0.8''\sim 1.0''$, Poor: $1.2''\sim 1.5''$) |
| `zenith_angle` | float | `45.0` | Target zenith angle (deg; $0^\circ$ at zenith, typical observations: $30^\circ\sim 45^\circ$) |
| `ebv` | float | `0.03` | Galactic foreground dust reddening $E(B-V)$ (mag; high galactic latitude: $\approx 0.02\sim 0.05$) |
| `field_angle` | float | `0.675` | Focal plane field angle distance (deg; $0^\circ$ at center, $0.6^\circ$ at edge) |
| `decenter` | float | `0.03` | Fiber positioning centering offset (arcsec; typical: $0.03''\sim 0.1''$) |
| `lunar_za` | float | `135.0` | Moon zenith angle ($>90^\circ$ denotes moon below horizon, Dark Time) |
| `lunar_angle` | float | `90.0` | Angular distance between target and moon (deg) |
| `lunar_phase` | float | `0.25` | Lunar phase fraction ($0.0$ = New Moon, $0.5$ = Full Moon) |
| `sysfrac` | float | `0.01` | Sky subtraction systematic floor (1% RMS per pixel) |
| `diffuse_stray` | float | `0.02` | Detector diffuse stray light fraction (2%) |
| `r_eff` | float | `0.0` | Target half-light radius (arcsec; `0.0` for point sources/stars) |
| `skytype_hex` | str | `'10003'` | 5-digit hexadecimal sky model control mask |

### 4.2 Hexadecimal Sky Model Mask Encoding (`skytype_hex`)
```text
skytype = 0x L  A  C  M  S
             │  │  │  │  └── S: Sky Continuum (0: Off, 3: Standard Dark Sky 21.55 mag/arcsec²)
             │  │  │  └───── M: Moonlight Scattering (0: Krisciunas & Schaefer Physical Model)
             │  │  └──────── C: Continuous Opacity (0: Standard Extinction Curve)
             │  └─────────── A: Atmospheric Line Absorption (0: KPNO table, 1: Add MK 3mm PWV bands)
             └────────────── L: Sky Emission Lines (0: Off, 1: Enable UVES 2816 lines + OH 698 lines)
```

---

## 5. Cross-Pipeline Comparison (`justspecsimu` vs. `JUST ETC`)

| Metric / Dimension | `justspecsimu` (DESI-based) | `JUST ETC` (PFS-based Python Vectorized) | Scientific & Implementation Note |
| :--- | :--- | :--- | :--- |
| **Geometric Throughput ($\eta_\text{geo}$)** | Grid Interpolation (`DESI-0347_blur.ecsv` + `offset.ecsv`) | **Analytic 2D Convolution** (Hankel/Bessel Integrals) | ETC supports continuous arbitrary $r_\text{eff}$ and seeing without grid truncation. |
| **Night Sky Modeling** | Fixed continuum table (`spec-sky.dat`) | **UVES 2,816 lines + OH airglow + Dynamic Moonlight** | ETC achieves significantly higher fidelity in OH-crowded NIR bands and bright moon conditions. |
| **Plate Scale** | $128.473\text{ }\mu\text{m/arcsec}$ | $128.473\text{ }\mu\text{m/arcsec}$ ($EFL=26.5\text{ m}$) | Hardware parameters strictly aligned and identical. |
| **Extinction Curves** | KPNO broadband extinction table | High-res KPNO transmission ($0.25\text{ \AA}$) + CCM Dust | Resolves sharp atmospheric molecular absorption features. |
| **Execution Performance** | Python static pipeline | **Numba JIT Vectorization + Multiprocessing** | Per-spectrum computation time $< 2\text{ ms}$, supports parallel 10k spectra batches. |
