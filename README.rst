JUST Exposure Time Calculator
=============================

``just_etc`` estimates exposure times and per-pixel signal-to-noise ratios
for the Jiaotong University Spectroscopic Telescope (JUST). It includes a
Numba-accelerated ETC engine, a Python API, a command-line calculator, the
instrument configuration, and 40 example FITS spectral templates.

Installation
------------

Use an environment containing compatible NumPy, SciPy, Astropy, Numba and
Matplotlib versions. A current Python environment is recommended for new installations.

.. code-block:: bash

    git clone https://github.com/RainW7/just_etc.git
    cd just_etc
    python -m pip install .
    just-etc --help

For development, use ``python -m pip install -e '.[test]'``. Invoke pip via the
same Python interpreter you will use to run the calculator. Do not upgrade an
existing research environment unnecessarily.

Quick start
-----------
For Python:

.. code-block:: python

    import numpy as np
    from just_etc import JUSTExposureTimeCalculator, load_template, normalize_to_mag

    wave, flux = load_template("galaxy/elliptical_001.fits")
    flux, scale = normalize_to_mag(wave, flux, target_mag=21.0, band="r")
    etc = JUSTExposureTimeCalculator(calc_mode="fast")
    etc.set_obs_conditions(seeing_fwhm_800=0.8, r_eff=0.0)
    arms = etc.compute_snr(wave, flux, t_exp=900.0, n_exp=4)
    for arm in arms:
        print(arm["arm"], np.median(arm["snr"]))

``wave`` is in Angstroms and ``flux`` in erg/s/cm²/Å. ``t_exp`` is the time of
each exposure in seconds; the example totals 3600 s. ``snr`` is the combined
per-pixel SNR, not an integrated line or resolution-element SNR. Magnitude
normalization uses approximate band windows, not full filter transmission curves.

For bash: 

Calculate SNR of a 
.. code-block:: bash

    just-etc --mag 20.5 --band r --texp 900 --nexp 4 --target point

.. code-block:: bash

    just-etc --mag 21 --target extended --reff 0.6 --target-snr 5 --ref-wave 600

The first invocation may take longer while Numba compiles numerical kernels.
The CLI uses a starburst template by default; ``--target point`` changes spatial
extent only. Supply ``--template /path/to/stellar.fits`` for a stellar SED.

Repository layout
-----------------

.. code-block:: text

    py/just_etc/          Python API, engine, CLI, model tables and tests
      data/spec.dat      Installed default instrument configuration
      data/templates/    Installed FITS template library
      test/              Resource, numerical and CLI regression tests
    bin/                 CLI launcher and configurable PBS example
    doc/                 Current documentation, API reference and historical notes
    etc/                 Editable spec.dat example and optional modulefile
    examples/            Research, plotting, batch and Redrock workflows
    pyproject.toml       PEP 517 build-system configuration
    setup.cfg            Package metadata, dependencies and entry points
    setup.py             Compatibility shim for the original Python 3.6 environment
    MANIFEST.in          Source-distribution file selection

Runtime data lives inside the package so it is included in wheels and works
outside the checkout. The ``etc/spec.dat`` example matches the packaged default;
pass ``spec_file=...`` to use a custom instrument configuration.

Documentation and validation
----------------------------

* `中文入门 <doc/quickstart_zh.md>`_
* `API and scientific conventions <doc/usage.md>`_
* `Migration and file inventory <doc/migration.md>`_
* `Git maintenance guide <doc/development.md>`_
* `Research workflows <examples/README.md>`_

Run ``python -m pytest py/just_etc/test`` after installation. Build documentation
with ``python -m pip install '.[doc]'`` followed by
``python -m sphinx -b html doc doc/_build/html``.

Research outputs, caches, screenshots and the large external SAGA catalog are
not required for the package and are not included. Historical manuals and
comparison reports are retained under ``doc/legacy`` and ``doc/validation``;
their historical results are not new validation of the model. Optional
DESI/Redrock workflows require their own dependencies and input datasets.
