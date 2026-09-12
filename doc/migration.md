# Migration from ETC_py_v1

The repository follows the desitemplate directory convention. Build metadata uses
`setup.cfg` plus a small `setup.py` compatibility shim rather than PEP 621 metadata
because the original execution environment is Python 3.6 with setuptools 59.6.
`pyproject.toml` still provides the PEP 517 build backend.

| Original content | Repository destination |
| --- | --- |
| `ETC_py_optimized.py`, `modeldata.py` | `py/just_etc/`, unchanged |
| `just_etc_api.py`, `dwarf_ssp_model.py` | `py/just_etc/`, package-relative imports/data |
| `cal_exp_time.py` | `py/just_etc/cli.py`; `just-etc` command |
| `spec.dat` | Installed `py/just_etc/data/spec.dat` and editable `etc/spec.dat` |
| `templates/*/*.fits` | `py/just_etc/data/templates/`, unchanged binary data |
| Three representative plots | `examples/`; specialized workflows archived in `examples/archive/` |
| `dwarf/simulate_dwarf_observability.py` | `examples/archive/simulate_dwarf_observability.py` |
| `submit_bgs10k_etc.qsub` | Parameterized `bin/submit_bgs10k_etc.qsub` |
| Existing and source README/manual files | `doc/legacy/`, marked historical |
| Comparison reports and plots | `doc/validation/` |
| Flowchart | `doc/flowchart.md` |

Not published: `.DS_Store`, Python/Numba caches, generated `output/` and `limit_mag/`
results, template screenshots, large SAGA catalogs, and comparison input datasets.
The duplicate historical `limit_mag/generate_just_limit.py` remains in the source
directory; `examples/archive/generate_just_limit.py` uses the top-level source version.
Archived programs preserve the original scientific workflows, while the showcase plots
are maintained as the recommended entry points.

Old imports such as `from just_etc_api import ...` should become
`from just_etc import ...`; low-level access is
`from just_etc import ETC_py_optimized`. Public function names and scientific
calculations are preserved. The loader also accepts FITS ASCII tables, used by
13 of the supplied infrared templates; numerical table values are not changed. Examples now use installed package data and write
under the current working directory. Optional Redrock workflows respect the
user's existing Redrock configuration instead of overriding it with a local path.

The inherited BSD license is retained. `modeldata.py` and the supplied spectral
templates retain their original data and FITS metadata. Template families include
the source library's galaxy, Kinney starburst, BC03 and infrared galaxy SEDs;
these data have not been newly generated or independently calibrated here.

## Validation scope

Packaging tests check bundled resources, installation-independent paths, the CLI,
approximate flux scaling, seeded simulation behavior and SNR consistency.
Full-arm numerical migration comparisons are run separately against an unchanged
copy of the supplied API/engine for point and extended source configurations.
They establish migration consistency, not new physical calibration or agreement
with the original C code. Full survey, HPC and optional Redrock fitting workflows
require external inputs and are not part of the package smoke test.
