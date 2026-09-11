"""Convenience launcher for the PFS ETC Python port.

Edit the CONFIGURATION section below as needed and then run:

    python Gemini/run_etc_preset.py

The script patches the interactive prompts in `ETC_py_optimized.main()` so the
exposure-time calculator runs end-to-end without manual input.
"""

from just_etc.resources import DATA_DIR

import time
import datetime as _dt
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from unittest.mock import patch

# ETC_py_v1 — all modules co-located in this directory
_V1_DIR = Path.cwd()
import sys

try:
    from just_etc import ETC_py_optimized as ETC_py
except ModuleNotFoundError as exc:
    if exc.name == "numba":
        raise RuntimeError(
            "ETC_py_optimized depends on the 'numba' package. Install it via 'pip install numba' "
            "before running this preset launcher."
        ) from exc
    raise

# ---------------------------------------------------------------------------
# CONFIGURATION – tweak these values to your desired scenario
# ---------------------------------------------------------------------------
PROJECT_ROOT = _V1_DIR
CONFIG_FILE  = DATA_DIR / "spec.dat"

# Observing conditions (mirrors prompts in ETC_py_old_safe.main)
SKYTYPE_HEX = "10003"          # see JUST_manual.txt for nibble breakdown
SEEING_ARCSEC = 0.8             # FWHM @ 800 nm
ZENITH_ANGLE_DEG = 45.0
EBV_MAG = 0.03
FIELD_ANGLE_DEG = 0.675
DECENTER_ARCSEC = 0.03

# Exposure setup
EXPOSURE_TIME_S = 450.0         # per exposure in seconds
NUM_EXPOSURES = 8
SYSTEMATIC_FLOOR = 0         # rms per 1D pixel
DIFFUSE_STRAY_FRACTION = 0

# Output control – files will be created under OUTPUT_BASE/<timestamp>
OUTPUT_BASE = _V1_DIR / "output" / "auto_runs"
WRITE_ELG_SNR_CURVE = True
WRITE_CONTINUUM_CURVE = True
WRITE_SINGLE_LINE_SNR_CURVE = True

# Single Line SNR Parameters
SINGLE_LINE_FLUX = 1.0e-17
SINGLE_LINE_SIGMA = 70.0

# Optional [OII] catalogue handling
OII_INPUT_CATALOG = None   # set to _V1_DIR / "galaxy_catalog.dat" to enable
OII_MIN_SNR = 5.0          # used only if catalogue is provided

# Optional Magnitude Input File
MAGNITUDE_INPUT_FILE = None # set Path(...) or string; None means skip ("-")

# ---------------------------------------------------------------------------


def _format_inputs(noise_file: Path,
                   elg_file: str,
                   cont_file: str,
                   single_line_file: str,
                   oii_input: str,
                   oii_output: Optional[str]) -> List[str]:
    """Build the ordered list of answers fed to ETC_py.main()."""
    inputs: List[str] = [
        str(CONFIG_FILE),
        SKYTYPE_HEX,
        f"{SEEING_ARCSEC:.6f}",
        f"{ZENITH_ANGLE_DEG:.6f}",
        f"{EBV_MAG:.6f}",
        f"{FIELD_ANGLE_DEG:.6f}",
        f"{DECENTER_ARCSEC:.6f}",
        f"{EXPOSURE_TIME_S:.6f}",
        str(int(NUM_EXPOSURES)),
        f"{SYSTEMATIC_FLOOR:.6f}",
        f"{DIFFUSE_STRAY_FRACTION:.6f}",
        "0", # Noise reused? (0=No)
        str(noise_file),
        elg_file,
        single_line_file,
        f"{SINGLE_LINE_FLUX:.6e}",
        f"{SINGLE_LINE_SIGMA:.6f}",
        cont_file,
        oii_input,
    ]

    if oii_output is not None:
        inputs.append(oii_output)
        inputs.append(f"{OII_MIN_SNR:.6f}")
    
    # Magnitude input file
    msg_input_val = "-"
    if MAGNITUDE_INPUT_FILE is not None:
        msg_input_val = str(MAGNITUDE_INPUT_FILE)
    inputs.append(msg_input_val)

    return inputs


def _resolve_files() -> Tuple[Path, str, str, str, str, Optional[str]]:
    """Determine output file paths and ensure their directories exist."""
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_BASE / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    noise_file = run_dir / "noise_vector.txt"
    elg_file = str(run_dir / "elg_snr.txt") if WRITE_ELG_SNR_CURVE else "-"
    cont_file = str(run_dir / "continuum_snr.txt") if WRITE_CONTINUUM_CURVE else "-"
    single_line_file = str(run_dir / "single_line_snr.txt") if WRITE_SINGLE_LINE_SNR_CURVE else "-"

    if OII_INPUT_CATALOG is None:
        oii_input = "-"
        oii_output = None  # type: Optional[str]
    else:
        oii_input_path = Path(OII_INPUT_CATALOG)
        oii_input = str(oii_input_path)
        oii_output_path = run_dir / "oii_results.txt"
        oii_output = str(oii_output_path)
        oii_output_path.parent.mkdir(parents=True, exist_ok=True)

    return noise_file, elg_file, cont_file, single_line_file, oii_input, oii_output


def _check_prerequisites() -> None:
    """Fail fast if required resources are missing."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Spectrograph config not found: {CONFIG_FILE}")
    if not ETC_py.MODEL_DATA_AVAILABLE:
        raise RuntimeError(
            "modeldata.py not available in ETC_py_v1 – copy it from Gemini/ before running.")


def run_with_presets(mode_name: str = 'balanced') -> None:
    """Execute ETC_py_optimized.main() using the preset answers defined above."""
    
    # Set calculation mode
    if hasattr(ETC_py, 'set_calculation_mode'):
        ETC_py.set_calculation_mode(mode_name)
    else:
        print("Warning: ETC_py module does not support 'set_calculation_mode'. Running with defaults.")
    _check_prerequisites()
    noise_file, elg_file, cont_file, single_line_file, oii_input, oii_output = _resolve_files()
    answers = _format_inputs(noise_file, elg_file, cont_file, single_line_file, oii_input, oii_output)

    def _answer_iterator(values: Iterable[str]):
        for value in values:
            yield value

    inputs_iter = _answer_iterator(answers)

    def _patched_input(prompt: str = "") -> str:
        try:
            response = next(inputs_iter)
        except StopIteration as exc:
            raise RuntimeError("Ran out of scripted inputs for ETC prompts") from exc
        print(prompt, end="")
        print(response)
        return response

    print(f"\n--- Starting ETC Run in [{mode_name.upper()}] mode ---")
    start_time = time.time()
    
    with patch("builtins.input", side_effect=_patched_input):
        ETC_py.main()
        
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n[{mode_name.upper()}] Mode Calculation Completed in {duration:.2f} seconds.")


if __name__ == "__main__":
    print("Select Calculation Mode:")
    print("1. Fast (Low Precision, High Speed)")
    print("2. Balanced (Standard)")
    print("3. Accurate (High Precision, Low Speed)")
    
    choice = input("Enter choice [1/2/3] (default 2): ").strip()
    
    mode_map = {'1': 'fast', '2': 'balanced', '3': 'accurate'}
    selected_mode = mode_map.get(choice, 'balanced')
    
    run_with_presets(selected_mode)

