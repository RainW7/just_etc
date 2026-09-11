"""Paths to data installed with JUST ETC (independent of the working directory)."""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

def template_path(filepath):
    """Resolve an existing user path first, then a bundled category/filename."""
    path = Path(filepath).expanduser()
    if path.exists() or path.is_absolute():
        return path
    if path.parts and path.parts[0] == "templates":
        path = Path(*path.parts[1:])
    return DATA_DIR / "templates" / path
