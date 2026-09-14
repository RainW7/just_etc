"""Exposure Time Calculator for the Jiaotong University Spectroscopic Telescope."""
from ._version import __version__
from .just_etc_api import (
    JUSTExposureTimeCalculator, load_template, list_templates, normalize_to_mag,
)
from .redrock import write_redrock_mock_spectrum
from .spectral_library import convet_to_redrock_format

__all__ = ["JUSTExposureTimeCalculator", "load_template", "list_templates",
           "normalize_to_mag", "write_redrock_mock_spectrum",
           "convet_to_redrock_format", "__version__"]
