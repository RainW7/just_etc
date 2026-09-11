"""Exposure Time Calculator for the Jiaotong University Spectroscopic Telescope."""
from ._version import __version__
from .just_etc_api import (
    JUSTExposureTimeCalculator, load_template, list_templates, normalize_to_mag,
)

__all__ = ["JUSTExposureTimeCalculator", "load_template", "list_templates",
           "normalize_to_mag", "__version__"]
