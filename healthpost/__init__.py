"""
HealthPost - Complete CHW Decision Support System

A comprehensive tool for Community Health Workers that supports the complete
patient visit workflow: intake, diagnosis, prescription, and dispensing.

Uses MedGemma for medical vision and reasoning, with DDInter API for drug interaction checking.
"""

from .core import HealthPost
from .config import Config

__version__ = "0.1.0"
__all__ = ["HealthPost", "Config"]
