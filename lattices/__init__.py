"""
lattices/__init__.py — Lattice construction for tight-binding models.

This module provides functions to construct tight-binding Hamiltonians
and correlation matrices for free fermion systems on various lattices.
"""

from .chain_1d import chain_1d
from .square_2d import square_2d
from .honeycomb_2d import honeycomb_2d
from .subsystems import (
    subsystem_left_half,
    subsystem_cylinder,
    subsystem_strip,
)

__all__ = [
    "chain_1d",
    "square_2d", 
    "honeycomb_2d",
    "subsystem_left_half",
    "subsystem_cylinder",
    "subsystem_strip",
]