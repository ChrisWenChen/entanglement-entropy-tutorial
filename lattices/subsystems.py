"""
lattices/subsystems.py — Subsystem construction utilities.

Functions to generate lists of site indices for various subsystem geometries.
"""

import numpy as np


def subsystem_left_half(Lx, Ly, sites_per_cell=1):
    """Return site indices for the left half of a lattice (x < Lx/2).

    This is the standard "cylinder cut" used for studying area laws.

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions in unit cells.
    sites_per_cell : int, optional
        Number of sites per unit cell (1 for square, 2 for honeycomb).

    Returns
    -------
    sub : list of int
        Site indices belonging to the left half (x < Lx//2).

    Notes
    -----
    The boundary length is |∂A| = Ly * sites_per_cell.
    For the square lattice: |∂A| = Ly.
    For the honeycomb lattice: |∂A| = 2 * Ly (since each cell has 2 sites).
    """
    sub = []
    for x in range(Lx // 2):
        for y in range(Ly):
            base = sites_per_cell * (x * Ly + y)
            for s in range(sites_per_cell):
                sub.append(base + s)
    return sub


def subsystem_cylinder(Lx, Ly, center_x, center_y, radius, sites_per_cell=1):
    """Return site indices for a cylindrical (circular) subsystem.

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions in unit cells.
    center_x, center_y : float
        Center of the cylinder in unit cell coordinates.
    radius : float
        Radius of the cylinder.
    sites_per_cell : int, optional
        Number of sites per unit cell.

    Returns
    -------
    sub : list of int
        Site indices inside the cylinder.

    Notes
    -----
    Useful for extracting topological entanglement entropy, which appears
    as a constant term: S = alpha * |∂A| - gamma.
    """
    sub = []
    for x in range(Lx):
        for y in range(Ly):
            # Calculate distance from center (in unit cell coordinates)
            dx = x - center_x
            dy = y - center_y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist <= radius:
                base = sites_per_cell * (x * Ly + y)
                for s in range(sites_per_cell):
                    sub.append(base + s)
    return sub


def subsystem_strip(Lx, Ly, y_start, width, sites_per_cell=1):
    """Return site indices for a horizontal strip subsystem.

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions.
    y_start : int
        Starting y index of the strip.
    width : int
        Width of the strip in y direction.
    sites_per_cell : int, optional
        Number of sites per unit cell.

    Returns
    -------
    sub : list of int
        Site indices in the strip y_start <= y < y_start + width.
    """
    sub = []
    for x in range(Lx):
        for y in range(y_start, min(y_start + width, Ly)):
            base = sites_per_cell * (x * Ly + y)
            for s in range(sites_per_cell):
                sub.append(base + s)
    return sub
