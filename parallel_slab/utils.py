#!/usr/bin/env python3

__doc__ = """Common Utilities"""

import numpy as np
from typing import Dict, Any


def generate_grid_like_pycfs(
    n_total_points: int, l_half_solid: float, l_half_fluid: float
):
    """
    Generates grid according to conventions of our in-house solver for comparison

    Parameters
    ----------
    n_total_points : Total number of points in the cfs domain (between 0.0 to 1.0)
    l_half_solid : Half-length of the solid
    l_half_fluid : Half-length of the fluid

    Returns
    -------
    a (2, ) tuple with the solid and the fluid grids
    The solid grids is from (0.0, l_solid)
    The fluid grid is from (0.0, l_fluid)
    We solve for the two grids separately and merge them together
    """
    assert l_half_solid < 0.5
    assert l_half_fluid < 0.5

    # Only look at one half of solution
    # n_effective_points = n_total_points // 2

    # Solid starts at 0.5 + dx / 2 in simulation
    dx = 1.0 / n_total_points
    n_solid = int(np.floor(l_half_solid / dx))

    solid_start = 0.5 * dx
    solid_end = solid_start + n_solid * dx
    y_solid = np.linspace(solid_start, solid_end, n_solid + 1, endpoint=True)

    fluid_start = solid_end + dx
    if solid_end > l_half_solid:
        # The last point is +0.5dx and is inside the fluid zone
        y_solid = y_solid[:-1]
        fluid_start -= dx

    # If at 0.0, solids end is fluids start
    # fluid_start = 0.0 * dx  # solid_end + dx
    n_fluid = int(np.floor(l_half_fluid / dx))

    fluid_end = fluid_start + n_fluid * dx
    # We add an extra grid point here because we start again at 0.0 and not 0.0 + dx
    y_fluid = np.linspace(fluid_start, fluid_end, n_fluid + 1, endpoint=True)

    # The last point is +0.5dx and is beyond the boundary
    y_fluid = y_fluid[:-1]

    # We solve separately for the fluid and solid parts, so shift it
    y_fluid -= l_half_solid

    return y_solid, y_fluid


def generate_regular_grid(
    n_total_points: int, l_half_solid: float, l_half_fluid: float
):
    """
    Generates regular grid

    Parameters
    ----------
    n_total_points : Total number of points in the domain (between 0.0 to 1.0)
    l_half_solid : Half-length of the solid
    l_half_fluid : Half-length of the fluid

    Returns
    -------
    a (2, ) tuple with the solid and the fluid grids
    The solid grids is from (0.0, l_solid)
    The fluid grid is from (0.0, l_fluid)
    We solve for the two grids separately and merge them together
    """
    n_solid_points = int(l_half_solid * n_total_points / (l_half_solid + l_half_fluid))
    n_fluid_points = n_total_points - n_solid_points

    ## Use this for the figure, detailed solution
    y_solid = np.linspace(0.0, l_half_solid, n_solid_points, endpoint=False)
    y_fluid = np.linspace(
        l_half_solid, l_half_solid + l_half_fluid, n_fluid_points, endpoint=False
    )

    # We solve separately for the fluid and solid parts, so shift it
    y_fluid -= l_half_solid

    return y_solid, y_fluid


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    import hashlib
    import json
    from pandas import Timestamp

    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}

    # convert unknown objects to strings first
    # used in psweep
    def myconverter(o):
        if isinstance(o, Timestamp):
            return o.__str__()

    encoded = json.dumps(dictionary, sort_keys=True, default=myconverter).encode()
    dhash.update(encoded)
    return dhash.hexdigest()
