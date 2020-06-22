#!/usr/bin/env python3
"""
The axes classes used for all ProPlot figures.
"""
import matplotlib.projections as mproj

from ..internals import warnings
from . import plot  # noqa: F401
from .base import Axes  # noqa: F401
from .cartesian import CartesianAxes
from .geo import GeoAxes  # noqa: F401
from .geo import BasemapAxes, CartopyAxes
from .polar import PolarAxes

XYAxes, ProjAxes = warnings._rename_objs(
    '0.6',
    XYAxes=CartesianAxes,
    ProjAxes=GeoAxes,
)

mproj.register_projection(CartesianAxes)
mproj.register_projection(PolarAxes)
mproj.register_projection(BasemapAxes)
mproj.register_projection(CartopyAxes)

# Prevent importing module names and set order of appearance for objects
__all__ = [
    'Axes', 'CartesianAxes', 'PolarAxes',
    'GeoAxes', 'CartopyAxes', 'BasemapAxes',
    'ProjAxes', 'XYAxes',  # deprecated
]
