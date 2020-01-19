#!/usr/bin/env python3
"""
Simple tools that may be useful in the context of plotting. They are also
used internally throughout this package.
"""
from .validators import _validate_units
from numbers import Integral
import numpy as np
try:  # use this for debugging instead of print()!
    from icecream import ic
except ImportError:  # graceful fallback if IceCream isn't installed
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

__all__ = ['arange', 'edges', 'edges2d', 'units']


def arange(min_, *args):
    """Identical to `numpy.arange` but with inclusive endpoints. For
    example, ``plot.arange(2,4)`` returns ``np.array([2,3,4])`` instead
    of ``np.array([2,3])``. This command is useful for generating lists of
    tick locations or colorbar level boundaries."""
    # Optional arguments just like np.arange
    if len(args) == 0:
        max_ = min_
        min_ = 0
        step = 1
    elif len(args) == 1:
        max_ = args[0]
        step = 1
    elif len(args) == 2:
        max_ = args[0]
        step = args[1]
    else:
        raise ValueError('Function takes from one to three arguments.')
    # All input is integer
    if all(isinstance(val, Integral) for val in (min_, max_, step)):
        min_, max_, step = np.int64(min_), np.int64(max_), np.int64(step)
        max_ += np.sign(step) * 1
    # Input is float or mixed, cast to float64
    # Don't use np.nextafter with np.finfo(np.dtype(np.float64)).max, because
    # round-off errors from continually adding step to min mess this up
    else:
        min_, max_, step = np.float64(min_), np.float64(max_), np.float64(step)
        max_ += np.sign(step) * (step / 2)
    return np.arange(min_, max_, step)


def edges(Z, axis=-1):
    """
    Calculate the approximate "edge" values along an arbitrary axis, given
    "center" values. This is used internally to calculate graticule edges when
    you supply centers to `~matplotlib.axes.Axes.pcolor` or
    `~matplotlib.axes.Axes.pcolormesh` and to calculate colormap levels
    when you supply centers to any method wrapped by
    `~proplot.wrappers.cmap_changer`.

    Parameters
    ----------
    Z : array-like
        Array of any shape or size. Generally, should be monotonically
        increasing or decreasing along `axis`.
    axis : int, optional
        The axis along which "edges" are calculated. The size of this axis
        will be increased by one.

    Returns
    -------
    `~numpy.ndarray`
        Array of "edge" coordinates.
    """
    Z = np.asarray(Z)
    Z = np.swapaxes(Z, axis, -1)
    Z = np.concatenate((
        Z[..., :1] - (Z[..., 1] - Z[..., 0]) / 2,
        (Z[..., 1:] + Z[..., :-1]) / 2,
        Z[..., -1:] + (Z[..., -1] - Z[..., -2]) / 2,
    ), axis=-1)
    return np.swapaxes(Z, axis, -1)


def edges2d(Z):
    """
    Like `edges` but for 2d arrays.
    The size of both axes are increased by one. This is used
    internally to calculate graitule edges when you supply centers to
    `~matplotlib.axes.Axes.pcolor` or `~matplotlib.axes.Axes.pcolormesh`.

    Parameters
    ----------
    Z : array-like
        A 2d array.

    Returns
    -------
    `~numpy.ndarray`
        Array of "edge" coordinates.
    """
    Z = np.asarray(Z)
    if Z.ndim != 2:
        raise ValueError(f'Input must be a 2d array, but got {Z.ndim}d.')
    ny, nx = Z.shape
    Zb = np.zeros((ny + 1, nx + 1))
    # Inner
    Zb[1:-1, 1:-1] = 0.25 * (
        Z[1:, 1:] + Z[:-1, 1:] + Z[1:, :-1] + Z[:-1, :-1]
    )
    # Lower and upper
    Zb[0] += edges(1.5 * Z[0] - 0.5 * Z[1])
    Zb[-1] += edges(1.5 * Z[-1] - 0.5 * Z[-2])
    # Left and right
    Zb[:, 0] += edges(1.5 * Z[:, 0] - 0.5 * Z[:, 1])
    Zb[:, -1] += edges(1.5 * Z[:, -1] - 0.5 * Z[:, -2])
    # Corners
    Zb[[0, 0, -1, -1], [0, -1, -1, 0]] *= 0.5
    return Zb


def units(value, out='in', **kwargs):
    """
    Convert values and lists of values between arbitrary physical units. This
    is used internally all over ProPlot, permitting flexible units for various
    keyword arguments.

    Parameters
    ----------
    value : float or str or list thereof
        A size specifier or *list thereof*. If numeric, nothing is done.
        If string, it is converted to `units` units. The string should look
        like ``'123.456unit'``, where the number is the magnitude and
        ``'unit'`` is one of the following.

        =========  =========================================================================================
        Key        Description
        =========  =========================================================================================
        ``'m'``    Meters
        ``'cm'``   Centimeters
        ``'mm'``   Millimeters
        ``'ft'``   Feet
        ``'in'``   Inches
        ``'pt'``   `Points <https://en.wikipedia.org/wiki/Point_(typography)>`__ (1/72 inches)
        ``'pc'``   `Pica <https://en.wikipedia.org/wiki/Pica_(typography)>`__ (1/6 inches)
        ``'px'``   Pixels on screen, uses dpi of :rcraw:`figure.dpi`
        ``'pp'``   Pixels once printed, uses dpi of :rcraw:`savefig.dpi`
        ``'em'``   `Em square <https://en.wikipedia.org/wiki/Em_(typography)>`__ for :rcraw:`font.size`
        ``'en'``   `En square <https://en.wikipedia.org/wiki/En_(typography)>`__ for :rcraw:`font.size`
        ``'Em'``   `Em square <https://en.wikipedia.org/wiki/Em_(typography)>`__ for :rcraw:`axes.titlesize`
        ``'En'``   `En square <https://en.wikipedia.org/wiki/En_(typography)>`__ for :rcraw:`axes.titlesize`
        ``'ax'``   Axes relative units. Not always available.
        ``'fig'``  Figure relative units. Not always available.
        ``'ly'``   Light years ;)
        =========  =========================================================================================

    out : str, optional
        The destination units. Default is inches, i.e. ``'in'``.
    axes : `~matplotlib.axes.Axes`, optional
        The axes to use for scaling units that look like ``0.1ax``.
    figure : `~matplotlib.figure.Figure`, optional
        The figure to use for scaling units that look like ``0.1fig``. If
        ``None`` we try to get the figure from ``axes.figure``.
    width : bool, optional
        Whether to use the width or height for the axes and figure relative
        coordinates.
    """  # noqa
    # Scale for converting inches to arbitrary other unit
    scale = _validate_units(
        '1' + out if isinstance(out, str) else out,
        prefix='Invalid destination units {out!r}. ',
        convert=True,
        **kwargs
    )

    # Convert units for each value in list
    prefix = (
        f'Invalid size spec {value!r}. '
        'Must be number or string or list thereof. '
    )
    if not np.iterable(value) or isinstance(value, str):
        return _validate_units(
            value, prefix=prefix, convert=True, **kwargs
        ) / scale
    else:
        return [
            _validate_units(
                val, prefix=prefix, convert=True, **kwargs
            ) / scale
            for val in value
        ]
