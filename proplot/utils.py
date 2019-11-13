#!/usr/bin/env python3
"""
Simple tools used in various places across this package.
"""
import re
import time
import functools
import warnings
import numpy as np
from matplotlib import rcParams
from numbers import Number, Integral
try:
    from icecream import ic
except ImportError:  # graceful fallback if IceCream isn't installed
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a) # noqa
__all__ = ['arange', 'edges', 'units']

NUMBER = re.compile('^([-+]?[0-9._]+([eE][-+]?[0-9_]+)?)(.*)$')
BENCHMARK = False

class _benchmark(object):
    """Context object that can be used to time import statements."""
    def __init__(self, message):
        self.message = message
    def __enter__(self):
        if BENCHMARK:
            self.time = time.clock()
    def __exit__(self, *args):
        if BENCHMARK:
            print(f'{self.message}: {time.clock() - self.time}s')

def _timer(func):
    """Decorator that prints the time a function takes to execute.
    See: https://stackoverflow.com/a/1594484/4970632"""
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if BENCHMARK:
            t = time.clock()
        res = func(*args, **kwargs)
        if BENCHMARK:
            print(f'  {func.__name__}() time: {time.clock()-t}s')
        return res
    return decorator

def _counter(func):
    """Decorator that counts and prints the cumulative time a function
    has been running. See: https://stackoverflow.com/a/1594484/4970632"""
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if BENCHMARK:
            t = time.clock()
        res = func(*args, **kwargs)
        if BENCHMARK:
            decorator.time += (time.clock() - t)
            decorator.count += 1
            print(f'  {func.__name__}() cumulative time: {decorator.time}s ({decorator.count} calls)')
        return res
    decorator.time = 0
    decorator.count = 0 # initialize
    return decorator

def _notNone(*args, names=None):
    """Return the first non-``None`` value. This is used with keyword arg
    aliases and for setting default values. Ugly name but clear purpose. Pass
    the `names` keyword arg to issue warning if multiple args were passed. Must
    be list of non-empty strings."""
    if names is None:
        for arg in args:
            if arg is not None:
                return arg
        return arg # last one
    else:
        first = None
        kwargs = {}
        if len(names) != len(args) - 1:
            raise ValueError(f'Need {len(args)+1} names for {len(args)} args, but got {len(names)} names.')
        names = [*names, '']
        for name,arg in zip(names,args):
            if arg is not None:
                if first is None:
                    first = arg
                if name:
                    kwargs[name] = arg
        if len(kwargs) > 1:
            warnings.warn(f'Got conflicting or duplicate keyword args: {kwargs}. Using the first one.')
        return first

# Accessible for user
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
    if all(isinstance(val,Integral) for val in (min_,max_,step)):
        min_, max_, step = np.int64(min_), np.int64(max_), np.int64(step)
        max_ += 1
    # Input is float or mixed, cast to float64
    # Don't use np.nextafter with np.finfo(np.dtype(np.float64)).max, because
    # round-off errors from continually adding step to min mess this up
    else:
        min_, max_, step = np.float64(min_), np.float64(max_), np.float64(step)
        max_ += step/2
    return np.arange(min_, max_, step)

def edges(array, axis=-1):
    """
    Calculate the approximate "edge" values along an arbitrary axis, given
    "center" values. This is used internally to calculate graticule edges when
    you supply centers to `~matplotlib.axes.Axes.pcolor` or
    `~matplotlib.axes.Axes.pcolormesh`, and to calculate colorbar level edges
    when you supply centers to any method wrapped by
    `~proplot.wrappers.cmap_changer`.

    Parameters
    ----------
    array : array-like
        Array of any shape or size. Generally, should be monotonically
        increasing or decreasing along `axis`.
    axis : int, optional
        The axis along which "edges" are calculated. The size of this axis
        will be augmented by one.

    Returns
    -------
    `~numpy.ndarray`
        Array of "edge" coordinates.
    """
    # First permute
    array = np.array(array)
    array = np.swapaxes(array, axis, -1)
    # Next operate
    flip = False
    idxs = [[0] for _ in range(array.ndim-1)] # must be list because we use it twice
    if array[np.ix_(*idxs, [1])] < array[np.ix_(*idxs, [0])]:
        flip = True
        array = np.flip(array, axis=-1)
    array = np.concatenate((
        array[...,:1]  - (array[...,1]-array[...,0])/2,
        (array[...,1:] + array[...,:-1])/2,
        array[...,-1:] + (array[...,-1]-array[...,-2])/2,
        ), axis=-1)
    if flip:
        array = np.flip(array, axis=-1)
    # Permute back and return
    array = np.swapaxes(array, axis, -1)
    return array

def units(value, units='in', axes=None, figure=None, width=True):
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

    units : str, optional
        The destination units. Default is inches, i.e. ``'in'``.
    axes : `~matplotlib.axes.Axes`, optional
        The axes to use for scaling units that look like ``0.1ax``.
    figure : `~matplotlib.figure.Figure`, optional
        The figure to use for scaling units that look like ``0.1fig``. If
        ``None`` we try to get the figure from ``axes.figure``.
    width : bool, optional
        Whether to use the width or height for the axes and figure relative
        coordinates.
    """
    # Font unit scales
    # NOTE: Delay font_manager import, because want to avoid rebuilding font
    # cache, which means import must come after TTFPATH added to environ
    # by styletools.register_fonts()!
    small = rcParams['font.size'] # must be absolute
    large = rcParams['axes.titlesize']
    if isinstance(large, str):
        import matplotlib.font_manager as mfonts
        scale = mfonts.font_scalings.get(large, 1) # error will be raised somewhere else if string name is invalid!
        large = small*scale

    # Scales for converting physical units to inches
    unit_dict = {
        'in': 1.0,
        'm':  39.37,
        'ft': 12.0,
        'cm': 0.3937,
        'mm': 0.03937,
        'pt': 1/72.0,
        'pc': 1/6.0,
        'em': small/72.0,
        'en': 0.5*small/72.0,
        'Em': large/72.0,
        'En': 0.5*large/72.0,
        'ly': 3.725e+17,
        }
    # Scales for converting display units to inches
    # WARNING: In ipython shell these take the value 'figure'
    if not isinstance(rcParams['figure.dpi'], str):
        unit_dict['px'] = 1/rcParams['figure.dpi'] # once generated by backend
    if not isinstance(rcParams['savefig.dpi'], str):
        unit_dict['pp'] = 1/rcParams['savefig.dpi'] # once 'printed' i.e. saved
    # Scales relative to axes and figure objects
    if axes is not None and hasattr(axes, 'get_size_inches'): # proplot axes
        unit_dict['ax'] = axes.get_size_inches()[1-int(width)]
    if figure is None:
        figure = getattr(axes, 'figure', None)
    if figure is not None and hasattr(figure, 'get_size_inches'): # proplot axes
        unit_dict['fig'] = fig.get_size_inches()[1-int(width)]
    # Scale for converting inches to arbitrary other unit
    try:
        scale = unit_dict[units]
    except KeyError:
        raise ValueError(f'Invalid destination units {units!r}. Valid units are {", ".join(map(repr, unit_dict.keys()))}.')

    # Convert units for each value in list
    result = []
    singleton = (not np.iterable(value) or isinstance(value, str))
    for val in ((value,) if singleton else value):
        if val is None or isinstance(val, Number):
            result.append(val)
            continue
        elif not isinstance(val, str):
            raise ValueError(f'Size spec must be string or number or list thereof. Got {value!r}.')
        regex = NUMBER.match(val)
        if not regex:
            raise ValueError(f'Invalid size spec {val!r}. Valid units are {", ".join(map(repr, unit_dict.keys()))}.')
        number, _, units = regex.groups() # second group is exponential
        try:
            result.append(float(number) * (unit_dict[units]/scale if units else 1))
        except (KeyError, ValueError):
            raise ValueError(f'Invalid size spec {val!r}. Valid units are {", ".join(map(repr, unit_dict.keys()))}.')
    if singleton:
        result = result[0]
    return result

