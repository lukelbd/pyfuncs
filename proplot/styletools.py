#!/usr/bin/env python3
"""
Tools for registering and visualizing colormaps, color cycles, color string
names, and fonts. New colormap classes, new colormap normalizer
classes, and new constructor functions for generating instances of these
classes. Related utilities for manipulating colors. See
:ref:`Colormaps`, :ref:`Color cycles`, and :ref:`Colors and fonts`
for details.
"""
# Potential bottleneck, loading all this stuff?  *No*. Try using @timer on
# register functions, turns out worst is colormap one at 0.1 seconds.
import os
import re
import json
import glob
import cycler
from xml.etree import ElementTree
from numbers import Number, Integral
from matplotlib import rcParams
import numpy as np
import numpy.ma as ma
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from .cbook import _notNone, _timer, _warn_proplot
from .external import hsluv
try:  # use this for debugging instead of print()!
    from icecream import ic
except ImportError:  # graceful fallback if IceCream isn't installed
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

__all__ = [
    'BinNorm', 'CmapDict', 'ColorDict',
    'LinearSegmentedNorm',
    'LinearSegmentedColormap',
    'ListedColormap',
    'MidpointNorm', 'PerceptuallyUniformColormap',
    'cmaps', 'colors', 'cycles', 'fonts',
    'make_mapping_array',
    'register_cmaps', 'register_colors', 'register_cycles', 'register_fonts',
    'saturate', 'shade', 'show_cmaps', 'show_channels',
    'show_colors', 'show_colorspaces', 'show_cycles', 'show_fonts',
    'to_rgb', 'to_xyz',
    'Colormap', 'Colors', 'Cycle', 'Norm',
]

# Colormap stuff
CYCLES_TABLE = {
    'Matplotlib originals': (
        'default', 'classic',
    ),
    'Matplotlib stylesheets': (
        'colorblind', 'colorblind10', 'ggplot', 'bmh', 'solarized', '538',
    ),
    'ColorBrewer2.0 qualitative': (
        'Accent', 'Dark2',
        'Paired', 'Pastel1', 'Pastel2',
        'Set1', 'Set2', 'Set3',
    ),
    'Other qualitative': (
        'FlatUI', 'Qual1', 'Qual2', 'Viz',
    ),
    'ProPlot originals': (
        'Cool', 'Warm', 'Hot',
        'Floral', 'Contrast', 'Sharp',
    ),
}
CMAPS_TABLE = {
    # Assorted origin, but these belong together
    'Grayscale': (
        'Grays', 'Mono', 'GrayCycle',
    ),
    # Builtin
    'Matplotlib originals': (
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'twilight', 'twilight_shifted',
    ),
    # seaborn
    'Seaborn originals': (
        'Rocket', 'Mako', 'IceFire', 'Vlag',
    ),
    # PerceptuallyUniformColormap
    'ProPlot sequential': (
        'Fire',
        'Stellar',
        'Boreal',
        'Marine',
        'Dusk',
        'Glacial',
        'Sunrise', 'Sunset',
    ),
    'ProPlot diverging': (
        'Div', 'NegPos', 'DryWet',
    ),
    # cmOcean
    'cmOcean sequential': (
        'Oxy', 'Thermal', 'Dense', 'Ice', 'Haline',
        'Deep', 'Algae', 'Tempo', 'Speed', 'Turbid', 'Solar', 'Matter',
        'Amp', 'Phase',
    ),
    'cmOcean diverging': (
        'Balance', 'Delta', 'Curl',
    ),
    # ColorBrewer
    'ColorBrewer2.0 sequential': (
        'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'PuBu', 'PuBuGn', 'BuGn', 'GnBu', 'YlGnBu', 'YlGn'
    ),
    'ColorBrewer2.0 diverging': (
        'Spectral', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGY',
        'RdBu', 'RdYlBu', 'RdYlGn',
    ),
    # Nice diverging maps
    'Other diverging': (
        'ColdHot', 'CoolWarm', 'BR',
    ),
    # SciVisColor
    'SciVisColor blues': (
        'Blue0', 'Blue1', 'Blue2', 'Blue3', 'Blue4', 'Blue5',
        'Blue6', 'Blue7', 'Blue8', 'Blue9', 'Blue10', 'Blue11',
    ),
    'SciVisColor greens': (
        'Green1', 'Green2', 'Green3', 'Green4', 'Green5',
        'Green6', 'Green7', 'Green8',
    ),
    'SciVisColor oranges': (
        'Orange1', 'Orange2', 'Orange3', 'Orange4', 'Orange5',
        'Orange6', 'Orange7', 'Orange8',
    ),
    'SciVisColor browns': (
        'Brown1', 'Brown2', 'Brown3', 'Brown4', 'Brown5',
        'Brown6', 'Brown7', 'Brown8', 'Brown9',
    ),
    'SciVisColor reds and purples': (
        'RedPurple1', 'RedPurple2', 'RedPurple3', 'RedPurple4',
        'RedPurple5', 'RedPurple6', 'RedPurple7', 'RedPurple8',
    ),
    # Builtin maps that will be deleted; categories are taken from comments in
    # matplotlib source code. Some of these are really bad, some are segmented
    # maps when the should be color cycles, and some are just uninspiring.
    'MATLAB': (
        'bone', 'cool', 'copper', 'autumn', 'flag', 'prism',
        'jet', 'hsv', 'hot', 'spring', 'summer', 'winter', 'pink', 'gray',
    ),
    'GNUplot': (
        'gnuplot', 'gnuplot2', 'ocean', 'afmhot', 'rainbow',
    ),
    'GIST': (
        'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar',
        'gist_rainbow', 'gist_stern', 'gist_yarg',
    ),
    'Other': (
        'binary', 'bwr', 'brg',  # appear to be custom matplotlib
        'cubehelix', 'Wistia', 'CMRmap',  # individually released
        'seismic', 'terrain', 'nipy_spectral',  # origin ambiguous
        'tab10', 'tab20', 'tab20b', 'tab20c',  # merged colormap cycles
    )
}
CMAPS_DIVERGING = tuple(
    (key1.lower(), key2.lower()) for key1, key2 in (
        ('PiYG', 'GYPi'),
        ('PRGn', 'GnRP'),
        ('BrBG', 'GBBr'),
        ('PuOr', 'OrPu'),
        ('RdGy', 'GyRd'),
        ('RdBu', 'BuRd'),
        ('RdYlBu', 'BuYlRd'),
        ('RdYlGn', 'GnYlRd'),
        ('BR', 'RB'),
        ('CoolWarm', 'WarmCool'),
        ('ColdHot', 'HotCold'),
        ('NegPos', 'PosNeg'),
        ('DryWet', 'WetDry')
    ))

# Named color filter props
COLORS_SPACE = 'hcl'  # color "distincness" is defined with this space
COLORS_THRESH = 0.10  # bigger number equals fewer colors
COLORS_TRANSLATIONS = tuple((re.compile(regex), sub) for regex, sub in (
    ('/', ' '),
    ('\'s', ''),
    (r'\s?majesty', ''),  # purple mountains majesty is too long
    ('reddish', 'red'),  # remove 'ish'
    ('purplish', 'purple'),
    ('bluish', 'blue'),
    (r'ish\b', ''),
    ('grey', 'gray'),
    ('pinky', 'pink'),
    ('greeny', 'green'),
    ('bluey', 'blue'),
    ('purply', 'purple'),
    ('purpley', 'purple'),
    ('yellowy', 'yellow'),
    ('robin egg', 'robins egg'),
    ('egg blue', 'egg'),
    ('bluegray', 'blue gray'),
    ('grayblue', 'gray blue'),
    ('lightblue', 'light blue'),
))  # prevent registering similar-sounding names
COLORS_IGNORE = re.compile('(' + '|'.join((
    'shit', 'poop', 'poo', 'pee', 'piss', 'puke', 'vomit', 'snot',
    'booger', 'bile', 'diarrhea',
)) + ')')  # filter these out, let's try to be professional here...
COLORS_INCLUDE = (
    'charcoal', 'sky blue', 'eggshell', 'sea blue', 'coral', 'aqua',
    'tomato red', 'brick red', 'crimson',
    'red orange', 'yellow orange', 'yellow green', 'blue green',
    'blue violet', 'red violet',
)  # common names that should always be included
COLORS_OPEN = (
    'red', 'pink', 'grape', 'violet',
    'indigo', 'blue', 'cyan', 'teal',
    'green', 'lime', 'yellow', 'orange', 'gray'
)
COLORS_BASE = {
    'blue': (0, 0, 1),
    'green': (0, 0.5, 0),
    'red': (1, 0, 0),
    'cyan': (0, 0.75, 0.75),
    'magenta': (0.75, 0, 0.75),
    'yellow': (0.75, 0.75, 0),
    'black': (0, 0, 0),
    'white': (1, 1, 1),
}


def _get_channel(color, channel, space='hcl'):
    """
    Get the hue, saturation, or luminance channel value from the input color.
    The color name `color` can optionally be a string with the format
    ``'color+x'`` or ``'color-x'``, where `x` specifies the offset from the
    channel value.

    Parameters
    ----------
    color : color-spec
        The color. Sanitized with `to_rgb`.
    channel : {'hue', 'chroma', 'saturation', 'luminance'}
        The HCL channel to be retrieved.
    space : {'hcl', 'hpl', 'hsl', 'hsv', 'rgb'}, optional
        The colorspace for the corresponding channel value.

    Returns
    -------
    value : float
        The channel value.
    """
    # Interpret channel
    if callable(color) or isinstance(color, Number):
        return color
    if channel == 'hue':
        channel = 0
    elif channel in ('chroma', 'saturation'):
        channel = 1
    elif channel == 'luminance':
        channel = 2
    else:
        raise ValueError(f'Unknown channel {channel!r}.')
    # Interpret string or RGB tuple
    offset = 0
    if isinstance(color, str):
        match = re.search('([-+][0-9.]+)$', color)
        if match:
            offset = float(match.group(0))
            color = color[:match.start()]
    return offset + to_xyz(color, space)[channel]


def shade(color, scale=1):
    """
    Scale the luminance channel of the input color.

    Parameters
    ----------
    color : color-spec
        The color. Sanitized with `to_rgb`.
    scale : float, optoinal
        The luminance channel is multiplied by this value.

    Returns
    -------
    color
        The new RGB tuple.
    """
    *color, alpha = to_rgb(color, alpha=True)
    color = [*hsluv.rgb_to_hsl(*color)]
    # multiply luminance by this value
    color[2] = max(0, min(color[2] * scale, 100))
    color = [*hsluv.hsl_to_rgb(*color)]
    return (*color, alpha)


def saturate(color, scale=0.5):
    """
    Scale the saturation channel of the input color.

    Parameters
    ----------
    color : color-spec
        The color. Sanitized with `to_rgb`.
    scale : float, optoinal
        The HCL saturation channel is multiplied by this value.

    Returns
    -------
    color
        The new RGB tuple.
    """
    *color, alpha = to_rgb(color, alpha=True)
    color = [*hsluv.rgb_to_hsl(*color)]
    # multiply luminance by this value
    color[1] = max(0, min(color[1] * scale, 100))
    color = [*hsluv.hsl_to_rgb(*color)]
    return (*color, alpha)


def to_rgb(color, space='rgb', cycle=None, alpha=False):
    """
    Translate the color in *any* format and from *any* colorspace to an RGB
    tuple. This is a generalization of `matplotlib.colors.to_rgb` and the
    inverse of `to_xyz`.

    Parameters
    ----------
    color : str or length-3 list
        The color specification. Can be a tuple of channel values for the
        `space` colorspace, a hex string, a registered color name, a cycle
        color, or a colormap color (see `ColorDict`).

        If `space` is ``'rgb'``, this is a tuple of RGB values, and any
        channels are larger than ``2``, the channels are assumed to be on
        a ``0`` to ``255`` scale and are therefore divided by ``255``.
    space : {'rgb', 'hsv', 'hsl', 'hpl', 'hcl'}, optional
        The colorspace for the input channel values. Ignored unless `color` is
        an container of numbers.
    cycle : str or list, optional
        The registered color cycle name used to interpret colors that
        look like ``'C0'``, ``'C1'``, etc. Default is :rc:`cycle`.
    alpha : bool, optional
        Whether to preserve the opacity channel, if it exists. Default
        is ``False``.

    Returns
    -------
    color
        The RGB tuple.
    """
    # Convert color cycle strings
    if isinstance(color, str) and re.match('^C[0-9]$', color):
        if isinstance(cycle, str):
            try:
                cycle = mcm.cmap_d[cycle].colors
            except (KeyError, AttributeError):
                cycles = sorted(name for name, cmap in mcm.cmap_d.items(
                ) if isinstance(cmap, ListedColormap))
                raise ValueError(
                    f'Invalid cycle {cycle!r}. Options are: '
                    + ', '.join(map(repr, cycles)) + '.')
        elif cycle is None:
            cycle = rcParams['axes.prop_cycle'].by_key()
            if 'color' not in cycle:
                cycle = ['k']
            else:
                cycle = cycle['color']
        else:
            raise ValueError(f'Invalid cycle {cycle!r}.')
        color = cycle[int(color[-1]) % len(cycle)]

    # Translate RGB strings and (cmap,index) tuples
    opacity = 1
    if isinstance(color, str) or (np.iterable(color) and len(color) == 2):
        try:
            *color, opacity = mcolors.to_rgba(color)  # ensure is valid color
        except (ValueError, TypeError):
            raise ValueError(f'Invalid RGB argument {color!r}.')

    # Pull out alpha channel
    if len(color) == 4:
        *color, opacity = color
    elif len(color) != 3:
        raise ValueError(f'Invalid RGB argument {color!r}.')

    # Translate arbitrary colorspaces
    if space == 'rgb':
        try:
            if any(c > 2 for c in color):
                color = [c / 255 for c in color]  # scale to within 0-1
            color = tuple(color)
        except (ValueError, TypeError):
            raise ValueError(f'Invalid RGB argument {color!r}.')
    elif space == 'hsv':
        color = hsluv.hsl_to_rgb(*color)
    elif space == 'hpl':
        color = hsluv.hpluv_to_rgb(*color)
    elif space == 'hsl':
        color = hsluv.hsluv_to_rgb(*color)
    elif space == 'hcl':
        color = hsluv.hcl_to_rgb(*color)
    else:
        raise ValueError('Invalid color {color!r} for colorspace {space!r}.')

    # Return RGB or RGBA
    if alpha:
        return (*color, opacity)
    else:
        return color


def to_xyz(color, space='hcl', alpha=False):
    """
    Translate color in *any* format to a tuple of channel values in *any*
    colorspace. This is the inverse of `to_rgb`.

    Parameters
    ----------
    color : color-spec
        The color. Sanitized with `to_rgb`.
    space : {'hcl', 'hpl', 'hsl', 'hsv', 'rgb'}, optional
        The colorspace for the output channel values.
    alpha : bool, optional
        Whether to preserve the opacity channel, if it exists. Default
        is ``False``.

    Returns
    -------
    color
        Tuple of colorspace `space` channel values.
    """
    # Run tuple conversions
    # NOTE: Don't pass color tuple, because we may want to permit
    # out-of-bounds RGB values to invert conversion
    *color, opacity = to_rgb(color, alpha=True)
    if space == 'rgb':
        pass
    elif space == 'hsv':
        color = hsluv.rgb_to_hsl(*color)  # rgb_to_hsv would also work
    elif space == 'hpl':
        color = hsluv.rgb_to_hpluv(*color)
    elif space == 'hsl':
        color = hsluv.rgb_to_hsluv(*color)
    elif space == 'hcl':
        color = hsluv.rgb_to_hcl(*color)
    else:
        raise ValueError(f'Invalid colorspace {space}.')
    if alpha:
        return (*color, opacity)
    else:
        return color


def _clip_colors(colors, clip=True, gray=0.2):
    """
    Clip impossible colors rendered in an HSL-to-RGB colorspace conversion.
    Used by `PerceptuallyUniformColormap`. If `mask` is ``True``, impossible
    colors are masked out.

    Parameters
    ----------
    colors : list of length-3 tuples
        The RGB colors.
    clip : bool, optional
        If `clip` is ``True`` (the default), RGB channel values >1 are clipped
        to 1. Otherwise, the color is masked out as gray.
    gray : float, optional
        The identical RGB channel values (gray color) to be used if `mask`
        is ``True``.
    """
    # Clip colors
    colors = np.array(colors)
    over = (colors > 1)
    under = (colors < 0)
    if clip:
        colors[under] = 0
        colors[over] = 1
    else:
        colors[(under | over)] = gray
    # Message
    # NOTE: Never print warning because happens when using builtin maps
    # message = 'Clipped' if clip else 'Invalid'
    # for i,name in enumerate('rgb'):
    #     if under[:,i].any():
    #         _warn_proplot(f'{message} {name!r} channel ( < 0).')
    #     if over[:,i].any():
    #         _warn_proplot(f'{message} {name!r} channel ( > 1).')
    return colors


def _make_segmentdata_array(values, coords=None, ratios=None):
    """
    Return a segmentdata array or callable given the input colors
    and coordinates.

    Parameters
    ----------
    values : list of float
        The channel values.
    coords : list of float, optional
        The segment coordinates.
    ratios : list of float, optional
        The relative length of each segment transition.
    """
    # Allow callables
    if callable(values):
        return values
    values = np.atleast_1d(values)
    if len(values) == 1:
        value = values[0]
        return [(0, value, value), (1, value, value)]

    # Get coordinates
    if not np.iterable(values):
        raise TypeError('Colors must be iterable, got {values!r}.')
    if coords is not None:
        coords = np.atleast_1d(coords)
        if ratios is not None:
            _warn_proplot(
                f'Segment coordinates were provided, ignoring '
                f'ratios={ratios!r}.'
            )
        if len(coords) != len(values) or coords[0] != 0 or coords[-1] != 1:
            raise ValueError(
                f'Coordinates must range from 0 to 1, got {coords!r}.'
            )
    elif ratios is not None:
        coords = np.atleast_1d(ratios)
        if len(coords) != len(values) - 1:
            raise ValueError(
                f'Need {len(values)-1} ratios for {len(values)} colors, '
                f'but got {len(ratios)} ratios.'
            )
        coords = np.concatenate(([0], np.cumsum(coords)))
        coords = coords / np.max(coords)  # normalize to 0-1
    else:
        coords = np.linspace(0, 1, len(values))

    # Build segmentdata array
    array = []
    for c, value in zip(coords, values):
        array.append((c, value, value))
    return array


def make_mapping_array(N, data, gamma=1.0, inverse=False):
    r"""
    Similar to `~matplotlib.colors.makeMappingArray` but permits
    *circular* hue gradations along 0-360, disables clipping of
    out-of-bounds channel values, and uses fancier "gamma" scaling.

    Parameters
    ----------
    N : int
        Number of points in the colormap lookup table.
    data : 2D array-like
        List of :math:`(x, y_0, y_1)` tuples specifying the channel jump (from
        :math:`y_0` to :math:`y_1`) and the :math:`x` coordinate of that
        transition (ranges between 0 and 1).
        See `~matplotlib.colors.LinearSegmentedColormap` for details.
    gamma : float or list of float, optional
        To obtain channel values between coordinates :math:`x_i` and
        :math:`x_{i+1}` in rows :math:`i` and :math:`i+1` of `data`,
        we use the formula:

        .. math::

            y = y_{1,i} + w_i^{\gamma_i}*(y_{0,i+1} - y_{1,i})

        where :math:`\gamma_i` corresponds to `gamma` and the weight
        :math:`w_i` ranges from 0 to 1 between rows ``i`` and ``i+1``.
        If `gamma` is float, it applies to every transition. Otherwise,
        its length must equal ``data.shape[0]-1``.

        This is like the `gamma` used with matplotlib's
        `~matplotlib.colors.makeMappingArray`, except it controls the
        weighting for transitions *between* each segment data coordinate rather
        than the coordinates themselves. This makes more sense for
        `PerceptuallyUniformColormap`\\ s because they usually consist of just
        one linear transition for *sequential* colormaps and two linear
        transitions for *diverging* colormaps -- and in the latter case, it
        is often desirable to modify both "halves" of the colormap in the
        same way.
    inverse : bool, optional
        If ``True``, :math:`w_i^{\gamma_i}` is replaced with
        :math:`1 - (1 - w_i)^{\gamma_i}` -- that is, when `gamma` is greater
        than 1, this weights colors toward *higher* channel values instead
        of lower channel values.

        This is implemented in case we want to apply *equal* "gamma scaling"
        to different HSL channels in different directions. Usually, this
        is done to weight low data values with higher luminance *and* lower
        saturation, thereby emphasizing "extreme" data values with stronger
        colors.
    """
    # Allow for *callable* instead of linearly interpolating between segments
    gammas = np.atleast_1d(gamma)
    if (gammas < 0.01).any() or (gammas > 10).any():
        raise ValueError('Gamma can only be in range [0.01,10].')
    if callable(data):
        if len(gammas) > 1:
            raise ValueError(
                'Only one gamma allowed for functional segmentdata.')
        x = np.linspace(0, 1, N)**gamma
        lut = np.array(data(x), dtype=float)
        return lut

    # Get array
    data = np.array(data)
    shape = data.shape
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError('Data must be nx3 format.')
    if len(gammas) != 1 and len(gammas) != shape[0] - 1:
        raise ValueError(
            f'Need {shape[0]-1} gammas for {shape[0]}-level mapping array, '
            f'but got {len(gamma)}.'
        )
    if len(gammas) == 1:
        gammas = np.repeat(gammas, shape[:1])

    # Get indices
    x = data[:, 0]
    y0 = data[:, 1]
    y1 = data[:, 2]
    if x[0] != 0.0 or x[-1] != 1.0:
        raise ValueError(
            'Data mapping points must start with x=0 and end with x=1.'
        )
    if (np.diff(x) < 0).any():
        raise ValueError(
            'Data mapping points must have x in increasing order.'
        )
    x = x * (N - 1)

    # Get distances from the segmentdata entry to the *left* for each requested
    # level, excluding ends at (0,1), which must exactly match segmentdata ends
    xq = (N - 1) * np.linspace(0, 1, N)
    # where xq[i] must be inserted so it is larger than x[ind[i]-1] but
    # smaller than x[ind[i]]
    ind = np.searchsorted(x, xq)[1:-1]
    distance = (xq[1:-1] - x[ind - 1]) / (x[ind] - x[ind - 1])

    # Scale distances in each segment by input gamma
    # The ui are starting-points, the ci are counts from that point
    # over which segment applies (i.e. where to apply the gamma), the relevant
    # 'segment' is to the *left* of index returned by searchsorted
    _, uind, cind = np.unique(ind, return_index=True, return_counts=True)
    for ui, ci in zip(uind, cind):  # length should be N-1
        # the relevant segment is to *left* of this number
        gamma = gammas[ind[ui] - 1]
        if gamma == 1:
            continue
        ireverse = False
        if ci > 1:  # i.e. more than 1 color in this 'segment'
            # by default want to weight toward a *lower* channel value
            ireverse = ((y0[ind[ui]] - y1[ind[ui] - 1]) < 0)
        if inverse:
            ireverse = (not ireverse)
        if ireverse:
            distance[ui:ui + ci] = 1 - (1 - distance[ui:ui + ci])**gamma
        else:
            distance[ui:ui + ci] **= gamma

    # Perform successive linear interpolations all rolled up into one equation
    lut = np.zeros((N,), float)
    lut[1:-1] = distance * (y0[ind] - y1[ind - 1]) + y1[ind - 1]
    lut[0] = y1[0]
    lut[-1] = y0[-1]
    return lut


class _Colormap(object):
    """Mixin class used to add some helper methods."""
    def _get_data(self, ext):
        """
        Return a string containing the colormap colors for saving.

        Parameters
        ----------
        ext : {'hex', 'txt', 'rgb', 'rgba'}
            The filename extension.
        colors : list of color-spec
            The colors.
        """
        # Get lookup table colors and filter out bad ones
        if not self._isinit:
            self._init()
        colors = self._lut[:-3, :]
        # Get data string
        if ext == 'hex':
            data = ', '.join(mcolors.to_hex(color) for color in colors)
        elif ext in ('txt', 'rgb', 'rgba'):
            rgb = mcolors.to_rgba if ext == 'rgba' else mcolors.to_rgb
            data = [rgb(color) for color in colors]
            data = '\n'.join(','.join(str(num) for num in line)
                             for line in data)
        else:
            raise ValueError(
                f'Invalid extension {ext!r}. Options are: '
                "'hex', 'txt', 'rgb', 'rgba'."
            )
        return data

    def _parse_path(self, path, dirname='.', ext=''):
        """
        Parse the user input path.

        Parameters
        ----------
        dirname : str, optional
            The default directory.
        ext : str, optional
            The default extension.
        """
        path = os.path.expanduser(path or '')
        dirname = os.path.expanduser(dirname or '')
        if not path or os.path.isdir(path):
            path = os.path.join(path or dirname, self.name)  # default name
        dirname, basename = os.path.split(path)  # default to current directory
        path = os.path.join(dirname or '.', basename)
        if not os.path.splitext(path)[-1]:
            path = path + '.' + ext  # default file extension
        return path


class LinearSegmentedColormap(mcolors.LinearSegmentedColormap, _Colormap):
    r"""
    New base class for all `~matplotlib.colors.LinearSegmentedColormap`\ s.
    """
    def __str__(self):
        return type(self).__name__ + f'(name={self.name!r})'

    def __repr__(self):
        string = f" 'name': {self.name!r},\n"
        if hasattr(self, '_space'):
            string += f" 'space': {self._space!r},\n"
        if hasattr(self, '_cyclic'):
            string += f" 'cyclic': {self._cyclic!r},\n"
        for key, data in self._segmentdata.items():
            if callable(data):
                string += f' {key!r}: <function>,\n'
            else:
                string += (f' {key!r}: [{data[0][2]:.3f}, '
                           f'..., {data[-1][1]:.3f}],\n')
        return type(self).__name__ + '({\n' + string + '})'

    def __init__(self, *args, cyclic=False, alpha=None, **kwargs):
        """
        Parameters
        ----------
        cyclic : bool, optional
            Whether the colormap is cyclic. If ``True``, this changes how the
            leftmost and rightmost color levels are selected, and `extend` can
            only be ``'neither'`` (a warning will be issued otherwise).
        alpha : float, optional
            The opacity for the entire colormap. Overrides the input
            segment data.
        *args, **kwargs
            Passed to `~matplotlib.colors.LinearSegmentedColormap`.
        """
        super().__init__(*args, **kwargs)
        self._cyclic = cyclic
        if alpha is not None:
            self.set_alpha(alpha)

    def _resample(self, N):
        """Returns a resampled copy of the colormap."""
        return self.updated(self, N=N)

    def concatenate(self, *args, ratios=1, name=None, **kwargs):
        """
        Return the concatenation of this colormap with the
        input colormaps.

        Parameters
        ----------
        *args
            Instances of `LinearSegmentedColormap`.
        ratios : list of float, optional
            Relative extent of each component colormap in the merged colormap.
            Length must equal ``len(args) + 1``.

            For example, ``cmap1.concatenate(cmap2, ratios=[2,1])`` generates
            a colormap with the left two-thrids containing colors from
            ``cmap1`` and the right one-third containing colors from ``cmap2``.
        name : str, optional
            The colormap name. Default is
            ``'_'.join(cmap.name for cmap in args)``.
        N : int, optional
            Number of points in the colormap lookup table.
            Default is :rc:`image.lut` times ``len(args)``.
        **kwargs
            Passed to `LinearSegmentedColormap.updated`
            or `PerceptuallyUniformColormap.updated`.

        Returns
        -------
        `LinearSegmentedColormap`
            The colormap.
        """
        # Try making a simple copy
        if not args:
            raise ValueError(
                f'Got zero positional args, you must provide at least one.'
            )
        if not all(isinstance(cmap, type(self)) for cmap in args):
            raise ValueError(
                f'Colormaps {cmap.name + ": " + repr(cmap) for cmap in args} '
                f'must all belong to the same class.'
            )
        cmaps = (self, *args)
        spaces = {cmap.name: getattr(cmap, '_space', None) for cmap in cmaps}
        if len({*spaces.values(), }) > 1:
            raise ValueError(
                'Cannot merge PerceptuallyUniformColormaps that use '
                'different colorspaces: '
                + ', '.join(map(repr, spaces)) + '.'
            )
        N = kwargs.pop('N', None)
        N = N or len(cmaps) * rcParams['image.lut']
        if name is None:
            name = '_'.join(cmap.name for cmap in cmaps)

        # Combine the segmentdata, and use the y1/y2 slots at merge points so
        # we never interpolate between end colors of different colormaps
        segmentdata = {}
        ratios = ratios or 1
        if isinstance(ratios, Number):
            ratios = [1] * len(cmaps)
        # so if 4 cmaps, will be 1/4
        ratios = np.array(ratios) / np.sum(ratios)
        x0 = np.concatenate([[0], np.cumsum(ratios)])  # coordinates for edges
        xw = x0[1:] - x0[:-1]  # widths between edges
        for key in self._segmentdata.keys():
            # Callable segments
            # WARNING: If just reference a global 'funcs' list from inside the
            # 'data' function it can get overwritten in this loop. Must
            # embed 'funcs' into the definition using a keyword argument.
            callable_ = [callable(cmap._segmentdata[key]) for cmap in cmaps]
            if all(callable_):  # expand range from x-to-w to 0-1
                funcs = [cmap._segmentdata[key] for cmap in cmaps]

                def xyy(ix, funcs=funcs):
                    ix = np.atleast_1d(ix)
                    kx = np.empty(ix.shape)
                    for j, jx in enumerate(ix.flat):
                        idx = max(np.searchsorted(x0, jx) - 1, 0)
                        kx.flat[j] = funcs[idx]((jx - x0[idx]) / xw[idx])
                    return kx
            # Concatenate segment arrays and make the transition at the
            # seam instant so we *never interpolate* between end colors
            # of different maps.
            elif not any(callable_):
                datas = []
                for x, w, cmap in zip(x0[:-1], xw, cmaps):
                    xyy = np.array(cmap._segmentdata[key])
                    xyy[:, 0] = x + w * xyy[:, 0]
                    datas.append(xyy)
                for i in range(len(datas) - 1):
                    datas[i][-1, 2] = datas[i + 1][0, 2]
                    datas[i + 1] = datas[i + 1][1:, :]
                xyy = np.concatenate(datas, axis=0)
                xyy[:, 0] = xyy[:, 0] / xyy[:, 0].max(axis=0)  # fix fp errors
            else:
                raise ValueError(
                    'Mixed callable and non-callable colormap values.'
                )
            segmentdata[key] = xyy
            # Handle gamma values
            if key == 'saturation':
                ikey = 'gamma1'
            elif key == 'luminance':
                ikey = 'gamma2'
            else:
                continue
            if ikey in kwargs:
                continue
            gamma = []
            for cmap in cmaps:
                igamma = getattr(cmap, '_' + ikey)
                if not np.iterable(igamma):
                    if all(callable_):
                        igamma = [igamma]
                    else:
                        igamma = (len(cmap._segmentdata[key]) - 1) * [igamma]
                gamma.extend(igamma)
            if all(callable_):
                if any(igamma != gamma[0] for igamma in gamma[1:]):
                    _warn_proplot(
                        'Cannot use multiple segment gammas when '
                        'concatenating callable segments. Using the first '
                        f'gamma of {gamma[0]}.'
                    )
                gamma = gamma[0]
            kwargs[ikey] = gamma

        # Return copy
        return self.updated(name=name, segmentdata=segmentdata, **kwargs)

    def punched(self, cut=None, name=None, **kwargs):
        """
        Return a version of the colormap with the center "punched out".
        This is great for making the transition from "negative" to "positive"
        in a diverging colormap more distinct.

        Parameters
        ----------
        cut : float, optional
            The proportion to cut from the center of the colormap.
            For example, ``center=0.1`` cuts the central 10%%.
        name : str, optional
            The name of the new colormap. Default is
            ``self.name + '_punched'``.
        **kwargs
            Passed to `LinearSegmentedColormap.updated`
            or `PerceptuallyUniformColormap.updated`.

        Returns
        -------
        `LinearSegmentedColormap`
            The colormap.
        """
        cut = _notNone(cut, 0)
        if cut == 0:
            return self
        if name is None:
            name = self.name + '_punched'

        # Decompose cut into two truncations followed by concatenation
        left_center = 0.5 - cut / 2
        right_center = 0.5 + cut / 2
        cmap_left = self.truncated(0, left_center)
        cmap_right = self.truncated(right_center, 1)
        return cmap_left.concatenate(cmap_right, name=name)

    def reversed(self, name=None, **kwargs):
        """
        Return a reversed copy of the colormap, as in
        `~matplotlib.colors.LinearSegmentedColormap`.

        Parameters
        ----------
        name : str, optional
            The new colormap name. Default is ``self.name + '_r'``.
        **kwargs
            Passed to `LinearSegmentedColormap.updated`
            or `PerceptuallyUniformColormap.updated`.
        """
        if name is None:
            name = self.name + '_r'

        def factory(dat):
            def func_r(x):
                return dat(1.0 - x)
            return func_r
        segmentdata = {key:
                       factory(data) if callable(data) else
                       [(1.0 - x, y1, y0) for x, y0, y1 in reversed(data)]
                       for key, data in self._segmentdata.items()}
        for key in ('gamma1', 'gamma2'):
            if key in kwargs:
                continue
            gamma = getattr(self, '_' + key, None)
            if gamma is not None and np.iterable(gamma):
                kwargs[key] = gamma[::-1]
        return self.updated(name, segmentdata, **kwargs)

    def save(self, path=None):
        """
        Save the colormap data to a file.

        Parameters
        ----------
        path : str, optional
            The output filename. If not provided, the colormap
            is saved under ``~/.proplot/cmaps/name.json`` where ``name``
            is the colormap name. Valid extensions are described in
            the below table.

            =====================  ==========================================================
            Extension              Description
            =====================  ==========================================================
            ``.json`` (default)    JSON database of the channel segment data.
            ``.hex``               Comma-delimited list of HEX strings.
            ``.rgb``, ``.txt``     3-column table of comma-delimited RGB values.
            ``.rgba``              As with ``.rgb``, but with an opacity (or "alpha") column.
            =====================  ==========================================================
        """  # noqa
        dirname = os.path.join('~', '.proplot', 'cmaps')
        filename = self._parse_path(path, dirname, 'json')
        # Save channel segment data in json file
        _, ext = os.path.splitext(filename)
        if ext[1:] == 'json':
            # Sanitize segmentdata values
            # Convert np.float to builtin float, np.array to list of lists,
            # and callable to list of lists. We tried encoding func.__code__
            # with base64 and marshal instead, but when cmap.concatenate()
            # embeds functions as keyword arguments, this seems to make it
            # *impossible* to load back up the function with FunctionType
            # (error message: arg 5 (closure) must be tuple). Instead use
            # this brute force workaround.
            data = {}
            for key, value in self._segmentdata.items():
                if callable(value):
                    x = np.linspace(0, 1, 256)  # just save the transitions
                    y = np.array([value(_) for _ in x]).squeeze()
                    value = np.vstack((x, y, y)).T
                data[key] = np.asarray(value).astype(float).tolist()
            # Add critical attributes to the dictionary
            keys = ()
            if isinstance(self, PerceptuallyUniformColormap):
                keys = ('cyclic', 'gamma1', 'gamma2', 'space')
            elif isinstance(self, LinearSegmentedColormap):
                keys = ('cyclic', 'gamma')
            for key in keys:
                data[key] = getattr(self, '_' + key)
            with open(filename, 'w') as file:
                json.dump(data, file, indent=4)
        # Save lookup table colors
        else:
            data = self._get_data(ext[1:])
            with open(filename, 'w') as f:
                f.write(data)
        print(f'Saved colormap to {filename!r}.')

    def set_alpha(self, alpha):
        """
        Set the opacity for the entire colormap.

        Parameters
        ----------
        alpha : float
            The opacity.
        """
        self._segmentdata['alpha'] = [(0, alpha, alpha), (1, alpha, alpha)]
        self._isinit = False

    def set_cyclic(self, b):
        """
        Set whether this colormap is "cyclic". See `LinearSegmentedColormap`
        for details.
        """
        self._cyclic = bool(b)
        self._isinit = False

    def shifted(self, shift=None, name=None, **kwargs):
        """
        Return a cyclicaly shifted version of the colormap. If the colormap
        cyclic property is set to ``False`` a warning will be raised.

        Parameters
        ----------
        shift : float, optional
            The number of degrees to shift, out of 360 degrees. If ``None``,
            the original colormap is returned.
        name : str, optional
            The name of the new colormap. Default is
            ``self.name + '_shifted'``.
        **kwargs
            Passed to `LinearSegmentedColormap.updated`
            or `PerceptuallyUniformColormap.updated`.
        """
        shift = ((shift or 0) / 360) % 1
        if shift == 0:
            return self
        if name is None:
            name = self.name + '_shifted'
        if not self._cyclic:
            _warn_proplot(
                f'Shifting non-cyclic colormap {self.name!r}. '
                f'Use cmap.set_cyclic(True) or Colormap(..., cyclic=True) to '
                'suppress this warning.'
            )
            self._cyclic = True

        # Decompose shift into two truncations followed by concatenation
        cmap_left = self.truncated(shift, 1)
        cmap_right = self.truncated(0, shift)
        return cmap_left.concatenate(
            cmap_right, ratios=(1 - shift, shift), name=name)

    def truncated(self, left=None, right=None, name=None, **kwargs):
        """
        Return a truncated version of the colormap.

        Parameters
        ----------
        left : float, optional
            The colormap index for the new "leftmost" color. Must fall between
            ``0`` and ``1``. For example,
            ``left=0.1`` cuts the leftmost 10%% of the colors.
        right : float, optional
            The colormap index for the new "rightmost" color. Must fall between
            ``0`` and ``1``. For example,
            ``right=0.9`` cuts the leftmost 10%% of the colors.
        name : str, optional
            The name of the new colormap. Default is
            ``self.name + '_truncated'``.
        **kwargs
            Passed to `LinearSegmentedColormap.updated`
            or `PerceptuallyUniformColormap.updated`.
        """
        # Bail out
        left = max(_notNone(left, 0), 0)
        right = min(_notNone(right, 1), 1)
        if left == 0 and right == 1:
            return self
        if name is None:
            name = self.name + '_truncated'

        # Resample the segmentdata arrays
        segmentdata = {}
        for key, xyy in self._segmentdata.items():
            # Callable array
            # WARNING: If just reference a global 'xyy' callable from inside
            # the lambda function it gets overwritten in the loop! Must embed
            # the old callable in the new one as a default keyword arg.
            if callable(xyy):
                def xyy(x, func=xyy):
                    return func(left + x * (right - left))
            # Slice
            # l is the first point where x > 0 or x > left, should be >= 1
            # r is the last point where r < 1 or r < right
            else:
                xyy = np.asarray(xyy)
                x = xyy[:, 0]
                l = np.searchsorted(x, left)  # first x value > left  # noqa
                r = np.searchsorted(x, right) - 1  # last x value < right
                xc = xyy[l:r + 1, :].copy()
                xl = xyy[l - 1, 1:] + (left - x[l - 1]) * (
                    (xyy[l, 1:] - xyy[l - 1, 1:]) / (x[l] - x[l - 1])
                )
                xr = xyy[r, 1:] + (right - x[r]) * (
                    (xyy[r + 1, 1:] - xyy[r, 1:]) / (x[r + 1] - x[r])
                )
                xyy = np.vstack(((left, *xl), xc, (right, *xr)))
                xyy[:, 0] = (xyy[:, 0] - left) / (right - left)
            segmentdata[key] = xyy
            # Retain the corresponding gamma *segments*
            if key == 'saturation':
                ikey = 'gamma1'
            elif key == 'luminance':
                ikey = 'gamma2'
            else:
                continue
            if ikey in kwargs:
                continue
            gamma = getattr(self, '_' + ikey)
            if np.iterable(gamma):
                if callable(xyy):
                    if any(igamma != gamma[0] for igamma in gamma[1:]):
                        _warn_proplot(
                            'Cannot use multiple segment gammas when '
                            'truncating colormap. Using the first gamma '
                            f'of {gamma[0]}.'
                        )
                    gamma = gamma[0]
                else:
                    gamma = gamma[l - 1:r + 1]
            kwargs[ikey] = gamma
        return self.updated(name, segmentdata, **kwargs)

    def updated(
        self, name=None, segmentdata=None, N=None, *,
        alpha=None, gamma=None, cyclic=None
    ):
        """
        Return a new colormap, with relevant properties copied from this one
        if they were not provided as keyword arguments.

        Parameters
        ----------
        name : str
            The colormap name. Default is ``self.name + '_updated'``.
        segmentdata, N, alpha, gamma, cyclic : optional
            See `LinearSegmentedColormap`. If not provided,
            these are copied from the current colormap.
        """
        if name is None:
            name = self.name + '_updated'
        if segmentdata is None:
            segmentdata = self._segmentdata
        if gamma is None:
            gamma = self._gamma
        if cyclic is None:
            cyclic = self._cyclic
        if N is None:
            N = self.N
        cmap = LinearSegmentedColormap(name, segmentdata, N,
                                       alpha=alpha, gamma=gamma, cyclic=cyclic)
        cmap._rgba_bad = self._rgba_bad
        cmap._rgba_under = self._rgba_under
        cmap._rgba_over = self._rgba_over
        return cmap

    @staticmethod
    def from_file(path, warn_on_failure=False):
        """
        Load colormap from a file.
        Valid file extensions are described in the below table.

        =====================  =============================================================================================================================================================================================================
        Extension              Description
        =====================  =============================================================================================================================================================================================================
        ``.hex``               List of HEX strings in any format (comma-separated, separate lines, with double quotes... anything goes).'ColorBlind10':
        ``.xml``               XML files with ``<Point .../>`` entries specifying ``x``, ``r``, ``g``, ``b``, and optionally, ``a`` values, where ``x`` is the colormap coordinate and the rest are the RGB and opacity (or "alpha") values.
        ``.rgb``               3-column table delimited by commas or consecutive spaces, each column indicating red, blue and green color values.
        ``.xrgb``              As with ``.rgb``, but with 4 columns. The first column indicates the colormap coordinate.
        ``.rgba``, ``.xrgba``  As with ``.rgb``, ``.xrgb``, but with a trailing opacity (or "alpha") column.
        =====================  =============================================================================================================================================================================================================

        Parameters
        ----------
        path : str
            The file path.
        warn_on_failure : bool, optional
            If ``True``, issue a warning when loading fails rather than
            raising an error.
        """  # noqa
        return _from_file(path, listed=False, warn_on_failure=warn_on_failure)

    @staticmethod
    def from_list(name, colors, ratios=None, **kwargs):
        """
        Make a `LinearSegmentedColormap` from a list of colors.

        Parameters
        ----------
        name : str
            The colormap name.
        colors : list of color-spec or (float, color-spec) tuples, optional
            If list of RGB[A] tuples or color strings, the colormap transitions
            evenly from ``colors[0]`` at the left-hand side to
            ``colors[-1]`` at the right-hand side.

            If list of (float, color-spec) tuples, the float values are the
            coordinate of each transition and must range from 0 to 1. This
            can be used to divide  the colormap range unevenly.
        ratios : list of float, optional
            Relative extents of each color transition. Must have length
            ``len(colors) - 1``. Larger numbers indicate a slower
            transition, smaller numbers indicate a faster transition.

        Other parameters
        ----------------
        **kwargs
            Passed to `LinearSegmentedColormap`.

        Returns
        -------
        `LinearSegmentedColormap`
            The colormap.
        """
        # Get coordinates
        coords = None
        if not np.iterable(colors):
            raise ValueError(f'Colors must be iterable, got colors={colors!r}')
        if (np.iterable(colors[0]) and len(colors[0]) == 2
                and not isinstance(colors[0], str)):
            coords, colors = zip(*colors)
        colors = [to_rgb(color, alpha=True) for color in colors]

        # Build segmentdata
        keys = ('red', 'green', 'blue', 'alpha')
        cdict = {}
        for key, values in zip(keys, zip(*colors)):
            cdict[key] = _make_segmentdata_array(values, coords, ratios)
        return LinearSegmentedColormap(name, cdict, **kwargs)


class ListedColormap(mcolors.ListedColormap, _Colormap):
    r"""
    New base class for all `~matplotlib.colors.ListedColormap`\ s.
    """
    def __str__(self):
        return f'ListedColormap(name={self.name!r})'

    def __repr__(self):
        return (
            'ListedColormap({\n'
            f" 'name': {self.name!r},\n"
            f" 'colors': {[mcolors.to_hex(color) for color in self.colors]},\n"
            '})')

    def __init__(self, *args, alpha=None, **kwargs):
        """
        Parameters
        ----------
        alpha : float, optional
            The opacity for the entire colormap. Overrides the input
            colors.
        *args, **kwargs
            Passed to `~matplotlib.colors.ListedColormap`.
        """
        super().__init__(*args, **kwargs)
        if alpha is not None:
            self.set_alpha(alpha)

    def concatenate(self, *args, name=None, N=None, **kwargs):
        """
        Append arbitrary colormaps onto this colormap.

        Parameters
        ----------
        *args
            Instances of `ListedColormap`.
        name : str, optional
            The colormap name. Default is
            ``'_'.join(cmap.name for cmap in args)``.
        N : int, optional
            The number of colors in the colormap lookup table. Default is
            the number of colors in the concatenated lists.
        """
        if not args:
            raise ValueError(
                f'Got zero positional args, you must provide at least one.'
            )
        if not all(isinstance(cmap, type(self)) for cmap in args):
            raise ValueError(
                f'Input arguments {args} must all be ListedColormap.'
            )
        cmaps = (self, *args)
        if name is None:
            name = '_'.join(cmap.name for cmap in cmaps)
        colors = [color for cmap in cmaps for color in cmap.colors]
        return self.updated(colors, name, N or len(colors))

    def save(self, path=None):
        """
        Save the colormap data to a file.

        Parameters
        ----------
        path : str, optional
            The output filename. If not provided, the colormap
            is saved under ``~/.proplot/cycles/name.hex`` where ``name``
            is the colormap name. Valid extensions are described in
            the below table.

            =====================  ==========================================================
            Extension              Description
            =====================  ==========================================================
            ``.hex`` (default)     Comma-delimited list of HEX strings.
            ``.rgb``, ``.txt``     3-column table of comma-delimited RGB values.
            ``.rgba``              As with ``.rgb``, but with an opacity (or "alpha") column.
            =====================  ==========================================================
        """  # noqa
        dirname = os.path.join('~', '.proplot', 'cmaps')
        filename = self._parse_path(path, dirname, 'hex')
        # Save lookup table colors
        _, ext = os.path.splitext(filename)
        data = self._get_data(ext[1:])
        with open(filename, 'w') as f:
            f.write(data)
        print(f'Saved colormap to {filename!r}.')

    def set_alpha(self, alpha):
        """
        Set the opacity for the entire colormap.

        Parameters
        ----------
        alpha : float
            The opacity.
        """
        colors = [list(mcolors.to_rgba(color)) for color in self.colors]
        for color in colors:
            color[3] = alpha
        self.colors = colors
        self._init()

    def shifted(self, shift=None, name=None):
        """
        Return a cyclically shifted version of the colormap.

        Parameters
        ----------
        shift : float, optional
            The number of places to shift, between ``-self.N`` and ``self.N``.
            If ``None``, the original colormap is returned.
        name : str, optional
            The new colormap name. Default is ``self.name + '_shifted'``.
        """
        if not shift:
            return self
        if name is None:
            name = self.name + '_shifted'
        shift = shift % len(self.colors)
        colors = [*self.colors]  # ensure list
        colors = colors[shift:] + colors[:shift]
        return self.updated(colors, name, len(colors))

    def truncated(self, left=None, right=None, name=None):
        """
        Return a truncated version of the colormap.

        Parameters
        ----------
        left : float, optional
            The colormap index for the new "leftmost" color. Must fall between
            ``0`` and ``self.N``. For example,
            ``left=2`` deletes the two first colors.
        right : float, optional
            The colormap index for the new "rightmost" color. Must fall between
            ``0`` and ``self.N``. For example,
            ``right=4`` deletes colors after the fourth color.
        name : str, optional
            The new colormap name. Default is ``self.name + '_truncated'``.
        """
        if left is None and right is None:
            return self
        if name is None:
            name = self.name + '_truncated'
        colors = self.colors[left:right]
        return self.updated(colors, name, len(colors))

    def updated(self, colors=None, name=None, N=None, *, alpha=None):
        """
        Return a new colormap with relevant properties copied from this one
        if they were not provided as keyword arguments.

        Parameters
        ----------
        name : str
            The colormap name. Default is ``self.name + '_updated'``.
        colors, N, alpha : optional
            See `ListedColormap`. If not provided,
            these are copied from the current colormap.
        """
        if name is None:
            name = self.name + '_updated'
        if colors is None:
            colors = self.colors
        if N is None:
            N = self.N
        cmap = ListedColormap(colors, name, N, alpha=alpha)
        cmap._rgba_bad = self._rgba_bad
        cmap._rgba_under = self._rgba_under
        cmap._rgba_over = self._rgba_over
        return cmap

    @staticmethod
    def from_file(path, warn_on_failure=False):
        """
        Load color cycle from a file.
        Valid file extensions are described in the below table.

        =====================  =============================================================================================================================================================================================================
        Extension              Description
        =====================  =============================================================================================================================================================================================================
        ``.hex``               List of HEX strings in any format (comma-separated, separate lines, with double quotes... anything goes).'ColorBlind10':
        ``.xml``               XML files with ``<Point .../>`` entries specifying ``x``, ``r``, ``g``, ``b``, and optionally, ``a`` values, where ``x`` is the colormap coordinate and the rest are the RGB and opacity (or "alpha") values.
        ``.rgb``               3-column table delimited by commas or consecutive spaces, each column indicating red, blue and green color values.
        ``.xrgb``              As with ``.rgb``, but with 4 columns. The first column indicates the colormap coordinate.
        ``.rgba``, ``.xrgba``  As with ``.rgb``, ``.xrgb``, but with a trailing opacity (or "alpha") column.
        =====================  =============================================================================================================================================================================================================

        Parameters
        ----------
        path : str
            The file path.
        warn_on_failure : bool, optional
            If ``True``, issue a warning when loading fails rather than
            raising an error.
        """  # noqa
        return _from_file(path, listed=True, warn_on_failure=warn_on_failure)


class PerceptuallyUniformColormap(LinearSegmentedColormap, _Colormap):
    """Similar to `~matplotlib.colors.LinearSegmentedColormap`, but instead
    of varying the RGB channels, we vary hue, saturation, and luminance in
    either the HCL colorspace or the HSL or HPL scalings of HCL."""
    def __init__(
        self, name, segmentdata, N=None, space=None, clip=True,
        gamma=None, gamma1=None, gamma2=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        name : str
            The colormap name.
        segmentdata : dict-like
            Mapping containing the keys ``'hue'``, ``'saturation'``, and
            ``'luminance'``. The key values can be callable functions that
            return channel values given a colormap index, or lists containing
            any of the following channel specifiers.

            1. Numbers, within the range 0-360 for hue and 0-100 for
               saturation and luminance.
            2. Color string names or hex tags, in which case the channel
               value for that color is looked up.

            See `~matplotlib.colors.LinearSegmentedColormap` for a more
            detailed explanation.
        N : int, optional
            Number of points in the colormap lookup table.
            Default is :rc:`image.lut`.
        space : {'hsl', 'hpl', 'hcl'}, optional
            The hue, saturation, luminance-style colorspace to use for
            interpreting the channels. See
            `this page <http://www.hsluv.org/comparison/>`__ for a description.
        clip : bool, optional
            Whether to "clip" impossible colors, i.e. truncate HCL colors
            with RGB channels with values >1, or mask them out as gray.
        cyclic : bool, optional
            Whether the colormap is cyclic. If ``True``, this changes how the
            leftmost and rightmost color levels are selected, and `extend` can
            only be ``'neither'`` (a warning will be issued otherwise).
        gamma : float, optional
            Sets `gamma1` and `gamma2` to this identical value.
        gamma1 : float, optional
            If >1, makes low saturation colors more prominent. If <1,
            makes high saturation colors more prominent. Similar to the
            `HCLWizard <http://hclwizard.org:64230/hclwizard/>`_ option.
            See `make_mapping_array` for details.
        gamma2 : float, optional
            If >1, makes high luminance colors more prominent. If <1,
            makes low luminance colors more prominent. Similar to the
            `HCLWizard <http://hclwizard.org:64230/hclwizard/>`_ option.
            See `make_mapping_array` for details.
        **kwargs
            Passed to `LinearSegmentedColormap`.

        Example
        -------
        The following generates a `PerceptuallyUniformColormap` from a
        `segmentdata` dictionary that uses color names for the hue data,
        instead of channel values between ``0`` and ``360``.

        >>> import proplot as plot
        >>> data = {
        ...     'hue': [[0, 'red', 'red'], [1, 'blue', 'blue']],
        ...     'saturation': [[0, 100, 100], [1, 100, 100]],
        ...     'luminance': [[0, 100, 100], [1, 20, 20]],
        ... }
        >>> cmap = plot.PerceptuallyUniformColormap(data)

        """
        # Checks
        space = _notNone(space, 'hsl').lower()
        if space not in ('rgb', 'hsv', 'hpl', 'hsl', 'hcl'):
            raise ValueError(f'Unknown colorspace {space!r}.')
        keys = {*segmentdata.keys()}
        target = {'hue', 'saturation', 'luminance', 'alpha'}
        if not keys <= target:
            raise ValueError(
                f'Invalid segmentdata dictionary with keys {keys!r}.'
            )
        # Convert color strings to channel values
        for key, array in segmentdata.items():
            if callable(array):  # permit callable
                continue
            for i, xyy in enumerate(array):
                xyy = list(xyy)  # make copy!
                for j, y in enumerate(xyy[1:]):  # modify the y values
                    xyy[j + 1] = _get_channel(y, key, space)
                segmentdata[key][i] = xyy
        # Initialize
        N = N or rcParams['image.lut']
        super().__init__(name, segmentdata, N, gamma=1.0, **kwargs)
        # Custom properties
        self._gamma1 = _notNone(gamma1, gamma, 1.0)
        self._gamma2 = _notNone(gamma2, gamma, 1.0)
        self._space = space
        self._clip = clip

    def _init(self):
        """As with `~matplotlib.colors.LinearSegmentedColormap`, but convert
        each value in the lookup table from ``self._space`` to RGB."""
        # First generate the lookup table
        channels = ('hue', 'saturation', 'luminance')
        # gamma weights *low chroma* and *high luminance*
        inverses = (False, False, True)
        gammas = (1.0, self._gamma1, self._gamma2)
        self._lut_hsl = np.ones((self.N + 3, 4), float)  # fill
        for i, (channel, gamma, inverse) in enumerate(
                zip(channels, gammas, inverses)):
            self._lut_hsl[:-3, i] = make_mapping_array(
                self.N, self._segmentdata[channel], gamma, inverse)
        if 'alpha' in self._segmentdata:
            self._lut_hsl[:-3, 3] = make_mapping_array(
                self.N, self._segmentdata['alpha'])
        self._lut_hsl[:-3, 0] %= 360
        # Make hues circular, set extremes i.e. copy HSL values
        self._lut = self._lut_hsl.copy()
        self._set_extremes()  # generally just used end values in segmentdata
        self._isinit = True
        # Now convert values to RGB and clip colors
        for i in range(self.N + 3):
            self._lut[i, :3] = to_rgb(self._lut[i, :3], self._space)
        self._lut[:, :3] = _clip_colors(self._lut[:, :3], self._clip)

    def _resample(self, N):
        """Return a new colormap with *N* entries."""
        return self.updated(N=N)

    @staticmethod
    def from_color(name, color, fade=None, space='hsl', **kwargs):
        """
        Return a monochromatic "sequential" colormap that blends from white
        or near-white to the input color.

        Parameters
        ----------
        name : str, optional
            The colormap name.
        color : color-spec
            RGB tuple, hex string, or named color string.
        fade : float or color-spec, optional
            If float, this is the luminance channel strength on the left-hand
            side of the colormap (default is ``100``), and the saturation
            channel is held constant throughout the colormap.

            If RGB tuple, hex string, or named color string, the luminance and
            saturation (but *not* the hue) from this color are used for the
            left-hand side of the colormap.
        space : {'hsl', 'hpl', 'hcl'}, optional
            The colorspace in which the luminance is varied.

        Other parameters
        ----------------
        **kwargs
            Passed to `PerceptuallyUniformColormap.from_hsl`.

        Returns
        -------
        `PerceptuallyUniformColormap`
            The colormap.
        """
        hue, saturation, luminance, alpha = to_xyz(color, space, alpha=True)
        if fade is None:
            fade = 100
        if isinstance(fade, Number):
            saturation_fade, luminance_fade = saturation, fade
        else:
            _, saturation_fade, luminance_fade = to_xyz(fade, space)
        return PerceptuallyUniformColormap.from_hsl(
            name, hue=hue, alpha=alpha, space=space,
            saturation=(saturation_fade, saturation),
            luminance=(luminance_fade, luminance),
            **kwargs)

    @staticmethod
    def from_hsl(
        name, hue=0, saturation=100, luminance=(100, 20), alpha=None,
        ratios=None, **kwargs
    ):
        """
        Make a `~PerceptuallyUniformColormap` by specifying the hue,
        saturation, and luminance transitions individually.

        Parameters
        ----------
        name : str, optional
            The colormap name.
        hue : float, str, or list thereof, optional
            Hue channel value or list of values. Values can be
            any of the following.

            1. Numbers, within the range 0-360 for hue and 0-100 for
               saturation and luminance.
            2. Color string names or hex strings, in which case the channel
               value for that color is looked up.

            If scalar, the hue does not change across the colormap.
        saturation, luminance, alpha : float, str, or list thereof, optional
            As with `hue`, but for the saturation, luminance, and alpha
            (opacity) channels, respectively.
        ratios : list of float, optional
            Relative extents of each color transition. Must have length
            ``len(colors) - 1``. Larger numbers indicate a slower
            transition, smaller numbers indicate a faster transition.

            For example, ``luminance=[100,50,0]`` with ``ratios=[2,1]``
            results in a colormap with the transition from luminance ``100``
            to ``50`` taking *twice as long* as the transition from luminance
            ``50`` to ``0``.

        Other parameters
        ----------------
        **kwargs
            Passed to `PerceptuallyUniformColormap`.

        Returns
        -------
        `PerceptuallyUniformColormap`
            The colormap.
        """
        cdict = {}
        alpha = _notNone(alpha, 1.0)
        for key, channel in zip(
            ('hue', 'saturation', 'luminance', 'alpha'),
            (hue, saturation, luminance, alpha)
        ):
            cdict[key] = _make_segmentdata_array(channel, ratios=ratios)
        return PerceptuallyUniformColormap(name, cdict, **kwargs)

    @staticmethod
    def from_list(name, colors, ratios=None, **kwargs):
        """
        Make a `PerceptuallyUniformColormap` from a list of colors.

        Parameters
        ----------
        name : str
            The colormap name.
        colors : list of color-spec or (float, color-spec) tuples, optional
            If list of RGB[A] tuples or color strings, the colormap transitions
            evenly from ``colors[0]`` at the left-hand side to
            ``colors[-1]`` at the right-hand side.

            If list of (float, color-spec) tuples, the float values are the
            coordinate of each transition and must range from 0 to 1. This
            can be used to divide  the colormap range unevenly.
        ratios : list of float, optional
            Relative extents of each color transition. Must have length
            ``len(colors) - 1``. Larger numbers indicate a slower
            transition, smaller numbers indicate a faster transition.

            For example, ``red=[1,0.5,0]`` with ``ratios=[2,1]``
            results in a colormap with the transition from red ``1``
            to ``0.5`` taking *twice as long* as the transition from red
            ``0.5`` to ``0``.

        Other parameters
        ----------------
        **kwargs
            Passed to `PerceptuallyUniformColormap`.

        Returns
        -------
        `PerceptuallyUniformColormap`
            The colormap.
        """
        # Get coordinates
        coords = None
        space = kwargs.get('space', 'hsl')  # use the builtin default
        if not np.iterable(colors):
            raise ValueError(f'Colors must be iterable, got colors={colors!r}')
        if (np.iterable(colors[0]) and len(colors[0]) == 2
                and not isinstance(colors[0], str)):
            coords, colors = zip(*colors)
        colors = [to_xyz(color, space, alpha=True) for color in colors]

        # Build segmentdata
        keys = ('hue', 'saturation', 'luminance', 'alpha')
        cdict = {}
        for key, values in zip(keys, zip(*colors)):
            cdict[key] = _make_segmentdata_array(values, coords, ratios)
        return PerceptuallyUniformColormap(name, cdict, **kwargs)

    def set_gamma(self, gamma=None, gamma1=None, gamma2=None):
        """
        Modify the gamma value(s) and refresh the lookup table.

        Parameters
        ----------
        gamma : float, optional
            Sets `gamma1` and `gamma2` to this identical value.
        gamma1 : float, optional
            If >1, makes low saturation colors more prominent. If <1,
            makes high saturation colors more prominent. Similar to the
            `HCLWizard <http://hclwizard.org:64230/hclwizard/>`_ option.
            See `make_mapping_array` for details.
        gamma2 : float, optional
            If >1, makes high luminance colors more prominent. If <1,
            makes low luminance colors more prominent. Similar to the
            `HCLWizard <http://hclwizard.org:64230/hclwizard/>`_ option.
            See `make_mapping_array` for details.
        """
        gamma1 = _notNone(gamma1, gamma)
        gamma2 = _notNone(gamma2, gamma)
        if gamma1 is not None:
            self._gamma1 = gamma1
        if gamma2 is not None:
            self._gamma2 = gamma2
        self._init()

    def updated(
        self, name=None, segmentdata=None, N=None, *,
        alpha=None, gamma=None, cyclic=None,
        clip=None, gamma1=None, gamma2=None, space=None
    ):
        """
        Return a new colormap with relevant properties copied from this one
        if they were not provided as keyword arguments.

        Parameters
        ----------
        name : str
            The colormap name. Default is ``self.name + '_updated'``.
        segmentdata, N, alpha, clip, cyclic, gamma, gamma1, gamma2, space : \
optional
            See `PerceptuallyUniformColormap`. If not provided,
            these are copied from the current colormap.
        """
        if name is None:
            name = self.name + '_updated'
        if segmentdata is None:
            segmentdata = self._segmentdata
        if space is None:
            space = self._space
        if clip is None:
            clip = self._clip
        if gamma is not None:
            gamma1 = gamma2 = gamma
        if gamma1 is None:
            gamma1 = self._gamma1
        if gamma2 is None:
            gamma2 = self._gamma2
        if cyclic is None:
            cyclic = self._cyclic
        if N is None:
            N = self.N
        cmap = PerceptuallyUniformColormap(
            name, segmentdata, N,
            alpha=alpha, clip=clip, cyclic=cyclic,
            gamma1=gamma1, gamma2=gamma2, space=space)
        cmap._rgba_bad = self._rgba_bad
        cmap._rgba_under = self._rgba_under
        cmap._rgba_over = self._rgba_over
        return cmap


class CmapDict(dict):
    """Dictionary subclass used to replace the `matplotlib.cm.cmap_d`
    colormap dictionary. See `~CmapDict.__getitem__` and
    `~CmapDict.__setitem__` for details."""
    def __init__(self, kwargs):
        """
        Parameters
        ----------
        kwargs : dict-like
            The source dictionary.
        """
        for key, value in kwargs.items():
            if not isinstance(key, str):
                raise KeyError(f'Invalid key {key}. Must be string.')
            self.__setitem__(key, value, sort=False)
        try:
            for record in (cmaps, cycles):
                record[:] = sorted(record)
        except NameError:
            pass

    def __delitem__(self, key):
        """Delete the item from the list records."""
        super().__delitem__(self, key)
        try:
            for record in (cmaps, cycles):
                try:
                    record.remove(key)
                except ValueError:
                    pass
        except NameError:
            pass

    def __getitem__(self, key):
        """Retrieve the colormap associated with the sanitized key name. The
        key name is case insensitive. If it ends in ``'_r'``, the result of
        ``cmap.reversed()`` is returned for the colormap registered under
        the name ``key[:-2]``. If it ends in ``'_shifted'``, the result of
        ``cmap.shifted(180)`` is returned for the colormap registered under
        the name ``cmap[:-8]``. Reversed diverging colormaps can be requested
        with their "reversed" name -- for example, ``'BuRd'`` is equivalent
        to ``'RdBu_r'``."""
        key = self._sanitize_key(key, mirror=True)
        shift = (key[-8:] == '_shifted')
        if shift:
            key = key[:-8]
        reverse = (key[-2:] == '_r')
        if reverse:
            key = key[:-2]
        value = super().__getitem__(key)  # may raise keyerror
        if shift:
            if hasattr(value, 'shifted'):
                value = value.shifted(180)
            else:
                raise KeyError(
                    f'Item of type {type(value).__name__!r} '
                    'does not have shifted() method.'
                )
        if reverse:
            if hasattr(value, 'reversed'):
                value = value.reversed()
            else:
                raise KeyError(
                    f'Item of type {type(value).__name__!r} '
                    'does not have reversed() method.'
                )
        return value

    def __setitem__(self, key, item, sort=True):
        """Store the colormap under its lowercase name. If the colormap is
        a matplotlib `~matplotlib.colors.ListedColormap` or
        `~matplotlib.colors.LinearSegmentedColormap`, it is converted to the
        ProPlot `ListedColormap` or `LinearSegmentedColormap` subclass."""
        if isinstance(item, (ListedColormap, LinearSegmentedColormap)):
            pass
        elif isinstance(item, mcolors.LinearSegmentedColormap):
            item = LinearSegmentedColormap(
                item.name, item._segmentdata, item.N, item._gamma)
        elif isinstance(item, mcolors.ListedColormap):
            item = ListedColormap(
                item.colors, item.name, item.N)
        elif item is None:
            return
        else:
            raise ValueError(
                f'Invalid colormap {item}. Must be instance of '
                'matplotlib.colors.ListedColormap or '
                'matplotlib.colors.LinearSegmentedColormap.'
            )
        key = self._sanitize_key(key, mirror=False)
        try:
            record = cycles if isinstance(item, ListedColormap) else cmaps
            record.append(key)
            if sort:
                record[:] = sorted(record)
        except NameError:
            pass
        return super().__setitem__(key, item)

    def __contains__(self, item):
        """Test for membership using the sanitized colormap name."""
        try:  # by default __contains__ ignores __getitem__ overrides
            self.__getitem__(item)
            return True
        except KeyError:
            return False

    def _sanitize_key(self, key, mirror=True):
        """Return the sanitized colormap name."""
        if not isinstance(key, str):
            raise KeyError(f'Invalid key {key!r}. Key must be a string.')
        key = key.lower()
        reverse = False
        if key[-2:] == '_r':
            key = key[:-2]
            reverse = True
        if mirror and not super().__contains__(key):  # search for mirrored key
            key_mirror = key
            for pair in CMAPS_DIVERGING:
                try:
                    idx = pair.index(key)
                    key_mirror = pair[1 - idx]
                except (ValueError, KeyError):
                    continue
            if super().__contains__(key_mirror):
                reverse = (not reverse)
                key = key_mirror
        if reverse:
            key = key + '_r'
        return key

    def get(self, key, *args):
        """Retrieve the sanitized colormap name."""
        key = self._sanitize_key(key, mirror=True)
        return super().get(key, *args)

    def pop(self, key, *args):
        """Pop the sanitized colormap name."""
        key = self._sanitize_key(key, mirror=True)
        try:
            for record in (cmaps, cycles):
                try:
                    record.remove(key)
                except ValueError:
                    pass
        except NameError:
            pass
        return super().pop(key, *args)

    def update(self, *args, **kwargs):
        """Update the dictionary with sanitized colormap names."""
        if len(args) == 1:
            kwargs.update(args[0])
        elif len(args) > 1:
            raise TypeError(
                f'update() expected at most 1 arguments, got {len(args)}.'
            )
        for key, value in kwargs.items():
            self[key] = value


class _ColorMappingOverride(mcolors._ColorMapping):
    """Mapping whose cache attribute is a `ColorDict` dictionary."""
    def __init__(self, mapping):
        super().__init__(mapping)
        self.cache = ColorDict({})


class ColorDict(dict):
    """This class overrides the builtin matplotlib color cache, allowing
    users to draw colors from *named colormaps and color cycles* for any
    plotting command that accepts a `color` keyword arg.
    See `~ColorDict.__getitem__` for details."""
    def __getitem__(self, key):
        """
        Allows user to select colors from arbitrary named colormaps and
        color cycles.

        * For a smooth colormap, usage is e.g. ``color=('Blues', 0.8)``. The
          number is the colormap index, and must be between 0 and 1.
        * For a color cycle, usage is e.g. ``color=('colorblind', 2)``. The
          number is the list index.

        These examples work with any
        matplotlib command that accepts a `color` keyword arg.
        """
        # Matplotlib 'color' args are passed to to_rgba, which tries to read
        # directly from cache and if that fails, sanitizes input, which
        # raises error on receiving (colormap, idx) tuple. So we *have* to
        # override cache instead of color dict itself.
        rgb, alpha = key
        if (not isinstance(rgb, str) and np.iterable(rgb) and len(rgb) == 2
                and isinstance(rgb[1], Number) and isinstance(rgb[0], str)):
            try:
                cmap = mcm.cmap_d[rgb[0]]
            except (TypeError, KeyError):
                pass
            else:
                if isinstance(cmap, ListedColormap):
                    if not 0 <= rgb[1] < len(cmap.colors):
                        raise ValueError(
                            f'Color cycle sample for {rgb[0]!r} cycle must be '
                            f'between 0 and {len(cmap.colors)-1}, '
                            f'got {rgb[1]}.'
                        )
                    # draw color from the list of colors, using index
                    rgb = cmap.colors[rgb[1]]
                else:
                    if not 0 <= rgb[1] <= 1:
                        raise ValueError(
                            f'Colormap sample for {rgb[0]!r} colormap must be '
                            f'between 0 and 1, got {rgb[1]}.'
                        )
                    # interpolate color from colormap, using key in range 0-1
                    rgb = cmap(rgb[1])
                rgba = mcolors.to_rgba(rgb, alpha)
                return rgba
        return super().__getitem__((rgb, alpha))


def Colors(*args, **kwargs):
    """Pass all arguments to `Cycle` and return the list of colors from
    the resulting `~cycler.Cycler` object."""
    cycle = Cycle(*args, **kwargs)
    return [dict_['color'] for dict_ in cycle]


def Colormap(
    *args, name=None, listmode='perceptual', fade=None, cycle=None,
    shift=None, cut=None, left=None, right=None, reverse=False,
    save=False, save_kw=None, **kwargs
):
    """
    Generate or retrieve colormaps and optionally merge and manipulate
    them in a variety of ways. Used to interpret the `cmap` and `cmap_kw`
    arguments when passed to any plotting method wrapped by
    `~proplot.wrappers.cmap_changer`.

    Parameters
    ----------
    *args : colormap-spec
        Positional arguments that individually generate colormaps. If more than
        one argument is passed, the resulting colormaps are merged. Arguments
        are interpreted as follows.

        * If `~matplotlib.colors.Colormap` or a registered colormap name, the
          colormap is simply returned.
        * If a filename string with valid extension, the colormap data will
          be loaded. See `register_cmaps` and `register_cycles`.
        * If RGB tuple or color string, a `PerceptuallyUniformColormap` is
          generated with `~PerceptuallyUniformColormap.from_color`. If the
          string ends in ``'_r'``, the monochromatic map will be *reversed*,
          i.e. will go from dark to light instead of light to dark.
        * If list of RGB tuples or color strings, a
          `PerceptuallyUniformColormap` is generated with
          `~PerceptuallyUniformColormap.from_list`.
        * If dictionary containing the keys ``'hue'``, ``'saturation'``, and
          ``'luminance'``, a `PerceptuallyUniformColormap` is generated with
          `~PerceptuallyUniformColormap.from_hsl`.

    name : str, optional
        Name under which the final colormap is registered. It can then be
        reused by passing ``cmap='name'`` to plotting functions like
        `~matplotlib.axes.Axes.contourf`.
    fade : float, optional
        The maximum luminosity used when generating colormaps with
        `PerceptuallyUniformColormap.from_color`. Default is ``100`` when
        calling `Colormap` directly, and ``90`` when `Colormap` is called by
        `Cycle` (this prevents having pure white in the color cycle).

        For example, ``plot.Colormap('blue', fade=80)`` generates a blue
        colormap that fades to a pale blue with 80% luminance.
    cycle : str or list of color-spec, optional
        The registered cycle name or a list of colors used to interpret cycle
        color strings like ``'C0'`` and ``'C2'`` when generating colormaps
        with `PerceptuallyUniformColormap.from_color`. Default is colors
        from the currently active property cycler.

        For example, ``plot.Colormap('C0', 'C1', 'C2', cycle='538')``
        generates a colormap using colors from the ``'538'`` color cycle.
    listmode : {'perceptual', 'linear', 'listed'}, optional
        Controls how colormaps are generated when you input list(s) of colors.
        If ``'perceptual'``, a `PerceptuallyUniformColormap` is generated with
        `PerceptuallyUniformColormap.from_list`. If ``'linear'``,
        a `~matplotlib.colors.LinearSegmentedColormap` is generated with
        `~matplotlib.colors.LinearSegmentedColormap.from_list`. If
        ``'listed'``, the `~matplotlib.colors.ListedColormap` is generated.

        Default is ``'perceptual'`` when calling `Colormap` directly, and
        ``'listed'`` when `Colormap` is called by `Cycle`.
    cut : float, optional
        Passed to `LinearSegmentedColormap.punched`.
        This applies to the final *merged* colormap.
    left, right : float, optional
        Passed to `LinearSegmentedColormap.truncated` or
        `ListedColormap.truncated`. These apply to *each colormap*
        individually.
    reverse : bool, optional
        Passed to `LinearSegmentedColormap.reversed` or
        `ListedColormap.reversed`. This applies to *each colormap*
        individually.
    shift : float, optional
        Passed to `LinearSegmentedColormap.shifted` or
        `ListedColormap.shifted`. This applies to the final *merged* colormap.
    save : bool, optional
        Whether to call the colormap save method, i.e.
        `LinearSegmentedColormap.save` or
        `ListedColormap.save`.
    save_kw : dict-like, optional
        Ignored if `save` is ``False``. Passed to the colormap save method,
        i.e. `LinearSegmentedColormap.save` or
        `ListedColormap.save`.
    **kwargs
        Passed to `LinearSegmentedColormap.concatenate` or
        `ListedColormap.concatenate`. Each of these functions accepts
        arbitrary colormap settings.

    Returns
    -------
    `~matplotlib.colors.Colormap`
        A `~matplotlib.colors.LinearSegmentedColormap` or
        `~matplotlib.colors.ListedColormap` instance.
    """
    # Initial stuff
    # TODO: Play with using "qualitative" colormaps in realistic examples,
    # how to make colormaps cyclic.
    if not args:
        raise ValueError(
            f'Colormap() requires at least one positional argument.'
        )
    if listmode not in ('listed', 'linear', 'perceptual'):
        raise ValueError(
            f'Invalid listmode={listmode!r}. Options are: '
            "'listed', 'linear', 'perceptual'."
        )
    tmp = '_no_name'
    cmaps = []
    for i, cmap in enumerate(args):
        # Load registered colormaps and maps on file
        # TODO: Document how 'listmode' also affects loaded files
        if isinstance(cmap, str):
            if '.' in cmap:
                if listmode == 'listed':
                    cmap = ListedColormap.from_file(cmap)
                else:
                    cmap = LinearSegmentedColormap.from_file(cmap)
            else:
                try:
                    cmap = mcm.cmap_d[cmap]
                except KeyError:
                    pass
        # Convert matplotlib colormaps to subclasses
        if isinstance(cmap, (ListedColormap, LinearSegmentedColormap)):
            pass
        elif isinstance(cmap, mcolors.LinearSegmentedColormap):
            cmap = LinearSegmentedColormap(
                cmap.name, cmap._segmentdata, cmap.N, cmap._gamma)
        elif isinstance(cmap, mcolors.ListedColormap):
            cmap = ListedColormap(
                cmap.colors, cmap.name, cmap.N)
        # Dictionary of hue/sat/luminance values or 2-tuples representing
        # linear transition
        elif isinstance(cmap, dict):
            cmap = PerceptuallyUniformColormap.from_hsl(tmp, **cmap)
        # List of color tuples or color strings, i.e. iterable of iterables
        elif (not isinstance(cmap, str) and np.iterable(cmap)
              and all(np.iterable(color) for color in cmap)):
            try:
                colors = [to_rgb(color, cycle=cycle, alpha=True)
                          for color in cmap]
            except (ValueError, TypeError):
                raise ValueError(f'Invalid color(s) in list {cmap!r}.')
            if listmode == 'listed':
                cmap = ListedColormap(colors, tmp)
            elif listmode == 'linear':
                cmap = LinearSegmentedColormap.from_list(tmp, colors)
            else:
                cmap = PerceptuallyUniformColormap.from_list(tmp, colors)
        # Monochrome colormap from input color
        else:
            ireverse = (isinstance(cmap, str) and cmap[-2:] == '_r')
            if ireverse:
                cmap = cmap[:-2]
            try:
                color = to_rgb(cmap, cycle=cycle, alpha=True)
            except (ValueError, TypeError):
                msg = f'Invalid cmap, cycle, or color {cmap!r}.'
                if isinstance(cmap, str):
                    msg += (
                        f'\nValid cmap and cycle names: '
                        + ', '.join(map(repr, sorted(mcm.cmap_d))) + '.'
                        f'\nValid color names: '
                        + ', '.join(map(repr, sorted(
                            mcolors.colorConverter.colors))
                        ) + '.'
                    )
                raise ValueError(msg)
            cmap = PerceptuallyUniformColormap.from_color(tmp, color, fade)
            if ireverse:
                cmap = cmap.reversed()

        # Cut the edges and/or reverse the map
        if left is not None or right is not None:
            cmap = cmap.truncated(left, right)
        if reverse:
            cmap = cmap.reversed()
        cmaps.append(cmap)

    # Merge the result of this arbitrary user input
    if len(cmaps) > 1:  # more than one map?
        cmap = cmaps[0].concatenate(*cmaps[1:], **kwargs)
    elif kwargs:  # modify any props?
        cmap = cmaps[0].updated(**kwargs)

    # Cut the center and roate the colormap
    if cut is not None:
        cmap = cmap.punched(cut)
    if shift is not None:
        cmap = cmap.shifted(shift)

    # Initialize
    if not cmap._isinit:
        cmap._init()

    # Register and save the colormap
    if name is None:
        name = cmap.name  # may have been modified by e.g. .shifted()
    else:
        cmap.name = name
    mcm.cmap_d[name] = cmap
    if save:
        save_kw = save_kw or {}
        cmap.save(**save_kw)
    return cmap


def Cycle(
    *args, N=None, name=None,
    marker=None, alpha=None, dashes=None, linestyle=None, linewidth=None,
    markersize=None, markeredgewidth=None,
    markeredgecolor=None, markerfacecolor=None,
    save=False, save_kw=None, **kwargs
):
    """
    Generate and merge `~cycler.Cycler` instances in a variety of ways.
    Used to interpret the `cycle` and `cycle_kw` arguments when passed to
    any plotting method wrapped by
    `~proplot.wrappers.cycle_changer`.

    If you just want a list of colors instead of a `~cycler.Cycler` instance,
    use the `colors` function. If you want a `~cycler.Cycler` instance that
    imposes black as the default color and cycles through properties like
    ``linestyle`` instead, call this function without any positional arguments.

    Parameters
    ----------
    *args : colormap-spec or cycle-spec, optional
        Positional arguments control the *colors* in the `~cycler.Cycler`
        object. If more than one argument is passed, the resulting cycles are
        merged. Arguments are interpreted as follows.

        * If `~cycler.Cycler`, nothing more
          is done.
        * If list of RGB tuples or color strings, these
          colors are used.
        * If `~matplotlib.colors.ListedColormap`, colors from the ``colors``
          attribute are used.
        * If string color cycle name, that `~matplotlib.colors.ListedColormap`
          is looked up and its ``colors`` attribute is used. See `cycles`.
        * Otherwise, the argument is passed to `Colormap`, and colors
          from the resulting `~matplotlib.colors.LinearSegmentedColormap`
          are used. See the `N` argument.

        If the last positional argument is numeric, it is used for the `N`
        keyword argument.
    N : float or list of float, optional
        For `~matplotlib.colors.ListedColormap`\ s, this is the number of
        colors to select. For example, ``Cycle('538', 4)`` returns the first 4
        colors of the ``'538'`` color cycle.

        For `~matplotlib.colors.LinearSegmentedColormap`\ s, this is either
        a *list of sample coordinates* used to draw colors from the map, or an
        *integer number of colors* to draw. If the latter, the sample
        coordinates are ``np.linspace(0, 1, samples)``. For example,
        ``Cycle('Reds', 5)`` divides the ``'Reds'`` colormap into five evenly
        spaced colors.
    name : str, optional
        Name of the resulting `~matplotlib.colors.ListedColormap` used to
        register the color cycle. Default name is ``'no_name'``.
    marker, alpha, dashes, linestyle, linewidth, markersize, markeredgewidth, markeredgecolor, markerfacecolor : list of specs, optional
        Lists of `~matplotlib.lines.Line2D` properties that can be added to
        the `~cycler.Cycler` instance. If the lists have unequal length, they
        will be filled to match the length of the longest list.  See
        `~matplotlib.axes.Axes.set_prop_cycle` for more info on cyclers.
        Also see the `line style reference \
<https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html>`__,
        the `marker reference \
<https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/marker_reference.html>`__,
        and the `custom dashes reference \
<https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/line_demo_dash_control.html>`__.
    save : bool, optional
        Whether to save the `ListedColormap` associated with this cycle.
        See `ListedColormap.save`.
    save_kw : dict-like, optional
        Ignored if `save` is ``False``. Passed to `ListedColormap.save`
        for the `ListedColormap` associated with this cycle.
    **kwargs
        Passed to `Colormap` when the input is not already a `~cycler.Cycler`
        instance.

    Returns
    -------
    `~cycler.Cycler`
        A cycler instance that can be passed to
        `~matplotlib.axes.Axes.set_prop_cycle`.
    """  # noqa
    # Add properties
    props = {}
    nprops = 0
    for key, value in (
        ('marker', marker),
        ('alpha', alpha),
        ('dashes', dashes),
        ('linestyle', linestyle),
        ('linewidth', linewidth),
        ('markersize', markersize),
        ('markeredgewidth', markeredgewidth),
        ('markeredgecolor', markeredgecolor),
        ('markerfacecolor', markerfacecolor),
    ):
        if value is not None:
            if isinstance(value, str) or not np.iterable(value):
                raise ValueError(
                    f'Invalid {key!r} property {value!r}. '
                    f'Must be list or tuple of properties.'
                )
            nprops = max(nprops, len(value))
            props[key] = [*value]  # ensure mutable list
    # If args is non-empty, means we want color cycle; otherwise is black
    if not args:
        props['color'] = ['k']  # ensures property cycler is non empty
        if kwargs:
            _warn_proplot(f'Ignoring Cycle() keyword arg(s) {kwargs}.')
    # Merge cycler objects
    elif all(isinstance(arg, cycler.Cycler) for arg in args):
        if kwargs:
            _warn_proplot(f'Ignoring Cycle() keyword arg(s) {kwargs}.')
        if len(args) == 1:
            return args[0]
        else:
            props = {}
            for arg in args:
                for key, value in arg.by_key():
                    if key not in props:
                        props[key] = []
                    props[key].extend([*value])
            return cycler.cycler(**props)
    # Build and register a ListedColormap
    else:
        # Collect samples
        if args and isinstance(args[-1], Number):
            # means we want to sample existing colormaps or cycles
            args, N = args[:-1], args[-1]
        kwargs.setdefault('fade', 90)
        kwargs.setdefault('listmode', 'listed')
        cmap = Colormap(*args, **kwargs)  # the cmap object itself
        if isinstance(cmap, ListedColormap):
            colors = cmap.colors[:N]  # if N is None, does nothing
        else:
            N = _notNone(N, 10)
            if isinstance(N, Integral):
                x = np.linspace(0, 1, N)  # from edge to edge
            elif np.iterable(N) and all(
                    isinstance(item, Number) for item in N):
                x = np.array(N)
            else:
                raise ValueError(f'Invalid samples {N!r}.')
            N = len(x)
            colors = cmap(x)

        # Register and save the samples as a ListedColormap
        name = name or '_no_name'
        cmap = ListedColormap(colors, name=name, N=N)
        mcm.cmap_d[name] = cmap
        if save:
            save_kw = save_kw or {}
            cmap.save(**save_kw)

        # Add to property dict
        nprops = max(nprops, len(colors))
        props['color'] = [
            tuple(color) if not isinstance(color, str) else color
            for color in cmap.colors
        ]  # save the tupled version!

    # Build cycler, make sure lengths are the same
    for key, value in props.items():
        if len(value) < nprops:
            value[:] = [
                value[i % len(value)] for i in range(nprops)
            ]  # make loop double back
    cycle = cycler.cycler(**props)
    cycle.name = name
    return cycle


def Norm(norm, levels=None, **kwargs):
    """
    Return an arbitrary `~matplotlib.colors.Normalize` instance.
    Used to interpret the `norm` and `norm_kw` arguments when passed to any
    plotting method wrapped by `~proplot.wrappers.cmap_changer`.

    Parameters
    ----------
    norm : str or `~matplotlib.colors.Normalize`
        Key name for the normalizer. The recognized normalizer key names
        are as follows.

        ===============================  ===============================
        Key(s)                           Class
        ===============================  ===============================
        ``'midpoint'``, ``'zero'``       `MidpointNorm`
        ``'segments'``, ``'segmented'``  `LinearSegmentedNorm`
        ``'none'``, ``'null'``           `~matplotlib.colors.NoNorm`
        ``'linear'``                     `~matplotlib.colors.Normalize`
        ``'log'``                        `~matplotlib.colors.LogNorm`
        ``'power'``                      `~matplotlib.colors.PowerNorm`
        ``'symlog'``                     `~matplotlib.colors.SymLogNorm`
        ===============================  ===============================

    levels : array-like, optional
        Level *edges*, passed to `LinearSegmentedNorm` or used to determine
        the `vmin` and `vmax` arguments for `MidpointNorm`.
    **kwargs
        Passed to the `~matplotlib.colors.Normalize` initializer.
        See `this tutorial \
<https://matplotlib.org/tutorials/colors/colormapnorms.html>`__
        for more info.

    Returns
    -------
    `~matplotlib.colors.Normalize`
        A `~matplotlib.colors.Normalize` instance.
    """
    if isinstance(norm, mcolors.Normalize):
        return norm
    if isinstance(norm, str):
        # Get class
        norm_out = normalizers.get(norm, None)
        if norm_out is None:
            raise ValueError(
                f'Unknown normalizer {norm!r}. Options are: '
                + ', '.join(map(repr, normalizers.keys())) + '.'
            )
        # Instantiate class
        if norm_out is LinearSegmentedNorm:
            if not np.iterable(levels):
                raise ValueError(
                    f'Need levels for normalizer {norm!r}. '
                    f'Received levels={levels!r}.'
                )
            kwargs.update({'levels': levels})
        norm_out = norm_out(**kwargs)  # initialize
    else:
        raise ValueError(f'Unknown norm {norm_out!r}.')
    return norm_out


class BinNorm(mcolors.BoundaryNorm):
    """
    This normalizer is used for all colormap plots. It can be thought of as a
    "meta-normalizer": It first scales the data according to any
    arbitrary `~matplotlib.colors.Normalize` class, then maps the normalized
    values ranging from 0-1 into **discrete** levels.

    Consider input levels of ``[0, 3, 6, 9, 12, 15]``. The algorithm is
    as follows.

    1. `levels` are normalized according to the input normalizer `norm`.
       If it is ``None``, they are not changed. Possible normalizers include
       `~matplotlib.colors.LogNorm`, which makes color transitions linear in
       the logarithm of the value, or `LinearSegmentedNorm`, which makes
       color transitions linear in the **index** of the level array.
    2. Possible colormap coordinates, corresponding to bins delimited by the
       normalized `levels` array, are calculated.  In this case, the bin
       centers are simply ``[1.5, 4.5, 7.5, 10.5, 13.5]``, which gives us
       normalized colormap coordinates of ``[0, 0.25, 0.5, 0.75, 1]``.
    3. Out-of-bounds coordinates are added. These depend on the value of the
       `extend` keyword argument. For `extend` equal to ``'neither'``,
       the coordinates including out-of-bounds values are
       ``[0, 0, 0.25, 0.5, 0.75, 1, 1]`` -- out-of-bounds values have the same
       color as the nearest in-bounds values. For `extend` equal to ``'both'``,
       the bins are ``[0, 0.16, 0.33, 0.5, 0.66, 0.83, 1]`` --
       out-of-bounds values are given distinct colors. This makes sure your
       colorbar always shows the **full range of colors** in the colormap.
    4. Whenever `BinNorm.__call__` is invoked, the input value normalized by
       `norm` is compared against the normalized `levels` array. Its bin index
       is determined with `numpy.searchsorted`, and its corresponding
       colormap coordinate is selected using this index.

    """
    # See this post: https://stackoverflow.com/a/48614231/4970632
    # WARNING: Must be child of BoundaryNorm. Many methods in ColorBarBase
    # test for class membership, crucially including _process_values(), which
    # if it doesn't detect BoundaryNorm will try to use BinNorm.inverse().
    def __init__(
        self, levels, norm=None, clip=False,
        step=1.0, extend='neither'
    ):
        """
        Parameters
        ----------
        levels : list of float
            The discrete data levels.
        norm : `~matplotlib.colors.Normalize`, optional
            The normalizer used to transform `levels` and all data passed
            to `BinNorm.__call__` *before* discretization.
        step : float, optional
            The intensity of the transition to out-of-bounds color, as a
            faction of the *average* step between in-bounds colors.
            Default is ``1``.
        extend : {'neither', 'both', 'min', 'max'}, optional
            Which direction colors will be extended. No matter the `extend`
            option, `BinNorm` ensures colors always extend through the
            extreme end colors.
        clip : bool, optional
            A `~matplotlib.colors.Normalize` option.

        Note
        ----
        If you are using a diverging colormap with ``extend='max'`` or
        ``extend='min'``, the center will get messed up. But that is very
        strange usage anyway... so please just don't do that :)
        """
        # Declare boundaries, vmin, vmax in True coordinates.
        # Notes:
        # * Idea is that we bin data into len(levels) discrete x-coordinates,
        #   and optionally make out-of-bounds colors the same or different
        # * Don't need to call parent __init__, this is own implementation
        #   Do need it to subclass BoundaryNorm, so ColorbarBase will detect it
        # See BoundaryNorm:
        # https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/colors.py
        levels = np.atleast_1d(levels)
        if levels.size <= 1:
            raise ValueError('Need at least two levels.')
        elif ((levels[1:] - levels[:-1]) <= 0).any():
            raise ValueError(
                f'Levels {levels} passed to Normalize() must be '
                'monotonically increasing.'
            )
        if extend not in ('both', 'min', 'max', 'neither'):
            raise ValueError(
                f'Unknown extend option {extend!r}. Choose from '
                '"min", "max", "both", "neither".'
            )

        # Determine color ids for levels, i.e. position in 0-1 space
        # Length of these ids should be N + 1 -- that is, N - 1 colors
        # for values in-between levels, plus 2 colors for out-of-bounds.
        # * For same out-of-bounds colors, looks like [0, 0, ..., 1, 1]
        # * For unique out-of-bounds colors, looks like [0, X, ..., 1 - X, 1]
        #   where the offset X equals step/len(levels).
        # First get coordinates
        if not norm:
            # WARNING: Normalization to 0-1 must always take place first,
            # required by colorbar_factory ticks manager.
            norm = mcolors.Normalize()
        elif not isinstance(norm, mcolors.Normalize):
            raise ValueError(
                'Normalizer must be matplotlib.colors.Normalize, '
                f'got {type(norm)}.'
            )
        elif isinstance(norm, mcolors.BoundaryNorm):
            raise ValueError(
                f'Normalizer cannot be an instance of '
                'matplotlib.colors.BoundaryNorm.'
            )
        x_b = norm(levels)
        x_m = (x_b[1:] + x_b[:-1]) / 2  # get level centers after norm scaling
        y = (x_m - x_m.min()) / (x_m.max() - x_m.min())
        if isinstance(y, ma.core.MaskedArray):
            y = y.filled(np.nan)
        y = y[np.isfinite(y)]
        # Account for out of bounds colors
        # WARNING: For some reason must clip manually for LogNorm, or
        # end up with unpredictable fill value, weird "out-of-bounds" colors
        offset = 0
        scale = 1
        eps = step / (y.size - 1)
        # eps = step/levels.size
        if extend in ('min', 'both'):
            offset = eps
            scale -= eps
        if extend in ('max', 'both'):
            scale -= eps
        # insert '0' (arg 3) before index '0' (arg 2)
        y = np.concatenate(([0], offset + scale * y, [1]))
        self._norm = norm
        self._x_b = x_b
        self._y = y
        if isinstance(norm, mcolors.LogNorm):
            self._norm_clip = (5e-249, None)
        else:
            self._norm_clip = None

        # Add builtin properties
        # NOTE: Are vmin/vmax even used?
        self.boundaries = levels
        self.vmin = levels.min()
        self.vmax = levels.max()
        self.clip = clip
        self.N = levels.size

    def __call__(self, xq, clip=None):
        """Normalize data values to 0-1."""
        # Follow example of LinearSegmentedNorm, but perform no interpolation,
        # just use searchsorted to bin the data.
        norm_clip = self._norm_clip
        if norm_clip:
            xq = np.clip(xq, *norm_clip)
        xq = self._norm(xq)
        # which x-bin does each point in xq belong to?
        yq = self._y[np.searchsorted(self._x_b, xq)]
        mask = ma.getmaskarray(xq)
        return ma.array(yq, mask=mask)

    def inverse(self, yq):
        """Raise an error. Inversion after discretization is impossible."""
        raise RuntimeError('BinNorm is not invertible.')


class LinearSegmentedNorm(mcolors.Normalize):
    """
    This is the default normalizer paired with `BinNorm` whenever `levels`
    are non-linearly spaced. The normalized value is linear with respect to
    its average index in the `levels` vector, allowing uniform color
    transitions across arbitrarily spaced monotonically increasing values.

    It accomplishes this following the example of the
    `~matplotlib.colors.LinearSegmentedColormap` source code, by performing
    efficient, vectorized linear interpolation between the provided boundary
    levels.

    Can be used by passing ``norm='segmented'`` or ``norm='segments'`` to any
    command accepting ``cmap``. The default midpoint is zero.
    """
    def __init__(self, levels, vmin=None, vmax=None, **kwargs):
        """
        Parameters
        ----------
        levels : list of float
            The discrete data levels.
        vmin, vmax : None
            Ignored, because `vmin` and `vmax` are set to the minimum and
            maximum of `levels`.
        **kwargs
            Passed to `~matplotlib.colors.Normalize`.
        """
        levels = np.atleast_1d(levels)
        if levels.size <= 1:
            raise ValueError('Need at least two levels.')
        elif ((levels[1:] - levels[:-1]) <= 0).any():
            raise ValueError(
                f'Levels {levels} passed to LinearSegmentedNorm must be '
                'monotonically increasing.'
            )
        vmin, vmax = levels.min(), levels.max()
        super().__init__(vmin, vmax, **kwargs)  # second level superclass
        self._x = levels
        self._y = np.linspace(0, 1, len(levels))

    def __call__(self, xq, clip=None):
        """Normalize the data values to 0-1. Inverse
        of `~LinearSegmentedNorm.inverse`."""
        # Follow example of make_mapping_array for efficient, vectorized
        # linear interpolation across multiple segments.
        # * Normal test puts values at a[i] if a[i-1] < v <= a[i]; for
        #   left-most data, satisfy a[0] <= v <= a[1]
        # * searchsorted gives where xq[i] must be inserted so it is larger
        #   than x[ind[i]-1] but smaller than x[ind[i]]
        x = self._x  # from arbitrarily spaced monotonic levels
        y = self._y  # to linear range 0-1
        xq = np.atleast_1d(xq)
        ind = np.searchsorted(x, xq)
        ind[ind == 0] = 1
        ind[ind == len(x)] = len(x) - 1
        distance = (xq - x[ind - 1]) / (x[ind] - x[ind - 1])
        yq = distance * (y[ind] - y[ind - 1]) + y[ind - 1]
        mask = ma.getmaskarray(xq)
        return ma.array(yq, mask=mask)

    def inverse(self, yq):
        """Inverse operation of `~LinearSegmentedNorm.__call__`."""
        x = self._x
        y = self._y
        yq = np.atleast_1d(yq)
        ind = np.searchsorted(y, yq)
        ind[ind == 0] = 1
        ind[ind == len(y)] = len(y) - 1
        distance = (yq - y[ind - 1]) / (y[ind] - y[ind - 1])
        xq = distance * (x[ind] - x[ind - 1]) + x[ind - 1]
        mask = ma.getmaskarray(yq)
        return ma.array(xq, mask=mask)


class MidpointNorm(mcolors.Normalize):
    """
    Ensures a "midpoint" always lies at the central colormap color.
    Can be used by passing ``norm='midpoint'`` to any command accepting
    ``cmap``. The default midpoint is zero.
    """

    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=None):
        """
        Parameters
        ----------
        midpoint : float, optional
            The midpoint, or the data value corresponding to the normalized
            value ``0.5`` -- halfway down the colormap.
        vmin, vmax, clip
            The minimum and maximum data values, and the clipping setting.
            Passed to `~matplotlib.colors.Normalize`.
        """
        # Bigger numbers are too one-sided
        super().__init__(vmin, vmax, clip)
        self._midpoint = midpoint

    def __call__(self, xq, clip=None):
        """Normalize data values to 0-1. Inverse of `~MidpointNorm.inverse`."""
        # Get middle point in 0-1 coords, and value
        # Notes:
        # * Look up these three values in case vmin/vmax changed; this is
        #   a more general normalizer than the others. Others are 'parent'
        #   normalizers, meant to be static more or less.
        # * searchsorted gives where xq[i] must be inserted so it is larger
        #   than x[ind[i]-1] but smaller than x[ind[i]]
        #   x, y = [self.vmin, self._midpoint, self.vmax], [0, 0.5, 1]
        if self.vmin >= self._midpoint or self.vmax <= self._midpoint:
            raise ValueError(
                f'Midpoint {self._midpoint} outside of vmin {self.vmin} '
                f'and vmax {self.vmax}.'
            )
        x = np.array([self.vmin, self._midpoint, self.vmax])
        y = np.array([0, 0.5, 1])
        xq = np.atleast_1d(xq)
        ind = np.searchsorted(x, xq)
        ind[ind == 0] = 1  # in this case will get normed value <0
        # in this case, will get normed value >0
        ind[ind == len(x)] = len(x) - 1
        distance = (xq - x[ind - 1]) / (x[ind] - x[ind - 1])
        yq = distance * (y[ind] - y[ind - 1]) + y[ind - 1]
        mask = ma.getmaskarray(xq)
        return ma.array(yq, mask=mask)

    def inverse(self, yq, clip=None):
        """Inverse operation of `~MidpointNorm.__call__`."""
        # Invert the above
        # x, y = [self.vmin, self._midpoint, self.vmax], [0, 0.5, 1]
        # return ma.masked_array(np.interp(yq, y, x))
        # Performs inverse operation of __call__
        x = np.array([self.vmin, self._midpoint, self.vmax])
        y = np.array([0, 0.5, 1])
        yq = np.atleast_1d(yq)
        ind = np.searchsorted(y, yq)
        ind[ind == 0] = 1
        ind[ind == len(y)] = len(y) - 1
        distance = (yq - y[ind - 1]) / (y[ind] - y[ind - 1])
        xq = distance * (x[ind] - x[ind - 1]) + x[ind - 1]
        mask = ma.getmaskarray(yq)
        return ma.array(xq, mask=mask)


def _get_data_paths(dirname):
    """Return data folder paths in reverse order of precedence."""
    # When loading colormaps, cycles, and colors, files in the latter
    # directories overwrite files in the former directories. When loading
    # fonts, the resulting paths need to be *reversed*.
    return [
        os.path.join(os.path.dirname(__file__), dirname),
        os.path.join(os.path.expanduser('~'), '.proplot', dirname)
    ]


def _from_file(filename, listed=False, warn_on_failure=False):
    """Read generalized colormap and color cycle files."""
    filename = os.path.expanduser(filename)
    if os.path.isdir(filename):  # no warning
        return

    # Warn if loading failed during `register_cmaps` or `register_cycles`
    # but raise error if user tries to load a file.
    def _warn_or_raise(msg, error=RuntimeError):
        if warn_on_failure:
            _warn_proplot(msg)
        else:
            raise error(msg)

    # Directly read segmentdata json file
    # NOTE: This is special case! Immediately return name and cmap
    if not os.path.exists(filename):
        _warn_or_raise(f'File {filename!r} not found.', FileNotFoundError)
        return
    N = rcParams['image.lut']
    name, ext = os.path.splitext(os.path.basename(filename))
    ext = ext[1:]
    cmap = None
    if ext == 'json':
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            _warn_or_raise(
                f'Failed to load {filename!r}.', json.JSONDecodeError
            )
            return
        kw = {}
        for key in ('cyclic', 'gamma', 'gamma1', 'gamma2', 'space'):
            kw[key] = data.pop(key, None)
        if 'red' in data:
            cmap = LinearSegmentedColormap(name, data, N=N)
        else:
            cmap = PerceptuallyUniformColormap(name, data, N=N, **kw)
        if name[-2:] == '_r':
            cmap = cmap.reversed(name[:-2])

    # Read .rgb, .rgba, .xrgb, and .xrgba files
    elif ext in ('txt', 'rgb', 'xrgb', 'rgba', 'xrgba'):
        # Load
        # NOTE: This appears to be biggest import time bottleneck! Increases
        # time from 0.05s to 0.2s, with numpy loadtxt or with this regex thing.
        delim = re.compile(r'[,\s]+')
        data = [
            delim.split(line.strip())
            for line in open(filename).readlines()
            if line.strip() and line.strip()[0] != '#'
        ]
        try:
            data = [[float(num) for num in line] for line in data]
        except ValueError:
            _warn_or_raise(
                f'Failed to load {filename!r}. Expected a table of comma '
                'or space-separated values.'
            )
            return
        # Build x-coordinates and standardize shape
        data = np.array(data)
        if data.shape[1] != len(ext):
            _warn_or_raise(
                f'Failed to load {filename!r}. Got {data.shape[1]} columns, '
                f'but expected {len(ext)}.'
            )
            return
        if ext[0] != 'x':  # i.e. no x-coordinates specified explicitly
            x = np.linspace(0, 1, data.shape[0])
        else:
            x, data = data[:, 0], data[:, 1:]

    # Load XML files created with scivizcolor
    # Adapted from script found here:
    # https://sciviscolor.org/matlab-matplotlib-pv44/
    elif ext == 'xml':
        try:
            doc = ElementTree.parse(filename)
        except ElementTree.ParseError:
            _warn_or_raise(
                f'Failed to load {filename!r}. Parsing error.',
                ElementTree.ParseError
            )
            return
        x, data = [], []
        for s in doc.getroot().findall('.//Point'):
            # Verify keys
            if any(key not in s.attrib for key in 'xrgb'):
                _warn_or_raise(
                    f'Failed to load {filename!r}. Missing an x, r, g, or b '
                    'specification inside one or more <Point> tags.'
                )
                return
            # Get data
            color = []
            for key in 'rgbao':  # o for opacity
                if key not in s.attrib:
                    continue
                color.append(float(s.attrib[key]))
            x.append(float(s.attrib['x']))
            data.append(color)
        # Convert to array
        if not all(len(data[0]) == len(color)
                   and len(color) in (3, 4) for color in data):
            _warn_or_raise(
                f'Failed to load {filename!r}. Unexpected number of channels '
                'or mixed channels across <Point> tags.'
            )
            return

    # Read hex strings
    elif ext == 'hex':
        # Read arbitrary format
        string = open(filename).read()  # into single string
        data = re.findall('#[0-9a-fA-F]{6}', string)  # list of strings
        if len(data) < 2:
            _warn_or_raise(
                f'Failed to load {filename!r}. Hex strings not found.'
            )
            return
        # Convert to array
        x = np.linspace(0, 1, len(data))
        data = [to_rgb(color) for color in data]
    else:
        _warn_or_raise(
            f'Colormap or cycle file {filename!r} has unknown extension.'
        )
        return

    # Standardize and reverse if necessary to cmap
    # TODO: Document the fact that filenames ending in _r return a reversed
    # version of the colormap stored in that file.
    if not cmap:
        x, data = np.array(x), np.array(data)
        # for some reason, some aren't in 0-1 range
        x = (x - x.min()) / (x.max() - x.min())
        if (data > 2).any():  # from 0-255 to 0-1
            data = data / 255
        if name[-2:] == '_r':
            name = name[:-2]
            data = data[::-1, :]
            x = 1 - x[::-1]
        if listed:
            cmap = ListedColormap(data, name, N=len(data))
        else:
            data = [(x, color) for x, color in zip(x, data)]
            cmap = LinearSegmentedColormap.from_list(name, data, N=N)

    # Return colormap or data
    return cmap


@_timer
def register_cmaps():
    """
    Register colormaps packaged with ProPlot or saved to the
    ``~/.proplot/cmaps`` folder. This is called on import. Maps are registered
    according to their filenames -- for example, ``name.xyz`` will be
    registered as ``'name'``.

    For a table of valid extensions, see `LinearSegmentedColormap.from_file`.
    To visualize the registered colormaps, use `show_cmaps`.
    """
    for i, path in enumerate(_get_data_paths('cmaps')):
        for filename in sorted(glob.glob(os.path.join(path, '*'))):
            cmap = LinearSegmentedColormap.from_file(
                filename, warn_on_failure=True
            )
            if not cmap:
                continue
            if i == 0 and cmap.name.lower() in ('phase', 'graycycle'):
                cmap._cyclic = True
            mcm.cmap_d[cmap.name] = cmap


@_timer
def register_cycles():
    """
    Register color cycles packaged with ProPlot or saved to the
    ``~/.proplot/cycles`` folder. This is called on import. Cycles are
    registered according to their filenames -- for example, ``name.hex`` will
    be registered under the name ``'name'`` as a
    `~matplotlib.colors.ListedColormap` map (see `Cycle` for details).

    For a table of valid extensions, see `ListedColormap.from_file`.
    To visualize the registered colormaps, use `show_cmaps`.
    """
    for path in _get_data_paths('cycles'):
        for filename in sorted(glob.glob(os.path.join(path, '*'))):
            cmap = ListedColormap.from_file(
                filename, warn_on_failure=True
            )
            if not cmap:
                continue
            if isinstance(cmap, LinearSegmentedColormap):
                cmap = ListedColormap(colors(cmap), name=cmap.name)
            mcm.cmap_d[cmap.name] = cmap
            cycles.append(cmap.name)


@_timer
def register_colors(nmax=np.inf):
    """
    Add color names packaged with ProPlot or saved to the ``~/.proplot/colors``
    folder. ProPlot loads the crowd-sourced XKCD color
    name database, Crayola crayon color database, and any user input
    files, then filters them to be "perceptually distinct" in the HCL
    colorspace. Files must just have one line per color in the format
    ``name : hex``. Whitespace is ignored.

    This is called on import. Use `show_colors` to generate a table of the
    resulting colors.
    """
    # Reset native colors dictionary and add some default groups
    # Add in CSS4 so no surprises for user, but we will not encourage this
    # usage and will omit CSS4 colors from the demo table.
    colors.clear()
    base = {}
    base.update(mcolors.BASE_COLORS)
    base.update(COLORS_BASE)  # full names
    mcolors.colorConverter.colors.clear()  # clean out!
    mcolors.colorConverter.cache.clear()  # clean out!
    for name, dict_ in (('base', base), ('css', mcolors.CSS4_COLORS)):
        mcolors.colorConverter.colors.update(dict_)
        colors[name] = sorted(dict_)

    # Load colors from file and get their HCL values
    dicts = {}
    seen = {*base}  # never overwrite base names, e.g. 'blue' and 'b'!
    hcls = []
    data = []
    for i, path in enumerate(_get_data_paths('colors')):
        if i == 0:
            paths = [  # be explicit because categories matter!
                os.path.join(path, base)
                for base in ('xkcd.txt', 'crayola.txt', 'opencolor.txt')
            ]
        else:
            paths = sorted(glob.glob(os.path.join(path, '*.txt')))
        for file in paths:
            cat, _ = os.path.splitext(os.path.basename(file))
            with open(file, 'r') as f:
                cnt = 0
                hex = re.compile(
                    r'\A#(?:[0-9a-fA-F]{3}){1,2}\Z'  # ?: prevents capture
                )
                pairs = []
                for line in f.readlines():
                    cnt += 1
                    stripped = line.strip()
                    if not stripped or stripped[0] == '#':
                        continue
                    pair = tuple(item.strip() for item in line.split(':'))
                    if len(pair) != 2 or not hex.match(pair[1]):
                        _warn_proplot(
                            f'Illegal line #{cnt} in file {file!r}:\n'
                            f'{line!r}\n'
                            f'Lines must be formatted as "name: hexcolor".'
                        )
                        continue
                    pairs.append(pair)

            # Categories for which we add *all* colors
            if cat == 'opencolor' or i == 1:
                dict_ = {name: color for name, color in pairs}
                mcolors.colorConverter.colors.update(dict_)
                colors[cat] = sorted(dict_)
                continue

            # Filter remaining colors to *unique* colors
            j = 0
            if cat not in dicts:
                dicts[cat] = {}
            for name, color in pairs:  # is list of name, color tuples
                j += 1
                if j > nmax:  # e.g. for xkcd colors
                    break
                for regex, sub in COLORS_TRANSLATIONS:
                    name = regex.sub(sub, name)
                if name in seen or COLORS_IGNORE.search(name):
                    continue
                seen.add(name)
                hcls.append(to_xyz(color, space=COLORS_SPACE))
                data.append((cat, name, color))  # category name pair

    # Remove colors that are 'too similar' by rounding to the nearest n units
    # WARNING: Unique axis argument requires numpy version >=1.13
    hcls = np.array(hcls)
    if hcls.size > 0:
        hcls = hcls / np.array([360, 100, 100])
        hcls = np.round(hcls / COLORS_THRESH).astype(np.int64)
        _, idxs, _ = np.unique(
            hcls, return_index=True, return_counts=True, axis=0)
        for idx, (cat, name, color) in enumerate(data):
            if name in COLORS_INCLUDE or idx in idxs:
                dicts[cat][name] = color
        for cat, dict_ in dicts.items():
            mcolors.colorConverter.colors.update(dict_)
            colors[cat] = sorted(dict_)


@_timer
def register_fonts():
    """Add fonts packaged with ProPlot or saved to the ``~/.proplot/fonts``
    folder, if they are not already added. Detects ``.ttf`` and ``.otf`` files
    -- see `this link \
<https://gree2.github.io/python/2015/04/27/python-change-matplotlib-font-on-mac>`__
    for a guide on converting various other font file types to ``.ttf`` and
    ``.otf`` for use with matplotlib."""
    # Add proplot path to TTFLIST and rebuild cache *only if necessary*
    # * Nice gallery of sans-serif fonts:
    #   https://www.lifewire.com/classic-sans-serif-fonts-clean-appearance-1077406 # noqa
    # * Sources for downloading more fonts:
    #   https://fonts.google.com/?category=Sans+Serif
    #   https://www.cufonfonts.com
    # WARNING: If you include a font file with an unrecognized style,
    # matplotlib may use that font instead of the 'normal' one! Valid styles:
    # 'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman',
    # 'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'
    # https://matplotlib.org/api/font_manager_api.html
    # For macOS the only fonts with 'Thin' in one of the .ttf file names
    # are Helvetica Neue and .SF NS Display Condensed. Never try to use these!
    paths = ':'.join(_get_data_paths('fonts')[::-1])  # user paths come first
    if 'TTFPATH' not in os.environ:
        os.environ['TTFPATH'] = paths
    elif paths not in os.environ['TTFPATH']:
        os.environ['TTFPATH'] += (':' + paths)

    # Detect user-input .ttc fonts
    import matplotlib.font_manager as mfonts
    fnames_proplot = {*mfonts.findSystemFonts(paths.split(':'))}
    fnames_proplot_ttc = {
        file for file in fnames_proplot if os.path.splitext(file)[1] == '.ttc'
    }
    if fnames_proplot_ttc:
        _warn_proplot(
            'Ignoring the following .ttc fonts because they cannot be '
            'saved into PDF or EPS files (see matplotlib issue #3135): '
            + ', '.join(map(repr, sorted(fnames_proplot_ttc)))
            + '. Please consider expanding them into separate .ttf files.'
        )

    # Rebuild font cache only if necessary! Can be >50% of total import time!
    fnames_all = {font.fname for font in mfonts.fontManager.ttflist}
    fnames_proplot -= fnames_proplot_ttc
    if not fnames_all >= fnames_proplot:
        _warn_proplot('Rebuilding font cache.')
        if hasattr(mfonts.fontManager, 'addfont'):
            for fname in fnames_proplot:
                mfonts.fontManager.addfont(fname)
            mfonts.json_dump(mfonts.fontManager, mfonts._fmcache)
        else:
            mfonts._rebuild()

    # Remove ttc files *after* rebuild
    mfonts.fontManager.ttflist = [
        font for font in mfonts.fontManager.ttflist
        if os.path.splitext(font.fname)[1] != '.ttc'
    ]

    # Populate font name lists, with proplot fonts *first*
    fonts_proplot = sorted({
        font.name for font in mfonts.fontManager.ttflist
        if any(path in font.fname for path in paths.split(':'))
    })
    fonts_system = sorted({
        font.name for font in mfonts.fontManager.ttflist
        if not any(path in font.fname for path in paths.split(':'))
    })
    fonts[:] = [*fonts_proplot, *fonts_system]


def _draw_bars(names, *, source, unknown='User', length=4.0, width=0.2):
    """
    Draw colorbars for "colormaps" and "color cycles". This is called by
    `show_cycles` and `show_cmaps`.
    """
    # Categorize the input names
    cmapdict = {}
    names_all = list(map(str.lower, names))
    names_known = list(map(str.lower, sum(map(list, source.values()), [])))
    names_unknown = [name for name in names if name not in names_known]
    if names_unknown:
        cmapdict[unknown] = names_unknown
    for cat, names in source.items():
        names_cat = [name for name in names if name.lower() in names_all]
        if names_cat:
            cmapdict[cat] = names_cat

    # Draw figure
    from . import subplots
    naxs = len(cmapdict) + sum(map(len, cmapdict.values()))
    fig, axs = subplots(
        nrows=naxs, axwidth=length, axheight=width,
        share=0, hspace=0.03,
    )
    iax = -1
    nheads = nbars = 0  # for deciding which axes to plot in
    for cat, names in cmapdict.items():
        nheads += 1
        for imap, name in enumerate(names):
            iax += 1
            if imap + nheads + nbars > naxs:
                break
            ax = axs[iax]
            if imap == 0:  # allocate this axes for title
                iax += 1
                ax.set_visible(False)
                ax = axs[iax]
            ax.colorbar(  # TODO: support this in public API
                mcm.cmap_d[name], loc='_fill',
                orientation='horizontal', locator='null', linewidth=0
            )
            ax.text(
                0 - (rcParams['axes.labelpad'] / 72) / length, 0.45, name,
                ha='right', va='center', transform='axes',
            )
            if imap == 0:
                ax.set_title(cat)
        nbars += len(names)


def show_channels(
    *args, N=100, rgb=True, saturation=True, minhue=0,
    maxsat=500, width=100, axwidth=1.7
):
    """
    Show how arbitrary colormap(s) vary with respect to the hue, chroma,
    luminance, HSL saturation, and HPL saturation channels, and optionally
    the red, blue and green channels. Adapted from `this example \
<https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html#lightness-of-matplotlib-colormaps>`__.

    Parameters
    ----------
    *args : colormap-spec, optional
        Positional arguments are colormap names or objects. Default is
        :rc:`image.cmap`.
    N : int, optional
        The number of markers to draw for each colormap.
    rgb : bool, optional
        Whether to also show the red, green, and blue channels in the bottom
        row. Default is ``True``.
    saturation : bool, optional
        Whether to show the HSL and HPL saturation channels alongside the
        raw chroma.
    minhue : float, optional
        The minimum hue. This lets you rotate the hue plot cyclically.
    maxsat : float, optional
        The maximum saturation. Use this to truncate large saturation values.
    width : int, optional
        The width of each colormap line in points.
    axwidth : int or str, optional
        The width of each subplot. Passed to `~proplot.subplots.subplots`.

    Returns
    -------
    `~proplot.subplots.Figure`
        The figure.
    """
    # Figure and plot
    from . import subplots
    if not args:
        raise ValueError(f'At least one positional argument required.')
    array = [[1, 1, 2, 2, 3, 3]]
    labels = ('Hue', 'Chroma', 'Luminance')
    if saturation:
        array += [[0, 4, 4, 5, 5, 0]]
        labels += ('HSL saturation', 'HPL saturation')
    if rgb:
        array += [np.array([4, 4, 5, 5, 6, 6]) + 2 * int(saturation)]
        labels += ('Red', 'Green', 'Blue')
    fig, axs = subplots(
        array=array, span=False, share=1,
        axwidth=axwidth, axpad='1em',
    )
    # Iterate through colormaps
    mc, ms, mp = 0, 0, 0
    cmaps = []
    for cmap in args:
        # Get colormap and avoid registering new names
        name = cmap if isinstance(cmap, str) else getattr(cmap, 'name', None)
        cmap = Colormap(cmap, N=N)  # arbitrary cmap argument
        if name is not None:
            cmap.name = name
        cmap._init()
        cmaps.append(cmap)
        # Get clipped RGB table
        x = np.linspace(0, 1, N)
        lut = cmap._lut[:-3, :3].copy()
        rgb_data = lut.T  # 3 by N
        hcl_data = np.array([to_xyz(color, space='hcl')
                             for color in lut]).T  # 3 by N
        hsl_data = [to_xyz(color, space='hsl')[1] for color in lut]
        hpl_data = [to_xyz(color, space='hpl')[1] for color in lut]
        # Plot channels
        # If rgb is False, the zip will just truncate the other iterables
        data = (*hcl_data,)
        if saturation:
            data += (hsl_data, hpl_data)
        if rgb:
            data += (*rgb_data,)
        for ax, y, label in zip(axs, data, labels):
            ylim, ylocator = None, None
            if label in ('Red', 'Green', 'Blue'):
                ylim = (0, 1)
                ylocator = 0.2
            elif label == 'Luminance':
                ylim = (0, 100)
                ylocator = 20
            elif label == 'Hue':
                ylim = (minhue, minhue + 360)
                ylocator = 90
                y = y - 720
                for _ in range(3):  # rotate up to 1080 degrees
                    y[y < minhue] += 360
            else:
                if label == 'Chroma':
                    mc = max(min(max(mc, max(y)), maxsat), 100)
                    m = mc
                elif 'HSL' in label:
                    ms = max(min(max(ms, max(y)), maxsat), 100)
                    m = ms
                else:
                    mp = max(min(max(mp, max(y)), maxsat), 100)
                    m = mp
                ylim = (0, m)
                ylocator = ('maxn', 5)
            ax.scatter(x, y, c=x, cmap=cmap, s=width, linewidths=0)
            ax.format(title=label, ylim=ylim, ylocator=ylocator)
    # Formatting
    suptitle = ', '.join(repr(cmap.name) for cmap in cmaps[:-1]) + (
        ', and ' if len(cmaps) > 2 else ' and ' if len(cmaps) == 2 else ' '
    ) + f'{repr(cmaps[-1].name)} colormap' + ('s' if len(cmaps) > 1 else '')
    axs.format(
        xlocator=0.25, xformatter='null',
        suptitle=f'{suptitle} by channel', ylim=None, ytickminor=False,
    )
    # Colorbar on the bottom
    for cmap in cmaps:
        fig.colorbar(cmap,
                     loc='b', span=(2, 5),
                     locator='null', label=cmap.name, labelweight='bold')
    return fig


def show_colorspaces(luminance=None, saturation=None, hue=None, axwidth=2):
    """
    Generate hue-saturation, hue-luminance, and luminance-saturation
    cross-sections for the HCL, HSL, and HPL colorspaces.

    Parameters
    ----------
    luminance : float, optional
        If passed, saturation-hue cross-sections are drawn for
        this luminance. Must be between ``0` and ``100``. Default is ``50``.
    saturation : float, optional
        If passed, luminance-hue cross-sections are drawn for this
        saturation. Must be between ``0` and ``100``.
    hue : float, optional
        If passed, luminance-saturation cross-sections
        are drawn for this hue. Must be between ``0` and ``360``.
    axwidth : str or float, optional
        Average width of each subplot. Units are interpreted by
        `~proplot.utils.units`.

    Returns
    -------
    `~proplot.subplots.Figure`
        The figure.
    """
    # Get colorspace properties
    hues = np.linspace(0, 360, 361)
    sats = np.linspace(0, 120, 120)
    lums = np.linspace(0, 99.99, 101)
    if luminance is None and saturation is None and hue is None:
        luminance = 50
    if luminance is not None:
        hsl = np.concatenate((
            np.repeat(hues[:, None], len(sats), axis=1)[..., None],
            np.repeat(sats[None, :], len(hues), axis=0)[..., None],
            np.ones((len(hues), len(sats)))[..., None] * luminance,
        ), axis=2)
        suptitle = f'Hue-saturation cross-section for luminance {luminance}'
        xlabel, ylabel = 'hue', 'saturation'
        xloc, yloc = 60, 20
    elif saturation is not None:
        hsl = np.concatenate((
            np.repeat(hues[:, None], len(lums), axis=1)[..., None],
            np.ones((len(hues), len(lums)))[..., None] * saturation,
            np.repeat(lums[None, :], len(hues), axis=0)[..., None],
        ), axis=2)
        suptitle = f'Hue-luminance cross-section for saturation {saturation}'
        xlabel, ylabel = 'hue', 'luminance'
        xloc, yloc = 60, 20
    elif hue is not None:
        hsl = np.concatenate((
            np.ones((len(lums), len(sats)))[..., None] * hue,
            np.repeat(sats[None, :], len(lums), axis=0)[..., None],
            np.repeat(lums[:, None], len(sats), axis=1)[..., None],
        ), axis=2)
        suptitle = 'Luminance-saturation cross-section'
        xlabel, ylabel = 'luminance', 'saturation'
        xloc, yloc = 20, 20

    # Make figure, with black indicating invalid values
    # Note we invert the x-y ordering for imshow
    from . import subplots
    fig, axs = subplots(
        ncols=3, share=0, axwidth=axwidth, aspect=1, axpad=0.05
    )
    for ax, space in zip(axs, ('hcl', 'hsl', 'hpl')):
        rgba = np.ones((*hsl.shape[:2][::-1], 4))  # RGBA
        for j in range(hsl.shape[0]):
            for k in range(hsl.shape[1]):
                rgb_jk = to_rgb(hsl[j, k, :].flat, space)
                if not all(0 <= c <= 1 for c in rgb_jk):
                    rgba[k, j, 3] = 0  # black cell
                else:
                    rgba[k, j, :3] = rgb_jk
        ax.imshow(rgba, origin='lower', aspect='auto')
        ax.format(
            xlabel=xlabel, ylabel=ylabel, suptitle=suptitle,
            grid=False, xtickminor=False, ytickminor=False,
            xlocator=xloc, ylocator=yloc, facecolor='k',
            title=space.upper(), titleweight='bold'
        )
    return fig


def show_colors(nhues=17, minsat=20):
    """
    Generate tables of the registered color names. Adapted from
    `this example <https://matplotlib.org/examples/color/named_colors.html>`__.

    Parameters
    ----------
    nhues : int, optional
        The number of breaks between hues for grouping "like colors" in the
        color table.
    minsat : float, optional
        The threshold saturation, between ``0`` and ``100``, for designating
        "gray colors" in the color table.

    Returns
    -------
    figs : list of `~proplot.subplots.Figure`
        The figure.
    """
    # Test used to "categories" colors
    breakpoints = np.linspace(0, 360, nhues)
    def _color_filter(i, hcl):  # noqa: E306
        gray = hcl[1] <= minsat
        if i == 0:
            return gray
        color = breakpoints[i - 1] <= hcl[0] < breakpoints[i]
        if i == nhues - 1:
            color = color or color == breakpoints[i]  # endpoint inclusive
        return not gray and color

    # Draw figures for different groups of colors
    figs = []
    from . import subplots
    for cats in (
            ('opencolor',),
            tuple(name for name in colors if name not in ('css', 'opencolor'))
    ):
        # Dictionary of colors for that category
        data = {}
        for cat in cats:
            for color in colors[cat]:
                data[color] = mcolors.colorConverter.colors[color]

        # Group colors together by discrete range of hue, then sort by value
        # For opencolors this is not necessary
        if cats == ('opencolor',):
            wscale = 0.5
            swatch = 1.5
            nrows, ncols = 10, len(COLORS_OPEN)  # rows and columns
            names = np.array([
                [name + str(i) for i in range(nrows)]
                for name in COLORS_OPEN
            ])
            nrows = nrows * 2
            ncols = (ncols + 1) // 2
            names.resize((ncols, nrows))

        # Get colors in perceptally uniform space, then group based on hue
        # thresholds
        else:
            ncols = 4
            wscale = 0.8
            swatch = 1.2
            hclpairs = [
                (name, to_xyz(color, COLORS_SPACE))
                for name, color in data.items()
            ]
            hclpairs = [
                sorted(
                    [pair for pair in hclpairs if _color_filter(i, pair[1])],
                    key=lambda x: x[1][2]
                )
                for i in range(nhues)
            ]
            names = np.array([
                name for ipairs in hclpairs for name, _ in ipairs
            ])
            nrows = len(names) // ncols + 1
            names.resize((ncols, nrows))

        # Draw swatches as lines
        fig, ax = subplots(
            width=8 * wscale * (ncols / 4),
            height=5 * (nrows / 40),
            left=0, right=0, top=0, bottom=0, tight=False
        )
        X, Y = fig.get_dpi() * fig.get_size_inches()  # size in dots
        hsep, wsep = Y / (nrows + 1), X / ncols  # height and width in dots
        for col, inames in enumerate(names):
            for row, name in enumerate(inames):
                if not name:
                    continue
                y = Y - hsep * (row + 1)
                xi = wsep * (col + 0.05)
                xf = wsep * (col + 0.25 * swatch)
                yline = y + hsep * 0.1
                xtext = wsep * (col + 0.25 * swatch + 0.03 * swatch)
                ax.text(xtext, y, name, ha='left', va='center')
                ax.plot(
                    [xi, xf], [yline, yline],
                    color=data[name], lw=hsep * 0.6,
                    solid_capstyle='butt',  # do not stick out
                )

        # Apply formatting
        ax.format(
            xlim=(0, X), ylim=(0, Y),
            grid=False, yloc='neither', xloc='neither'
        )
        figs.append(fig)
    return figs


def show_cmaps(*args, N=None, **kwargs):
    """
    Generate a table of the registered colormaps or the input colormaps
    categorized by source. Adapted from `this example \
<http://matplotlib.org/examples/color/colormaps_reference.html>`__.

    Parameters
    ----------
    *args : colormap-spec, optional
        Colormap names or objects.
    N : int, optional
        The number of levels in each colorbar. Default is
        :rc:`image.lut`.
    unknown : str, optional
        Category name for colormaps that are unknown to ProPlot. The
        default is ``'User'``.
    length : float or str, optional
        The length of the colorbars. Units are interpreted by
        `~proplot.utils.units`.
    width : float or str, optional
        The width of the colorbars. Units are interpreted by
        `~proplot.utils.units`.

    Returns
    -------
    `~proplot.subplots.Figure`
        The figure.
    """
    # Have colormaps separated into categories
    N = _notNone(N, rcParams['image.lut'])
    if args:
        names = [Colormap(cmap, N=N).name for cmap in args]
    else:
        names = [
            name for name in mcm.cmap_d.keys() if
            isinstance(mcm.cmap_d[name], LinearSegmentedColormap)
        ]

    # Return figure of colorbars
    kwargs.setdefault('source', CMAPS_TABLE)
    return _draw_bars(names, **kwargs)


def show_cycles(*args, **kwargs):
    """
    Generate a table of registered color cycles or the input color cycles
    categorized by source. Adapted from `this example \
<http://matplotlib.org/examples/color/colormaps_reference.html>`__.

    Parameters
    ----------
    *args : colormap-spec, optional
        Cycle names or objects.
    unknown : str, optional
        Category name for cycles that are unknown to ProPlot. The
        default is ``'User'``.
    length : float or str, optional
        The length of the colorbars. Units are interpreted by
        `~proplot.utils.units`.
    width : float or str, optional
        The width of the colorbars. Units are interpreted by
        `~proplot.utils.units`.

    Returns
    -------
    `~proplot.subplots.Figure`
        The figure.
    """
    # Get the list of cycles
    if args:
        names = [cmap.name for cmap in args]
    else:
        names = [
            name for name in mcm.cmap_d.keys() if
            isinstance(mcm.cmap_d[name], ListedColormap)
        ]

    # Return figure of colorbars
    kwargs.setdefault('source', CYCLES_TABLE)
    return _draw_bars(names, **kwargs)


def show_fonts(*args, family=None, text=None, size=12):
    """
    Generate a table of fonts. If a glyph for a particular font is unavailable,
    it is replaced with the "¤" dummy character.

    Parameters
    ----------
    *args
        The font name(s). If none are provided and the `family` keyword
        argument was not provided, the *available* :rcraw:`font.sans-serif`
        fonts and the fonts in your ``.proplot/fonts`` folder are shown.
    family : {'serif', 'sans-serif', 'monospace', 'cursive', 'fantasy', \
'tex-gyre'}, optional
        If provided, the *available* fonts in the corresponding families
        are shown. The fonts belonging to these families are listed under the
        :rcraw:`font.serif`, :rcraw:`font.sans-serif`, :rcraw:`font.monospace`,
        :rcraw:`font.cursive`, and :rcraw:`font.fantasy` settings. The special
        family ``'tex-gyre'`` draws the `TeX Gyre \
<http://www.gust.org.pl/projects/e-foundry/tex-gyre>`__ fonts.
    text : str, optional
        The sample text. The default sample text includes the Latin letters,
        Greek letters, Arabic numerals, and some simple mathematical symbols.
    size : float, optional
        The font size in points.
    """
    from . import subplots
    import matplotlib.font_manager as mfonts
    if not args and family is None:
        # User fonts and sans-serif fonts. Note all proplot sans-serif fonts
        # are added to 'font.sans-serif' by default
        args = sorted({
            font.name for font in mfonts.fontManager.ttflist
            if font.name in rcParams['font.sans-serif']
            or _get_data_paths('fonts')[1] == os.path.dirname(font.fname)
        })
    elif family is not None:
        options = (
            'serif', 'sans-serif', 'monospace', 'cursive', 'fantasy',
            'tex-gyre',
        )
        if family not in options:
            raise ValueError(
                f'Invalid family {family!r}. Options are: '
                + ', '.join(map(repr, options)) + '.'
            )
        if family == 'tex-gyre':
            family_fonts = (
                'TeX Gyre Adventor',
                'TeX Gyre Bonum',
                'TeX Gyre Cursor',
                'TeX Gyre Chorus',
                'TeX Gyre Heros',
                'TeX Gyre Pagella',
                'TeX Gyre Schola',
                'TeX Gyre Termes',
            )
        else:
            family_fonts = rcParams['font.' + family]
        args = (
            *args, *sorted({
                font.name for font in mfonts.fontManager.ttflist
                if font.name in family_fonts
            })
        )

    # Text
    if text is None:
        text = (
            'the quick brown fox jumps over a lazy dog' '\n'
            'THE QUICK BROWN FOX JUMPS OVER A LAZY DOG' '\n'
            '(0) + {1\N{DEGREE SIGN}} \N{MINUS SIGN} [2*] - <3> / 4,0 '
            r'$\geq\gg$ 5.0 $\leq\ll$ ~6 $\times$ 7 '
            r'$\equiv$ 8 $\approx$ 9 $\propto$' '\n'
            r'$\alpha\beta$ $\Gamma\gamma$ $\Delta\delta$ '
            r'$\epsilon\zeta\eta$ $\Theta\theta$ $\kappa\mu\nu$ '
            r'$\Lambda\lambda$ $\Pi\pi$ $\xi\rho\tau\chi$ $\Sigma\sigma$ '
            r'$\Phi\phi$ $\Psi\psi$ $\Omega\omega$ !?&#%'
        )

    # Create figure
    f, axs = subplots(
        ncols=1, nrows=len(args), space=0,
        axwidth=4.5, axheight=1.2 * (text.count('\n') + 2.5) * size / 72,
        fallback_to_cm=False
    )
    axs.format(
        xloc='neither', yloc='neither',
        xlocator='null', ylocator='null', alpha=0
    )
    for i, ax in enumerate(axs):
        font = args[i]
        ax.text(
            0, 0.5, f'{font}:\n{text}',
            fontfamily=font, fontsize=size,
            weight='normal', ha='left', va='center'
        )
    return f


# Apply custom changes
if 'Greys' in mcm.cmap_d:  # 'Murica (and consistency with registered colors)
    mcm.cmap_d['Grays'] = mcm.cmap_d.pop('Greys')
if 'Spectral' in mcm.cmap_d:  # make spectral go from 'cold' to 'hot'
    mcm.cmap_d['Spectral'] = mcm.cmap_d['Spectral'].reversed(name='Spectral')
for _name in CMAPS_TABLE['Matplotlib originals']:
    if _name == 'twilight_shifted':  # we can generate shifted maps on the fly
        mcm.cmap_d.pop(_name, None)
    else:  # convert ListedColormaps to LinearSegmentedColormaps
        _cmap = mcm.cmap_d.get(_name, None)
        if _cmap and isinstance(_cmap, mcolors.ListedColormap):
            mcm.cmap_d.pop(_name, None)
            mcm.cmap_d[_name] = LinearSegmentedColormap.from_list(
                _name, _cmap.colors, cyclic=('twilight' in _name)
            )
for _cat in ('MATLAB', 'GNUplot', 'GIST', 'Other'):
    for _name in CMAPS_TABLE[_cat]:
        mcm.cmap_d.pop(_name, None)

# Initialize customization folders
_rc_folder = os.path.join(os.path.expanduser('~'), '.proplot')
if not os.path.isdir(_rc_folder):
    os.mkdir(_rc_folder)
for _rc_sub in ('cmaps', 'cycles', 'colors', 'fonts'):
    _rc_sub = os.path.join(_rc_folder, _rc_sub)
    if not os.path.isdir(_rc_sub):
        os.mkdir(_rc_sub)

# Apply monkey patches to top level modules
if not isinstance(mcm.cmap_d, CmapDict):
    _dict = {
        key: value for key, value in mcm.cmap_d.items() if key[-2:] != '_r'
    }
    mcm.cmap_d = CmapDict(_dict)
if not isinstance(mcolors._colors_full_map, _ColorMappingOverride):
    _map = _ColorMappingOverride(mcolors._colors_full_map)
    mcolors._colors_full_map = _map
    mcolors.colorConverter.cache = _map.cache  # re-instantiate
    mcolors.colorConverter.colors = _map  # re-instantiate

#: List of registered colormap names.
cmaps = []

#: List of registered color cycle names.
cycles = []

#: Lists of registered color names by category.
colors = {}

#: Registered font names.
fonts = []

# Call driver funcs
register_colors()
register_cmaps()
register_cycles()
register_fonts()

#: Dictionary of possible normalizers. See `Norm` for a table.
normalizers = {
    'none': mcolors.NoNorm,
    'null': mcolors.NoNorm,
    'zero': MidpointNorm,
    'midpoint': MidpointNorm,
    'segments': LinearSegmentedNorm,
    'segmented': LinearSegmentedNorm,
    'log': mcolors.LogNorm,
    'linear': mcolors.Normalize,
    'power': mcolors.PowerNorm,
    'symlog': mcolors.SymLogNorm,
}
