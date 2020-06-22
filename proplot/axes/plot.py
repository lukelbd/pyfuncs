#!/usr/bin/env python3
"""
The plotting wrappers that add functionality to various `~matplotlib.axes.Axes`
methods. "Wrapped" `~matplotlib.axes.Axes` methods accept the additional keyword
arguments documented by the wrapper function. In a future version, these features will
be documented on the individual `~proplot.axes.Axes` methods themselves.
"""
import functools
import inspect
import re
import sys
from numbers import Number

import matplotlib.artist as martist
import matplotlib.axes as maxes
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.container as mcontainer
import matplotlib.contour as mcontour
import matplotlib.legend as mlegend
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import numpy.ma as ma

from .. import colors as pcolors
from .. import constructor
from ..config import rc
from ..internals import ic  # noqa: F401
from ..internals import _not_none, docstring, warnings
from ..utils import edges, edges2d, to_rgb, to_xyz, units

try:
    from cartopy.crs import PlateCarree
except ModuleNotFoundError:
    PlateCarree = object


docstring.snippets['plot.1d_args'] = """
*args : (x,) or (x, y)
    The data passed as positional arguments. Interpreted as follows:

    * If only *y* data are passed, try to infer the *x* data from the
      `~pandas.Series` or `~pandas.DataFrame` indices or the `~xarray.DataArray`
      coordinate variables. Otherwise, use ``np.arange(0, y.shape[0])``.
    * If a 2D array of *y* data are passed, each column of data is
      plotted in succession (except for ``boxplot`` and ``violinplot``, where
      each column is interpreted as a separate distribution).

    ProPlot also tries to infer the default metadata from the *x* and *y* data
    containers; see `autoformat` for details.
"""

docstring.snippets['plot.2d_args'] = """
*args : (z1, ...) or (x, y, z1, ...)
    The data passed as positional arguments. Interpreted as follows:

    * If only *z* data are passed, try to infer the *x* and *y* data from
      the `~pandas.DataFrame` indices and columns or the `~xarray.DataArray`
      coordinate variable. Otherwise, use ``np.arange(0, y.shape[0])`` and
      ``np.arange(0, y.shape[1])``.
    * For ``pcolor`` and ``pcolormesh``, coordinate *edges* are calculated
      if *centers* were provided using `~proplot.utils.edges` or
      `~proplot.utils.edges2d`. For all other methods, coordinate *centers*
      are calculated if *edges* were provided.
"""

docstring.snippets['plot.2d_kwargs'] = """
order : {{'C', 'F'}}, optional
    If ``'C'``, arrays should be shaped ``(y, x)``. If ``'F'``, arrays
    should be shaped ``(x, y)``. Default is ``'C'``.
globe : bool, optional
    *For `~proplot.axes.GeoAxes` only*. Whether to ensure global coverage.
    Default is ``False``. When ``True`` this does the following:

    #. Interpolates input data to the North and South poles by setting the data
       values at the poles to the mean from latitudes nearest each pole.
    #. Makes meridional coverage "circular", i.e. the last longitude coordinate
       equals the first longitude coordinate plus 360\N{DEGREE SIGN}.
    #. (*For `~proplot.axes.BasemapAxes` only*.) 1D longitude vectors are cycled to
       fit within the map edges. For example, if the projection central longitude
       is 90\N{DEGREE SIGN}, the data is shifted so that it spans -90\N{DEGREE SIGN}
       to 270\N{DEGREE SIGN}.

extend : {{'neither', 'min', 'max', 'both'}}, optional
    Where to assign unique colors to out-of-bounds data and draw
    "extensions" (triangles, by default) on the colorbar.
edgefix : bool, optional
    Whether to fix the the `white-lines-between-filled-contours \
<https://stackoverflow.com/q/8263769/4970632>`__
    and `white-lines-between-pcolor-rectangles \
<https://stackoverflow.com/q/27092991/4970632>`__
    issues. This slows down figure rendering by a bit. Default is
    :rc:`image.edgefix`.
lw, linewidth, linewidths
    The width of `~matplotlib.axes.Axes.contour` lines and
    `~proplot.axes.Axes.parametric` lines, or the width of lines
    *between* `~matplotlib.axes.Axes.pcolor` boxes,
    `~matplotlib.axes.Axes.pcolormesh` boxes, and
    `~matplotlib.axes.Axes.contourf` filled contours.
ls, linestyle, linestyles
    As above, but for the line style.
color, colors, edgecolor, edgecolors
    As above, but for the line color. For `~matplotlib.axes.Axes.contourf`
    plots, if you provide `colors` without specifying the `linewidths`
    or `linestyles`, this argument is used to manually specify the *fill
    colors*. See the `~matplotlib.axes.Axes.contourf` documentation for
    details.
"""

docstring.snippets['plot.autoformat'] = """
autoformat : bool, optional
    Whether to automatically modify *x* axis labels, *y* axis labels, axis
    formatters, axes titles, colorbar labels, and legend labels when
    a `~pandas.Series`, `~pandas.DataFrame`, or `~xarray.DataArray` is passed
    to the plotting command. Default is :rc:`autoformat`.
"""

docstring.snippets['plot.auto_colorbar'] = """
colorbar : bool, int, or str, optional
    If not ``None``, this is a location specifying where to draw an *inset*
    or *panel* colorbar from the resulting object. If ``True``, the
    default location is used. Valid locations are described in
    `~proplot.axes.Axes.colorbar`.
colorbar_kw : dict-like, optional
    Ignored if `colorbar` is ``None``. Extra keyword args for our call
    to `~proplot.axes.Axes.colorbar`.
"""

docstring.snippets['plot.auto_legend'] = """
legend : bool, int, or str, optional
    If not ``None``, this is a location specifying where to draw an *inset*
    or *panel* legend from the resulting handle(s). If ``True``, the
    default location is used. Valid locations are described in
    `~proplot.axes.Axes.legend`.
legend_kw : dict-like, optional
    Ignored if `legend` is ``None``. Extra keyword args for our call
    to `~proplot.axes.Axes.legend`.
"""

docstring.snippets['plot.auto_labels'] = """
labels : bool, optional
    Whether to add labels. For contour plots, labels are added with
    `~matplotlib.axes.Axes.clabel`. For pcolor plots, labels are added
    with `~matplotlib.axes.Axes.text` by placing text at the center of
    each grid box. For filled contour and pcolor plots, the text will
    be colored black when the luminance of the underlying color is >50%%
    and white otherwise.
labels_kw : dict-like, optional
    Ignored if `labels` is ``False``. Extra keyword arguments for the labels.
    For contour plots, these are passed to `~matplotlib.axes.Axes.clabel`.
    or applied to the `~matplotlib.text.Text` objects. For pcolor plots, these
    can only be `~matplotlib.text.Text` properties.
precision : int, optional
    Maximum number of decimal places for the number labels. Trailing
    zeros will be trimmed by default.
fmt : format-spec, optional
    Passed to the `~proplot.constructor.Norm` constructor, used to format
    number labels. You can also use the `precision` keyword arg.
"""

docstring.snippets['plot.cmap_args'] = """
cmap : colormap spec, optional
    The colormap specifer, passed to the `~proplot.constructor.Colormap`
    constructor.
cmap_kw : dict-like, optional
    Passed to `~proplot.constructor.Colormap`.
norm : normalizer spec, optional
    The colormap normalizer, used to warp data before passing it
    to `~proplot.colors.DiscreteNorm`. This is passed to the
    `~proplot.constructor.Norm` constructor.
norm_kw : dict-like, optional
    Passed to `~proplot.constructor.Norm`.
vmin, vmax : float, optional
    Used to determine level locations if `levels` is an integer. Actual
    levels may not fall exactly on `vmin` and `vmax`, but the minimum
    level will be no smaller than `vmin` and the maximum level will be
    no larger than `vmax`. If `vmin` or `vmax` is not provided, the
    minimum and maximum data values are used.
levels, N : int or list of float, optional
    The number of level edges, or a list of level edges. If the former,
    `locator` is used to generate this many levels at "nice" intervals.
    If the latter, the levels should be monotonically increasing or
    decreasing (note that decreasing levels will only work with ``pcolor``
    plots, not ``contour`` plots). Default is :rc:`image.levels`.
    Note this means you can now discretize your colormap colors in a
    ``pcolor`` plot just like with ``contourf``.
values : int or list of float, optional
    The number of level centers, or a list of level centers. If provided,
    levels are inferred using `~proplot.utils.edges`. This will override
    any `levels` input.
symmetric : bool, optional
    If ``True``, automatically generated levels are symmetric about zero.
locator : locator-spec, optional
    The locator used to determine level locations if `levels` or `values`
    is an integer and `vmin` and `vmax` were not provided. Passed to the
    `~proplot.constructor.Locator` constructor. Default is
    `~matplotlib.ticker.MaxNLocator` with ``levels`` integer levels.
locator_kw : dict-like, optional
    Passed to `~proplot.constructor.Locator`.
"""

docstring.snippets['plot.cmap_note'] = """
The `~proplot.colors.DiscreteNorm` normalizer, used with all colormap
plots, makes sure that your levels always span the full range of colors
in the colormap, whether `extend` is set to ``'min'``, ``'max'``,
``'neither'``, or ``'both'``. By default, when `extend` is not ``'both'``,
matplotlib seems to just cut off the most intense colors (reserved for
coloring "out of bounds" data), even though they are not being used.

This could also be done by limiting the number of colors in the colormap
lookup table by selecting a smaller ``N`` (see
`~matplotlib.colors.LinearSegmentedColormap`). Instead, we prefer to
always build colormaps with high resolution lookup tables, and leave it
to the `~matplotlib.colors.Normalize` instance to handle discretization
of the color selections.
"""

docstring.snippets['plot.cycle_args'] = """
cycle : cycle-spec, optional
    The cycle specifer, passed to the `~proplot.constructor.Cycle`
    constructor. If the returned list of colors is unchanged from the
    current axes color cycler, the axes cycle will **not** be reset to the
    first position.
cycle_kw : dict-like, optional
    Passed to `~proplot.constructor.Cycle`.
label : float or str, optional
    The legend label to be used for this plotted element.
labels, values : list of float or list of str, optional
    Used with 2D input arrays. The legend labels or colorbar coordinates
    for each column in the array. Can be numeric or string, and must match
    the number of columns in the 2D array.
errobjs : `~matplotlib.artist.Artist` or list thereof, optional
    Error bar objects to add to the legend. This is used internally and
    should not be necessary for users. See `indicate_error`.
"""

docstring.snippets['plot.colorbar_args'] = """
mappable : mappable, list of plot handles, list of color-spec, or colormap-spec
    There are four options here:

    1. A mappable object. Basically, any object with a ``get_cmap`` method,
        like the objects returned by `~matplotlib.axes.Axes.contourf` and
        `~matplotlib.axes.Axes.pcolormesh`.
    2. A list of "plot handles". Basically, any object with a ``get_color``
        method, like `~matplotlib.lines.Line2D` instances. A colormap will
        be generated from the colors of these objects, and colorbar levels
        will be selected using `values`.  If `values` is ``None``, we try
        to infer them by converting the handle labels returned by
        `~matplotlib.artist.Artist.get_label` to `float`. Otherwise, it is
        set to ``np.linspace(0, 1, len(mappable))``.
    3. A list of hex strings, color string names, or RGB tuples. A colormap
        will be generated from these colors, and colorbar levels will be
        selected using `values`. If `values` is ``None``, it is set to
        ``np.linspace(0, 1, len(mappable))``.
    4. A `~matplotlib.colors.Colormap` instance. In this case, a colorbar
        will be drawn using this colormap and with levels determined by
        `values`. If `values` is ``None``, it is set to
        ``np.linspace(0, 1, cmap.N)``.

values : list of float, optional
    Ignored if `mappable` is a mappable object. This maps each color or
    plot handle in the `mappable` list to numeric values, from which a
    colormap and normalizer are constructed.
norm : normalizer spec, optional
    Ignored if `values` is ``None``. The normalizer for converting `values`
    to colormap colors. Passed to `~proplot.constructor.Norm`.
norm_kw : dict-like, optional
    The normalizer settings. Passed to `~proplot.constructor.Norm`.
"""

docstring.snippets['plot.colorbar_kwargs'] = """
extend : {None, 'neither', 'both', 'min', 'max'}, optional
    Direction for drawing colorbar "extensions" (i.e. references to
    out-of-bounds data with a unique color). These are triangles by
    default. If ``None``, we try to use the ``extend`` attribute on the
    mappable object. If the attribute is unavailable, we use ``'neither'``.
extendsize : float or str, optional
    The length of the colorbar "extensions" in *physical units*.
    If float, units are inches. If string, units are interpreted
    by `~proplot.utils.units`. Default is :rc:`colorbar.insetextend`
    for inset colorbars and :rc:`colorbar.extend` for outer colorbars.
reverse : bool, optional
    Whether to reverse the direction of the colorbar.
tickloc, ticklocation : {'bottom', 'top', 'left', 'right'}, optional
    Where to draw tick marks on the colorbar.
tickminor : bool, optional
    Whether to add minor ticks to the colorbar with
    `~matplotlib.colorbar.ColorbarBase.minorticks_on`.
grid : bool, optional
    Whether to draw "gridlines" between each level of the colorbar.
    Default is :rc:`colorbar.grid`.
label, title : str, optional
    The colorbar label. The `title` keyword is also accepted for
    consistency with `legend`.
locator, ticks : locator spec, optional
    Used to determine the colorbar tick positions. Passed to the
    `~proplot.constructor.Locator` constructor.
maxn : int, optional
    Used if `locator` is ``None``. Determines the maximum number of levels
    that are ticked. Default depends on the colorbar length relative
    to the font size. The keyword name "maxn" is meant to mimic
    the `~matplotlib.ticker.MaxNLocator` class name.
locator_kw : dict-like, optional
    The locator settings. Passed to `~proplot.constructor.Locator`.
minorlocator, minorticks, maxn_minor, minorlocator_kw
    As with `locator`, `maxn`, and `locator_kw`, but for the minor ticks.
formatter, ticklabels : formatter spec, optional
    The tick label format. Passed to the `~proplot.constructor.Formatter`
    constructor.
formatter_kw : dict-like, optional
    The formatter settings. Passed to `~proplot.constructor.Formatter`.
rotation : float, optional
    The tick label rotation. Default is ``0``.
edgecolor, linewidth : optional
    The edge color and line width for the colorbar outline.
labelsize, labelweight, labelcolor : optional
    The font size, weight, and color for colorbar label text.
ticklabelsize, ticklabelweight, ticklabelcolor : optional
    The font size, weight, and color for colorbar tick labels.
orientation : {{'horizontal', 'vertical'}}, optional
    The colorbar orientation. You should not have to explicitly set this.
"""

docstring.snippets['plot.error_args'] = """
means : bool, optional
    Whether to plot the means of each column in the input data.
medians : bool, optional
    Whether to plot the medians of each column in the input data.
barstds : (float, float) or bool, optional
    Standard deviation multiples for *thin error bars* with optional whiskers
    (i.e. caps). If ``True``, the default standard deviation multiples ``(-3, 3)``
    are used. This argument is only valid if `means` or `medians` is ``True``.
barpctiles : (float, float) or bool, optional
    As with `barstds`, but instead using *percentiles* for the error bars.
    The percentiles are calculated with `numpy.percentile`. If ``True``, the
    default percentiles ``(0, 100)`` are used.
bardata : 2 x N array or 1D array, optional
    If shape is 2 x N these are the lower and upper bounds for the thin error bars.
    If array is 1D these are the absolute, symmetric deviations from the central
    points.  This should be used if `means` and `medians` are both ``False`` (i.e.
    you did not provide dataset columns from which statistical properties can be
    calculated automatically).
boxstds, boxpctiles, boxdata : optional
    As with `barstds`, `barpctiles`, and `bardata`, but for *thicker error bars*
    representing a smaller interval than the thin error bars. If `boxstds` is
    ``True``, the default standard deviation multiples ``(-1, 1)`` are used.
    If `boxpctiles` is ``True``, the default percentile multiples ``(25, 75)``
    are used (i.e. the interquartile range). When boxes and bars are combined, this
    has the effect of drawing miniature box-and-whisker plots.
shadestds, shadepctiles, shadedata : optional
    As with `barstds`, `barpctiles`, and `bardata`, but using *shading* to indicate
    the error range. If `shadestds` is ``True``, the default standard deviation
    multiples ``(-2, 2)`` are used. If `shadepctiles` is ``True``, the default
    percentile multiples ``(10, 90)`` are used. Shading is generally useful for
    `~matplotlib.axes.Axes.plot` plots and not `~matplotlib.axes.Axes.bar` plots.
fadestds, fadepctiles, fadedata : optional
    As with `shadestds`, `shadepctiles`, and `shadedata`, but for an additional,
    more faded, *secondary* shaded region. If `fadestds` is ``True``, the default
    standard deviation multiples ``(-3, 3)`` are used. If `fadepctiles` is ``True``,
    the default percentile multiples ``(0, 100)`` are used.
barcolor, boxcolor, shadecolor, fadecolor : color-spec, optional
    Colors for the different error indicators. For error bars, the default is
    ``'k'``. For shading, the default behavior is to inherit color from the
    primary `~matplotlib.artist.Artist`.
shadelabel, fadelabel : bool or str, optional
    Labels for the shaded regions to be used as separate legend entries. To toggle
    labels "on" and apply a *default* label, use e.g. ``shadelabel=True``. To apply
    a *custom* label, use e.g. ``shadelabel='label'``. Otherwise, the shading is
    drawn underneath the line and/or marker in the legend entry.
barlinewidth, boxlinewidth, barlw, boxlw : float, optional
    Line widths for the thin and thick error bars, in points. The defaults
    are ``barlw=0.8`` and ``boxlw=4 * barlw``.
boxmarker : bool, optional
    Whether to draw a small marker in the middle of the box denoting the mean or
    median position. Ignored if `boxes` is ``False``. Default is ``True``.
boxmarkercolor : color-spec, optional
    Color for the `boxmarker` marker. Default is ``'w'``.
capsize : float, optional
    The cap size for thin error bars in points.
barzorder, boxzorder, shadezorder, fadezorder : float, optional
    The "zorder" for the thin error bars, thick error bars, and shading.
"""

docstring.snippets['plot.legend_args'] = """
handles : list of `~matplotlib.artist.Artist`, optional
    List of artists instances, or list of lists of artist instances (see
    the `center` keyword). If ``None``, the artists are retrieved with
    `~matplotlib.axes.Axes.get_legend_handles_labels`.
labels : list of str, optional
    Matching list of string labels, or list of lists of string labels (see
    the `center` keywod). If ``None``, the labels are retrieved by calling
    `~matplotlib.artist.Artist.get_label` on each
    `~matplotlib.artist.Artist` in `handles`.
"""

docstring.snippets['legend_kwargs'] = """
ncol, ncols : int, optional
    The number of columns. `ncols` is an alias, added
    for consistency with `~matplotlib.pyplot.subplots`.
order : {'C', 'F'}, optional
    Whether legend handles are drawn in row-major (``'C'``) or column-major
    (``'F'``) order. Analagous to `numpy.array` ordering. For some reason
    ``'F'`` was the original matplotlib default. Default is ``'C'``.
center : bool, optional
    Whether to center each legend row individually. If ``True``, we
    actually draw successive single-row legends stacked on top of each
    other.

    If ``None``, we infer this setting from `handles`. Default is ``True``
    if `handles` is a list of lists; each sublist is used as a *row*
    in the legend. Otherwise, default is ``False``.
label, title : str, optional
    The legend title. The `label` keyword is also accepted, for consistency
    with `colorbar`.
fontsize, fontweight, fontcolor : optional
    The font size, weight, and color for legend text.
color, lw, linewidth, linestyle, dashes, marker, markersize : property-spec, optional
    Properties used to override the legend handles. For example, if you
    want a legend that describes variations in line style ignoring
    variations in color, you might want to use ``color='k'``. For now this
    does not include `facecolor`, `edgecolor`, and `alpha`, because
    `~matplotlib.axes.Axes.legend` uses these keyword args to modify the
    frame properties.
"""

docstring.snippets['plot.negpos_args'] = """
negpos : bool, optional
    Whether to color regions greater than zero with `poscolor` and
    regions less than zero with `negcolor`.
negcolor, poscolor : color-spec, optional
    Colors to use for the negative and positive regions. Ignored if `negpos`
    is ``False``. Defaults are :rc:`negcolor` and :rc:`poscolor`.
"""


def _concatenate_docstrings(func):
    """
    Concatenate docstrings from a matplotlib axes method with a ProPlot axes
    method and obfuscate the call signature to avoid misleading users. Requires
    that ProPlot documentation has no "other parameters", notes, or examples
    sections.
    """
    # NOTE: Originally had idea to use numpydoc.docscrape.NumpyDocString to interpolate
    # docstrings but *enormous* number of assupmtions would go into this.
    # Get matplotlib axes func
    name = func.__name__
    func_orig = getattr(maxes.Axes, name, None)
    if not func_orig:  # should never happen
        return func
    doc_orig = inspect.getdoc(func_orig)
    if not doc_orig:  # should never happen
        return func

    # Prepend summary
    doc = inspect.getdoc(func) or ''  # also dedents
    regex = re.search(r'\.( | *\n|\Z)', doc_orig)
    if regex:
        doc = doc_orig[:regex.start() + 1] + '\n\n' + doc

    # Do not concatenate when running sphinx
    if rc['docstring.hardcopy']:  # True when running sphinx
        func.__doc__ = doc
        return func

    # Obfuscate signature by converting to *args **kwargs. Note this does
    # not change behavior of function! Copy parameters from a dummy function
    # because I'm too lazy to figure out inspect.Parameters API
    # See: https://stackoverflow.com/a/33112180/4970632
    sig = inspect.signature(func)
    sig_obfuscated = inspect.signature(lambda *args, **kwargs: None)
    func.__signature__ = (
        sig.replace(parameters=tuple(sig_obfuscated.parameters.values()))
    )

    # Concatenate docstrings and copy summary
    # Make sure different sections are very visible
    nequal = '=' * len(name)
    doc = f"""
================================{nequal}
proplot.axes.Axes.{name} documentation
================================{nequal}
{doc}
==================================={nequal}
matplotlib.axes.Axes.{name} documentation
==================================={nequal}
{doc_orig}
"""
    func.__doc__ = doc

    # Return
    return func


def _load_objects():
    """
    Delay loading expensive modules. We just want to detect if *input
    arrays* belong to these types -- and if this is the case, it means the
    module has already been imported! So, we only try loading these classes
    within autoformat calls. This saves >~500ms of import time.
    """
    global DataArray, DataFrame, Series, Index, ndarray, ARRAY_TYPES
    ndarray = np.ndarray
    DataArray = getattr(sys.modules.get('xarray', None), 'DataArray', ndarray)
    DataFrame = getattr(sys.modules.get('pandas', None), 'DataFrame', ndarray)
    Series = getattr(sys.modules.get('pandas', None), 'Series', ndarray)
    Index = getattr(sys.modules.get('pandas', None), 'Index', ndarray)
    ARRAY_TYPES = (ndarray, DataArray, DataFrame, Series, Index)


_load_objects()

# Make keywords for styling cmap_args-overridden plots *consistent*
# TODO: Consider deprecating linewidth and linestyle interpretation. Think
# these already have flexible interpretation for all plotting funcs.
STYLE_ARGS_TRANSLATE = {
    'contour': {
        'colors': 'colors',
        'linewidths': 'linewidths',
        'linestyles': 'linestyles',
    },
    'tricontour': {
        'colors': 'colors',
        'linewidths': 'linewidths',
        'linestyles': 'linestyles',
    },
    'pcolor': {
        'colors': 'edgecolors',
        'linewidths': 'linewidth',
        'linestyles': 'linestyle',
    },
    'pcolormesh': {
        'colors': 'edgecolors',
        'linewidths': 'linewidth',
        'linestyles': 'linestyle',
    },
    'pcolorfast': {
        'colors': 'edgecolors',
        'linewidths': 'linewidth',
        'linestyles': 'linestyle',
    },
    'tripcolor': {
        'colors': 'edgecolors',
        'linewidths': 'linewidth',
        'linestyles': 'linestyle',
    },
    'parametric': {
        'colors': 'color',
        'linewidths': 'linewidth',
        'linestyles': 'linestyle',
    },
    'hexbin': {
        'colors': 'edgecolors',
        'linewidths': 'linewidths',
        'linestyles': 'linestyles',
    },
    'hist2d': {
        'colors': 'edgecolors',
        'linewidths': 'linewidths',
        'linestyles': 'linestyles',
    },
    'barbs': {
        'colors': 'barbcolor',
        'linewidths': 'linewidth',
        'linestyles': 'linestyle',
    },
    'quiver': {  # NOTE: linewidth/linestyle apply to *arrow outline*
        'colors': 'color',
        'linewidths': 'linewidth',
        'linestyles': 'linestyle',
    },
    'streamplot': {
        'colors': 'color',
        'linewidths': 'linewidth',
        'linestyles': 'linestyle',
    },
    'spy': {
        'colors': 'color',
        'linewidths': 'linewidth',
        'linestyles': 'linestyle',
    },
    'matshow': {
        'colors': 'color',
        'linewidths': 'linewidth',
        'linestyles': 'linestyle',
    },
    'imshow': None,
}


def _is_number(data):
    """
    Test whether input is numeric array rather than datetime or strings.
    """
    return len(data) and np.issubdtype(_to_ndarray(data).dtype, np.number)


def _is_string(data):
    """
    Test whether input is array of strings.
    """
    return len(data) and isinstance(_to_ndarray(data).flat[0], str)


def _iter_legend_children(children):
    """
    Iterate recursively through `_children` attributes of various `HPacker`,
    `VPacker`, and `DrawingArea` classes.
    """
    for obj in children:
        if hasattr(obj, '_children'):
            yield from _iter_legend_children(obj._children)
        else:
            yield obj


def _iter_legend_objects(objs):
    """
    Retrieve the (object, label) pairs for objects with actual labels
    from nested lists and tuples of objects.
    """
    # Account for (1) multiple columns of data, (2) functions that return
    # multiple values (e.g. hist() returns (bins, values, patches)), and
    # (3) matplotlib.Collection list subclasses.
    if hasattr(objs, 'get_label'):
        label = objs.get_label()
        if label and label[:1] != '_':
            yield (objs, label)
    elif isinstance(objs, (list, tuple)):
        for obj in objs:
            yield from _iter_legend_objects(obj)


def _to_arraylike(data):
    """
    Convert list of lists to array-like type.
    """
    _load_objects()
    if not isinstance(data, (ndarray, DataArray, DataFrame, Series, Index)):
        data = np.array(data)
    if not np.iterable(data):
        data = np.atleast_1d(data)
    return data


def _to_indexer(data):
    """
    Return indexible attribute of array-like type.
    """
    return getattr(data, 'iloc', data)


def _to_ndarray(data):
    """
    Convert arbitrary input to ndarray cleanly.
    """
    return np.atleast_1d(getattr(data, 'values', data))


def _axis_labels_title(data, axis=None, units=True):
    """
    Get data and label for pandas or xarray objects or their coordinates along axis
    `axis`. If `units` is ``True`` also look for units on xarray data arrays.
    """
    label = ''
    _load_objects()
    if isinstance(data, ndarray):
        if axis is not None and data.ndim > axis:
            data = np.arange(data.shape[axis])

    # Xarray with common NetCDF attribute names
    elif isinstance(data, DataArray):
        if axis is not None and data.ndim > axis:
            data = data.coords[data.dims[axis]]
        label = getattr(data, 'name', '') or ''
        for key in ('standard_name', 'long_name'):
            label = data.attrs.get(key, label)
        if units:
            units = data.attrs.get('units', '')
            if label and units:
                label = f'{label} ({units})'
            elif units:
                label = units

    # Pandas object with name attribute
    # if not label and isinstance(data, DataFrame) and data.columns.size == 1:
    elif isinstance(data, (DataFrame, Series, Index)):
        if axis == 0 and isinstance(data, Index):
            pass
        elif axis == 0 and isinstance(data, (DataFrame, Series)):
            data = data.index
        elif axis == 1 and isinstance(data, DataFrame):
            data = data.columns
        elif axis == 1 and isinstance(data, (Series, Index)):
            data = np.array([data.name])  # treat series name as the "column" data
        # DataFrame has no native name attribute but user can add one:
        # https://github.com/pandas-dev/pandas/issues/447
        label = getattr(data, 'name', '') or ''

    return data, str(label).strip()


def _update_text(self, props):
    """
    Monkey patch that adds pseudo "border" properties to text objects
    without wrapping the entire class. We override update to facilitate
    updating inset titles.
    """
    props = props.copy()  # shallow copy
    border = props.pop('border', None)
    bordercolor = props.pop('bordercolor', 'w')
    borderinvert = props.pop('borderinvert', False)
    borderwidth = props.pop('borderwidth', 2)
    if border:
        facecolor, bgcolor = self.get_color(), bordercolor
        if borderinvert:
            facecolor, bgcolor = bgcolor, facecolor
        kwargs = {
            'linewidth': borderwidth,
            'foreground': bgcolor,
            'joinstyle': 'miter',
        }
        self.update({
            'color': facecolor,
            'path_effects': [mpatheffects.Stroke(**kwargs), mpatheffects.Normal()],
        })
    return type(self).update(self, props)


def _add_colorbar(
    self, mappable, values=None,
    extend=None, extendsize=None,
    title=None, label=None,
    grid=None, tickminor=None,
    reverse=False, tickloc=None, ticklocation=None,
    locator=None, ticks=None, maxn=None, maxn_minor=None,
    minorlocator=None, minorticks=None,
    locator_kw=None, minorlocator_kw=None,
    formatter=None, ticklabels=None, formatter_kw=None, rotation=None,
    norm=None, norm_kw=None,  # normalizer to use when passing colors/lines
    orientation='horizontal',
    edgecolor=None, linewidth=None,
    labelsize=None, labelweight=None, labelcolor=None,
    ticklabelsize=None, ticklabelweight=None, ticklabelcolor=None,
    **kwargs
):
    """
    Draw a colorbar with extra features.
    """
    # NOTE: There is a weird problem with colorbars when simultaneously
    # passing levels and norm object to a mappable; fixed by passing vmin/vmax
    # instead of levels. (see: https://stackoverflow.com/q/40116968/4970632).
    # NOTE: Often want levels instead of vmin/vmax, while simultaneously
    # using a Normalize (for example) to determine colors between the levels
    # (see: https://stackoverflow.com/q/42723538/4970632). Workaround makes
    # sure locators are in vmin/vmax range exclusively; cannot match values.
    # NOTE: In legend_wrapper() we try to add to the objects accepted by
    # legend() using handler_map. We can't really do anything similar for
    # colorbars; input must just be insnace of mixin class cm.ScalarMappable
    # Mutable args
    norm_kw = norm_kw or {}
    formatter_kw = formatter_kw or {}
    locator_kw = locator_kw or {}
    minorlocator_kw = minorlocator_kw or {}

    # Parse input args
    label = _not_none(title=title, label=label)
    locator = _not_none(ticks=ticks, locator=locator)
    minorlocator = _not_none(minorticks=minorticks, minorlocator=minorlocator)
    ticklocation = _not_none(tickloc=tickloc, ticklocation=ticklocation)
    formatter = _not_none(ticklabels=ticklabels, formatter=formatter)

    # Colorbar kwargs
    # WARNING: PathCollection scatter objects have an extend method!
    # WARNING: Matplotlib 3.3 deprecated 'extend' parameter passed to colorbar()
    # but *also* fails to read 'extend' parameter when added to a pcolor mappable!
    # Need to figure out workaround!
    grid = _not_none(grid, rc['colorbar.grid'])
    if extend is None:
        if isinstance(getattr(mappable, 'extend', None), str):
            extend = mappable.extend or 'neither'
        else:
            extend = 'neither'
    kwargs.update({
        'cax': self,
        'use_gridspec': True,
        'orientation': orientation,
        'spacing': 'uniform',
        'extend': extend,
    })
    kwargs.setdefault('drawedges', grid)

    # Text property keyword args
    kw_label = {}
    for key, value in (
        ('size', labelsize),
        ('weight', labelweight),
        ('color', labelcolor),
    ):
        if value is not None:
            kw_label[key] = value
    kw_ticklabels = {}
    for key, value in (
        ('size', ticklabelsize),
        ('weight', ticklabelweight),
        ('color', ticklabelcolor),
        ('rotation', rotation),
    ):
        if value is not None:
            kw_ticklabels[key] = value

    # Special case where auto colorbar is generated from 1d methods, a list is
    # always passed, but some 1d methods (scatter) do have colormaps.
    if (
        np.iterable(mappable)
        and len(mappable) == 1
        and hasattr(mappable[0], 'get_cmap')
    ):
        mappable = mappable[0]

    # For container objects, we just assume color is the same for every item.
    # Works for ErrorbarContainer, StemContainer, BarContainer.
    if (
        np.iterable(mappable)
        and len(mappable) > 0
        and all(isinstance(obj, mcontainer.Container) for obj in mappable)
    ):
        mappable = [obj[0] for obj in mappable]

    # Test if we were given a mappable, or iterable of stuff; note Container
    # and PolyCollection matplotlib classes are iterable.
    cmap = None
    if not isinstance(mappable, (martist.Artist, mcontour.ContourSet)):
        # A colormap instance
        # TODO: Pass remaining arguments through Colormap()? This is really
        # niche usage so maybe not necessary.
        if isinstance(mappable, mcolors.Colormap):
            # NOTE: 'Values' makes no sense if this is just a colormap. Just
            # use unique color for every segmentdata / colors color.
            cmap = mappable
            values = np.linspace(0, 1, cmap.N)

        # List of colors
        elif np.iterable(mappable) and all(
            isinstance(obj, str) or (np.iterable(obj) and len(obj) in (3, 4))
            for obj in mappable
        ):
            colors = list(mappable)
            cmap = mcolors.ListedColormap(colors, '_no_name')
            if values is None:
                values = np.arange(len(colors))
            locator = _not_none(locator, values)  # tick *all* values by default

        # List of artists
        # NOTE: Do not check for isinstance(Artist) in case it is an mpl collection
        elif np.iterable(mappable) and all(
            hasattr(obj, 'get_color') or hasattr(obj, 'get_facecolor')
            for obj in mappable
        ):
            # Generate colormap from colors and infer tick labels
            colors = []
            for obj in mappable:
                if hasattr(obj, 'get_color'):
                    color = obj.get_color()
                else:
                    color = obj.get_facecolor()
                if isinstance(color, np.ndarray):
                    color = color.squeeze()  # e.g. scatter plot
                    if color.ndim != 1:
                        raise ValueError(
                            'Cannot make colorbar from list of artists '
                            f'with more than one color: {color!r}.'
                        )
                colors.append(to_rgb(color))
            cmap = mcolors.ListedColormap(colors, '_no_name')

            # Try to infer tick values and tick labels from Artist labels
            if values is None:
                # Get object labels and values (avoid overwriting colorbar 'label')
                labs = []
                values = []
                for obj in mappable:
                    lab = value = None
                    if hasattr(obj, 'get_label'):
                        lab = obj.get_label() or None
                        if lab and lab[:1] == '_':  # intended to be ignored by legend
                            lab = None
                    if lab:
                        try:
                            value = float(lab)
                        except (TypeError, ValueError):
                            pass
                    labs.append(lab)
                    values.append(value)
                # Use default values if labels are non-numeric (numeric labels are
                # common when making on-the-fly colorbars). Try to use object labels
                # for ticks with default vertical rotation, like datetime axes.
                if any(value is None for value in values):
                    values = np.arange(len(mappable))
                    if formatter is None and any(lab is not None for lab in labs):
                        formatter = labs  # use these fixed values for ticks
                        if orientation == 'horizontal':
                            kw_ticklabels.setdefault('rotation', 90)
            locator = _not_none(locator, values)  # tick *all* values by default

        else:
            raise ValueError(
                'Input mappable must be a matplotlib artist, '
                'list of objects, list of colors, or colormap. '
                f'Got {mappable!r}.'
            )

        # Build ad hoc ScalarMappable object from colors
        if cmap is not None:
            if np.iterable(mappable) and len(values) != len(mappable):
                raise ValueError(
                    f'Passed {len(values)} values, but only {len(mappable)} '
                    f'objects or colors.'
                )
            norm, *_ = _build_discrete_norm(
                values=values, extend='neither',
                cmap=cmap, norm=norm, norm_kw=norm_kw,
            )
            mappable = mcm.ScalarMappable(norm, cmap)

    # Try to get tick locations from *levels* or from *values* rather than
    # random points along the axis.
    # NOTE: Do not necessarily want e.g. minor tick locations at logminor
    # for LogNorm! In _build_discrete_norm we sometimes select evenly spaced
    # levels in log-space *between* powers of 10, so logminor ticks would be
    # misaligned with levels.
    if locator is None:
        locator = getattr(mappable, 'ticks', None)
        if locator is None:
            # This should only happen if user calls plotting method on native
            # matplotlib axes.
            if isinstance(norm, mcolors.LogNorm):
                locator = 'log'
            elif isinstance(norm, mcolors.SymLogNorm):
                locator = 'symlog'
                locator_kw.setdefault('linthresh', norm.linthresh)
            else:
                locator = 'auto'

        elif not isinstance(locator, mticker.Locator):
            # Get default maxn, try to allot 2em squares per label maybe?
            # NOTE: Cannot use Axes.get_size_inches because this is a
            # native matplotlib axes
            width, height = self.figure.get_size_inches()
            if orientation == 'horizontal':
                scale = 3  # em squares alotted for labels
                length = width * abs(self.get_position().width)
                fontsize = kw_ticklabels.get('size', rc['xtick.labelsize'])
            else:
                scale = 1
                length = height * abs(self.get_position().height)
                fontsize = kw_ticklabels.get('size', rc['ytick.labelsize'])
            fontsize = rc._scale_font(fontsize)
            maxn = _not_none(maxn, int(length / (scale * fontsize / 72)))
            maxn_minor = _not_none(
                maxn_minor, int(length / (0.5 * fontsize / 72))
            )

            # Get locator
            if tickminor and minorlocator is None:
                step = 1 + len(locator) // max(1, maxn_minor)
                minorlocator = locator[::step]
            step = 1 + len(locator) // max(1, maxn)
            locator = locator[::step]

    # Get extend triangles in physical units
    width, height = self.figure.get_size_inches()
    if orientation == 'horizontal':
        scale = width * abs(self.get_position().width)
    else:
        scale = height * abs(self.get_position().height)
    extendsize = units(_not_none(extendsize, rc['colorbar.extend']))
    extendsize = extendsize / (scale - 2 * extendsize)

    # Draw the colorbar
    # NOTE: Set default formatter here because we optionally apply a FixedFormatter
    # using *labels* from handle input.
    locator = constructor.Locator(locator, **locator_kw)
    formatter = constructor.Formatter(_not_none(formatter, 'auto'), **formatter_kw)
    kwargs.update({
        'ticks': locator,
        'format': formatter,
        'ticklocation': ticklocation,
        'extendfrac': extendsize
    })
    mappable.extend = extend  # matplotlib >=3.3
    cb = self.figure.colorbar(mappable, **kwargs)
    axis = self.xaxis if orientation == 'horizontal' else self.yaxis

    # The minor locator
    # TODO: Document the improved minor locator functionality!
    # NOTE: Colorbar._use_auto_colorbar_locator() is never True because we use
    # the custom DiscreteNorm normalizer. Colorbar._ticks() always called.
    if minorlocator is None:
        if tickminor:
            cb.minorticks_on()
        else:
            cb.minorticks_off()
    elif not hasattr(cb, '_ticker'):
        warnings._warn_proplot(
            'Matplotlib colorbar API has changed. '
            f'Cannot use custom minor tick locator {minorlocator!r}.'
        )
        cb.minorticks_on()  # at least turn them on
    else:
        # Set the minor ticks just like matplotlib internally sets the
        # major ticks. Private API is the only way!
        minorlocator = constructor.Locator(minorlocator, **minorlocator_kw)
        ticks, *_ = cb._ticker(minorlocator, mticker.NullFormatter())
        axis.set_ticks(ticks, minor=True)
        axis.set_ticklabels([], minor=True)

    # Label and tick label settings
    # WARNING: Must use colorbar set_label to set text, calling set_text on
    # the axis will do nothing!
    if label is not None:
        cb.set_label(label)
    axis.label.update(kw_label)
    for obj in axis.get_ticklabels():
        obj.update(kw_ticklabels)

    # Ticks
    xy = axis.axis_name
    for which in ('minor', 'major'):
        kw = rc.category(xy + 'tick.' + which)
        kw.pop('visible', None)
        if edgecolor:
            kw['color'] = edgecolor
        if linewidth:
            kw['width'] = linewidth
        axis.set_tick_params(which=which, **kw)
    axis.set_ticks_position(ticklocation)

    # Fix alpha-blending issues.
    # Cannot set edgecolor to 'face' if alpha non-zero because blending will
    # occur, will get colored lines instead of white ones. Need manual blending
    # NOTE: For some reason cb solids uses listed colormap with always 1.0
    # alpha, then alpha is applied after.
    # See: https://stackoverflow.com/a/35672224/4970632
    cmap = cb.cmap
    if not cmap._isinit:
        cmap._init()
    if any(cmap._lut[:-1, 3] < 1):
        warnings._warn_proplot(
            f'Using manual alpha-blending for {cmap.name!r} colorbar solids.'
        )
        # Generate "secret" copy of the colormap!
        lut = cmap._lut.copy()
        cmap = mcolors.Colormap('_cbar_fix', N=cmap.N)
        cmap._isinit = True
        cmap._init = lambda: None
        # Manually fill lookup table with alpha-blended RGB colors!
        for i in range(lut.shape[0] - 1):
            alpha = lut[i, 3]
            lut[i, :3] = (1 - alpha) * 1 + alpha * lut[i, :3]  # blend *white*
            lut[i, 3] = 1
        cmap._lut = lut
        # Update colorbar
        cb.cmap = cmap
        cb.draw_all()

    # Fix colorbar outline
    kw_outline = {
        'edgecolor': _not_none(edgecolor, rc['axes.edgecolor']),
        'linewidth': _not_none(linewidth, rc['axes.linewidth']),
    }
    if cb.outline is not None:
        cb.outline.update(kw_outline)
    if cb.dividers is not None:
        cb.dividers.update(kw_outline)

    # *Never* rasterize because it causes misalignment with border lines
    if cb.solids:
        cb.solids.set_rasterized(False)
        cb.solids.set_linewidth(0.4)
        cb.solids.set_edgecolor('face')

    # Invert the axis if descending DiscreteNorm
    norm = mappable.norm
    if getattr(norm, '_descending', None):
        axis.set_inverted(True)
    if reverse:  # potentially double reverse, although that would be weird...
        axis.set_inverted(True)
    return cb


def _add_legend(
    self, handles=None, labels=None, *, ncol=None, ncols=None,
    center=None, order='C', loc=None, label=None, title=None,
    fontsize=None, fontweight=None, fontcolor=None,
    color=None, marker=None, lw=None, linewidth=None,
    dashes=None, linestyle=None, markersize=None, frameon=None, frame=None,
    **kwargs
):
    """
    Draw a legend with extra features.
    """
    # Parse input args
    # TODO: Legend entries for colormap or scatterplot objects! Idea is we
    # pass a scatter plot or contourf or whatever, and legend is generated by
    # drawing patch rectangles or markers using data values and their
    # corresponding cmap colors! For scatterplots just test get_facecolor()
    # to see if it contains more than one color.
    # TODO: It is *also* often desirable to label a colormap object with
    # one data value. Maybe add a legend option for the *number of samples*
    # or the *sample points* when drawing legends for colormap objects.
    # Look into "legend handlers", might just want to add own handlers by
    # passing handler_map to legend() and get_legend_handles_labels().
    if order not in ('F', 'C'):
        raise ValueError(
            f'Invalid order {order!r}. Choose from '
            '"C" (row-major, default) and "F" (column-major).'
        )
    ncol = _not_none(ncols=ncols, ncol=ncol)
    title = _not_none(label=label, title=title)
    frameon = _not_none(
        frame=frame, frameon=frameon, default=rc['legend.frameon']
    )
    if handles is not None and not np.iterable(handles):  # e.g. a mappable object
        handles = [handles]
    if labels is not None and (not np.iterable(labels) or isinstance(labels, str)):
        labels = [labels]
    if title is not None:
        kwargs['title'] = title
    if frameon is not None:
        kwargs['frameon'] = frameon
    if fontsize is not None:
        kwargs['fontsize'] = rc._scale_font(fontsize)

    # Handle and text properties that are applied after-the-fact
    # NOTE: Set solid_capstyle to 'butt' so line does not extend past error bounds
    # shading in legend entry. This change is not noticeable in other situations.
    kw_text = {}
    for key, value in (
        ('color', fontcolor),
        ('weight', fontweight),
    ):
        if value is not None:
            kw_text[key] = value
    kw_handle = {'solid_capstyle': 'butt'}
    for key, value in (
        ('color', color),
        ('marker', marker),
        ('linewidth', lw),
        ('linewidth', linewidth),
        ('markersize', markersize),
        ('linestyle', linestyle),
        ('dashes', dashes),
    ):
        if value is not None:
            kw_handle[key] = value

    # Legend box properties
    outline = rc.fill(
        {
            'linewidth': 'axes.linewidth',
            'edgecolor': 'axes.edgecolor',
            'facecolor': 'axes.facecolor',
            'alpha': 'legend.framealpha',
        }
    )
    for key in (*outline,):
        if key != 'linewidth':
            if kwargs.get(key, None):
                outline.pop(key, None)

    # Get axes for legend handle detection
    # TODO: Update this when no longer use "filled panels" for outer legends
    axs = [self]
    if self._panel_hidden:
        if self._panel_parent:  # axes panel
            axs = list(self._panel_parent._iter_axes(hidden=False, children=True))
        else:
            axs = list(self.figure._iter_axes(hidden=False, children=True))

    # Handle list of lists (centered row legends)
    # NOTE: Avoid very common plot() error where users draw individual lines
    # with plot() and add singleton tuples to a list of handles. If matplotlib
    # gets a list like this but gets no 'labels' argument, it raises error.
    list_of_lists = False
    if handles is not None:
        handles = [
            handle[0] if type(handle) is tuple and len(handle) == 1 else handle
            for handle in handles
        ]
        list_of_lists = any(type(handle) in (list, np.ndarray) for handle in handles)
    if handles is not None and labels is not None and len(handles) != len(labels):
        raise ValueError(
            f'Got {len(handles)} handles and {len(labels)} labels.'
        )
    if list_of_lists:
        if any(not np.iterable(_) for _ in handles):
            raise ValueError(f'Invalid handles={handles!r}.')
        if not labels:
            labels = [None] * len(handles)
        elif not all(np.iterable(_) and not isinstance(_, str) for _ in labels):
            # e.g. handles=[obj1, [obj2, obj3]] requires labels=[lab1, [lab2, lab3]]
            raise ValueError(
                f'Invalid labels={labels!r} for handles={handles!r}.'
            )

    # Parse handles and legends with native matplotlib parser
    if not list_of_lists:
        if isinstance(handles, np.ndarray):
            handles = handles.tolist()
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        handles, labels, *_ = mlegend._parse_legend_args(
            axs, handles=handles, labels=labels,
        )
        pairs = list(zip(handles, labels))
    else:
        pairs = []
        for ihandles, ilabels in zip(handles, labels):
            if isinstance(ihandles, np.ndarray):
                ihandles = ihandles.tolist()
            if isinstance(ilabels, np.ndarray):
                ilabels = ilabels.tolist()
            ihandles, ilabels, *_ = mlegend._parse_legend_args(
                axs, handles=ihandles, labels=ilabels,
            )
            pairs.append(list(zip(handles, labels)))

    # Manage pairs in context of 'center' option
    center = _not_none(center, list_of_lists)
    if not center and list_of_lists:  # standardize format based on input
        list_of_lists = False  # no longer is list of lists
        pairs = [pair for ipairs in pairs for pair in ipairs]
    elif center and not list_of_lists:
        list_of_lists = True
        ncol = _not_none(ncol, 3)
        pairs = [
            pairs[i * ncol:(i + 1) * ncol] for i in range(len(pairs))
        ]  # to list of iterables
        ncol = None
    if list_of_lists:  # remove empty lists, pops up in some examples
        pairs = [ipairs for ipairs in pairs if ipairs]

    # Bail if no pairs
    if not pairs:
        return mlegend.Legend(self, [], [], ncol=ncol, loc=loc, **kwargs)

    # Individual legend
    legs = []
    width, height = self.get_size_inches()
    if not center:
        # Optionally change order
        # See: https://stackoverflow.com/q/10101141/4970632
        # Example: If 5 columns, but final row length 3, columns 0-2 have
        # N rows but 3-4 have N-1 rows.
        ncol = _not_none(ncol, 3)
        if order == 'C':
            split = [  # split into rows
                pairs[i * ncol:(i + 1) * ncol]
                for i in range(len(pairs) // ncol + 1)
            ]
            nrowsmax = len(split)  # max possible row count
            nfinalrow = len(split[-1])  # columns in final row
            nrows = (
                [nrowsmax] * nfinalrow + [nrowsmax - 1] * (ncol - nfinalrow)
            )
            fpairs = []
            for col, nrow in enumerate(nrows):  # iterate through cols
                fpairs.extend(split[row][col] for row in range(nrow))
            pairs = fpairs

        # Draw legend
        leg = mlegend.Legend(self, *zip(*pairs), ncol=ncol, loc=loc, **kwargs)
        legs = [leg]

    # Legend with centered rows, accomplished by drawing separate legends for
    # each row. The label spacing/border spacing will be exactly replicated.
    else:
        # Message when overriding some properties
        overridden = []
        kwargs.pop('frameon', None)  # then add back later!
        for override in ('bbox_transform', 'bbox_to_anchor'):
            prop = kwargs.pop(override, None)
            if prop is not None:
                overridden.append(override)
        if ncol is not None:
            warnings._warn_proplot(
                'Detected list of *lists* of legend handles. '
                'Ignoring user input property "ncol".'
            )
        if overridden:
            warnings._warn_proplot(
                'Ignoring user input properties '
                + ', '.join(map(repr, overridden))
                + ' for centered-row legend.'
            )

        # Determine space we want sub-legend to occupy as fraction of height
        # NOTE: Empirical testing shows spacing fudge factor necessary to
        # exactly replicate the spacing of standard aligned legends.
        fontsize = kwargs.get('fontsize', None) or rc['legend.fontsize']
        fontsize = rc._scale_font(fontsize)
        spacing = kwargs.get('labelspacing', None) or rc['legend.labelspacing']
        if pairs:
            interval = 1 / len(pairs)  # split up axes
            interval = (((1 + spacing * 0.85) * fontsize) / 72) / height

        # Iterate and draw
        # NOTE: We confine possible bounding box in *y*-direction, but do not
        # confine it in *x*-direction. Matplotlib will automatically move
        # left-to-right if you request this.
        ymin, ymax = None, None
        if order == 'F':
            raise NotImplementedError(
                'When center=True, ProPlot vertically stacks successive '
                'single-row legends. Column-major (order="F") ordering '
                'is un-supported.'
            )
        loc = _not_none(loc, 'upper center')
        if not isinstance(loc, str):
            raise ValueError(
                f'Invalid location {loc!r} for legend with center=True. '
                'Must be a location *string*.'
            )
        elif loc == 'best':
            warnings._warn_proplot(
                'For centered-row legends, cannot use "best" location. '
                'Using "upper center" instead.'
            )

        # Iterate through sublists
        for i, ipairs in enumerate(pairs):
            if i == 1:
                kwargs.pop('title', None)
            if i >= 1 and title is not None:
                i += 1  # extra space!

            # Legend position
            if 'upper' in loc:
                y1 = 1 - (i + 1) * interval
                y2 = 1 - i * interval
            elif 'lower' in loc:
                y1 = (len(pairs) + i - 2) * interval
                y2 = (len(pairs) + i - 1) * interval
            else:  # center
                y1 = 0.5 + interval * len(pairs) / 2 - (i + 1) * interval
                y2 = 0.5 + interval * len(pairs) / 2 - i * interval
            ymin = min(y1, _not_none(ymin, y1))
            ymax = max(y2, _not_none(ymax, y2))

            # Draw legend
            bbox = mtransforms.Bbox([[0, y1], [1, y2]])
            leg = mlegend.Legend(
                self, *zip(*ipairs), loc=loc, ncol=len(ipairs),
                bbox_transform=self.transAxes, bbox_to_anchor=bbox,
                frameon=False, **kwargs
            )
            legs.append(leg)

    # Add legends manually so matplotlib does not remove old ones
    for leg in legs:
        self.add_artist(leg)
        leg.legendPatch.update(outline)  # or get_frame()

    # Apply *overrides* to legend elements
    # WARNING: legendHandles only contains the *first* artist per legend because
    # HandlerBase.legend_artist() called in Legend._init_legend_box() only
    # returns the first artist. Instead we try to iterate through offset boxes.
    # TODO: Remove this feature? Idea was this lets users create *categorical*
    # legends in clunky way, e.g. entries denoting *colors* and entries denoting
    # *markers*. But would be better to add capacity for categorical labels in a
    # *single* legend like seaborn rather than multiple legends.
    for leg in legs:
        try:
            children = leg._legend_handle_box._children
        except AttributeError:  # older versions maybe?
            children = []
        for obj in _iter_legend_children(children):
            # account for mixed legends, e.g. line on top of
            # error bounds shading.
            if isinstance(obj, mtext.Text):
                leg.update(kw_text)
            else:
                for key, value in kw_handle.items():
                    getattr(obj, f'set_{key}', lambda value: None)(value)

    # Draw manual fancy bounding box for un-aligned legend
    # WARNING: The matplotlib legendPatch transform is the default transform,
    # i.e. universal coordinates in points. Means we have to transform
    # mutation scale into transAxes sizes.
    # WARNING: Tempting to use legendPatch for everything but for some reason
    # coordinates are messed up. In some tests all coordinates were just result
    # of get window extent multiplied by 2 (???). Anyway actual box is found in
    # _legend_box attribute, which is accessed by get_window_extent.
    if center and frameon:
        if len(legs) == 1:
            # Use builtin frame
            legs[0].set_frame_on(True)
        else:
            # Get coordinates
            renderer = self.figure._get_renderer()
            bboxs = [
                leg.get_window_extent(renderer).transformed(self.transAxes.inverted())
                for leg in legs
            ]
            xmin = min(bbox.xmin for bbox in bboxs)
            xmax = max(bbox.xmax for bbox in bboxs)
            ymin = min(bbox.ymin for bbox in bboxs)
            ymax = max(bbox.ymax for bbox in bboxs)
            fontsize = (fontsize / 72) / width  # axes relative units
            fontsize = renderer.points_to_pixels(fontsize)

            # Draw and format patch
            patch = mpatches.FancyBboxPatch(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                snap=True, zorder=4.5,
                mutation_scale=fontsize,
                transform=self.transAxes
            )
            if kwargs.get('fancybox', rc['legend.fancybox']):
                patch.set_boxstyle('round', pad=0, rounding_size=0.2)
            else:
                patch.set_boxstyle('square', pad=0)
            patch.set_clip_on(False)
            patch.update(outline)
            self.add_artist(patch)

            # Add shadow
            # TODO: This does not work, figure out
            if kwargs.get('shadow', rc['legend.shadow']):
                shadow = mpatches.Shadow(patch, 20, -20)
                self.add_artist(shadow)

            # Add patch to list
            legs = (patch, *legs)

    # Append attributes and return, and set clip property!!! This is critical
    # for tight bounding box calcs!
    for leg in legs:
        leg.set_clip_on(False)
    return legs[0] if len(legs) == 1 else tuple(legs)


def _add_labels(func):
    """
    Add labels to the contour plot.
    """
    @functools.wraps(func)
    def wrapper(
        self, *args, fmt=None, labels=None, labels_kw=None, precision=None, **kwargs,
    ):
        # Call main funtion
        name = func.__name__
        obj = func(self, *args, **kwargs)
        if not labels:
            return obj

        # Default formatter
        labels_kw = labels_kw or {}
        fmt = _not_none(labels_kw.pop('fmt', None), fmt, 'simple')
        fmt = constructor.Formatter(fmt, precision=precision)

        # Add contour labels
        if name in ('contour', 'tricontour', 'contourf', 'tricontourf'):
            cobj = obj
            cmap = obj.get_cmap()
            norm = obj.get_norm()
            levels = obj.levels
            colors = None
            if name in ('contourf', 'tricontourf'):
                lums = [to_xyz(cmap(norm(level)), 'hcl')[2] for level in levels]
                cobj = self.contour(*args, levels=levels, linewidths=0)
                colors = ['w' if lum < 50 else 'k' for lum in lums]
            text_kw = {}
            for key in tuple(labels_kw):  # allow dict to change size
                if key in (
                    'levels', 'fontsize', 'colors', 'inline', 'inline_spacing',
                    'manual', 'rightside_up', 'use_clabeltext',
                ):
                    text_kw[key] = labels_kw.pop(key)
            labels_kw.setdefault('colors', colors)
            labels_kw.setdefault('inline_spacing', 3)
            labels_kw.setdefault('fontsize', rc['text.labelsize'])
            labs = cobj.clabel(fmt=fmt, **labels_kw)
            for lab in labs:
                lab.update(text_kw)

        # Add gridbox labels
        # See: https://stackoverflow.com/a/20998634/4970632
        elif name in (
            'pcolor', 'pcolormesh', 'pcolorfast', 'tripcolor', 'tripcolormesh'
        ):
            # Populate the _facecolors attribute, which is initially filled
            # with just a single color
            obj.update_scalarmappable()

            # Get text positions and colors
            labels_kw_ = {'size': rc['text.labelsize'], 'ha': 'center', 'va': 'center'}
            labels_kw_.update(labels_kw)
            array = obj.get_array()
            paths = obj.get_paths()
            colors = np.asarray(obj.get_facecolors())
            edgecolors = np.asarray(obj.get_edgecolors())
            if len(colors) == 1:  # weird flex but okay
                colors = np.repeat(colors, len(array), axis=0)
            if len(edgecolors) == 1:
                edgecolors = np.repeat(edgecolors, len(array), axis=0)
            for i, (color, path, num) in enumerate(zip(colors, paths, array)):
                if not np.isfinite(num):
                    edgecolors[i, :] = 0
                    continue
                bbox = path.get_extents()
                x = (bbox.xmin + bbox.xmax) / 2
                y = (bbox.ymin + bbox.ymax) / 2
                if 'color' not in labels_kw:
                    _, _, lum = to_xyz(color, 'hcl')
                    if lum < 50:
                        color = 'w'
                    else:
                        color = 'k'
                    labels_kw_['color'] = color
                self.text(x, y, fmt(num), **labels_kw_)
            obj.set_edgecolors(edgecolors)

        else:
            raise ValueError(f'Invalid wrapped function: {name}()')

        return obj

    return wrapper


def _auto_colorbar_legend(func):
    """
    Add a colorbar or legend from the resulting plot.
    """
    @functools.wraps(func)
    def wrapper(
        self, *args,
        colorbar=None, colorbar_kw=None, legend=None, legend_kw=None,
        label=None, labels=None, values=None,
        **kwargs,
    ):
        # Parse input args
        name = func.__name__
        autoformat = rc['autoformat']
        legend_kw = legend_kw or {}
        colorbar_kw = colorbar_kw or {}
        labels = _not_none(
            values=values,
            labels=labels,
            label=label,
            legend_kw_labels=legend_kw.pop('labels', None),
        )
        if name in ('pie',):  # add x coordinates as default pie chart labels
            labels = _not_none(labels, x)  # TODO: move to pie wrapper?
        colorbar_legend_label = None  # for colorbar or legend

        # Handle legend labels. Several scenarios:
        # 1. Always prefer input labels
        # 2. Always add labels if this is a *named* dimension.
        # 3. Even if not *named* dimension add labels if labels are string
        # WARNING: Most methods that accept 2D arrays use columns of data, but when
        # pandas DataFrame passed to hist, boxplot, or violinplot, rows of data
        # assumed! This is fixed in parse_1d by converting to values.
        sample = args[-1]
        ncols = 1
        if name in ('pie', 'boxplot', 'violinplot'):
            # Functions handle multiple labels on their own
            if labels is not None:
                kwargs['labels'] = labels  # error raised down the line
        else:
            # Get column count and sanitize labels
            ncols = 1 if y.ndim == 1 else y.shape[1]
            if not np.iterable(labels) or isinstance(labels, str):
                labels = [labels] * ncols
            if len(labels) != ncols:
                raise ValueError(
                    f'Got {ncols} columns in data array, but {len(labels)} labels.'
                )

            # Get automatic legend labels and legend title
            # NOTE: Only apply labels if they are string labels *or* the
            # legend or colorbar has a title (latter is more common for colorbars)
            if autoformat:
                ilabels, colorbar_legend_label = _axis_labels_title(sample, axis=1)
                ilabels = _to_ndarray(ilabels)  # may be empty!
                for i, (ilabel, label) in enumerate(zip(ilabels, labels)):
                    if label is None and (colorbar_legend_label or isinstance(ilabel, str)):
                        labels[i] = ilabel

        # Sanitize labels
        # WARNING: Must convert labels to string here because e.g. scatter() applies
        # default label if input is False-ey. So numeric '0' would be overridden.
        if labels is None:
            labels = [''] * ncols
        else:
            labels = [str(_not_none(label, '')) for label in labels]

            # Call main function
            objs = func(self, *args, **kwargs)

            # Add colorbar
            if colorbar:
                # Add handles
                loc = self._loc_translate(colorbar, 'colorbar', allow_manual=False)
                if loc not in self._auto_colorbar:
                    self._auto_colorbar[loc] = ([], {})
                self._auto_colorbar[loc][0].extend(objs)

                # Add keywords
                if loc != 'fill':
                    colorbar_kw.setdefault('loc', loc)
                if colorbar_legend_label:
                    colorbar_kw.setdefault('label', colorbar_legend_label)
                self._auto_colorbar[loc][1].update(colorbar_kw)

        # Add legend
        if legend:
            # Get error objects. If they have separate label, allocate separate
            # legend entry. If not, try to combine with current legend entry.
            if type(errobjs) not in (list, tuple):
                errobjs = (errobjs,)
            errobjs = list(filter(None, errobjs))
            errobjs_join = [obj for obj in errobjs if not obj.get_label()]
            errobjs_separate = [obj for obj in errobjs if obj.get_label()]

            # Get legend objects
            # NOTE: It is not yet possible to draw error bounds *and* draw lines
            # with multiple columns of data.
            # NOTE: Put error bounds objects *before* line objects in the tuple,
            # so that line gets drawn on top of bounds.
            legobjs = objs.copy()
            if errobjs_join:
                legobjs = [(*legobjs, *errobjs_join)[::-1]]
            legobjs.extend(errobjs_separate)
            try:
                legobjs, labels = list(zip(*_iter_legend_objects(legobjs)))
            except ValueError:
                legobjs = labels = ()

            # Add handles and labels
            # NOTE: Important to add labels as *keyword* so users can override
            # NOTE: Use legend(handles, labels) syntax so we can assign labels
            # for tuples of artists. Otherwise they are label-less.
            loc = self._loc_translate(legend, 'legend', allow_manual=False)
            if loc not in self._auto_legend:
                self._auto_legend[loc] = ([], {'labels': []})
            self._auto_legend[loc][0].extend(legobjs)
            self._auto_legend[loc][1]['labels'].extend(labels)

            # Add other keywords
            if loc != 'fill':
                legend_kw.setdefault('loc', loc)
            if colorbar_legend_label:
                legend_kw.setdefault('label', colorbar_legend_label)
            self._auto_legend[loc][1].update(legend_kw)


def _build_discrete_norm(
    data=None, N=None, levels=None, values=None,
    norm=None, norm_kw=None, locator=None, locator_kw=None,
    cmap=None, vmin=None, vmax=None, extend=None, symmetric=False,
    minlength=2,
):
    """
    Build a `~proplot.colors.DiscreteNorm` or `~proplot.colors.BoundaryNorm`
    from the input arguments. This automatically calculates "nice" level
    boundaries if they were not provided.

    Parameters
    ----------
    data, vmin, vmax, levels, values
        Used to determine the level boundaries.
    norm, norm_kw
        Passed to `~proplot.constructor.Norm`.
    locator, locator_kw
        Passed to `~proplot.constructor.Locator`.
    minlength : int
        The minimum length for level lists.

    Returns
    -------
    norm : `matplotlib.colors.Normalize`
        The normalizer.
    ticks : `numpy.ndarray` or `matplotlib.locator.Locator`
        The axis locator or the tick location candidates.
    """
    # Parse flexible keyword args
    norm_kw = norm_kw or {}
    locator_kw = locator_kw or {}
    levels = _not_none(
        N=N, levels=levels, norm_kw_levels=norm_kw.pop('levels', None),
        default=rc['image.levels']
    )
    vmin = _not_none(vmin=vmin, norm_kw_vmin=norm_kw.pop('vmin', None))
    vmax = _not_none(vmax=vmax, norm_kw_vmax=norm_kw.pop('vmax', None))
    if norm == 'segments':  # TODO: remove
        norm = 'segmented'

    # NOTE: Matplotlib colorbar algorithm *cannot* handle descending levels
    # so this function reverses them and adds special attribute to the
    # normalizer. Then colorbar_wrapper reads this attribute and flips the
    # axis and the colormap direction.
    # Check input levels and values
    for key, val in (('levels', levels), ('values', values)):
        if not np.iterable(val):
            continue
        if len(val) < minlength or len(val) >= 2 and any(
            np.sign(np.diff(val)) != np.sign(val[1] - val[0])
        ):
            raise ValueError(
                f'{key!r} must be monotonically increasing or decreasing '
                f'and at least length {minlength}, got {val}.'
            )

    # Get level edges from level centers
    ticks = None
    if isinstance(values, Number):
        levels = np.atleast_1d(values)[0] + 1
    elif np.iterable(values) and len(values) == 1:
        levels = [values[0] - 1, values[0] + 1]  # weird but why not
    elif np.iterable(values) and len(values) > 1:
        # Try to generate levels such that a LinearSegmentedNorm will
        # place values ticks at the center of each colorbar level.
        # utils.edges works only for evenly spaced values arrays.
        # We solve for: (x1 + x2)/2 = y --> x2 = 2*y - x1
        # with arbitrary starting point x1. We also start the algorithm
        # on the end with *smaller* differences.
        if norm is None or norm == 'segmented':
            reverse = abs(values[-1] - values[-2]) < abs(values[1] - values[0])
            if reverse:
                values = values[::-1]
            levels = [values[0] - (values[1] - values[0]) / 2]
            for val in values:
                levels.append(2 * val - levels[-1])
            if reverse:
                levels = levels[::-1]
            if any(np.sign(np.diff(levels)) != np.sign(levels[1] - levels[0])):
                levels = edges(values)  # backup plan, weird tick locations
        # Generate levels by finding in-between points in the
        # normalized numeric space, e.g. LogNorm space.
        else:
            inorm = constructor.Norm(norm, **norm_kw)
            levels = inorm.inverse(edges(inorm(values)))
    elif values is not None:
        raise ValueError(
            f'Unexpected input values={values!r}. '
            'Must be integer or list of numbers.'
        )

    # Get default normalizer
    # Only use LinearSegmentedNorm if necessary, because it is slow
    descending = False
    if np.iterable(levels):
        if len(levels) == 1:
            norm = mcolors.Normalize(vmin=levels[0] - 1, vmax=levels[0] + 1)
        else:
            levels, descending = pcolors._check_levels(levels)
    if norm is None:
        norm = 'linear'
        if np.iterable(levels) and len(levels) > 2:
            steps = np.abs(np.diff(levels))
            eps = np.mean(steps) / 1e3
            if np.any(np.abs(np.diff(steps)) >= eps):
                norm = 'segmented'
    if norm == 'segmented':
        if not np.iterable(levels):
            norm = 'linear'  # has same result
        else:
            norm_kw['levels'] = levels
    norm = constructor.Norm(norm, **norm_kw)

    # Use the locator to determine levels
    # Mostly copied from the hidden contour.ContourSet._autolev
    # NOTE: Subsequently, we *only* use the locator to determine ticks if
    # *levels* and *values* were not passed.
    if isinstance(norm, mcolors.BoundaryNorm):
        # Get levels from bounds
        # TODO: Test this feature?
        # NOTE: No warning because we get here internally?
        levels = norm.boundaries
    elif np.iterable(values):
        # Prefer ticks in center
        ticks = np.asarray(values)
    elif np.iterable(levels):
        # Prefer ticks on level edges
        ticks = np.asarray(levels)
    else:
        # Determine levels automatically
        N = levels
        if locator is not None:
            locator = constructor.Locator(locator, **locator_kw)
            ticks = locator
        elif isinstance(norm, mcolors.LogNorm):
            locator = mticker.LogLocator(**locator_kw)
            ticks = locator
        elif isinstance(norm, mcolors.SymLogNorm):
            locator_kw.setdefault('linthresh', norm.linthresh)
            locator = mticker.SymmetricalLogLocator(**locator_kw)
            ticks = locator
        else:
            locator_kw.setdefault('symmetric', symmetric)
            locator = mticker.MaxNLocator(N, min_n_ticks=1, **locator_kw)

        # Get locations
        automin = vmin is None
        automax = vmax is None
        if automin or automax:
            data = ma.masked_invalid(data, copy=False)
            if automin:
                vmin = float(data.min())
            if automax:
                vmax = float(data.max())
            if vmin == vmax or ma.is_masked(vmin) or ma.is_masked(vmax):
                vmin, vmax = 0, 1
        try:
            levels = locator.tick_values(vmin, vmax)
        except RuntimeError:  # too-many-ticks error
            levels = np.linspace(vmin, vmax, N)  # TODO: _autolev used N+1

        # Trim excess levels the locator may have supplied
        # NOTE: This part is mostly copied from _autolev
        if not locator_kw.get('symmetric', None):
            i0, i1 = 0, len(levels)  # defaults
            under, = np.where(levels < vmin)
            if len(under):
                i0 = under[-1]
                if not automin or extend in ('min', 'both'):
                    i0 += 1  # permit out-of-bounds data
            over, = np.where(levels > vmax)
            if len(over):
                i1 = over[0] + 1 if len(over) else len(levels)
                if not automax or extend in ('max', 'both'):
                    i1 -= 1  # permit out-of-bounds data
            if i1 - i0 < 3:
                i0, i1 = 0, len(levels)  # revert
            levels = levels[i0:i1]

        # Compare the no. of levels we *got* (levels) to what we *wanted* (N)
        # If we wanted more than 2 times the result, then add nn - 1 extra
        # levels in-between the returned levels *in normalized space*.
        # Example: A LogNorm gives too few levels, so we select extra levels
        # here, but use the locator for determining tick locations.
        nn = N // len(levels)
        if nn >= 2:
            olevels = norm(levels)
            nlevels = []
            for i in range(len(levels) - 1):
                l1, l2 = olevels[i], olevels[i + 1]
                nlevels.extend(np.linspace(l1, l2, nn + 1)[:-1])
            nlevels.append(olevels[-1])
            levels = norm.inverse(nlevels)

        # Use auto-generated levels for ticks if still None
        if ticks is None:
            ticks = levels

    # Generate DiscreteNorm and update "child" norm with vmin and vmax from
    # levels. This lets the colorbar set tick locations properly!
    if not isinstance(norm, mcolors.BoundaryNorm) and len(levels) > 1:
        norm = pcolors.DiscreteNorm(
            levels, cmap=cmap, norm=norm, descending=descending
        )
    if descending:
        cmap = cmap.reversed()
    return norm, cmap, levels, ticks


def _fix_edges(func):
    """
    Fix white lines between filled contours/mesh and fix issues with colormaps
    that are not perfectly opaque.
    """
    @functools.wraps(func)
    def wrapper(self, *args, edgefix=None, **kwargs):
        # Call main function
        obj = func(self, *args, **kwargs)

        # 0.4pt is thick enough to hide lines but thin enough to not add "dots" in
        # corner of pcolor plots.
        # See: https://github.com/jklymak/contourfIssues
        # See: https://stackoverflow.com/q/15003353/4970632
        cmap = obj.get_cmap()
        if not cmap._isinit:
            cmap._init()
        edgecolor = 'face' if all(cmap._lut[:-1, 3] == 1) else 'none'
        if isinstance(mcontour.ContourSet):
            for contour in obj.collections:
                contour.set_edgecolor(edgecolor)
                contour.set_linewidth(0.4)
                contour.set_linestyle('-')
        else:
            if hasattr(obj, 'set_linewidth'):  # not always true for pcolorfast
                obj.set_linewidth(0.4)
            if hasattr(obj, 'set_edgecolor'):  # not always true for pcolorfast
                obj.set_edgecolor(edgecolor)

        return obj

    return wrapper


def _indicate_error_data(
    data, y, errdata=None, stds=None, pctiles=False,
    stds_default=None, pctiles_default=None,
    means_or_medians=True, absolute=False, label=False,
):
    """
    Return values that can be passed to the `~matplotlib.axes.Axes.errorbar`
    `xerr` and `yerr` keyword args.
    """
    # Parse arguments
    # NOTE: Have to guard against "truth value of an array is ambiguous" errors
    if not isinstance(stds, ARRAY_TYPES):
        if stds in (1, True):
            stds = stds_default
        elif stds in (0, False):
            stds = None
    if not isinstance(pctiles, ARRAY_TYPES):
        if pctiles in (1, True):
            pctiles = pctiles_default
        elif pctiles in (0, False):
            pctiles = None

    # Incompatible settings
    if stds is not None and pctiles is not None:
        warnings._warn_proplot(
            'You passed both a standard deviation range and a percentile range for '
            'drawing error indicators. Using the former.'
        )
        pctiles = None
    if not means_or_medians and (stds is not None or pctiles is not None):
        raise ValueError(
            'To automatically compute standard deviations or percentiles on columns '
            'of data you must pass means=True or medians=True.'
        )
    if means_or_medians and errdata is not None:
        stds = pctiles = None
        warnings._warn_proplot(
            'You explicitly provided the error bounds but also requested '
            'automatically calculating means or medians on data columns. '
            'It may make more sense to use the "stds" or "pctiles" keyword args '
            'and have *proplot* calculate the error bounds.'
        )

    # Compute error data in format that can be passed to matplotlib.axes.Axes.errorbar()
    # NOTE: Include option to pass symmetric deviation from central points
    y = _to_ndarray(y)
    data = _to_ndarray(data)
    if errdata is not None:
        label_default = 'error range'
        err = _to_ndarray(errdata)
        if (
            err.ndim not in (1, 2)
            or err.shape[-1] != y.shape[-1]
            or err.ndim == 2 and err.shape[0] != 2
        ):
            raise ValueError(
                f'errdata must have shape (2, {y.shape[-1]}), but got {err.shape}.'
            )
        if err.ndim == 1:
            abserr = err
            err = np.empty((2, err.size))
            err[0, :] = y - abserr  # translated back to absolute deviations below
            err[1, :] = y + abserr
    elif stds is not None:
        label_default = fr'{stds[1]}$\sigma$ range'
        err = y + np.std(data, axis=0)[None, :] * np.asarray(stds)[:, None]
    elif pctiles is not None:
        label_default = f'{pctiles[0]}-{pctiles[1]} percentile range'
        err = np.percentile(data, pctiles, axis=0)
    else:
        raise ValueError('You must provide error bounds.')
    if label == True:  # noqa: E712 e.g. 1, 1.0, True
        label = label_default
    elif not label:
        label = None
    if not absolute:
        err = err - y
        err[0, :] *= -1  # absolute deviations from central points

    # Return data with default legend entry
    return err, label


def _indicate_error_deprecate(func):
    """
    Translate old-style keyword arguments to new-style in way that is too complex
    for _rename_kwargs. Use a decorator to avoid call signature pollution.
    """
    @functools.wraps(func)
    def wrapper(
        *args,
        bars=None, boxes=None, barstd=None, boxstd=None, barrange=None, boxrange=None,
        **kwargs
    ):
        for (prefix, b, std, span) in zip(
            ('bar', 'box'), (bars, boxes), (barstd, boxstd), (barrange, boxrange),
        ):
            if b is not None or std is not None or span is not None:
                warnings._warn_proplot(
                    f"Keyword args '{prefix}s', '{prefix}std', and '{prefix}range' "
                    'are deprecated and will be removed in a future version. '
                    f"Please use '{prefix}stds' or '{prefix}pctiles' instead."
                )
            if span is None and b:  # means 'use the default range'
                span = b
            if std:
                kwargs.setdefault(prefix + 'stds', span)
            else:
                kwargs.setdefault(prefix + 'pctiles', span)
        return func(*args, **kwargs)
    return wrapper


@_indicate_error_deprecate
def _indicate_error(
    self, func, *args,
    medians=False, means=False,
    boxdata=None, bardata=None, shadedata=None, fadedata=None,
    boxstds=None, barstds=None, shadestds=None, fadestds=None,
    boxpctiles=None, barpctiles=None, shadepctiles=None, fadepctiles=None,
    boxmarker=True, boxmarkercolor='white',
    boxcolor=None, barcolor=None, shadecolor=None, fadecolor=None,
    shadelabel=False, fadelabel=False, shadealpha=0.4, fadealpha=0.2,
    boxlinewidth=None, boxlw=None, barlinewidth=None, barlw=None, capsize=None,
    boxzorder=2.5, barzorder=2.5, shadezorder=1.5, fadezorder=1.5,
    **kwargs
):
    """
    Add error bars and/or error shading on-the-fly.
    """
    name = func.__name__
    x, data, *args = args
    x = _to_arraylike(x)
    data = _to_arraylike(data)

    # Get means or medians for plotting
    # NOTE: We can *only* use pctiles and stds if one of these was true
    # TODO: Add support for 3D arrays.
    y = data
    bars = any(_ is not None for _ in (barstds, barpctiles, bardata))
    boxes = any(_ is not None for _ in (boxstds, boxpctiles, boxdata))
    shading = any(_ is not None for _ in (shadestds, shadepctiles, shadedata))
    fading = any(_ is not None for _ in (fadestds, fadepctiles, fadedata))
    if means or medians:
        # Take means or medians while preserving metadata for legends
        # NOTE: Permit 3d array with error dimension coming first
        if not (bars or boxes or shading or fading):
            bars = boxes = True  # toggle these on
            barstds = boxstds = True  # error bars and boxes with default stdev ranges
        if data.ndim != 2:
            raise ValueError(
                f'Need 2D data array for means=True or medians=True, '
                f'got {data.ndim}D array.'
            )
        keep = {}
        if DataArray is not ndarray and isinstance(data, DataArray):
            keep['keep_attrs'] = True
        if means:
            y = data.mean(axis=0, **keep)
        elif medians:
            if hasattr(data, 'quantile'):  # DataFrame and DataArray
                y = data.quantile(0.5, axis=0, **keep)
                if Series is not ndarray and isinstance(y, Series):
                    y.name = ''  # do not set name to quantile number
            else:
                y = np.percentile(data, 50, axis=0, **keep)
        if getattr(data, 'name', '') and not getattr(y, 'name', ''):
            y.name = data.name  # copy DataFrame name to Series name

    # Infer width of error elements
    # NOTE: violinplot_wrapper passes some invalid keyword args with expectation
    # that indicate_error wrapper pops them and uses them for error bars.
    lw = None
    if name == 'bar':
        lw = _not_none(kwargs.get('linewidth', None), kwargs.get('lw', None))
    elif name == 'violinplot':
        lw = _not_none(kwargs.pop('linewidth', None), kwargs.pop('lw', None))
    lw = _not_none(lw, 0.8)
    barlw = _not_none(barlinewidth=barlinewidth, barlw=barlw, default=lw)
    boxlw = _not_none(boxlinewidth=boxlinewidth, boxlw=boxlw, default=4 * barlw)
    capsize = _not_none(capsize, 3.0)

    # Infer color for error bars
    edgecolor = None
    if name == 'bar':
        edgecolor = kwargs.get('edgecolor', None)
    elif name == 'violinplot':
        edgecolor = kwargs.pop('edgecolor', None)
    edgecolor = _not_none(edgecolor, 'k')
    barcolor = _not_none(barcolor, edgecolor)
    boxcolor = _not_none(boxcolor, barcolor)

    # Infer color for shading
    shadecolor_infer = shadecolor is None
    shadecolor = _not_none(
        shadecolor, kwargs.get('color', None), kwargs.get('facecolor', None), edgecolor
    )
    fadecolor_infer = fadecolor is None
    fadecolor = _not_none(fadecolor, shadecolor)

    # Draw dark and light shading
    vert = kwargs.get('vert', kwargs.get('orientation', 'vertical') == 'vertical')
    axis = 'y' if vert else 'x'  # yerr
    errargs = (x, y) if vert else (y, x)
    errobjs = []
    means_or_medians = means or medians
    if fading:
        err, label = _indicate_error_data(
            data, y, fadedata, fadestds, fadepctiles,
            stds_default=(-3, 3), pctiles_default=(0, 100), absolute=True,
            means_or_medians=means_or_medians, label=fadelabel,
        )
        errfunc = self.fill_between if vert else self.fill_betweenx
        errobj = errfunc(
            x, *err, linewidth=0, color=fadecolor,
            alpha=fadealpha, zorder=fadezorder,
        )
        errobj.set_label(label)
        errobjs.append(errobj)
    if shading:
        err, label = _indicate_error_data(
            data, y, shadedata, shadestds, shadepctiles,
            stds_default=(-2, 2), pctiles_default=(10, 90), absolute=True,
            means_or_medians=means_or_medians, label=shadelabel,
        )
        errfunc = self.fill_between if vert else self.fill_betweenx
        errobj = errfunc(
            x, *err, linewidth=0, color=shadecolor,
            alpha=shadealpha, zorder=shadezorder,
        )
        errobj.set_label(label)  # shadelabel=False
        errobjs.append(errobj)

    # Draw thin error bars and thick error boxes
    if boxes:
        err, label = _indicate_error_data(
            data, y, boxdata, boxstds, boxpctiles,
            stds_default=(-1, 1), pctiles_default=(25, 75),
            means_or_medians=means_or_medians,
        )
        if boxmarker:
            self.scatter(*errargs, s=boxlw, marker='o', color=boxmarkercolor, zorder=5)
        errkw = {axis + 'err': err}
        errobj = self.errorbar(
            *errargs, color=boxcolor, linewidth=boxlw, linestyle='none',
            capsize=0, zorder=boxzorder, **errkw,
        )
        errobjs.append(errobj)
    if bars:  # now impossible to make thin bar width different from cap width!
        err, label = _indicate_error_data(
            data, y, bardata, barstds, barpctiles,
            stds_default=(-3, 3), pctiles_default=(0, 100),
            means_or_medians=means_or_medians,
        )
        errkw = {axis + 'err': err}
        errobj = self.errorbar(
            *errargs, color=barcolor, linewidth=barlw, linestyle='none',
            markeredgecolor=barcolor, markeredgewidth=barlw,
            capsize=capsize, zorder=barzorder, **errkw
        )
        errobjs.append(errobj)

    # Call main function
    # NOTE: Provide error objects for inclusion in legend, but *only* provide
    # the shading. Never want legend entries for error bars.
    xy = (x, data) if name == 'violinplot' else (x, y)
    kwargs.setdefault('errobjs', errobjs[:int(shading + fading)])
    result = obj = func(self, *xy, *args, **kwargs)

    # Apply inferrred colors to objects
    if type(result) in (tuple, list):  # avoid BarContainer
        obj = result[0]
    i = 0
    for b, infer in zip((fading, shading), (fadecolor_infer, shadecolor_infer)):
        if b and infer:
            if hasattr(obj, 'get_facecolor'):
                color = obj.get_facecolor()
            elif hasattr(obj, 'get_color'):
                color = obj.get_color()
            else:
                color = None
            if color is not None:
                errobjs[i].set_facecolor(color)
            i += 1

    # Return objects
    # NOTE: This should not affect internal matplotlib calls to these funcs
    # NOTE: Avoid expanding matplolib collections that are list subclasses here
    if errobjs:
        if type(result) in (tuple, list):  # e.g. result of plot
            return (*result, *errobjs)
        else:
            return (result, *errobjs)
    else:
        return result


def _parse_1d(self, func, *args, **kwargs):
    """
    Standardize the positional arguments for 1D data.
    """
    # Sanitize input
    # TODO: Add exceptions for methods other than 'hist'?
    name = func.__name__
    autoformat = rc['autoformat']
    _load_objects()
    if not args:
        return func(self, *args, **kwargs)
    elif len(args) == 1:
        x = None
        y, *args = args
    elif len(args) <= 4:  # max signature is x, y, z, color
        x, y, *args = args
    else:
        raise ValueError(
            f'{name}() takes up to 4 positional arguments but {len(args)} was given.'
        )
    vert = kwargs.get('vert', None)
    if vert is not None:
        orientation = ('vertical' if vert else 'horizontal')
    else:
        orientation = kwargs.get('orientation', 'vertical')

    # Iterate through list of ys that we assume are identical
    # Standardize based on the first y input
    if len(args) >= 1 and 'fill_between' in name:
        ys, args = (y, args[0]), args[1:]
    else:
        ys = (y,)
    ys = [_to_arraylike(y) for y in ys]

    # Auto x coords
    y = ys[0]  # test the first y input
    if x is None:
        axis = int(
            name in ('hist', 'boxplot', 'violinplot')
            or any(kwargs.get(s, None) for s in ('means', 'medians'))
        )
        x, _ = _axis_labels_title(y, axis=axis)
    x = _to_arraylike(x)
    if x.ndim != 1:
        raise ValueError(
            f'x coordinates must be 1-dimensional, but got {x.ndim}.'
        )

    # Auto formatting
    x_index = None  # index version of 'x'
    if not hasattr(self, 'projection'):
        # First handle string-type x-coordinates
        kw = {}
        xname = 'y' if orientation == 'horizontal' else 'x'
        yname = 'x' if xname == 'y' else 'y'
        if _is_string(x):
            if name in ('hist',):
                kwargs.setdefault('labels', list(x))
            else:
                x_index = np.arange(len(x))
                kw[xname + 'locator'] = mticker.FixedLocator(x_index)
                kw[xname + 'formatter'] = mticker.IndexFormatter(x)
                kw[xname + 'minorlocator'] = mticker.NullLocator()
                if name == 'boxplot':  # otherwise IndexFormatter is overridden
                    kwargs['labels'] = x

        # Next handle labels if 'autoformat' is on
        # NOTE: Do not overwrite existing labels!
        if autoformat:
            # Ylabel
            y, label = _axis_labels_title(y)
            iname = xname if name in ('hist',) else yname
            if label and not getattr(self, f'get_{iname}label')():
                # For histograms, this label is used for *x* coordinates
                kw[iname + 'label'] = label
            if name not in ('hist',):
                # Xlabel
                x, label = _axis_labels_title(x)
                if label and not getattr(self, f'get_{xname}label')():
                    kw[xname + 'label'] = label
                # Reversed axis
                if name not in ('scatter',):
                    if x_index is None and len(x) > 1 and x[1] < x[0]:
                        kw[xname + 'reverse'] = True

        # Appply
        if kw:
            self.format(**kw)

    # Standardize args
    if x_index is not None:
        x = x_index
    if name in ('boxplot', 'violinplot'):
        ys = [_to_ndarray(yi) for yi in ys]  # store naked array
        kwargs['positions'] = x

    # Basemap shift x coordiantes without shifting y, we fix this!
    if getattr(self, 'name', '') == 'basemap' and kwargs.get('latlon', None):
        ix, iys = x, []
        xmin, xmax = self.projection.lonmin, self.projection.lonmax
        for y in ys:
            # Ensure data is monotonic and falls within map bounds
            x, y = _parse_2d_monotonic_lons(x, y)
            ix, iy = _parse_2d_enforce_bounds(x, y, xmin, xmax)
            iys.append(iy)
        x, ys = ix, iys

    # WARNING: For some functions, e.g. boxplot and violinplot, we *require*
    # cycle_changer is also applied so it can strip 'x' input.
    return func(self, x, *ys, *args, **kwargs)


def _parse_2d_interp_poles(y, Z):
    """
    Add data points on the poles as the average of highest latitude data.
    """
    # Get means
    with np.errstate(all='ignore'):
        p1 = Z[0, :].mean()  # pole 1, make sure is not 0D DataArray!
        p2 = Z[-1, :].mean()  # pole 2
    if hasattr(p1, 'item'):
        p1 = np.asscalar(p1)  # happens with DataArrays
    if hasattr(p2, 'item'):
        p2 = np.asscalar(p2)
    # Concatenate
    ps = (-90, 90) if (y[0] < y[-1]) else (90, -90)
    Z1 = np.repeat(p1, Z.shape[1])[None, :]
    Z2 = np.repeat(p2, Z.shape[1])[None, :]
    y = ma.concatenate((ps[:1], y, ps[1:]))
    Z = ma.concatenate((Z1, Z, Z2), axis=0)
    return y, Z


def _parse_2d_enforce_bounds(x, y, xmin, xmax):
    """
    Ensure data for basemap plots is restricted between the minimum and
    maximum longitude of the projection. Input is the ``x`` and ``y``
    coordinates. The ``y`` coordinates are rolled along the rightmost axis.
    """
    if x.ndim != 1:
        return x, y
    # Roll in same direction if some points on right-edge extend
    # more than 360 above min longitude; *they* should be on left side
    lonroll = np.where(x > xmin + 360)[0]  # tuple of ids
    if lonroll.size:  # non-empty
        roll = x.size - lonroll.min()
        x = np.roll(x, roll)
        y = np.roll(y, roll, axis=-1)
        x[:roll] -= 360  # make monotonic

    # Set NaN where data not in range xmin, xmax. Must be done
    # for regional smaller projections or get weird side-effects due
    # to having valid data way outside of the map boundaries
    y = y.copy()
    if x.size - 1 == y.shape[-1]:  # test western/eastern grid cell edges
        y[..., (x[1:] < xmin) | (x[:-1] > xmax)] = np.nan
    elif x.size == y.shape[-1]:  # test the centers and pad by one for safety
        where = np.where((x < xmin) | (x > xmax))[0]
        y[..., where[1:-1]] = np.nan
    return x, y


def _parse_2d_monotonic_lons(x, y):
    """
    Ensure longitudes are monotonic and make `~numpy.ndarray` copies so the
    contents can be modified. Ignores 2D coordinate arrays.
    """
    # Sanitization and bail if 2d
    if x.ndim == 1:
        x = ma.array(x)
    if y.ndim == 1:
        y = ma.array(y)
    if x.ndim != 1 or all(x < x[0]):  # skip monotonic backwards data
        return x, y
    # Enforce monotonic longitudes
    lon1 = x[0]
    while True:
        filter_ = (x < lon1)
        if filter_.sum() == 0:
            break
        x[filter_] += 360
    return x, y


def _parse_2d(self, func, *args, order='C', globe=False, **kwargs):
    """
    Standardize the positional arguments for 2D data.
    """
    # Sanitize input
    name = func.__name__
    autoformat = rc['autoformat']
    _load_objects()
    if not args:
        return func(self, *args, **kwargs)
    elif len(args) > 5:
        raise ValueError(
            f'{name}() takes up to 5 positional arguments but {len(args)} was given.'
        )
    x, y = None, None
    if len(args) > 2:
        x, y, *args = args

    # Ensure DataArray, DataFrame or ndarray
    Zs = []
    for Z in args:
        Z = _to_arraylike(Z)
        if Z.ndim != 2:
            raise ValueError(f'Z must be 2-dimensional, got shape {Z.shape}.')
        Zs.append(Z)
    if not all(Zs[0].shape == Z.shape for Z in Zs):
        raise ValueError(
            f'Zs must be same shape, got shapes {[Z.shape for Z in Zs]}.'
        )

    # Retrieve coordinates
    if x is None and y is None:
        Z = Zs[0]
        if order == 'C':
            idx, idy = 1, 0
        else:
            idx, idy = 0, 1
        # x = np.arange(Z.shape[idx])
        # y = np.arange(Z.shape[idy])
        if isinstance(Z, ndarray):
            x = np.arange(Z.shape[idx])
            y = np.arange(Z.shape[idy])
        elif isinstance(Z, DataArray):  # DataArray
            x = Z.coords[Z.dims[idx]]
            y = Z.coords[Z.dims[idy]]
        else:  # DataFrame; never Series or Index because these are 1d
            if order == 'C':
                x = Z.columns
                y = Z.index
            else:
                x = Z.index
                y = Z.columns

    # Optionally re-order
    # TODO: Double check this
    if order == 'F':
        x, y = x.T, y.T  # in case they are 2-dimensional
        Zs = tuple(Z.T for Z in Zs)
    elif order != 'C':
        raise ValueError(
            f'Invalid order {order!r}. Choose from '
            '"C" (row-major, default) and "F" (column-major).'
        )

    # Check coordinates
    x, y = _to_arraylike(x), _to_arraylike(y)
    if x.ndim != y.ndim:
        raise ValueError(
            f'x coordinates are {x.ndim}-dimensional, '
            f'but y coordinates are {y.ndim}-dimensional.'
        )
    for s, array in zip(('x', 'y'), (x, y)):
        if array.ndim not in (1, 2):
            raise ValueError(
                f'{s} coordinates are {array.ndim}-dimensional, '
                f'but must be 1 or 2-dimensional.'
            )

    # Auto axis labels
    # TODO: Check whether isinstance(GeoAxes) instead of checking projection attribute
    kw = {}
    xi = yi = None
    if not hasattr(self, 'projection'):
        # First handle string-type x and y-coordinates
        if _is_string(x):
            xi = np.arange(len(x))
            kw['xlocator'] = mticker.FixedLocator(xi)
            kw['xformatter'] = mticker.IndexFormatter(x)
            kw['xminorlocator'] = mticker.NullLocator()
        if _is_string(y):
            yi = np.arange(len(y))
            kw['ylocator'] = mticker.FixedLocator(yi)
            kw['yformatter'] = mticker.IndexFormatter(y)
            kw['yminorlocator'] = mticker.NullLocator()

        # Handle labels if 'autoformat' is on
        # NOTE: Do not overwrite existing labels!
        if autoformat:
            for key, xy in zip(('xlabel', 'ylabel'), (x, y)):
                # Axis label
                _, label = _axis_labels_title(xy)
                if label and not getattr(self, f'get_{key}')():
                    kw[key] = label
                # Reversed axis
                if (
                    len(xy) > 1
                    and all(isinstance(xy, Number) for xy in xy[:2])
                    and xy[1] < xy[0]
                ):
                    kw[key[0] + 'reverse'] = True
    if kw:
        self.format(**kw)

    # Use *index coordinates* from here on out if input was array of strings
    if xi is not None:
        x = xi
    if yi is not None:
        y = yi

    # Auto axes title and colorbar label
    # NOTE: Do not overwrite existing title!
    # NOTE: Must apply default colorbar label *here* rather than in
    # cmap_args in case metadata is stripped by globe=True.
    colorbar_kw = kwargs.pop('colorbar_kw', None) or {}
    if autoformat:
        _, colorbar_label = _axis_labels_title(Zs[0], units=True)
        colorbar_kw.setdefault('label', colorbar_label)
    kwargs['colorbar_kw'] = colorbar_kw

    # Enforce edges
    if name in ('pcolor', 'pcolormesh', 'pcolorfast'):
        Z = Zs[0]  # already enforced that shapes must be identical (see above)
        xlen, ylen = x.shape[-1], y.shape[0]
        if Z.ndim != 2:
            raise ValueError(
                f'Input arrays must be 2D, instead got shape {Z.shape}.'
            )
        elif Z.shape[1] == xlen and Z.shape[0] == ylen:
            # Get edges given centers
            if all(z.ndim == 1 and z.size > 1 and _is_number(z) for z in (x, y)):
                x = edges(x)
                y = edges(y)
            else:
                if (
                    x.ndim == 2 and x.shape[0] > 1 and x.shape[1] > 1
                    and _is_number(x)
                ):
                    x = edges2d(x)
                if (
                    y.ndim == 2 and y.shape[0] > 1 and y.shape[1] > 1
                    and _is_number(y)
                ):
                    y = edges2d(y)
        elif Z.shape[1] != xlen - 1 or Z.shape[0] != ylen - 1:
            raise ValueError(
                f'Input shapes x {x.shape} and y {y.shape} must match '
                f'Z centers {Z.shape} or '
                f'Z borders {tuple(i+1 for i in Z.shape)}.'
            )

    # Enforce centers
    else:
        Z = Zs[0]  # already enforced that shapes must be identical (see above)
        xlen, ylen = x.shape[-1], y.shape[0]
        if Z.ndim != 2:
            raise ValueError(
                f'Input arrays must be 2d, instead got shape {Z.shape}.'
            )
        elif Z.shape[1] == xlen - 1 and Z.shape[0] == ylen - 1:
            # Get centers given edges.
            if all(z.ndim == 1 and z.size > 1 and _is_number(z) for z in (x, y)):
                x = 0.5 * (x[1:] + x[:-1])
                y = 0.5 * (y[1:] + y[:-1])
            else:
                if (
                    x.ndim == 2 and x.shape[0] > 1 and x.shape[1] > 1
                    and _is_number(x)
                ):
                    x = 0.25 * (x[:-1, :-1] + x[:-1, 1:] + x[1:, :-1] + x[1:, 1:])
                if (
                    y.ndim == 2 and y.shape[0] > 1 and y.shape[1] > 1
                    and _is_number(y)
                ):
                    y = 0.25 * (y[:-1, :-1] + y[:-1, 1:] + y[1:, :-1] + y[1:, 1:])
        elif Z.shape[1] != xlen or Z.shape[0] != ylen:
            raise ValueError(
                f'Input shapes x {x.shape} and y {y.shape} '
                f'must match Z centers {Z.shape} '
                f'or Z borders {tuple(i+1 for i in Z.shape)}.'
            )

    # Cartopy projection axes
    if (
        getattr(self, 'name', '') == 'cartopy'
        and isinstance(kwargs.get('transform', None), PlateCarree)
    ):
        x, y = _parse_2d_monotonic_lons(x, y)
        ix, iZs = x, []
        for Z in Zs:
            if globe and x.ndim == 1 and y.ndim == 1:
                # Fix holes over poles by *interpolating* there
                y, Z = _parse_2d_interp_poles(y, Z)

                # Fix seams by ensuring circular coverage. Unlike basemap,
                # cartopy can plot across map edges.
                if x[0] % 360 != (x[-1] + 360) % 360:
                    ix = ma.concatenate((x, [x[0] + 360]))
                    Z = ma.concatenate((Z, Z[:, :1]), axis=1)
            iZs.append(Z)
        x, Zs = ix, iZs

    # Basemap projection axes
    elif getattr(self, 'name', '') == 'basemap' and kwargs.get('latlon', None):
        # Fix grid
        xmin, xmax = self.projection.lonmin, self.projection.lonmax
        x, y = _parse_2d_monotonic_lons(x, y)
        ix, iZs = x, []
        for Z in Zs:
            # Ensure data is within map bounds
            ix, Z = _parse_2d_enforce_bounds(x, Z, xmin, xmax)

            # Globe coverage fixes
            if globe and ix.ndim == 1 and y.ndim == 1:
                # Fix holes over poles by interpolating there (equivalent to
                # simple mean of highest/lowest latitude points)
                y, Z = _parse_2d_interp_poles(y, Z)

                # Fix seams at map boundary; 3 scenarios here:
                # Have edges (e.g. for pcolor), and they fit perfectly against
                # basemap seams. Does not augment size.
                if ix[0] == xmin and ix.size - 1 == Z.shape[1]:
                    pass  # do nothing
                # Have edges (e.g. for pcolor), and the projection edge is
                # in-between grid cell boundaries. Augments size by 1.
                elif ix.size - 1 == Z.shape[1]:  # just add grid cell
                    ix = ma.append(xmin, ix)
                    ix[-1] = xmin + 360
                    Z = ma.concatenate((Z[:, -1:], Z), axis=1)
                # Have centers (e.g. for contourf), and we need to interpolate
                # to left/right edges of the map boundary. Augments size by 2.
                elif ix.size == Z.shape[1]:
                    xi = np.array([ix[-1], ix[0] + 360])  # x
                    if xi[0] != xi[1]:
                        Zq = ma.concatenate((Z[:, -1:], Z[:, :1]), axis=1)
                        xq = xmin + 360
                        Zq = (
                            Zq[:, :1] * (xi[1] - xq) + Zq[:, 1:] * (xq - xi[0])
                        ) / (xi[1] - xi[0])
                        ix = ma.concatenate(([xmin], ix, [xmin + 360]))
                        Z = ma.concatenate((Zq, Z, Zq), axis=1)
                else:
                    raise ValueError(
                        'Unexpected shape of longitude, latitude, and/or data array(s).'
                    )
            iZs.append(Z)
        x, Zs = ix, iZs

        # Convert to projection coordinates
        if x.ndim == 1 and y.ndim == 1:
            x, y = np.meshgrid(x, y)
        x, y = self.projection(x, y)
        kwargs['latlon'] = False

    # Finally return result
    return func(self, x, y, *Zs, **kwargs)


@warnings._rename_kwargs('0.6', centers='values')
def _parse_cmap(
    self, func, *args, extend='neither',
    cmap=None, cmap_kw=None, norm=None, norm_kw=None,
    vmin=None, vmax=None, N=None, levels=None, values=None,
    symmetric=False, locator=None, locator_kw=None,
    edgefix=None,
    colorbar=False, colorbar_kw=None,
    lw=None, linewidth=None, linewidths=None,
    ls=None, linestyle=None, linestyles=None,
    color=None, colors=None, edgecolor=None, edgecolors=None,
    **kwargs
):
    """
    Interpret colormap keyword arguments.
    """
    name = func.__name__
    if not args:
        return func(self, *args, **kwargs)

    # Mutable inputs
    cmap_kw = cmap_kw or {}
    norm_kw = norm_kw or {}
    locator_kw = locator_kw or {}
    colorbar_kw = colorbar_kw or {}

    # Flexible user input
    Z_sample = args[-1]
    edgefix = _not_none(edgefix, rc['image.edgefix'])
    linewidths = _not_none(lw=lw, linewidth=linewidth, linewidths=linewidths)
    linestyles = _not_none(ls=ls, linestyle=linestyle, linestyles=linestyles)
    colors = _not_none(
        color=color, colors=colors, edgecolor=edgecolor, edgecolors=edgecolors,
    )

    # Get colormap, but do not use cmap when 'colors' are passed to contour()
    # or to contourf() -- the latter only when 'linewidths' and 'linestyles'
    # are also *not* passed. This wrapper lets us add "edges" to contourf
    # plots by calling contour() after contourf() if 'linewidths' or
    # 'linestyles' are explicitly passed, but do not want to disable the
    # native matplotlib feature for manually coloring filled contours.
    # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.contourf
    add_contours = (
        name in ('contourf', 'tricontourf')
        and (linewidths is not None or linestyles is not None)
    )
    no_cmap = colors is not None and (
        name in ('contour', 'tricontour')
        or name in ('contourf', 'tricontourf') and not add_contours
    )
    if no_cmap:
        if cmap is not None:
            warnings._warn_proplot(
                f'Ignoring input colormap cmap={cmap!r}, using input colors '
                f'colors={colors!r} instead.'
            )
            cmap = None
        if name in ('contourf', 'tricontourf'):
            kwargs['colors'] = colors  # this was not done above
            colors = None
    else:
        if cmap is None:
            if name == 'spy':
                cmap = pcolors.ListedColormap(['w', 'k'], '_binary')
            else:
                cmap = rc['image.cmap']
        cmap = constructor.Colormap(cmap, **cmap_kw)
        if getattr(cmap, '_cyclic', None) and extend != 'neither':
            warnings._warn_proplot(
                f'Cyclic colormap requires extend="neither". '
                f'Overriding user input extend={extend!r}.'
            )
            extend = 'neither'

    # Translate standardized keyword arguments back into the keyword args
    # accepted by native matplotlib methods. Also disable edgefix if user want
    # to customize the "edges".
    style_kw = STYLE_ARGS_TRANSLATE.get(name, None)
    for key, value in (
        ('colors', colors),
        ('linewidths', linewidths),
        ('linestyles', linestyles)
    ):
        if value is None or add_contours:
            continue
        if not style_kw or key not in style_kw:  # no known conversion table
            raise TypeError(f'{name}() got an unexpected keyword argument {key!r}')
        edgefix = False  # disable edgefix when specifying borders!
        kwargs[style_kw[key]] = value

    # Build colormap normalizer and update keyword args
    # NOTE: Standard algorithm for obtaining default levels does not work
    # for hexbin, because it colors *counts*, not data values!
    ticks = None
    if cmap is not None and name not in ('hexbin',):
        norm, cmap, levels, ticks = _build_discrete_norm(
            Z_sample,  # sample data for getting suitable levels
            N=N, levels=levels, values=values,
            norm=norm, norm_kw=norm_kw,
            locator=locator, locator_kw=locator_kw,
            cmap=cmap, vmin=vmin, vmax=vmax, extend=extend,
            symmetric=symmetric,
            minlength=(1 if name in ('contour', 'tricontour') else 2),
        )
    if not no_cmap:
        kwargs['cmap'] = cmap
    if norm is not None:
        kwargs['norm'] = norm
    if name in ('contour', 'contourf', 'tricontour', 'tricontourf'):
        kwargs['levels'] = levels
        kwargs['extend'] = extend
    if name in ('parametric',):
        kwargs['values'] = values

    # Call function, possibly twice to add 'edges' to contourf plot
    obj = func(self, *args, **kwargs)
    if not isinstance(obj, tuple):  # hist2d
        obj.extend = extend  # normally 'extend' is just for contour/contourf
        if ticks is not None:
            obj.ticks = ticks  # a Locator or ndarray used for controlling ticks
    if add_contours:
        colors = _not_none(colors, 'k')
        self.contour(
            *args, levels=levels, linewidths=linewidths,
            linestyles=linestyles, colors=colors
        )

    return obj


def _parse_cycle(
    self, func, *args,
    cycle=None, cycle_kw=None,
    label=None, labels=None, values=None,
    errobjs=None,
    **kwargs
):
    """
    Parse the cycle argument.
    """
    # Parse positional args
    # NOTE: Requires standardize_1d wrapper before reaching this. Also note
    # that the 'x' coordinates are sometimes ignored below.
    name = func.__name__
    if not args:
        return func(self, *args, **kwargs)
    x, y, *args = args
    ys = (y,)
    if len(args) >= 1 and name in ('fill_between', 'fill_betweenx'):
        ys, args = (y, args[0]), args[1:]

    # Parse keyword args
    autoformat = rc['autoformat']  # possibly manipulated by standardize_[12]d
    barh = stacked = False
    cycle_kw = cycle_kw or {}
    if name in ('bar', 'fill_between', 'fill_betweenx'):
        stacked = kwargs.pop('stacked', False)
    if name in ('bar',):
        barh = kwargs.get('orientation', None) == 'horizontal'
        width = kwargs.pop('width', 0.8)  # 'width' for bar *and* barh (see bar_wrapper)
        bottom = 'x' if barh else 'bottom'
        kwargs.setdefault(bottom, 0)  # 'x' required even though 'y' isn't for bar plots

    # Determine and temporarily set cycler
    # NOTE: Axes cycle has no getter, only set_prop_cycle, which sets a
    # prop_cycler attribute on the hidden _get_lines and _get_patches_for_fill
    # objects. This is the only way to query current axes cycler! Should not
    # wrap set_prop_cycle because would get messy and fragile.
    # NOTE: The _get_lines cycler is an *itertools cycler*. Has no length, so
    # we must cycle over it with next(). We try calling next() the same number
    # of times as the length of input cycle. If the input cycle *is* in fact
    # the same, below does not reset the color position, cycles us to start!
    if cycle is not None or cycle_kw:
        # Get the new cycler
        cycle_args = () if cycle is None else (cycle,)
        if y.ndim > 1 and y.shape[1] > 1:  # default samples count
            cycle_kw.setdefault('N', y.shape[1])
        cycle = constructor.Cycle(*cycle_args, **cycle_kw)

        # Get the original property cycle
        # NOTE: Matplotlib saves itertools.cycle(cycler), not the original
        # cycler object, so we must build up the keys again.
        i = 0
        by_key = {}
        cycle_orig = self._get_lines.prop_cycler
        for i in range(len(cycle)):  # use the cycler object length as a guess
            prop = next(cycle_orig)
            for key, value in prop.items():
                if key not in by_key:
                    by_key[key] = set()
                if isinstance(value, (list, np.ndarray)):
                    value = tuple(value)
                by_key[key].add(value)

        # Reset property cycler if it differs
        reset = set(by_key) != set(cycle.by_key())
        if not reset:  # test individual entries
            for key, value in cycle.by_key().items():
                if by_key[key] != set(value):
                    reset = True
                    break
        if reset:
            self.set_prop_cycle(cycle)

    # Custom property cycler additions
    # NOTE: By default matplotlib uses _get_patches_for_fill.get_next_color
    # for scatter next scatter color, but cannot get anything else! We simultaneously
    # iterate through the _get_lines property cycler and apply relevant properties.
    apply_from_cycler = set()  # which keys to apply from property cycler
    if name in ('scatter',):
        # Figure out which props should be updated
        prop_keys = set(self._get_lines._prop_keys) - {'color', 'linestyle', 'dashes'}
        for key, prop in (
            ('markersize', 's'),
            ('linewidth', 'linewidths'),
            ('markeredgewidth', 'linewidths'),
            ('markeredgecolor', 'edgecolors'),
            ('alpha', 'alpha'),
            ('marker', 'marker'),
        ):
            prop = kwargs.get(prop, None)
            if key in prop_keys and prop is None:  # if key in cycler and property unset
                apply_from_cycler.add(key)

    # Plot susccessive columns
    y1 = ys[0]
    objs = []
    for i in range(ncols):
        # Prop cycle properties
        kw = kwargs.copy()
        if apply_from_cycler:
            props = next(self._get_lines.prop_cycler)
            for key in apply_from_cycler:
                value = props[key]
                if key in ('size', 'markersize'):
                    key = 's'
                elif key in ('linewidth', 'markeredgewidth'):  # translate
                    key = 'linewidths'
                elif key == 'markeredgecolor':
                    key = 'edgecolors'
                kw[key] = value

        # Get x coordinates for bar plot
        ix = x  # samples
        if name in ('bar',):  # adjust
            if not stacked:
                offset = width * (i - 0.5 * (ncols - 1))
                ix = x + offset
            elif stacked and y1.ndim > 1:
                key = 'x' if barh else 'bottom'
                kw[key] = _to_indexer(y1)[:, :i].sum(axis=1)

        # Get y coordinates and labels
        if name in ('pie', 'boxplot', 'violinplot'):
            # Only ever have one y value, cannot have legend labels
            iys = (y1,)

        else:
            # The coordinates
            # WARNING: If stacked=True then we always *ignore* second
            # argument passed to fill_between. Warning should be issued
            # by fill_between_wrapper in this case.
            if stacked and name in ('fill_between', 'fill_betweenx'):
                iys = tuple(
                    y1 if y1.ndim == 1
                    else _to_indexer(y1)[:, :ii].sum(axis=1)
                    for ii in (i, i + 1)
                )
            else:
                iys = tuple(
                    y_i if y_i.ndim == 1 else _to_indexer(y_i)[:, i]
                    for y_i in ys
                )
            kw['label'] = labels[i] or ''

        # Build coordinate arguments
        ixy = ()
        if barh:  # special case, use kwargs only!
            kw.update({'bottom': ix, 'width': iys[0]})
        elif name in ('pie', 'hist', 'boxplot', 'violinplot'):
            ixy = iys
        else:  # has x-coordinates, and maybe more than one y
            ixy = (ix, *iys)
        obj = func(self, *ixy, *args, **kw)
        if type(obj) in (list, tuple) and len(obj) == 1:
            obj = obj[0]
        objs.append(obj)

    # Return
    # WARNING: Make sure plot always returns tuple of objects, and bar always
    # returns singleton unless we have bulk drawn bar plots! Other matplotlib
    # methods call these internally and expect a certain output format!
    if name == 'plot':
        return tuple(objs)  # always return tuple of objects
    elif name in ('boxplot', 'violinplot'):
        return objs[0]  # always return singleton
    else:
        return objs[0] if len(objs) == 1 else tuple(objs)


def _with_autoformat(func):
    """
    Decorator that adds an `autoformat` keyword and executed the rest
    of the block inside a `plot.config.RcConfiguroator.context`.
    """
    @functools.wraps(func)
    def wrapper(*args, autoformat=None, **kwargs):
        autoformat = _not_none(autoformat, rc['autoformat'])
        with rc.context(autoformat=autoformat):
            return func(*args, **kwargs)
