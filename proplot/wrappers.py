#!/usr/bin/env python3
"""
Wrapper functions applied to various `~proplot.axes.Axes` plotting methods.
In a future version these features will be documented on the individual
plotting methods, but for now they are documented separately.
"""
import sys
import numpy as np
import numpy.ma as ma
import functools
from . import axistools, styletools, utils
from .cbook import (
    _notNone, _warn_proplot, _validate_colorbar_loc, _validate_legend_loc,
)
import matplotlib.axes as maxes
import matplotlib.contour as mcontour
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as mpatheffects
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.artist as martist
import matplotlib.legend as mlegend
from numbers import Number
from .rctools import rc
try:  # use this for debugging instead of print()!
    from icecream import ic
except ImportError:  # graceful fallback if IceCream isn't installed
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa
try:
    from cartopy.crs import PlateCarree
except ModuleNotFoundError:
    PlateCarree = object

__all__ = [
    'add_errorbars', 'bar_wrapper', 'barh_wrapper', 'boxplot_wrapper',
    'default_crs', 'default_latlon', 'default_transform',
    'cmap_changer',
    'cycle_changer',
    'colorbar_wrapper',
    'fill_between_wrapper', 'fill_betweenx_wrapper', 'hist_wrapper',
    'legend_wrapper', 'plot_wrapper', 'scatter_wrapper',
    'standardize_1d', 'standardize_2d', 'text_wrapper',
    'violinplot_wrapper',
]


def _load_objects():
    """Delay loading expensive modules. We just want to detect if *input
    arrays* belong to these types -- and if this is the case, it means the
    module has already been imported! So, we only try loading these classes
    within autoformat calls. This saves >~500ms of import time."""
    global DataArray, DataFrame, Series, Index, ndarray
    ndarray = np.ndarray
    DataArray = getattr(sys.modules.get('xarray', None), 'DataArray', ndarray)
    DataFrame = getattr(sys.modules.get('pandas', None), 'DataFrame', ndarray)
    Series = getattr(sys.modules.get('pandas', None), 'Series', ndarray)
    Index = getattr(sys.modules.get('pandas', None), 'Index', ndarray)


_load_objects()

# Keywords for styling cmap overridden plots
# TODO: Deprecate this when #45 merged! Pcolor *already* accepts lw,
# linewidth, *and* linewidths!
STYLE_ARGS_TRANSLATE = {
    'contour': {
        'colors': 'colors',
        'linewidths': 'linewidths',
        'linestyles': 'linestyles'},
    'hexbin': {
        'colors': 'edgecolors',
        'linewidths': 'linewidths'},
    'tricontour': {
        'colors': 'colors',
        'linewidths': 'linewidths',
        'linestyles': 'linestyles'},
    'parametric': {
        'colors': 'color',
        'linewidths': 'linewidth',
        'linestyles': 'linestyle'},
    'pcolor': {
        'colors': 'edgecolors',
        'linewidths': 'linewidth',
        'linestyles': 'linestyle'},
    'tripcolor': {
        'colors': 'edgecolors',
        'linewidths': 'linewidth',
        'linestyles': 'linestyle'},
    'pcolormesh': {
        'colors': 'edgecolors',
        'linewidths': 'linewidth',
        'linestyles': 'linestyle'},
}


def default_latlon(self, func, *args, latlon=True, **kwargs):
    """
    Wraps %(methods)s for `~proplot.axes.BasemapAxes`.
    With the default `~mpl_toolkits.basemap` API, you need to pass
    ``latlon=True`` if your data coordinates are longitude and latitude
    instead of map projection coordinates. Now, this is the default.
    """
    return func(self, *args, latlon=latlon, **kwargs)


def default_transform(self, func, *args, transform=None, **kwargs):
    """
    Wraps %(methods)s for `~proplot.axes.GeoAxes`.
    With the default `~cartopy.mpl.geoaxes.GeoAxes` API, you need to pass
    ``transform=cartopy.crs.PlateCarree()`` if your data coordinates are
    longitude and latitude instead of map projection coordinates. Now,
    this is the default.
    """
    # Apply default transform
    # TODO: Do some cartopy methods reset backgroundpatch or outlinepatch?
    # Deleted comment reported this issue
    if transform is None:
        transform = PlateCarree()
    result = func(self, *args, transform=transform, **kwargs)
    return result


def default_crs(self, func, *args, crs=None, **kwargs):
    """
    Wraps %(methods)s for `~proplot.axes.GeoAxes` and fixes a
    `~cartopy.mpl.geoaxes.GeoAxes.set_extent` bug associated with tight
    bounding boxes.

    With the default `~cartopy.mpl.geoaxes.GeoAxes` API, you need to pass
    ``crs=cartopy.crs.PlateCarree()`` if your data coordinates are
    longitude and latitude instead of map projection coordinates. Now,
    this is the default.
    """
    # Apply default crs
    name = func.__name__
    if crs is None:
        crs = PlateCarree()
    try:
        result = func(self, *args, crs=crs, **kwargs)
    except TypeError as err:  # duplicate keyword args, i.e. crs is positional
        if not args:
            raise err
        result = func(self, *args[:-1], crs=args[-1], **kwargs)
    # Fix extent, so axes tight bounding box gets correct box!
    # From this issue:
    # https://github.com/SciTools/cartopy/issues/1207#issuecomment-439975083
    if name == 'set_extent':
        clipped_path = self.outline_patch.orig_path.clip_to_bbox(self.viewLim)
        self.outline_patch._path = clipped_path
        self.background_patch._path = clipped_path
    return result


def _to_iloc(data):
    """Get indexible attribute of array, so we can perform axis
    wise operations."""
    return getattr(data, 'iloc', data)


def _to_array(data):
    """Convert to ndarray cleanly."""
    data = getattr(data, 'values', data)
    return np.array(data)


def _atleast_array(data):
    """Converts list of lists to array."""
    _load_objects()
    if not isinstance(data, (ndarray, DataArray, DataFrame, Series, Index)):
        data = np.array(data)
    if not np.iterable(data):
        data = np.atleast_1d(data)
    return data


def _auto_label(data, axis=None, units=True):
    """Gets data and label for pandas or xarray objects or
    their coordinates."""
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
        if axis == 0 and isinstance(data, (DataFrame, Series)):
            data = data.index
        elif axis == 1 and isinstance(data, DataFrame):
            data = data.columns
        elif axis is not None:
            data = np.arange(len(data))  # e.g. for Index
        # DataFrame has no native name attribute but user can add one:
        # https://github.com/pandas-dev/pandas/issues/447
        label = getattr(data, 'name', '') or ''
    return data, str(label).strip()


def standardize_1d(self, func, *args, **kwargs):
    """
    Wraps %(methods)s, standardizes acceptable positional args and optionally
    modifies the x axis label, y axis label, title, and axis ticks
    if a `~xarray.DataArray`, `~pandas.DataFrame`, or `~pandas.Series` is
    passed.

    Positional args are standardized as follows.

    * If a 2D array is passed, the corresponding plot command is called for
      each column of data (except for ``boxplot`` and ``violinplot``, in which
      case each column is interpreted as a distribution).
    * If *x* and *y* or *latitude* and *longitude* coordinates were not
      provided, and a `~pandas.DataFrame` or `~xarray.DataArray`, we
      try to infer them from the metadata. Otherwise,
      ``np.arange(0, data.shape[0])`` is used.
   """
    # Sanitize input
    # TODO: Add exceptions for methods other than 'hist'?
    name = func.__name__
    _load_objects()
    if not args:
        return func(self, *args, **kwargs)
    elif len(args) == 1:
        x = None
        y, *args = args
    elif len(args) in (2, 3, 4):
        x, y, *args = args  # same
    else:
        raise ValueError(f'Too many arguments passed to {name}. Max is 4.')
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
    ys = [_atleast_array(y) for y in ys]

    # Auto x coords
    y = ys[0]  # test the first y input
    if x is None:
        axis = 1 if (name in ('hist', 'boxplot', 'violinplot') or any(
            kwargs.get(s, None) for s in ('means', 'medians'))) else 0
        x, _ = _auto_label(y, axis=axis)
    x = _atleast_array(x)
    if x.ndim != 1:
        raise ValueError(
            f'x coordinates must be 1-dimensional, but got {x.ndim}.'
        )

    # Auto formatting
    xi = None  # index version of 'x'
    if not hasattr(self, 'projection'):
        # First handle string-type x-coordinates
        kw = {}
        xaxis = 'y' if (orientation == 'horizontal') else 'x'
        yaxis = 'x' if xaxis == 'y' else 'y'
        if _to_array(x).dtype == 'object':
            xi = np.arange(len(x))
            kw[xaxis + 'locator'] = mticker.FixedLocator(xi)
            kw[xaxis + 'formatter'] = mticker.IndexFormatter(x)
            kw[xaxis + 'minorlocator'] = mticker.NullLocator()
            if name == 'boxplot':
                kwargs['labels'] = x
            elif name == 'violinplot':
                kwargs['positions'] = xi
        if name in ('boxplot', 'violinplot'):
            kwargs['positions'] = xi
        # Next handle labels if 'autoformat' is on
        if self.figure._auto_format:
            # Ylabel
            y, label = _auto_label(y)
            if label:
                # for histogram, this indicates x coordinate
                iaxis = xaxis if name in ('hist',) else yaxis
                kw[iaxis + 'label'] = label
            # Xlabel
            x, label = _auto_label(x)
            if label and name not in ('hist',):
                kw[xaxis + 'label'] = label
            if name != 'scatter' and len(x) > 1 and xi is None and x[1] < x[0]:
                kw[xaxis + 'reverse'] = True
        # Appply
        if kw:
            self.format(**kw)

    # Standardize args
    if xi is not None:
        x = xi
    if name in ('boxplot', 'violinplot'):
        ys = [_to_array(yi) for yi in ys]  # store naked array

    # Basemap shift x coordiantes without shifting y, we fix this!
    if getattr(self, 'name', '') == 'basemap' and kwargs.get('latlon', None):
        ix, iys = x, []
        xmin, xmax = self.projection.lonmin, self.projection.lonmax
        for y in ys:
            # Ensure data is monotonic and falls within map bounds
            ix, iy = _enforce_bounds(*_standardize_latlon(x, y), xmin, xmax)
            iys.append(iy)
        x, ys = ix, iys

    # WARNING: For some functions, e.g. boxplot and violinplot, we *require*
    # cycle_changer is also applied so it can strip 'x' input.
    return func(self, x, *ys, *args, **kwargs)


def _interp_poles(y, Z):
    """Adds data points on the poles as the average of highest
    latitude data."""
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


def _standardize_latlon(x, y):
    """Ensures monotonic longitudes and makes `~numpy.ndarray` copies so the
    contents can be modified. Ignores 2D coordinate arrays."""
    # Sanitization and bail if 2D
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


def _enforce_bounds(x, y, xmin, xmax):
    """Ensures data for basemap plots is restricted between the minimum and
    maximum longitude of the projection. Input is the ``x`` and ``y``
    coordinates. The ``y`` coordinates are rolled along the rightmost axis."""
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


def standardize_2d(self, func, *args, order='C', globe=False, **kwargs):
    """
    Wraps %(methods)s, standardizes acceptable positional args and optionally
    modifies the x axis label, y axis label, title, and axis ticks if the
    a `~xarray.DataArray`, `~pandas.DataFrame`, or `~pandas.Series` is passed.

    Positional args are standardized as follows.

    * If *x* and *y* or *latitude* and *longitude* coordinates were not
      provided, and a `~pandas.DataFrame` or `~xarray.DataArray` is passed, we
      try to infer them from the metadata. Otherwise,
      ``np.arange(0, data.shape[0])`` and ``np.arange(0, data.shape[1])``
      are used.
    * For ``pcolor`` and ``pcolormesh``, coordinate *edges* are calculated
      if *centers* were provided. For all other methods, coordinate *centers*
      are calculated if *edges* were provided.

    For `~proplot.axes.GeoAxes` and `~proplot.axes.BasemapAxes`, the
    `globe` keyword arg is added, suitable for plotting datasets with global
    coverage. Passing ``globe=True`` does the following.

    1. "Interpolates" input data to the North and South poles.
    2. Makes meridional coverage "circular", i.e. the last longitude coordinate
       equals the first longitude coordinate plus 360\N{DEGREE SIGN}.

    For `~proplot.axes.BasemapAxes`, 1D longitude vectors are also cycled to
    fit within the map edges. For example, if the projection central longitude
    is 90\N{DEGREE SIGN}, the data is shifted so that it spans
    -90\N{DEGREE SIGN} to 270\N{DEGREE SIGN}.
    """
    # Sanitize input
    name = func.__name__
    _load_objects()
    if not args:
        return func(self, *args, **kwargs)
    elif len(args) > 4:
        raise ValueError(f'Too many arguments passed to {name}. Max is 4.')
    x, y = None, None
    if len(args) > 2:
        x, y, *args = args

    # Ensure DataArray, DataFrame or ndarray
    Zs = []
    for Z in args:
        Z = _atleast_array(Z)
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
        if order == 'C':  # TODO: check order stuff works
            idx, idy = 1, 0
        else:
            idx, idy = 0, 1
        if isinstance(Z, ndarray):
            x = np.arange(Z.shape[idx])
            y = np.arange(Z.shape[idy])
        elif isinstance(Z, DataArray):  # DataArray
            x = Z.coords[Z.dims[idx]]
            y = Z.coords[Z.dims[idy]]
        else:  # DataFrame; never Series or Index because these are 1D
            x = Z.index
            y = Z.columns

    # Check coordinates
    x, y = _atleast_array(x), _atleast_array(y)
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

    # Auto formatting
    kw = {}
    xi, yi = None, None
    if not hasattr(self, 'projection'):
        # First handle string-type x and y-coordinates
        if _to_array(x).dtype == 'object':
            xi = np.arange(len(x))
            kw['xlocator'] = mticker.FixedLocator(xi)
            kw['xformatter'] = mticker.IndexFormatter(x)
            kw['xminorlocator'] = mticker.NullLocator()
        if _to_array(y).dtype == 'object':
            yi = np.arange(len(y))
            kw['ylocator'] = mticker.FixedLocator(yi)
            kw['yformatter'] = mticker.IndexFormatter(y)
            kw['yminorlocator'] = mticker.NullLocator()
        # Handle labels if 'autoformat' is on
        if self.figure._auto_format:
            for key, xy in zip(('xlabel', 'ylabel'), (x, y)):
                _, label = _auto_label(xy)
                if label:
                    kw[key] = label
                if len(xy) > 1 and all(isinstance(xy, Number)
                                       for xy in xy[:2]) and xy[1] < xy[0]:
                    kw[key[0] + 'reverse'] = True
    if xi is not None:
        x = xi
    if yi is not None:
        y = yi
    # Handle figure titles
    if self.figure._auto_format:
        _, colorbar_label = _auto_label(Zs[0], units=True)
        _, title = _auto_label(Zs[0], units=False)
        if title:
            kw['title'] = title
        if kw:
            self.format(**kw)

    # Enforce edges
    if name in ('pcolor', 'pcolormesh'):
        # Get centers or raise error. If 2D, don't raise error, but don't fix
        # either, because matplotlib pcolor just trims last column and row.
        xlen, ylen = x.shape[-1], y.shape[0]
        for Z in Zs:
            if Z.ndim != 2:
                raise ValueError(
                    f'Input arrays must be 2D, instead got shape {Z.shape}.'
                )
            elif Z.shape[1] == xlen and Z.shape[0] == ylen:
                if all(z.ndim == 1 and z.size > 1
                       and z.dtype != 'object' for z in (x, y)):
                    x = utils.edges(x)
                    y = utils.edges(y)
                else:
                    if (x.ndim == 2 and x.shape[0] > 1 and x.shape[1] > 1
                            and x.dtype != 'object'):
                        x = utils.edges2d(x)
                    if (y.ndim == 2 and y.shape[0] > 1 and y.shape[1] > 1
                            and y.dtype != 'object'):
                        y = utils.edges2d(y)
            elif Z.shape[1] != xlen - 1 or Z.shape[0] != ylen - 1:
                raise ValueError(
                    f'Input shapes x {x.shape} and y {y.shape} must match '
                    f'Z centers {Z.shape} or '
                    f'Z borders {tuple(i+1 for i in Z.shape)}.'
                )
        # Optionally re-order
        # TODO: Double check this
        if order == 'F':
            x, y = x.T, y.T  # in case they are 2-dimensional
            Zs = (Z.T for Z in Zs)
        elif order != 'C':
            raise ValueError(
                f'Invalid order {order!r}. Choose from '
                '"C" (row-major, default) and "F" (column-major).'
            )

    # Enforce centers
    else:
        # Get centers given edges. If 2D, don't raise error, let matplotlib
        # raise error down the line.
        xlen, ylen = x.shape[-1], y.shape[0]
        for Z in Zs:
            if Z.ndim != 2:
                raise ValueError(
                    f'Input arrays must be 2D, instead got shape {Z.shape}.'
                )
            elif Z.shape[1] == xlen - 1 and Z.shape[0] == ylen - 1:
                if all(z.ndim == 1 and z.size > 1
                        and z.dtype != 'object' for z in (x, y)):
                    x = (x[1:] + x[:-1]) / 2
                    y = (y[1:] + y[:-1]) / 2
                else:
                    if (x.ndim == 2 and x.shape[0] > 1 and x.shape[1] > 1
                            and x.dtype != 'object'):
                        x = 0.25 * (x[:-1, :-1] + x[:-1, 1:]
                                    + x[1:, :-1] + x[1:, 1:])
                    if (y.ndim == 2 and y.shape[0] > 1 and y.shape[1] > 1
                            and y.dtype != 'object'):
                        y = 0.25 * (y[:-1, :-1] + y[:-1, 1:]
                                    + y[1:, :-1] + y[1:, 1:])
            elif Z.shape[1] != xlen or Z.shape[0] != ylen:
                raise ValueError(
                    f'Input shapes x {x.shape} and y {y.shape} '
                    f'must match Z centers {Z.shape} '
                    f'or Z borders {tuple(i+1 for i in Z.shape)}.'
                )
        # Optionally re-order
        # TODO: Double check this
        if order == 'F':
            x, y = x.T, y.T  # in case they are 2-dimensional
            Zs = (Z.T for Z in Zs)
        elif order != 'C':
            raise ValueError(
                f'Invalid order {order!r}. Choose from '
                '"C" (row-major, default) and "F" (column-major).'
            )

    # Cartopy projection axes
    if (getattr(self, 'name', '') == 'geo'
            and isinstance(kwargs.get('transform', None), PlateCarree)):
        x, y = _standardize_latlon(x, y)
        ix, iZs = x, []
        for Z in Zs:
            if globe and x.ndim == 1 and y.ndim == 1:
                # Fix holes over poles by *interpolating* there
                y, Z = _interp_poles(y, Z)

                # Fix seams by ensuring circular coverage. Unlike basemap,
                # cartopy can plot across map edges.
                if (x[0] % 360) != ((x[-1] + 360) % 360):
                    ix = ma.concatenate((x, [x[0] + 360]))
                    Z = ma.concatenate((Z, Z[:, :1]), axis=1)
            iZs.append(Z)
        x, Zs = ix, iZs

    # Basemap projection axes
    elif getattr(self, 'name', '') == 'basemap' and kwargs.get('latlon', None):
        # Fix grid
        xmin, xmax = self.projection.lonmin, self.projection.lonmax
        x, y = _standardize_latlon(x, y)
        ix, iZs = x, []
        for Z in Zs:
            # Ensure data is within map bounds
            ix, Z = _enforce_bounds(x, Z, xmin, xmax)

            # Globe coverage fixes
            if globe and ix.ndim == 1 and y.ndim == 1:
                # Fix holes over poles by interpolating there (equivalent to
                # simple mean of highest/lowest latitude points)
                y, Z = _interp_poles(y, Z)

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
                        Zq = (Zq[:, :1] * (xi[1] - xq) + Zq[:, 1:]
                              * (xq - xi[0])) / (xi[1] - xi[0])
                        ix = ma.concatenate(([xmin], ix, [xmin + 360]))
                        Z = ma.concatenate((Zq, Z, Zq), axis=1)
                else:
                    raise ValueError(
                        'Unexpected shape of longitude/latitude/data arrays.'
                    )
            iZs.append(Z)
        x, Zs = ix, iZs

        # Convert to projection coordinates
        if x.ndim == 1 and y.ndim == 1:
            x, y = np.meshgrid(x, y)
        x, y = self.projection(x, y)
        kwargs['latlon'] = False

    # Finally return result
    # WARNING: Must apply default colorbar label *here* in case metadata
    # was stripped by globe=True.
    colorbar_kw = kwargs.pop('colorbar_kw', None) or {}
    colorbar_kw.setdefault('label', colorbar_label)
    return func(self, x, y, *Zs, colorbar_kw=colorbar_kw, **kwargs)


def _errorbar_values(data, idata, bardata=None, barrange=None, barstd=False):
    """Returns values that can be passed to the `~matplotlib.axes.Axes.errorbar`
    `xerr` and `yerr` keyword args."""
    if bardata is not None:
        err = np.array(bardata)
        if err.ndim == 1:
            err = err[:, None]
        if err.ndim != 2 or err.shape[0] != 2 \
                or err.shape[1] != idata.shape[-1]:
            raise ValueError(
                f'bardata must have shape (2, {idata.shape[-1]}), '
                f'but got {err.shape}.'
            )
    elif barstd:
        err = np.array(idata) + np.std(
            data, axis=0)[None, :] * np.array(barrange)[:, None]
    else:
        err = np.percentile(data, barrange, axis=0)
    err = err - np.array(idata)
    err[0, :] *= -1  # array now represents error bar sizes
    return err


def add_errorbars(
    self, func, *args,
    medians=False, means=False,
    boxes=None, bars=None,
    boxdata=None, bardata=None,
    boxstd=False, barstd=False,
    boxmarker=True, boxmarkercolor='white',
    boxrange=(25, 75), barrange=(5, 95), boxcolor=None, barcolor=None,
    boxlw=None, barlw=None, capsize=None,
    boxzorder=3, barzorder=3,
    **kwargs
):
    """
    Wraps %(methods)s, adds support for drawing error bars. Includes
    options for interpreting columns of data as ranges, representing the mean
    or median of each column with lines, points, or bars, and drawing error
    bars representing percentile ranges or standard deviation multiples for
    the data in each column.

    Parameters
    ----------
    *args
        The input data.
    bars : bool, optional
        Toggles *thin* error bars with optional "whiskers" (i.e. caps). Default
        is ``True`` when `means` is ``True``, `medians` is ``True``, or
        `bardata` is not ``None``.
    boxes : bool, optional
        Toggles *thick* boxplot-like error bars with a marker inside
        representing the mean or median. Default is ``True`` when `means` is
        ``True``, `medians` is ``True``, or `boxdata` is not ``None``.
    means : bool, optional
        Whether to plot the means of each column in the input data.
    medians : bool, optional
        Whether to plot the medians of each column in the input data.
    bardata, boxdata : 2xN ndarray, optional
        Arrays that manually specify the thin and thick error bar coordinates.
        The first row contains lower bounds, and the second row contains
        upper bounds. Columns correspond to points in the dataset.
    barstd, boxstd : bool, optional
        Whether `barrange` and `boxrange` refer to multiples of the standard
        deviation, or percentile ranges. Default is ``False``.
    barrange : (float, float), optional
        Percentile ranges or standard deviation multiples for drawing thin
        error bars. The defaults are ``(-3,3)`` (i.e. +/-3 standard deviations)
        when `barstd` is ``True``, and ``(0,100)`` (i.e. the full data range)
        when `barstd` is ``False``.
    boxrange : (float, float), optional
        Percentile ranges or standard deviation multiples for drawing thick
        error bars. The defaults are ``(-1,1)`` (i.e. +/-1 standard deviation)
        when `boxstd` is ``True``, and ``(25,75)`` (i.e. the middle 50th
        percentile) when `boxstd` is ``False``.
    barcolor, boxcolor : color-spec, optional
        Colors for the thick and thin error bars. Default is ``'k'``.
    barlw, boxlw : float, optional
        Line widths for the thin and thick error bars, in points. Default
        `barlw` is ``0.7`` and default `boxlw` is ``4*barlw``.
    boxmarker : bool, optional
        Whether to draw a small marker in the middle of the box denoting
        the mean or median position. Ignored if `boxes` is ``False``.
        Default is ``True``.
    boxmarkercolor : color-spec, optional
        Color for the `boxmarker` marker. Default is ``'w'``.
    capsize : float, optional
        The cap size for thin error bars, in points.
    barzorder, boxzorder : float, optional
        The "zorder" for the thin and thick error bars.
    lw, linewidth : float, optional
        If passed, this is used for the default `barlw`.
    edgecolor : float, optional
        If passed, this is used for the default `barcolor` and `boxcolor`.
    """
    name = func.__name__
    x, y, *args = args
    # Sensible defaults
    if boxdata is not None:
        bars = _notNone(bars, True)
    if bardata is not None:
        boxes = _notNone(boxes, True)
    if boxdata is not None or bardata is not None:
        # e.g. if boxdata passed but bardata not passed, use bars=False
        bars = _notNone(bars, False)
        boxes = _notNone(boxes, False)

    # Get means or medians for plotting
    iy = y
    if (means or medians):
        bars = _notNone(bars, True)
        boxes = _notNone(boxes, True)
        if y.ndim != 2:
            raise ValueError(
                f'Need 2D data array for means=True or medians=True, '
                f'got {y.ndim}D array.'
            )
        if means:
            iy = np.mean(y, axis=0)
        elif medians:
            iy = np.percentile(y, 50, axis=0)

    # Call function, accounting for different signatures of plot and violinplot
    get = kwargs.pop if name == 'violinplot' else kwargs.get
    lw = _notNone(get('lw', None), get('linewidth', None), 0.7)
    get = kwargs.pop if name != 'bar' else kwargs.get
    edgecolor = _notNone(get('edgecolor', None), 'k')
    if name == 'violinplot':
        xy = (x, y)  # full data
    else:
        xy = (x, iy)  # just the stats
    obj = func(self, *xy, *args, **kwargs)
    if not boxes and not bars:
        return obj

    # Account for horizontal bar plots
    if 'vert' in kwargs:
        orientation = 'vertical' if kwargs['vert'] else 'horizontal'
    else:
        orientation = kwargs.get('orientation', 'vertical')
    if orientation == 'horizontal':
        axis = 'x'  # xerr
        xy = (iy, x)
    else:
        axis = 'y'  # yerr
        xy = (x, iy)

    # Defaults settings
    barlw = _notNone(barlw, lw)
    boxlw = _notNone(boxlw, 4 * barlw)
    capsize = _notNone(capsize, 3)
    barcolor = _notNone(barcolor, edgecolor)
    boxcolor = _notNone(boxcolor, edgecolor)

    # Draw boxes and bars
    if boxes:
        default = (-1, 1) if barstd else (25, 75)
        boxrange = _notNone(boxrange, default)
        err = _errorbar_values(y, iy, boxdata, boxrange, boxstd)
        if boxmarker:
            self.scatter(*xy, marker='o', color=boxmarkercolor,
                         s=boxlw, zorder=5)
        self.errorbar(*xy, **{
            axis + 'err': err, 'capsize': 0, 'zorder': boxzorder,
            'color': boxcolor, 'linestyle': 'none', 'linewidth': boxlw})
    if bars:  # now impossible to make thin bar width different from cap width!
        default = (-3, 3) if barstd else (0, 100)
        barrange = _notNone(barrange, default)
        err = _errorbar_values(y, iy, bardata, barrange, barstd)
        self.errorbar(*xy, **{
            axis + 'err': err, 'capsize': capsize, 'zorder': barzorder,
            'color': barcolor, 'linewidth': barlw, 'linestyle': 'none',
            'markeredgecolor': barcolor, 'markeredgewidth': barlw})
    return obj


def plot_wrapper(self, func, *args, cmap=None, values=None, **kwargs):
    """
    Wraps %(methods)s, draws a "colormap line" if the `cmap` argument was
    passed. "Colormap lines" change color as a function of the parametric
    coordinate `values` using the input colormap `cmap`.

    Parameters
    ----------
    *args : (y,), (x,y), or (x,y,fmt)
        Passed to `~matplotlib.axes.Axes.plot`.
    cmap, values : optional
        Passed to `~proplot.axes.Axes.parametric`.
    **kwargs
        `~matplotlib.lines.Line2D` properties.
    """
    if len(args) > 3:  # e.g. with fmt string
        raise ValueError(f'Expected 1-3 positional args, got {len(args)}.')
    if cmap is None:
        lines = func(self, *args, values=values, **kwargs)
    else:
        lines = self.parametric(*args, cmap=cmap, values=values, **kwargs)
    return lines


def scatter_wrapper(
    self, func, *args,
    s=None, size=None, markersize=None,
    c=None, color=None, markercolor=None,
    smin=None, smax=None,
    cmap=None, cmap_kw=None, vmin=None, vmax=None, norm=None, norm_kw=None,
    lw=None, linewidth=None, linewidths=None,
    markeredgewidth=None, markeredgewidths=None,
    edgecolor=None, edgecolors=None,
    markeredgecolor=None, markeredgecolors=None,
    **kwargs
):
    """
    Wraps `~matplotlib.axes.Axes.scatter`, adds optional keyword args
    more consistent with the `~matplotlib.axes.Axes.plot` keywords.

    Parameters
    ----------
    s, size, markersize : float or list of float, optional
        Aliases for the marker size.
    smin, smax : float, optional
        Used to scale the `s` array. These are the minimum and maximum marker
        sizes. Defaults are the minimum and maximum of the `s` array.
    c, color, markercolor : color-spec or list thereof, or array, optional
        Aliases for the marker fill color. If just an array of values, the
        colors will be generated by passing the values through the `norm`
        normalizer and drawing from the `cmap` colormap.
    cmap : colormap-spec, optional
        The colormap specifer, passed to the `~proplot.styletools.Colormap`
        constructor.
    cmap_kw : dict-like, optional
        Passed to `~proplot.styletools.Colormap`.
    vmin, vmax : float, optional
        Used to generate a `norm` for scaling the `c` array. These are the
        values corresponding to the leftmost and rightmost colors in the
        colormap. Defaults are the minimum and maximum values of the `c` array.
    norm : normalizer spec, optional
        The colormap normalizer, passed to the `~proplot.styletools.Norm`
        constructor.
    norm_kw : dict, optional
        Passed to `~proplot.styletools.Norm`.
    lw, linewidth, linewidths, markeredgewidth, markeredgewidths : \
float or list thereof, optional
        Aliases for the marker edge width.
    edgecolors, markeredgecolor, markeredgecolors : \
color-spec or list thereof, optional
        Aliases for the marker edge color.
    **kwargs
        Passed to `~matplotlib.axes.Axes.scatter`.
    """
    # Manage input arguments
    # NOTE: Parse 1D must come before this
    nargs = len(args)
    if len(args) > 4:
        raise ValueError(f'Expected 1-4 positional args, got {nargs}.')
    args = list(args)
    if len(args) == 4:
        c = args.pop(1)
    if len(args) == 3:
        s = args.pop(0)

    # Format cmap and norm
    cmap_kw = cmap_kw or {}
    norm_kw = norm_kw or {}
    if cmap is not None:
        cmap = styletools.Colormap(cmap, **cmap_kw)
    if norm is not None:
        norm = styletools.Norm(norm, **norm_kw)

    # Apply some aliases for keyword arguments
    c = _notNone(c, color, markercolor, None,
                 names=('c', 'color', 'markercolor'))
    s = _notNone(s, size, markersize, None,
                 names=('s', 'size', 'markersize'))
    lw = _notNone(
        lw, linewidth, linewidths, markeredgewidth, markeredgewidths, None,
        names=(
            'lw', 'linewidth', 'linewidths',
            'markeredgewidth', 'markeredgewidths'
        ))
    ec = _notNone(
        edgecolor, edgecolors, markeredgecolor, markeredgecolors, None,
        names=(
            'edgecolor', 'edgecolors', 'markeredgecolor', 'markeredgecolors'
        ))

    # Scale s array
    if np.iterable(s):
        smin_true, smax_true = min(s), max(s)
        if smin is None:
            smin = smin_true
        if smax is None:
            smax = smax_true
        s = smin + (smax - smin) * (np.array(s) - smin_true) / \
            (smax_true - smin_true)
    return func(self, *args, c=c, s=s,
                cmap=cmap, vmin=vmin, vmax=vmax,
                norm=norm, linewidths=lw, edgecolors=ec,
                **kwargs)


def _fill_between_apply(
    self, func, *args,
    negcolor='blue', poscolor='red', negpos=False,
    **kwargs
):
    """Parse args and call function."""
    # Allow common keyword usage
    x = 'y' if 'x' in func.__name__ else 'y'
    y = 'x' if x == 'y' else 'y'
    if x in kwargs:
        args = (kwargs.pop(x), *args)
    for y in (y + '1', y + '2'):
        if y in kwargs:
            args = (*args, kwargs.pop(y))
    if len(args) == 1:
        args = (np.arange(len(args[0])), *args)
    if len(args) == 2:
        if kwargs.get('stacked', False):
            args = (*args, 0)
        else:
            args = (args[0], 0, args[1])  # default behavior
    if len(args) != 3:
        raise ValueError(f'Expected 2-3 positional args, got {len(args)}.')
    if not negpos:
        obj = func(self, *args, **kwargs)
        return obj

    # Get zero points
    objs = []
    kwargs.setdefault('interpolate', True)
    y1, y2 = np.atleast_1d(
        args[-2]).squeeze(), np.atleast_1d(args[-1]).squeeze()
    if y1.ndim > 1 or y2.ndim > 1:
        raise ValueError(f'When "negpos" is True, y must be 1-dimensional.')
    if kwargs.get('where', None) is not None:
        raise ValueError(
            'When "negpos" is True, you cannot set the "where" keyword.'
        )
    for i in range(2):
        kw = {**kwargs}
        kw.setdefault('color', negcolor if i == 0 else poscolor)
        where = (y2 < y1) if i == 0 else (y2 >= y1)
        obj = func(self, *args, where=where, **kw)
        objs.append(obj)
    return (*objs,)


def fill_between_wrapper(self, func, *args, **kwargs):
    """
    Wraps `~matplotlib.axes.Axes.fill_between`, also accessible via the
    `~proplot.axes.Axes.area` alias.

    Parameters
    ----------
    *args : (y1,), (x,y1), or (x,y1,y2)
        The *x* and *y* coordinates. If `x` is not provided, it will be
        inferred from `y1`. If `y1` and `y2` are provided, their shapes
        must be identical, and we fill between respective columns of these
        arrays.
    stacked : bool, optional
        If `y2` is ``None``, this indicates whether to "stack" successive
        columns of the `y1` array.
    negpos : bool, optional
        Whether to shade where `y2` is greater than `y1` with the color
        `poscolor`, and where `y1` is greater than `y2` with the color
        `negcolor`. For example, to shade positive values red and negtive blue,
        use ``ax.fill_between(x, 0, y)``.
    negcolor, poscolor : color-spec, optional
        Colors to use for the negative and positive values. Ignored if `negpos`
        is ``False``.
    where : ndarray, optional
        Boolean ndarray mask for points you want to shade. See `this example \
<https://matplotlib.org/3.1.0/gallery/pyplots/whats_new_98_4_fill_between.html#sphx-glr-gallery-pyplots-whats-new-98-4-fill-between-py>`__.
    **kwargs
        Passed to `~matplotlib.axes.Axes.fill_between`.
    """  # noqa
    return _fill_between_apply(self, func, *args, **kwargs)


def fill_betweenx_wrapper(self, func, *args, **kwargs):
    """Wraps %(methods)s, also accessible via the `~proplot.axes.Axes.areax`
    alias. Usage is same as `fill_between_wrapper`."""
    return _fill_between_apply(self, func, *args, **kwargs)


def hist_wrapper(self, func, x, bins=None, **kwargs):
    """Wraps %(methods)s, enforces that all arguments after `bins` are
    keyword-only and sets the default patch linewidth to ``0``."""
    kwargs.setdefault('linewidth', 0)
    return func(self, x, bins=bins, **kwargs)


def barh_wrapper(
    self, func, y=None, width=None, height=0.8, left=None, **kwargs
):
    """Wraps %(methods)s, usage is same as `bar_wrapper`."""
    kwargs.setdefault('orientation', 'horizontal')
    if y is None and width is None:
        raise ValueError(
            f'barh() requires at least 1 positional argument, got 0.'
        )
    return self.bar(x=left, height=height, width=width, bottom=y, **kwargs)


def bar_wrapper(
    self, func, x=None, height=None, width=0.8, bottom=None, *, left=None,
    vert=None, orientation='vertical', stacked=False,
    lw=None, linewidth=0.7, edgecolor='k',
    **kwargs
):
    """
    Wraps %(methods)s, permits bar stacking and bar grouping.

    Parameters
    ----------
    x, height, width, bottom : float or list of float, optional
        The dimensions of the bars. If the *x* coordinates are not provided,
        they are set to ``np.arange(0, len(height))``.
    orientation : {'vertical', 'horizontal'}, optional
        The orientation of the bars.
    vert : bool, optional
        Alternative to the `orientation` keyword arg. If ``False``, horizontal
        bars are drawn. This is for consistency with
        `~matplotlib.axes.Axes.boxplot` and `~matplotlib.axes.Axes.violinplot`.
    stacked : bool, optional
        Whether to stack columns of input data, or plot the bars side-by-side.
    edgecolor : color-spec, optional
        The edge color for the bar patches.
    lw, linewidth : float, optional
        The edge width for the bar patches.
    """
    # Barh converts y-->bottom, left-->x, width-->height, height-->width.
    # Convert back to (x, bottom, width, height) so we can pass stuff through
    # cycle_changer.
    # NOTE: You *must* do juggling of barh keyword order --> bar keyword order
    # --> barh keyword order, because horizontal hist passes arguments to bar
    # directly and will not use a 'barh' method with overridden argument order!
    if vert is not None:
        orientation = ('vertical' if vert else 'horizontal')
    if orientation == 'horizontal':
        x, bottom = bottom, x
        width, height = height, width

    # Parse args
    # TODO: Stacked feature is implemented in `cycle_changer`, but makes more
    # sense do document here; figure out way to move it here?
    if left is not None:
        _warn_proplot(
            f'The "left" keyword with bar() is deprecated. Use "x" instead.'
        )
        x = left
    if x is None and height is None:
        raise ValueError(
            f'bar() requires at least 1 positional argument, got 0.'
        )
    elif height is None:
        x, height = None, x

    # Call func
    # TODO: This *must* also be wrapped by cycle_changer, which ultimately
    # permutes back the x/bottom args for horizontal bars! Need to clean up.
    lw = _notNone(lw, linewidth, None, names=('lw', 'linewidth'))
    return func(self, x, height, width=width, bottom=bottom,
                linewidth=lw, edgecolor=edgecolor,
                stacked=stacked, orientation=orientation,
                **kwargs)


def boxplot_wrapper(
    self, func, *args,
    color='k', fill=True, fillcolor=None, fillalpha=0.7,
    lw=None, linewidth=0.7, orientation=None,
    marker=None, markersize=None,
    boxcolor=None, boxlw=None,
    capcolor=None, caplw=None,
    meancolor=None, meanlw=None,
    mediancolor=None, medianlw=None,
    whiskercolor=None, whiskerlw=None,
    fliercolor=None, flierlw=None,
    **kwargs
):
    """
    Wraps %(methods)s, adds convenient keyword args.
    Fills the objects with a cycle color by default.

    Parameters
    ----------
    *args : 1D or 2D ndarray
        The data array.
    color : color-spec, optional
        The color of all objects.
    fill : bool, optional
        Whether to fill the box with a color.
    fillcolor : color-spec, optional
        The fill color for the boxes. Default is the next color cycler color.
    fillalpha : float, optional
        The opacity of the boxes. Default is ``1``.
    lw, linewidth : float, optional
        The linewidth of all objects.
    orientation : {None, 'horizontal', 'vertical'}, optional
        Alternative to the native `vert` keyword arg. Controls orientation.
    marker : marker-spec, optional
        Marker style for the 'fliers', i.e. outliers.
    markersize : float, optional
        Marker size for the 'fliers', i.e. outliers.
    boxcolor, capcolor, meancolor, mediancolor, whiskercolor : \
color-spec, optional
        The color of various boxplot components. These are shorthands so you
        don't have to pass e.g. a ``boxprops`` dictionary.
    boxlw, caplw, meanlw, medianlw, whiskerlw : float, optional
        The line width of various boxplot components. These are shorthands so
        you don't have to pass e.g. a ``boxprops`` dictionary.
    """
    # Call function
    if len(args) > 2:
        raise ValueError(f'Expected 1-2 positional args, got {len(args)}.')
    if orientation is not None:
        if orientation == 'horizontal':
            kwargs['vert'] = False
        elif orientation != 'vertical':
            raise ValueError(
                'Orientation must be "horizontal" or "vertical", '
                f'got {orientation!r}.'
            )
    obj = func(self, *args, **kwargs)
    if not args:
        return obj

    # Modify results
    # TODO: Pass props keyword args instead? Maybe does not matter.
    lw = _notNone(lw, linewidth, None, names=('lw', 'linewidth'))
    if fillcolor is None:
        cycler = next(self._get_lines.prop_cycler)
        fillcolor = cycler.get('color', None)
    for key, icolor, ilw in (
        ('boxes', boxcolor, boxlw),
        ('caps', capcolor, caplw),
        ('whiskers', whiskercolor, whiskerlw),
        ('means', meancolor, meanlw),
        ('medians', mediancolor, medianlw),
        ('fliers', fliercolor, flierlw),
    ):
        if key not in obj:  # possible if not rendered
            continue
        artists = obj[key]
        ilw = _notNone(ilw, lw)
        icolor = _notNone(icolor, color)
        for artist in artists:
            if icolor is not None:
                artist.set_color(icolor)
                artist.set_markeredgecolor(icolor)
            if ilw is not None:
                artist.set_linewidth(ilw)
                artist.set_markeredgewidth(ilw)
            if key == 'boxes' and fill:
                patch = mpatches.PathPatch(
                    artist.get_path(), color=fillcolor,
                    alpha=fillalpha, linewidth=0)
                self.add_artist(patch)
            if key == 'fliers':
                if marker is not None:
                    artist.set_marker(marker)
                if markersize is not None:
                    artist.set_markersize(markersize)
    return obj


def violinplot_wrapper(
    self, func, *args,
    lw=None, linewidth=0.7, fillcolor=None, edgecolor='k',
    fillalpha=0.7, orientation=None,
    **kwargs
):
    """
    Wraps %(methods)s, adds convenient keyword args.
    Makes the style shown in right plot of `this matplotlib example \
<https://matplotlib.org/3.1.0/gallery/statistics/customized_violin.html>`__
    the default. It is also no longer possible to show minima and maxima with
    whiskers, because this is redundant.

    Parameters
    ----------
    *args : 1D or 2D ndarray
        The data array.
    lw, linewidth : float, optional
        The linewidth of the line objects. Default is ``1``.
    edgecolor : color-spec, optional
        The edge color for the violin patches. Default is ``'k'``.
    fillcolor : color-spec, optional
        The violin plot fill color. Default is the next color cycler color.
    fillalpha : float, optional
        The opacity of the violins. Default is ``1``.
    orientation : {None, 'horizontal', 'vertical'}, optional
        Alternative to the native `vert` keyword arg. Controls orientation.
    boxrange, barrange : (float, float), optional
        Percentile ranges for the thick and thin central bars. The defaults
        are ``(25, 75)`` and ``(5, 95)``, respectively.
    """
    # Orientation and checks
    if len(args) > 2:
        raise ValueError(f'Expected 1-2 positional args, got {len(args)}.')
    if orientation is not None:
        if orientation == 'horizontal':
            kwargs['vert'] = False
        elif orientation != 'vertical':
            raise ValueError(
                'Orientation must be "horizontal" or "vertical", '
                f'got {orientation!r}.'
            )

    # Sanitize input
    lw = _notNone(lw, linewidth, None, names=('lw', 'linewidth'))
    if kwargs.pop('showextrema', None):
        _warn_proplot(f'Ignoring showextrema=True.')
    if 'showmeans' in kwargs:
        kwargs.setdefault('means', kwargs.pop('showmeans'))
    if 'showmedians' in kwargs:
        kwargs.setdefault('medians', kwargs.pop('showmedians'))
    kwargs.setdefault('capsize', 0)
    obj = func(self, *args,
               showmeans=False, showmedians=False, showextrema=False,
               edgecolor=edgecolor, lw=lw, **kwargs)
    if not args:
        return obj

    # Modify body settings
    for artist in obj['bodies']:
        artist.set_alpha(fillalpha)
        artist.set_edgecolor(edgecolor)
        artist.set_linewidths(lw)
        if fillcolor is not None:
            artist.set_facecolor(fillcolor)
    return obj


def _get_transform(self, transform):
    """Translates user input transform. Also used in an axes method."""
    try:
        from cartopy.crs import CRS
    except ModuleNotFoundError:
        CRS = None
    cartopy = (getattr(self, 'name', '') == 'geo')
    if (isinstance(transform, mtransforms.Transform)
            or CRS and isinstance(transform, CRS)):
        return transform
    elif transform == 'figure':
        return self.figure.transFigure
    elif transform == 'axes':
        return self.transAxes
    elif transform == 'data':
        return PlateCarree() if cartopy else self.transData
    elif cartopy and transform == 'map':
        return self.transData
    else:
        raise ValueError(f'Unknown transform {transform!r}.')


def text_wrapper(
    self, func,
    x=0, y=0, text='', transform='data',
    fontfamily=None, fontname=None, fontsize=None, size=None,
    border=False, bordercolor='w', invert=False, lw=None, linewidth=2,
    **kwargs
):
    """
    Wraps %(methods)s, and enables specifying `tranform` with a string name and
    adds feature for drawing borders around text.

    Parameters
    ----------
    x, y : float
        The *x* and *y* coordinates for the text.
    text : str
        The text string.
    transform : {'data', 'axes', 'figure'} or \
`~matplotlib.transforms.Transform`, optional
        The transform used to interpret `x` and `y`. Can be a
        `~matplotlib.transforms.Transform` object or a string representing the
        `~matplotlib.axes.Axes.transData`, `~matplotlib.axes.Axes.transAxes`,
        or `~matplotlib.figure.Figure.transFigure` transforms. Default is
        ``'data'``, i.e. the text is positioned in data coordinates.
    size, fontsize : float or str, optional
        The font size. If float, units are inches. If string, units are
        interpreted by `~proplot.utils.units`.
    fontname, fontfamily : str, optional
        Aliases for the ``fontfamily`` `~matplotlib.text.Text` property.
    border : bool, optional
        Whether to draw border around text.
    bordercolor : color-spec, optional
        The color of the border. Default is ``'w'``.
    invert : bool, optional
        If ``False``, ``'color'`` is used for the text and ``bordercolor``
        for the border. If ``True``, this is inverted.
    lw, linewidth : float, optional
        Ignored if `border` is ``False``. The width of the text border.

    Other parameters
    ----------------
    **kwargs
        Passed to `~matplotlib.text.Text` instantiator.
    """
    # Default transform by string name
    if not transform:
        transform = self.transData
    else:
        transform = _get_transform(self, transform)

    # More flexible keyword args and more helpful warning if invalid font
    # is specified
    fontname = _notNone(
        fontfamily, fontname, None,
        names=('fontfamily', 'fontname')
    )
    if fontname is not None:
        if not isinstance(fontname, str) and np.iterable(
                fontname) and len(fontname) == 1:
            fontname = fontname[0]
        if fontname.lower() in list(map(str.lower, styletools.fonts)):
            kwargs['fontfamily'] = fontname
        else:
            _warn_proplot(
                f'Font {fontname!r} unavailable. Available fonts are '
                + ', '.join(map(repr, styletools.fonts)) + '.'
            )
    size = _notNone(fontsize, size, None, names=('fontsize', 'size'))
    if size is not None:
        kwargs['fontsize'] = utils.units(size, 'pt')
    # text.color is ignored sometimes unless we apply this
    kwargs.setdefault('color', rc['text.color'])
    obj = func(self, x, y, text, transform=transform, **kwargs)

    # Optionally draw border around text
    if border:
        linewidth = lw or linewidth
        facecolor, bgcolor = kwargs['color'], bordercolor
        if invert:
            facecolor, bgcolor = bgcolor, facecolor
        kwargs = {'linewidth': linewidth,
                  'foreground': bgcolor, 'joinstyle': 'miter'}
        obj.update({
            'color': facecolor,
            'zorder': 100,
            'path_effects': [
                mpatheffects.Stroke(**kwargs), mpatheffects.Normal()
            ]
        })
    return obj


def cycle_changer(
    self, func, *args,
    cycle=None, cycle_kw=None,
    markers=None, linestyles=None,
    label=None, labels=None, values=None,
    legend=None, legend_kw=None,
    colorbar=None, colorbar_kw=None,
    **kwargs
):
    """
    Wraps methods that use the property cycler (%(methods)s),
    adds features for controlling colors in the property cycler and drawing
    legends or colorbars in one go.

    This wrapper also *standardizes acceptable input* -- these methods now all
    accept 2D arrays holding columns of data, and *x*-coordinates are always
    optional. Note this alters the behavior of `~matplotlib.axes.Axes.boxplot`
    and `~matplotlib.axes.Axes.violinplot`, which now compile statistics on
    *columns* of data instead of *rows*.

    Parameters
    ----------
    cycle : cycle-spec, optional
        The cycle specifer, passed to the `~proplot.styletools.Cycle`
        constructor. If the returned list of colors is unchanged from the
        current axes color cycler, the axes cycle will **not** be reset to the
        first position.
    cycle_kw : dict-like, optional
        Passed to `~proplot.styletools.Cycle`.
    label : float or str, optional
        The legend label to be used for this plotted element.
    labels, values : list of float or list of str, optional
        Used with 2D input arrays. The legend labels or colorbar coordinates
        for each column in the array. Can be numeric or string, and must match
        the number of columns in the 2D array.
    legend : bool, int, or str, optional
        If not ``None``, this is a location specifying where to draw an *inset*
        or *panel* legend from the resulting handle(s). If ``True``, the
        default location is used. Valid locations are described in
        `~proplot.axes.Axes.legend`.
    legend_kw : dict-like, optional
        Ignored if `legend` is ``None``. Extra keyword args for our call
        to `~proplot.axes.Axes.legend`.
    colorbar : bool, int, or str, optional
        If not ``None``, this is a location specifying where to draw an *inset*
        or *panel* colorbar from the resulting handle(s). If ``True``, the
        default location is used. Valid locations are described in
        `~proplot.axes.Axes.colorbar`.
    colorbar_kw : dict-like, optional
        Ignored if `colorbar` is ``None``. Extra keyword args for our call
        to `~proplot.axes.Axes.colorbar`.

    Other parameters
    ----------------
    *args, **kwargs
        Passed to the matplotlib plotting method.

    See also
    --------
    `~proplot.styletools.Cycle`, `~proplot.styletools.colors`

    Notes
    -----
    See the `matplotlib source \
<https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/axes/_base.py>`_.
    The `set_prop_cycle` command modifies underlying
    `_get_lines` and `_get_patches_for_fill`.
    """
    # No mutable defaults
    cycle_kw = cycle_kw or {}
    legend_kw = legend_kw or {}
    colorbar_kw = colorbar_kw or {}

    # Test input
    # NOTE: Requires standardize_1d wrapper before reaching this. Also note
    # that the 'x' coordinates are sometimes ignored below.
    name = func.__name__
    if not args:
        return func(self, *args, **kwargs)
    barh = (name == 'bar' and kwargs.get('orientation', None) == 'horizontal')
    x, y, *args = args
    if len(args) >= 1 and 'fill_between' in name:
        ys = (y, args[0])
        args = args[1:]
    else:
        ys = (y,)
    is1d = (y.ndim == 1)

    # Determine and temporarily set cycler
    # NOTE: Axes cycle has no getter, only set_prop_cycle, which sets a
    # prop_cycler attribute on the hidden _get_lines and _get_patches_for_fill
    # objects. This is the only way to query current axes cycler! Should not
    # wrap set_prop_cycle because would get messy and fragile.
    # NOTE: The _get_lines cycler is an *itertools cycler*. Has no length, so
    # we must cycle over it with next(). We try calling next() the same number
    # of times as the length of user input cycle. If the input cycle *is* in
    # fact the same, below does not reset the color position, cycles us to
    # start!
    if cycle is not None or cycle_kw:
        # Get the new cycler
        cycle_args = () if cycle is None else (cycle,)
        if not is1d and y.shape[1] > 1:  # default samples count
            cycle_kw.setdefault('N', y.shape[1])
        cycle = styletools.Cycle(*cycle_args, **cycle_kw)
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
                    by_key[key] = {*()}  # set
                if isinstance(value, (list, np.ndarray)):
                    value = tuple(value)
                by_key[key].add(value)
        # Reset property cycler if it differs
        reset = ({*by_key} != {*cycle.by_key()})  # reset if keys are different
        if not reset:  # test individual entries
            for key, value in cycle.by_key().items():
                if by_key[key] != {*value}:
                    reset = True
                    break
        if reset:
            self.set_prop_cycle(cycle)

    # Custom property cycler additions
    # NOTE: By default matplotlib uses _get_patches_for_fill.get_next_color
    # for scatter properties! So we simultaneously iterate through the
    # _get_lines property cycler and apply them.
    apply = {*()}  # which keys to apply from property cycler
    if name == 'scatter':
        # Figure out which props should be updated
        keys = {*self._get_lines._prop_keys} - {'color', 'linestyle', 'dashes'}
        for key, prop in (
            ('markersize', 's'),
            ('linewidth', 'linewidths'),
            ('markeredgewidth', 'linewidths'),
            ('markeredgecolor', 'edgecolors'),
            ('alpha', 'alpha'),
            ('marker', 'marker'),
        ):
            prop = kwargs.get(prop, None)
            if key in keys and prop is None:
                apply.add(key)

    # Plot susccessive columns
    # WARNING: Most methods that accept 2D arrays use columns of data, but when
    # pandas DataFrame passed to hist, boxplot, or violinplot, rows of data
    # assumed! This is fixed in parse_1d by converting to values.
    objs = []
    ncols = 1
    label_leg = None  # for colorbar or legend
    labels = _notNone(values, labels, label, None,
                      names=('values', 'labels', 'label'))
    stacked = kwargs.pop('stacked', False)
    if name in ('pie', 'boxplot', 'violinplot'):
        if labels is not None:
            kwargs['labels'] = labels
    else:
        ncols = (1 if is1d else y.shape[1])
        if labels is None or isinstance(labels, str):
            labels = [labels] * ncols
    if name in ('bar',):
        # for bar plots; 0.8 is matplotlib default
        width = kwargs.pop('width', 0.8)
        kwargs['height' if barh else 'width'] = (
            width if stacked else width / ncols)
    for i in range(ncols):
        # Prop cycle properties
        kw = {**kwargs}  # copy
        if apply:
            props = next(self._get_lines.prop_cycler)
            for key in apply:
                value = props[key]
                if key in ('size', 'markersize'):
                    key = 's'
                elif key in ('linewidth', 'markeredgewidth'):  # translate
                    key = 'linewidths'
                elif key == 'markeredgecolor':
                    key = 'edgecolors'
                kw[key] = value
        # Get x coordinates
        ix, iy = x, ys[0]  # samples
        if name in ('pie',):
            kw['labels'] = _notNone(labels, ix)  # TODO: move to pie wrapper?
        if name in ('bar',):  # adjust
            if not stacked:
                ix = x + (i - ncols / 2 + 0.5) * width / ncols
            elif stacked and not is1d:
                key = 'x' if barh else 'bottom'
                # sum of empty slice will be zero
                kw[key] = _to_iloc(iy)[:, :i].sum(axis=1)
        # Get y coordinates and labels
        if name in ('pie', 'boxplot', 'violinplot'):
            iys = (iy,)  # only ever have one y value, cannot have legend labs
        else:
            # The coordinates
            if stacked and 'fill_between' in name:
                iys = tuple(iy if is1d else _to_iloc(
                    iy)[:, :j].sum(axis=1) for j in (i, i + 1))
            else:
                iys = tuple(iy if is1d else _to_iloc(iy)[:, i] for iy in ys)
            # Possible legend labels
            if len(labels) != ncols:
                raise ValueError(
                    f'Got {ncols} columns in data array, '
                    f'but {len(labels)} labels.'
                )
            label = labels[i]
            values, label_leg = _auto_label(iy, axis=1)
            if label_leg and label is None:
                label = _to_array(values)[i]
            if label is not None:
                kw['label'] = label
        # Call with correct args
        xy = ()
        if barh:  # special, use kwargs only!
            kw.update({'bottom': ix, 'width': iys[0]})
            # must always be provided
            kw.setdefault('x', kwargs.get('bottom', 0))
        elif name in ('pie', 'hist', 'boxplot', 'violinplot'):
            xy = (*iys,)
        else:  # has x-coordinates, and maybe more than one y
            xy = (ix, *iys)
        obj = func(self, *xy, *args, **kw)
        # plot always returns list or tuple
        if isinstance(obj, (list, tuple)) and len(obj) == 1:
            obj = obj[0]
        objs.append(obj)

    # Add colorbar and/or legend
    if colorbar:
        # Add handles
        loc = _validate_colorbar_loc(colorbar, default=rc['colorbar.loc'])
        if loc not in self._auto_colorbar:
            self._auto_colorbar[loc] = ([], {})
        self._auto_colorbar[loc][0].extend(objs)
        # Add keywords
        if loc != 'fill':
            colorbar_kw.setdefault('loc', loc)
        if label_leg:
            colorbar_kw.setdefault('label', label_leg)
        self._auto_colorbar[loc][1].update(colorbar_kw)
    if legend:
        # Add handles
        loc = _validate_legend_loc(legend, default=rc['legend.loc'])
        if loc not in self._auto_legend:
            self._auto_legend[loc] = ([], {})
        self._auto_legend[loc][0].extend(objs)
        # Add keywords
        if loc != 'fill':
            legend_kw.setdefault('loc', loc)
        if label_leg:
            legend_kw.setdefault('label', label_leg)
        self._auto_legend[loc][1].update(legend_kw)

    # Return
    # WARNING: Make sure plot always returns tuple of objects, and bar always
    # returns singleton unless we have bulk drawn bar plots! Other matplotlib
    # methods call these internally!
    if name == 'plot':
        return (*objs,)  # always return tuple of objects
    elif name in ('boxplot', 'violinplot'):
        # always singleton, because these methods accept the whole 2D object
        return objs[0]
    else:
        return objs[0] if is1d else (*objs,)  # sensible default behavior


def cmap_changer(
    self, func, *args, cmap=None, cmap_kw=None,
    extend='neither', norm=None, norm_kw=None,
    N=None, levels=None, values=None, centers=None, vmin=None, vmax=None,
    locator=None, symmetric=False, locator_kw=None,
    edgefix=None, labels=False, labels_kw=None, fmt=None, precision=2,
    colorbar=False, colorbar_kw=None,
    lw=None, linewidth=None, linewidths=None,
    ls=None, linestyle=None, linestyles=None,
    color=None, colors=None, edgecolor=None, edgecolors=None,
    **kwargs
):
    """
    Wraps methods that take a `cmap` argument (%(methods)s),
    adds several new keyword args and features.
    Uses the `~proplot.styletools.BinNorm` normalizer to bin data into
    discrete color levels (see notes).

    Parameters
    ----------
    cmap : colormap spec, optional
        The colormap specifer, passed to the `~proplot.styletools.Colormap`
        constructor.
    cmap_kw : dict-like, optional
        Passed to `~proplot.styletools.Colormap`.
    norm : normalizer spec, optional
        The colormap normalizer, used to warp data before passing it
        to `~proplot.styletools.BinNorm`. This is passed to the
        `~proplot.styletools.Norm` constructor.
    norm_kw : dict-like, optional
        Passed to `~proplot.styletools.Norm`.
    extend : {'neither', 'min', 'max', 'both'}, optional
        Where to assign unique colors to out-of-bounds data and draw
        "extensions" (triangles, by default) on the colorbar.
    levels, N : int or list of float, optional
        The number of level edges, or a list of level edges. If the former,
        `locator` is used to generate this many levels at "nice" intervals.
        Default is :rc:`image.levels`.

        Since this function also wraps `~matplotlib.axes.Axes.pcolor` and
        `~matplotlib.axes.Axes.pcolormesh`, this means they now
        accept the `levels` keyword arg. You can now discretize your
        colors in a ``pcolor`` plot just like with ``contourf``.
    values, centers : int or list of float, optional
        The number of level centers, or a list of level centers. If provided,
        levels are inferred using `~proplot.utils.edges`. This will override
        any `levels` input.
    vmin, vmax : float, optional
        Used to determine level locations if `levels` is an integer. Actual
        levels may not fall exactly on `vmin` and `vmax`, but the minimum
        level will be no smaller than `vmin` and the maximum level will be
        no larger than `vmax`.

        If `vmin` or `vmax` is not provided, the minimum and maximum data
        values are used.
    locator : locator-spec, optional
        The locator used to determine level locations if `levels` or `values`
        is an integer and `vmin` and `vmax` were not provided. Passed to the
        `~proplot.axistools.Locator` constructor. Default is
        `~matplotlib.ticker.MaxNLocator` with ``levels`` or ``values+1``
        integer levels.
    locator_kw : dict-like, optional
        Passed to `~proplot.axistools.Locator`.
    symmetric : bool, optional
        Toggle this to make automatically generated levels symmetric
        about zero.
    edgefix : bool, optional
        Whether to fix the the `white-lines-between-filled-contours \
<https://stackoverflow.com/q/8263769/4970632>`__
        and `white-lines-between-pcolor-rectangles \
<https://stackoverflow.com/q/27092991/4970632>`__
        issues. This slows down figure rendering by a bit. Default is
        :rc:`image.edgefix`.
    labels : bool, optional
        For `~matplotlib.axes.Axes.contour`, whether to add contour labels
        with `~matplotlib.axes.Axes.clabel`. For `~matplotlib.axes.Axes.pcolor`
        or `~matplotlib.axes.Axes.pcolormesh`, whether to add labels to the
        center of grid boxes. In the latter case, the text will be black
        when the luminance of the underlying grid box color is >50%%, and
        white otherwise (see the `~proplot.styletools` documentation).
    labels_kw : dict-like, optional
        Ignored if `labels` is ``False``. Extra keyword args for the labels.
        For `~matplotlib.axes.Axes.contour`, passed to
        `~matplotlib.axes.Axes.clabel`.  For `~matplotlib.axes.Axes.pcolor`
        or `~matplotlib.axes.Axes.pcolormesh`, passed to
        `~matplotlib.axes.Axes.text`.
    fmt : format-spec, optional
        Passed to the `~proplot.styletools.Norm` constructor, used to format
        number labels. You can also use the `precision` keyword arg.
    precision : int, optional
        Maximum number of decimal places for the number labels.
        Number labels are generated with the
        `~proplot.axistools.SimpleFormatter` formatter, which allows us to
        limit the precision.
    colorbar : bool, int, or str, optional
        If not ``None``, this is a location specifying where to draw an *inset*
        or *panel* colorbar from the resulting mappable. If ``True``, the
        default location is used. Valid locations are described in
        `~proplot.axes.Axes.colorbar`.
    colorbar_kw : dict-like, optional
        Ignored if `colorbar` is ``None``. Extra keyword args for our call
        to `~proplot.axes.Axes.colorbar`.

    Other parameters
    ----------------
    lw, linewidth, linewidths
        The width of `~matplotlib.axes.Axes.contour` lines and
        `~proplot.axes.Axes.parametric` lines. Also the width of lines
        *between* `~matplotlib.axes.Axes.pcolor` boxes,
        `~matplotlib.axes.Axes.pcolormesh` boxes, and
        `~matplotlib.axes.Axes.contourf` filled contours.
    ls, linestyle, linestyles
        As above, but for the line style.
    color, colors, edgecolor, edgecolors
        As above, but for the line color.
    *args, **kwargs
        Passed to the matplotlib plotting method.

    Notes
    -----
    The `~proplot.styletools.BinNorm` normalizer, used with all colormap
    plots, makes sure that your "levels" always span the full range of colors
    in the colormap, whether you are extending max, min, neither, or both. By
    default, when you select `extend` not ``'both'``, matplotlib seems to just
    cut off the most intense colors (reserved for coloring "out of bounds"
    data), even though they are not being used.

    This could also be done by limiting the number of colors in the colormap
    lookup table by selecting a smaller ``N`` (see
    `~matplotlib.colors.LinearSegmentedColormap`).  But I prefer the approach
    of always building colormaps with hi-res lookup tables, and leaving the job
    of normalizing data values to colormap locations to the
    `~matplotlib.colors.Normalize` object.

    See also
    --------
    `~proplot.styletools.Colormap`, `~proplot.styletools.Norm`,
    `~proplot.styletools.BinNorm`
    """
    # No mutable defaults
    cmap_kw = cmap_kw or {}
    norm_kw = norm_kw or {}
    locator_kw = locator_kw or {}
    labels_kw = labels_kw or {}
    colorbar_kw = colorbar_kw or {}

    # Parse args
    # Disable edgefix=True for certain keyword combos e.g. if user wants
    # white lines around their pcolor mesh.
    name = func.__name__
    if not args:
        return func(self, *args, **kwargs)
    vmin = _notNone(
        vmin, norm_kw.pop('vmin', None), None,
        names=('vmin', 'norm_kw={"vmin":value}'))
    vmax = _notNone(
        vmax, norm_kw.pop('vmax', None), None,
        names=('vmax', 'norm_kw={"vmax":value}'))
    levels = _notNone(
        N, levels, norm_kw.pop('levels', None), rc['image.levels'],
        names=('N', 'levels', 'norm_kw={"levels":value}'))
    values = _notNone(
        values, centers, None,
        names=('values', 'centers'))
    colors = _notNone(
        color, colors, edgecolor, edgecolors, None,
        names=('color', 'colors', 'edgecolor', 'edgecolors'))
    linewidths = _notNone(
        lw, linewidth, linewidths, None,
        names=('lw', 'linewidth', 'linewidths'))
    linestyles = _notNone(
        ls, linestyle, linestyles, None,
        names=('ls', 'linestyle', 'linestyles'))
    style_kw = STYLE_ARGS_TRANSLATE.get(name, {})
    edgefix = _notNone(edgefix, rc['image.edgefix'])
    for key, value in (
            ('colors', colors),
            ('linewidths', linewidths),
            ('linestyles', linestyles)):
        if value is None:
            continue
        elif 'contourf' in name:  # special case, we re-draw our own contours
            continue
        if key in style_kw:
            edgefix = False  # override!
            kwargs[style_kw[key]] = value
        else:
            raise ValueError(
                f'Unknown keyword arg {key!r} for function {name!r}.'
            )
    # Check input
    for key, val in (('levels', levels), ('values', values)):
        if not np.iterable(val):
            continue
        if 'contour' in name and 'contourf' not in name:
            continue
        if len(val) < 2 or any(np.diff(val) <= 0):
            raise ValueError(
                f'{key!r} must be monotonically increasing and '
                f'at least length 2, got {val}.'
            )

    # Get level edges from level centers
    if values is not None:
        if isinstance(values, Number):
            levels = values + 1
        elif np.iterable(values):
            # Try to generate levels such that a LinearSegmentedNorm will
            # place values ticks at the center of each colorbar level.
            # utile.edges works only for evenly spaced values arrays.
            # We solve for: (x1 + x2)/2 = y --> x2 = 2*y - x1
            # with arbitrary starting point x1.
            if norm is None or norm in ('segments', 'segmented'):
                levels = [values[0] - (values[1] - values[0]) / 2]
                for i, val in enumerate(values):
                    levels.append(2 * val - levels[-1])
                if any(np.diff(levels) <= 0):  # algorithm failed
                    levels = utils.edges(values)
            # Generate levels by finding in-between points in the
            # normalized numeric space
            else:
                inorm = styletools.Norm(norm, **norm_kw)
                levels = inorm.inverse(utils.edges(inorm(values)))
            if name in ('parametric',):
                kwargs['values'] = values
        else:
            raise ValueError(
                f'Unexpected input values={values!r}. '
                'Must be integer or list of numbers.'
            )

    # Input colormap, for methods that accept a colormap and normalizer
    # contour, tricontour, i.e. not a method where cmap is optional
    if not ('contour' in name and 'contourf' not in name):
        cmap = _notNone(cmap, rc['image.cmap'])
    if cmap is not None:
        # Get colormap object
        cmap = styletools.Colormap(cmap, **cmap_kw)
        cyclic = getattr(cmap, '_cyclic', False)
        if cyclic and extend != 'neither':
            _warn_proplot(
                f'Cyclic colormap requires extend="neither". '
                'Overriding user input extend={extend!r}.'
            )
            extend = 'neither'
        kwargs['cmap'] = cmap

        # Get default normalizer
        # Only use LinearSegmentedNorm if necessary, because it is slow
        if name not in ('hexbin',):
            if norm is None:
                if not np.iterable(levels) or len(levels) == 1:
                    norm = 'linear'
                else:
                    diff = np.diff(levels)
                    eps = diff.mean() / 1e3
                    if (np.abs(np.diff(diff)) >= eps).any():
                        norm = 'segmented'
                        norm_kw.setdefault('levels', levels)
                    else:
                        norm = 'linear'
            elif norm in ('segments', 'segmented'):
                norm_kw.setdefault('levels', levels)
            norm = styletools.Norm(norm, **norm_kw)

    # Get default levels
    # TODO: Add kernel density plot to hexbin!
    if isinstance(levels, Number):
        # Cannot infer counts a priori, so do nothing
        if name in ('hexbin',):
            levels = None
        # Use the locator to determine levels
        # Mostly copied from the hidden contour.ContourSet._autolev
        else:
            # Get the locator
            N = levels
            if locator is not None:
                locator = axistools.Locator(locator, **locator_kw)
            elif isinstance(norm, mcolors.LogNorm):
                locator = mticker.LogLocator(**locator_kw)
            elif isinstance(norm, getattr(mcolors, 'SymLogNorm', type(None))):
                locator = mticker.SymmetricalLogLocator(**locator_kw)
            else:
                locator_kw.setdefault('symmetric', symmetric)
                locator = mticker.MaxNLocator(N, min_n_ticks=1, **locator_kw)
            # Get locations
            automin = (vmin is None)
            automax = (vmax is None)
            if automin or automax:
                Z = ma.masked_invalid(args[-1], copy=False)
                if automin:
                    vmin = float(Z.min())
                if automax:
                    vmax = float(Z.max())
                if vmin == vmax or ma.is_masked(vmin) or ma.is_masked(vmax):
                    vmin, vmax = 0, 1
            try:
                levels = locator.tick_values(vmin, vmax)
            except RuntimeError:
                levels = np.linspace(vmin, vmax, N)  # TODO: orig used N+1
            # Trim excess levels the locator may have supplied
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
            # Special consideration if not enough levels
            # how many times more levels did we want than what we got?
            nn = N // len(levels)
            if nn >= 2:
                olevels = norm(levels)
                nlevels = []
                for i in range(len(levels) - 1):
                    l1, l2 = olevels[i], olevels[i + 1]
                    nlevels.extend(np.linspace(l1, l2, nn + 1)[:-1])
                nlevels.append(olevels[-1])
                levels = norm.inverse(nlevels)

    # Norm settings
    # Generate BinNorm and update child norm object with vmin/vmax from levels
    # This is important for the colorbar setting tick locations properly!
    if norm is not None:
        if levels is not None:
            norm.vmin, norm.vmax = min(levels), max(levels)
        if levels is not None:
            bin_kw = {'extend': extend}
            if cyclic:
                bin_kw.update({'step': 0.5, 'extend': 'both'})
            norm = styletools.BinNorm(norm=norm, levels=levels, **bin_kw)
        kwargs['norm'] = norm

    # Call function
    if 'contour' in name:  # contour, contourf, tricontour, tricontourf
        kwargs.update({'levels': levels, 'extend': extend})
    obj = func(self, *args, **kwargs)
    obj.extend = extend  # for colorbar to determine 'extend' property
    if values is not None:
        obj.values = values  # preferred tick locations
    if levels is not None:
        obj.levels = levels  # for colorbar to determine tick locations
    if locator is not None and not isinstance(locator, mticker.MaxNLocator):
        obj.locator = locator  # for colorbar to determine tick locations

    # Call again for contourf plots with edges
    if 'contourf' in name and (linewidths is not None or colors is not None
                               or linestyles is not None):
        colors = _notNone(colors, 'k')
        cobj = self.contour(*args, levels=levels, linewidths=linewidths,
                            linestyles=linestyles, colors=colors)

    # Apply labels
    # TODO: Add quiverkey to this!
    if labels:
        # Formatting for labels
        # Respect if 'fmt' was passed in labels_kw instead of as a main
        # argument
        fmt = _notNone(labels_kw.pop('fmt', None), fmt, 'simple')
        fmt = axistools.Formatter(fmt, precision=precision)
        # Use clabel method
        if 'contour' in name:
            if 'contourf' in name:
                lums = [styletools.to_xyz(cmap(norm(level)), 'hcl')[
                    2] for level in levels]
                colors = ['w' if lum < 50 else 'k' for lum in lums]
                cobj = self.contour(*args, levels=levels, linewidths=0)
            else:
                cobj = obj
                colors = None
            text_kw = {}
            for key in (
                    *labels_kw,):  # allow dict to change size during iteration
                if key not in (
                    'levels', 'fontsize', 'colors', 'inline', 'inline_spacing',
                    'manual', 'rightside_up', 'use_clabeltext',
                ):
                    text_kw[key] = labels_kw.pop(key)
            labels_kw.setdefault('colors', colors)
            labels_kw.setdefault('inline_spacing', 3)
            labels_kw.setdefault('fontsize', rc['small'])
            labs = self.clabel(cobj, fmt=fmt, **labels_kw)
            for lab in labs:
                lab.update(text_kw)
        # Label each box manually
        # See: https://stackoverflow.com/a/20998634/4970632
        elif 'pcolor' in name:
            # populates the _facecolors attribute, initially filled with just a
            # single color
            obj.update_scalarmappable()
            labels_kw_ = {'size': rc['small'], 'ha': 'center', 'va': 'center'}
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
                    _, _, lum = styletools.to_xyz(color, 'hcl')
                    if lum < 50:
                        color = 'w'
                    else:
                        color = 'k'
                    labels_kw_['color'] = color
                self.text(x, y, fmt(num), **labels_kw_)
            obj.set_edgecolors(edgecolors)
        else:
            raise RuntimeError(f'Not possible to add labels to {name!r} plot.')

    # Fix white lines between filled contours/mesh, allow user to override!
    # 0.4 points is thick enough to hide lines but thin enough to not
    # add "dots" in corner of pcolor plots
    # *Never* use this when colormap has opacity
    # See: https://stackoverflow.com/q/15003353/4970632
    if 'pcolor' in name or 'contourf' in name:
        cmap = obj.get_cmap()
        if not cmap._isinit:
            cmap._init()
        if edgefix and all(cmap._lut[:-1, 3] == 1):
            if 'pcolor' in name:  # 'pcolor', 'pcolormesh', 'tripcolor'
                obj.set_edgecolor('face')
                obj.set_linewidth(0.4)
            elif 'contourf' in name:  # 'contourf', 'tricontourf'
                for contour in obj.collections:
                    contour.set_edgecolor('face')
                    contour.set_linewidth(0.4)
                    contour.set_linestyle('-')

    # Add colorbar
    if colorbar:
        loc = _validate_colorbar_loc(colorbar, default=rc['colorbar.loc'])
        if 'label' not in colorbar_kw and self.figure._auto_format:
            _, label = _auto_label(args[-1])  # last one is data, we assume
            if label:
                colorbar_kw.setdefault('label', label)
        if name in ('parametric',) and values is not None:
            colorbar_kw.setdefault('values', values)
        if loc != 'fill':
            colorbar_kw.setdefault('loc', loc)
        self.colorbar(obj, **colorbar_kw)
    return obj


def legend_wrapper(
    self, handles=None, labels=None, ncol=None, ncols=None,
    center=None, order='C', loc=None, label=None, title=None,
    fontsize=None, fontweight=None, fontcolor=None,
    color=None, marker=None, lw=None, linewidth=None,
    dashes=None, linestyle=None, markersize=None, frameon=None, frame=None,
    **kwargs
):
    """
    Wraps `~proplot.axes.Axes` `~proplot.axes.Axes.legend` and
    `~proplot.subplots.Figure` `~proplot.subplots.Figure.legend`, adds some
    handy features.

    Parameters
    ----------
    handles : list of `~matplotlib.artist.Artist`, optional
        List of artists instances, or list of lists of artist instances (see
        the `center` keyword). If ``None``, the artists are retrieved with
        `~matplotlib.axes.Axes.get_legend_handles_labels`.
    labels : list of str, optional
        Matching list of string labels, or list of lists of string labels (see
        the `center` keywod). If ``None``, the labels are retrieved by calling
        `~matplotlib.artist.Artist.get_label` on each
        `~matplotlib.artist.Artist` in `handles`.
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
    loc : int or str, optional
        The legend location. The following location keys are valid.

        ==================  ================================================
        Location            Valid keys
        ==================  ================================================
        "best" possible     ``0``, ``'best'``, ``'b'``, ``'i'``, ``'inset'``
        upper right         ``1``, ``'upper right'``, ``'ur'``
        upper left          ``2``, ``'upper left'``, ``'ul'``
        lower left          ``3``, ``'lower left'``, ``'ll'``
        lower right         ``4``, ``'lower right'``, ``'lr'``
        center left         ``5``, ``'center left'``, ``'cl'``
        center right        ``6``, ``'center right'``, ``'cr'``
        lower center        ``7``, ``'lower center'``, ``'lc'``
        upper center        ``8``, ``'upper center'``, ``'uc'``
        center              ``9``, ``'center'``, ``'c'``
        ==================  ================================================

    label, title : str, optional
        The legend title. The `label` keyword is also accepted, for consistency
        with `colorbar`.
    fontsize, fontweight, fontcolor : optional
        The font size, weight, and color for legend text.
    color, lw, linewidth, marker, linestyle, dashes, markersize : \
property-spec, optional
        Properties used to override the legend handles. For example, if you
        want a legend that describes variations in line style ignoring
        variations in color, you might want to use ``color='k'``. For now this
        does not include `facecolor`, `edgecolor`, and `alpha`, because
        `~matplotlib.axes.Axes.legend` uses these keyword args to modify the
        frame properties.

    Other parameters
    ----------------
    **kwargs
        Passed to `~matplotlib.axes.Axes.legend`.
    """
    # First get legend settings and interpret kwargs.
    if order not in ('F', 'C'):
        raise ValueError(
            f'Invalid order {order!r}. Choose from '
            '"C" (row-major, default) and "F" (column-major).'
        )
    # may still be None, wait till later
    ncol = _notNone(ncols, ncol, None, names=('ncols', 'ncol'))
    title = _notNone(label, title, None, names=('label', 'title'))
    frameon = _notNone(
        frame, frameon, rc['legend.frameon'], names=('frame', 'frameon'))
    if title is not None:
        kwargs['title'] = title
    if frameon is not None:
        kwargs['frameon'] = frameon
    if fontsize is not None:
        kwargs['fontsize'] = fontsize
    # Text properties, some of which have to be set after-the-fact
    kw_text = {}
    if fontcolor is not None:
        kw_text['color'] = fontcolor
    if fontweight is not None:
        kw_text['weight'] = fontweight

    # Automatically get labels and handles
    # TODO: Use legend._parse_legend_args instead? This covers functionality
    # just fine, _parse_legend_args seems overkill.
    if handles is None:
        if self._filled:
            raise ValueError(
                'You must pass a handles list for panel axes '
                '"filled" with a legend.'
            )
        else:
            # ignores artists with labels '_nolegend_'
            handles, labels_default = self.get_legend_handles_labels()
            if labels is None:
                labels = labels_default
            if not handles:
                raise ValueError(
                    'No labeled artists found. To generate a legend without '
                    'providing the artists explicitly, pass label="label" in '
                    'your plotting commands.'
                )
    if not np.iterable(handles):  # e.g. a mappable object
        handles = [handles]
    if labels is not None and (not np.iterable(
            labels) or isinstance(labels, str)):
        labels = [labels]

    # Legend entry for colormap or scatterplot object
    # TODO: Idea is we pass a scatter plot or contourf or whatever, and legend
    # is generating by drawing patch rectangles or markers with different
    # colors.
    if any(not hasattr(handle, 'get_facecolor') and hasattr(handle, 'get_cmap')
           for handle in handles) and len(handles) > 1:
        raise ValueError(
            f'Handles must be objects with get_facecolor attributes or '
            'a single mappable object from which we can draw colors.'
        )

    # Build pairs of handles and labels
    # This allows alternative workflow where user specifies labels when
    # creating the legend.
    pairs = []
    # e.g. not including BarContainer
    list_of_lists = (not hasattr(handles[0], 'get_label'))
    if labels is None:
        for handle in handles:
            if list_of_lists:
                ipairs = []
                for ihandle in handle:
                    if not hasattr(ihandle, 'get_label'):
                        raise ValueError(
                            f'Object {ihandle} must have "get_label" method.'
                        )
                    ipairs.append((ihandle, ihandle.get_label()))
                pairs.append(ipairs)
            else:
                if not hasattr(handle, 'get_label'):
                    raise ValueError(
                        f'Object {handle} must have "get_label" method.'
                    )
                pairs.append((handle, handle.get_label()))
    else:
        if len(labels) != len(handles):
            raise ValueError(
                f'Got {len(labels)} labels, but {len(handles)} handles.'
            )
        for label, handle in zip(labels, handles):
            if list_of_lists:
                ipairs = []
                if not np.iterable(label) or isinstance(label, str):
                    raise ValueError(
                        f'Got list of lists of handles, but list of labels.'
                    )
                elif len(label) != len(handle):
                    raise ValueError(
                        f'Got {len(label)} labels in sublist, '
                        f'but {len(handle)} handles.'
                    )
                for ilabel, ihandle in zip(label, handle):
                    ipairs.append((ihandle, ilabel))
                pairs.append(ipairs)
            else:
                if not isinstance(label, str) and np.iterable(label):
                    raise ValueError(
                        f'Got list of lists of labels, but list of handles.'
                    )
                pairs.append((handle, label))

    # Manage pairs in context of 'center' option
    if center is None:  # automatically guess
        center = list_of_lists
    elif center and list_of_lists and ncol is not None:
        _warn_proplot(
            'Detected list of *lists* of legend handles. '
            'Ignoring user input property "ncol".'
        )
    elif not center and list_of_lists:  # standardize format based on input
        list_of_lists = False  # no longer is list of lists
        pairs = [pair for ipairs in pairs for pair in ipairs]
    elif center and not list_of_lists:
        list_of_lists = True
        ncol = _notNone(ncol, 3)
        pairs = [pairs[i * ncol:(i + 1) * ncol]
                 for i in range(len(pairs))]  # to list of iterables
    if list_of_lists:  # remove empty lists, pops up in some examples
        pairs = [ipairs for ipairs in pairs if ipairs]

    # Now draw legend(s)
    legs = []
    width, height = self.get_size_inches()
    # Individual legend
    if not center:
        # Optionally change order
        # See: https://stackoverflow.com/q/10101141/4970632
        # Example: If 5 columns, but final row length 3, columns 0-2 have
        # N rows but 3-4 have N-1 rows.
        ncol = _notNone(ncol, 3)
        if order == 'C':
            fpairs = []
            # split into rows
            split = [pairs[i * ncol:(i + 1) * ncol]
                     for i in range(len(pairs) // ncol + 1)]
            # max possible row count, and columns in final row
            nrowsmax, nfinalrow = len(split), len(split[-1])
            nrows = [nrowsmax] * nfinalrow + \
                [nrowsmax - 1] * (ncol - nfinalrow)
            for col, nrow in enumerate(nrows):  # iterate through cols
                fpairs.extend(split[row][col] for row in range(nrow))
            pairs = fpairs
        # Make legend object
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
        if overridden:
            _warn_proplot(
                f'For centered-row legends, must override '
                'user input properties '
                + ', '.join(map(repr, overridden)) + '.'
            )
        # Determine space we want sub-legend to occupy as fraction of height
        # NOTE: Empirical testing shows spacing fudge factor necessary to
        # exactly replicate the spacing of standard aligned legends.
        fontsize = kwargs.get('fontsize', None) or rc['legend.fontsize']
        spacing = kwargs.get('labelspacing', None) or rc['legend.labelspacing']
        interval = 1 / len(pairs)  # split up axes
        interval = (((1 + spacing * 0.85) * fontsize) / 72) / height
        # Iterate and draw
        # NOTE: We confine possible bounding box in *y*-direction, but do not
        # confine it in *x*-direction. Matplotlib will automatically move
        # left-to-right if you request this.
        ymin, ymax = None, None
        if order == 'F':
            raise NotImplementedError(
                f'When center=True, ProPlot vertically stacks successive '
                'single-row legends. Column-major (order="F") ordering '
                'is un-supported.'
            )
        loc = _notNone(loc, 'upper center')
        if not isinstance(loc, str):
            raise ValueError(
                f'Invalid location {loc!r} for legend with center=True. '
                'Must be a location *string*.'
            )
        elif loc == 'best':
            _warn_proplot(
                'For centered-row legends, cannot use "best" location. '
                'Using "upper center" instead.'
            )
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
            ymin = min(y1, _notNone(ymin, y1))
            ymax = max(y2, _notNone(ymax, y2))
            # Draw legend
            bbox = mtransforms.Bbox([[0, y1], [1, y2]])
            leg = mlegend.Legend(
                self, *zip(*ipairs), loc=loc, ncol=len(ipairs),
                bbox_transform=self.transAxes, bbox_to_anchor=bbox,
                frameon=False, **kwargs)
            legs.append(leg)

    # Add legends manually so matplotlib does not remove old ones
    # Also apply override settings
    kw_handle = {}
    outline = rc.fill({
        'linewidth': 'axes.linewidth',
        'edgecolor': 'axes.edgecolor',
        'facecolor': 'axes.facecolor',
        'alpha': 'legend.framealpha',
    })
    for key in (*outline,):
        if key != 'linewidth':
            if kwargs.get(key, None):
                outline.pop(key, None)
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
    for leg in legs:
        self.add_artist(leg)
        leg.legendPatch.update(outline)  # or get_frame()
        for obj in leg.legendHandles:
            if isinstance(obj, martist.Artist):
                obj.update(kw_handle)
        for obj in leg.get_texts():
            if isinstance(obj, martist.Artist):
                obj.update(kw_text)
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
            legs[0].set_frame_on(True)  # easy!
        else:
            # Get coordinates
            renderer = self.figure._get_renderer()
            bboxs = [leg.get_window_extent(renderer).transformed(
                self.transAxes.inverted()) for leg in legs]
            xmin, xmax = min(bbox.xmin for bbox in bboxs), max(
                bbox.xmax for bbox in bboxs)
            ymin, ymax = min(bbox.ymin for bbox in bboxs), max(
                bbox.ymax for bbox in bboxs)
            fontsize = (fontsize / 72) / width  # axes relative units
            fontsize = renderer.points_to_pixels(fontsize)
            # Draw and format patch
            patch = mpatches.FancyBboxPatch(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                snap=True, zorder=4.5,
                mutation_scale=fontsize, transform=self.transAxes)
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
    return legs[0] if len(legs) == 1 else (*legs,)


def colorbar_wrapper(
    self, mappable, values=None,
    extend=None, extendsize=None,
    title=None, label=None,
    grid=None, tickminor=None,
    tickloc=None, ticklocation=None,
    locator=None, ticks=None, maxn=None, maxn_minor=None,
    minorlocator=None, minorticks=None,
    locator_kw=None, minorlocator_kw=None,
    formatter=None, ticklabels=None, formatter_kw=None,
    norm=None, norm_kw=None,  # normalizer to use when passing colors/lines
    orientation='horizontal',
    edgecolor=None, linewidth=None,
    labelsize=None, labelweight=None, labelcolor=None,
    ticklabelsize=None, ticklabelweight=None, ticklabelcolor=None,
    **kwargs
):
    """
    Wraps `~proplot.axes.Axes` `~proplot.axes.Axes.colorbar` and
    `~proplot.subplots.Figure` `~proplot.subplots.Figure.colorbar`, adds some
    handy features.

    Parameters
    ----------
    mappable : mappable, list of plot handles, list of color-spec, \
or colormap-spec
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

        This is handy if you have multiple colorbars in one figure.
        With the matplotlib API, it is really hard to get triangle
        sizes to match, because the `extendsize` units are *relative*.
    tickloc, ticklocation : {'bottom', 'top', 'left', 'right'}, optional
        Where to draw tick marks on the colorbar.
    label, title : str, optional
        The colorbar label. The `title` keyword is also accepted for
        consistency with `legend`.
    grid : bool, optional
        Whether to draw "gridlines" between each level of the colorbar.
        Default is :rc:`colorbar.grid`.
    tickminor : bool, optional
        Whether to add minor ticks to the colorbar with
        `~matplotlib.colorbar.ColorbarBase.minorticks_on`. Default is
        ``False``.
    locator, ticks : locator spec, optional
        Used to determine the colorbar tick positions. Passed to the
        `~proplot.axistools.Locator` constructor.
    maxn : int, optional
        Used if `locator` is ``None``. Determines the maximum number of levels
        that are ticked. Default depends on the colorbar length relative
        to the font size. The keyword name "maxn" is meant to mimic
        the `~matplotlib.ticker.MaxNLocator` class name.
    maxn_minor : int, optional
        As with `maxn`, but for minor tick positions. Default depends
        on the colorbar length.
    locator_kw : dict-like, optional
        The locator settings. Passed to `~proplot.axistools.Locator`.
    minorlocator, minorticks
        As with `locator`, but for the minor ticks.
    minorlocator_kw
        As for `locator_kw`, but for the minor ticks.
    formatter, ticklabels : formatter spec, optional
        The tick label format. Passed to the `~proplot.axistools.Formatter`
        constructor.
    formatter_kw : dict-like, optional
        The formatter settings. Passed to `~proplot.axistools.Formatter`.
    norm : normalizer spec, optional
        Ignored if `values` is ``None``. The normalizer
        for converting `values` to colormap colors. Passed to the
        `~proplot.styletools.Norm` constructor. As an example, if your
        values are logarithmically spaced but you want the level boundaries
        to appear halfway in-between the colorbar ticks, try
        ``norm='log'``.
    norm_kw : dict-like, optional
        The normalizer settings. Passed to `~proplot.styletools.Norm`.
    edgecolor, linewidth : optional
        The edge color and line width for the colorbar outline.
    labelsize, labelweight, labelcolor : optional
        The font size, weight, and color for colorbar label text.
    ticklabelsize, ticklabelweight, ticklabelcolor : optional
        The font size, weight, and color for colorbar tick labels.
    orientation : {'horizontal', 'vertical'}, optional
        The colorbar orientation. You should not have to explicitly set this.

    Other parameters
    ----------------
    **kwargs
        Passed to `~matplotlib.figure.Figure.colorbar`.
    """
    # Developer notes
    # * Colorbar axes must be of type `matplotlib.axes.Axes`,
    #   not `~proplot.axes.Axes`, because colorbar uses some internal methods
    #   that are wrapped by `~proplot.axes.Axes`.
    # * There is an insanely weird problem with colorbars when simultaneously
    #   passing levels and norm object to a mappable; fixed by passing
    #   vmin/vmax instead of levels.
    #   (see: https://stackoverflow.com/q/40116968/4970632).
    # * Problem is often want levels instead of vmin/vmax, while simultaneously
    #   using a Normalize (for example) to determine colors between the levels
    #   (see: https://stackoverflow.com/q/42723538/4970632). Workaround makes
    #   sure locators are in vmin/vmax range exclusively; cannot match values.
    # No mutable defaults
    locator_kw = locator_kw or {}
    minorlocator_kw = minorlocator_kw or {}
    formatter_kw = formatter_kw or {}
    norm_kw = norm_kw or {}
    # Parse flexible input
    label = _notNone(title, label, None, names=('title', 'label'))
    locator = _notNone(ticks, locator, None, names=('ticks', 'locator'))
    formatter = _notNone(
        ticklabels, formatter, 'auto',
        names=('ticklabels', 'formatter')
    )
    minorlocator = _notNone(
        minorticks, minorlocator, None,
        names=('minorticks', 'minorlocator')
    )
    ticklocation = _notNone(
        tickloc, ticklocation, None,
        names=('tickloc', 'ticklocation')
    )

    # Colorbar kwargs
    # WARNING: PathCollection scatter objects have an extend method!
    grid = _notNone(grid, rc['colorbar.grid'])
    if extend is None:
        if isinstance(getattr(mappable, 'extend', None), str):
            extend = mappable.extend or 'neither'
        else:
            extend = 'neither'
    kwargs.update({
        'cax': self,
        'use_gridspec': True,
        'orientation': orientation,
        'extend': extend,
        'spacing': 'uniform'})
    kwargs.setdefault('drawedges', grid)

    # Text property keyword args
    kw_label = {}
    if labelsize is not None:
        kw_label['size'] = labelsize
    if labelweight is not None:
        kw_label['weight'] = labelweight
    if labelcolor is not None:
        kw_label['color'] = labelcolor
    kw_ticklabels = {}
    if ticklabelsize is not None:
        kw_ticklabels['size'] = ticklabelsize
    if ticklabelweight is not None:
        kw_ticklabels['weight'] = ticklabelweight
    if ticklabelcolor is not None:
        kw_ticklabels['color'] = ticklabelcolor

    # Special case where auto colorbar is generated from 1D methods, a list is
    # always passed but some 1D methods (scatter) do have colormaps.
    if np.iterable(mappable) and len(
            mappable) == 1 and hasattr(mappable[0], 'get_cmap'):
        mappable = mappable[0]

    # Test if we were given a mappable, or iterable of stuff; note Container
    # and PolyCollection matplotlib classes are iterable.
    cmap = None
    tick_all = (values is not None)
    if not isinstance(mappable, martist.Artist) and not isinstance(
            mappable, mcontour.ContourSet):
        # Object for testing
        obj = mappable[0] if np.iterable(mappable) else mappable
        try:
            obj = obj[0]  # e.g. for BarContainer, which is not numpy.iterable
        except (TypeError, KeyError):
            pass
        # List of handles
        if hasattr(obj, 'get_color') or hasattr(obj, 'get_facecolor'):
            # Make colormap
            colors = []
            for obj in mappable:
                if np.iterable(obj):
                    obj = obj[0]
                color = getattr(obj, 'get_color', None) or getattr(
                    obj, 'get_facecolor')
                colors.append(color())
            cmap = styletools.Colormap(colors, listmode='listed')
            # Infer values
            if values is None:
                values = []
                for obj in mappable:
                    val = obj.get_label()
                    try:
                        val = float(val)
                    except ValueError:
                        values = None
                        break
                    values.append(val)
            if values is None:
                values = np.arange(0, len(mappable))
            tick_all = True
        # Any colormap spec, including a list of colors, colormap name, or
        # colormap instance
        elif isinstance(mappable, mcolors.Colormap):
            cmap = mappable
            if values is None:
                if np.iterable(mappable) and not isinstance(
                        mappable, str):  # e.g. list of colors
                    values = np.linspace(0, 1, len(mappable))
                else:
                    values = np.linspace(0, 1, cmap.N)
        else:
            raise ValueError(
                'Input mappable must be a matplotlib artist, '
                'list of objects, list of colors, or colormap. '
                f'Got {mappable!r}.'
            )

    # Build new ad hoc mappable object from handles
    # NOTE: Need to use wrapped contourf but this might be native matplotlib
    # axes. Call on self.axes, which is child if child axes, self otherwise.
    if cmap is not None:
        if np.iterable(mappable) and len(values) != len(mappable):
            raise ValueError(
                f'Passed {len(values)} values, but only {len(mappable)} '
                f'objects or colors.'
            )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mappable = self.axes.contourf(
                [0, 0], [0, 0], ma.array([[0, 0], [0, 0]], mask=True),
                cmap=cmap, extend='neither', values=np.array(values),
                norm=norm, norm_kw=norm_kw)  # workaround

    # Try to get tick locations from *levels* or from *values* rather than
    # random points along the axis. If values were provided as keyword arg,
    # this is colorbar from lines/colors, and we label *all* values by default.
    # TODO: Handle more of the log locator stuff here instead of cmap_changer?
    if tick_all and locator is None:
        locator = values
        tickminor = False
    if locator is None:
        for attr in ('values', 'locator', 'levels'):
            locator = getattr(mappable, attr, None)
            if locator is not None:
                break
        if locator is None:  # i.e. no attributes found
            if isinstance(getattr(mappable, 'norm', None), mcolors.LogNorm):
                locator = 'log'
            else:
                locator = 'auto'
        # i.e. was a 'values' or 'levels' attribute
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
            maxn = _notNone(maxn, int(length / (scale * fontsize / 72)))
            maxn_minor = _notNone(maxn_minor, int(
                length / (0.5 * fontsize / 72)))
            # Get locator
            if tickminor and minorlocator is None:
                step = 1 + len(locator) // max(1, maxn_minor)
                minorlocator = locator[::step]
            step = 1 + len(locator) // max(1, maxn)
            locator = locator[::step]

    # Final settings
    locator = axistools.Locator(locator, **locator_kw)
    formatter = axistools.Formatter(formatter, **formatter_kw)
    width, height = self.figure.get_size_inches()
    if orientation == 'horizontal':
        scale = width * abs(self.get_position().width)
    else:
        scale = height * abs(self.get_position().height)
    extendsize = utils.units(_notNone(extendsize, rc['colorbar.extend']))
    extendsize = extendsize / (scale - 2 * extendsize)
    kwargs.update({
        'ticks': locator,
        'format': formatter,
        'ticklocation': ticklocation,
        'extendfrac': extendsize
    })

    # Draw the colorbar
    # NOTE: self._use_auto_colorbar_locator() is never True because
    # we use the custom BinNorm normalizer. Colorbar._ticks() always called.
    try:
        self.figure._locked = False
        cb = self.figure.colorbar(mappable, **kwargs)
    except Exception as err:
        self.figure._locked = True
        raise err
    axis = self.xaxis if orientation == 'horizontal' else self.yaxis

    # The minor locator
    # TODO: Document the improved minor locator functionality!
    if tickminor and minorlocator is None:
        cb.minorticks_on()
    elif minorlocator is None:
        cb.minorticks_off()
    elif not hasattr(cb, '_ticker'):
        _warn_proplot(
            'Matplotlib colorbar API has changed. '
            'Cannot use custom minor tick locator.'
        )
        if tickminor:
            cb.minorticks_on()
    else:
        # Private API is the only way!
        minorlocator = axistools.Locator(minorlocator, **minorlocator_kw)
        ticks, *_ = cb._ticker(minorlocator, mticker.NullFormatter())
        axis.set_ticks(ticks, minor=True)
        axis.set_ticklabels([], minor=True)

    # Outline
    kw_outline = {
        'edgecolor': _notNone(edgecolor, rc['axes.edgecolor']),
        'linewidth': _notNone(linewidth, rc['axes.linewidth']),
    }
    if cb.outline is not None:
        cb.outline.update(kw_outline)
    if cb.dividers is not None:
        cb.dividers.update(kw_outline)

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
        _warn_proplot(
            f'Using manual alpha-blending for {cmap.name!r} colorbar solids.'
        )
        # Generate "secret" copy of the colormap!
        lut = cmap._lut.copy()
        cmap = mcolors.Colormap('_colorbar_fix', N=cmap.N)
        cmap._isinit = True
        cmap._init = (lambda: None)
        # Manually fill lookup table with alpha-blended RGB colors!
        for i in range(lut.shape[0] - 1):
            alpha = lut[i, 3]
            lut[i, :3] = (1 - alpha) * 1 + alpha * \
                lut[i, :3]  # blend with *white*
            lut[i, 3] = 1
        cmap._lut = lut
        # Update colorbar
        cb.cmap = cmap
        cb.draw_all()

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

    # *Never* rasterize because it causes misalignment with border lines
    if cb.solids:
        cb.solids.set_rasterized(False)
        cb.solids.set_linewidth(0.4)
        cb.solids.set_edgecolor('face')
    return cb


def _redirect(func):
    """Docorator that calls the basemap version of the function of the
    same name. This must be applied as innermost decorator, which means it must
    be applied on the base axes class, not the basemap axes."""
    name = func.__name__
    @functools.wraps(func)
    def _wrapper(self, *args, **kwargs):
        if getattr(self, 'name', '') == 'basemap':
            return getattr(self.projection, name)(*args, ax=self, **kwargs)
        else:
            return func(self, *args, **kwargs)
    _wrapper.__doc__ = None
    return _wrapper


def _norecurse(func):
    """Decorator to prevent recursion in basemap method overrides.
    See `this post https://stackoverflow.com/a/37675810/4970632`__."""
    name = func.__name__
    func._has_recurred = False
    @functools.wraps(func)
    def _wrapper(self, *args, **kwargs):
        if func._has_recurred:
            # Return the *original* version of the matplotlib method
            func._has_recurred = False
            result = getattr(maxes.Axes, name)(self, *args, **kwargs)
        else:
            # Return the version we have wrapped
            func._has_recurred = True
            result = func(self, *args, **kwargs)
        func._has_recurred = False  # cleanup, in case recursion never occurred
        return result
    return _wrapper


def _wrapper_decorator(driver):
    """Generate generic wrapper decorator and dynamically modify the docstring
    to list methods wrapped by this function. Also set `__doc__` to ``None`` so
    that ProPlot fork of automodapi doesn't add these methods to the website
    documentation. Users can still call help(ax.method) because python looks
    for superclass method docstrings if a docstring is empty."""
    driver._docstring_orig = driver.__doc__ or ''
    driver._methods_wrapped = []
    proplot_methods = ('parametric', 'heatmap', 'area', 'areax')
    cartopy_methods = ('get_extent', 'set_extent')

    def decorator(func):
        # Define wrapper and suppress documentation
        # We only document wrapper functions, not the methods they wrap
        @functools.wraps(func)
        def _wrapper(self, *args, **kwargs):
            return driver(self, func, *args, **kwargs)
        name = func.__name__
        if name not in proplot_methods:
            _wrapper.__doc__ = None

        # List wrapped methods in the driver function docstring
        # Prevents us from having to both explicitly apply decorators in
        # axes.py and explicitly list functions *again* in this file
        docstring = driver._docstring_orig
        if '%(methods)s' in docstring:
            if name in proplot_methods:
                link = f'`~proplot.axes.Axes.{name}`'
            elif name in cartopy_methods:
                link = f'`~cartopy.mpl.geoaxes.GeoAxes.{name}`'
            else:
                link = f'`~matplotlib.axes.Axes.{name}`'
            methods = driver._methods_wrapped
            if link not in methods:
                methods.append(link)
                string = (
                    ', '.join(methods[:-1])
                    + ',' * int(len(methods) > 2)  # Oxford comma bitches
                    + ' and ' * int(len(methods) > 1)
                    + methods[-1])
                driver.__doc__ = docstring % {'methods': string}
        return _wrapper
    return decorator


# Auto generated decorators. Each wrapper internally calls
# func(self, ...) somewhere.
_add_errorbars = _wrapper_decorator(add_errorbars)
_bar_wrapper = _wrapper_decorator(bar_wrapper)
_barh_wrapper = _wrapper_decorator(barh_wrapper)
_default_latlon = _wrapper_decorator(default_latlon)
_boxplot_wrapper = _wrapper_decorator(boxplot_wrapper)
_default_crs = _wrapper_decorator(default_crs)
_default_transform = _wrapper_decorator(default_transform)
_cmap_changer = _wrapper_decorator(cmap_changer)
_cycle_changer = _wrapper_decorator(cycle_changer)
_fill_between_wrapper = _wrapper_decorator(fill_between_wrapper)
_fill_betweenx_wrapper = _wrapper_decorator(fill_betweenx_wrapper)
_hist_wrapper = _wrapper_decorator(hist_wrapper)
_plot_wrapper = _wrapper_decorator(plot_wrapper)
_scatter_wrapper = _wrapper_decorator(scatter_wrapper)
_standardize_1d = _wrapper_decorator(standardize_1d)
_standardize_2d = _wrapper_decorator(standardize_2d)
_text_wrapper = _wrapper_decorator(text_wrapper)
_violinplot_wrapper = _wrapper_decorator(violinplot_wrapper)
