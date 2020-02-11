#!/usr/bin/env python3
"""
The starting point for creating custom ProPlot figures. Includes
pyplot-inspired functions for creating figures and related classes.
"""
# NOTE: Importing backend causes issues with sphinx, and anyway not sure it's
# always included, so make it optional
import os
import numpy as np
import functools
import inspect
from matplotlib import docstring
import matplotlib.artist as martist
import matplotlib.figure as mfigure
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
from numbers import Integral
from .rctools import rc
from .utils import _warn_proplot, _notNone, _counter, _setstate, units  # noqa
from . import projs, axes
try:  # use this for debugging instead of print()!
    from icecream import ic
except ImportError:  # graceful fallback if IceCream isn't installed
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

__all__ = [
    'subplot_grid', 'close', 'figure', 'show', 'subplots',
    'EdgeStack', 'Figure', 'GeometrySolver',
    'GridSpec', 'SubplotSpec',
]

# Dimensions of figures for common journals
JOURNAL_SPECS = {
    'aaas1': '5.5cm',
    'aaas2': '12cm',
    'agu1': ('95mm', '115mm'),
    'agu2': ('190mm', '115mm'),
    'agu3': ('95mm', '230mm'),
    'agu4': ('190mm', '230mm'),
    'ams1': 3.2,
    'ams2': 4.5,
    'ams3': 5.5,
    'ams4': 6.5,
    'nat1': '89mm',
    'nat2': '183mm',
    'pnas1': '8.7cm',
    'pnas2': '11.4cm',
    'pnas3': '17.8cm',
}

# Documentation
_figure_doc = """
figsize : length-2 tuple, optional
    Tuple specifying the figure ``(width, height)``.
width, height : float or str, optional
    The figure width and height. Units are interpreted by
    `~proplot.utils.units`.
axwidth, axheight : float or str, optional
    The width and height of the `ref` axes. Units are interpreted by
    `~proplot.utils.units`. Default is :rc:`subplots.axwidth`.
    These arguments are convenient where you don't care about the figure
    dimensions and just want your axes to have enough "room".
aspect : float or length-2 list of floats, optional
    The aspect ratio of the `ref` axes as a ``width/height`` number or a
    ``(width, height)`` tuple. If you do not provide the `hratios` or
    `wratios` keyword args, this will control the aspect ratio of *all* axes.
ref : int, optional
    The reference axes number. The `axwidth`, `axheight`, and `aspect` keyword
    args are applied to this axes and conserved during tight layout adjustment.
tight : bool, optional
    Toggles whether the gridspec spaces `left`, `right`, `bottom`,
    `top`, `wspace`, and `hspace` are determined automatically to
    make room for labels and plotted content. Default is :rc:`tight`.
pad, axpad, panelpad : float or str, optional
    Padding around the edge of the figure, between subplots in adjacent
    rows and columns, and between subplots and axes panels or between
    "stacked" panels. Units are interpreted by `~proplot.utils.units`.
    Defaults are :rc:`subplots.pad`, :rc:`subplots.axpad`, and
    :rc:`subplots.panelpad`.
left, right, top, bottom, wspace, hspace : float or str, optional
    The spacing parameters passed to `GridSpec`. If `tight` is ``True``
    and you pass any of these, the tight layout algorithm will be
    ignored for that particular spacing. See the following examples.

    * With ``plot.figure(left='3em')``, the left margin is
      fixed but the other margins are variable.
    * With ``plot.subplots(ncols=3, wspace=0)``, the space between
      columns is fixed at zero, but between rows is variable.
    * With ``plot.subplots(ncols=3, wspace=(0, None))``, the space
      between the first and second columns is fixed, but the space
      between the second and third columns is variable.
sharex, sharey, share : {3, 2, 1, 0}, optional
    The "axis sharing level" for the *x* axis, *y* axis, or both axes.
    Default is ``3``. This can considerably reduce redundancy in your
    figure. Options are as follows.

    0. No axis sharing. Also sets the default `spanx` and `spany`
       values to ``False``.
    1. Only draw *axis label* on the leftmost column (*y*) or
       bottommost row (*x*) of subplots. Axis tick labels
       still appear on every subplot.
    2. As in 1, but forces the axis limits to be identical. Axis
       tick labels still appear on every subplot.
    3. As in 2, but only show the *axis tick labels* on the
       leftmost column (*y*) or bottommost row (*x*) of subplots.

spanx, spany, span : bool or {0, 1}, optional
    Toggles "spanning" axis labels for the *x* axis, *y* axis, or both
    axes.  Default is ``False`` if `sharex`, `sharey`, or `share` are
    ``0``, ``True`` otherwise. When ``True``, a single, centered axis
    label is used for all axes with bottom and left edges in the same
    row or column.  This can considerably redundancy in your figure.

    "Spanning" labels integrate with "shared" axes. For example,
    for a 3-row, 3-column figure, with ``sharey > 1`` and ``spany=1``,
    your figure will have 1 ylabel instead of 9.
alignx, aligny, align : bool or {0, 1}, optional
    Default is ``False``. Whether to `align axis labels \
<https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/align_labels_demo.html>`__
    for the *x* axis, *y* axis, or both axes. Only has an effect when
    `spanx`, `spany`, or `span` are ``False``.
includepanels : bool, optional
    Whether to include panels when centering *x* axis labels,
    *y* axis labels, and figure "super titles" along the edge of the
    subplot grid. Default is ``False``.
autoformat : bool, optional
    Whether to automatically configure *x* axis labels, *y* axis
    labels, axis formatters, axes titles, colorbar labels, and legend
    labels when a `~pandas.Series`, `~pandas.DataFrame` or
    `~xarray.DataArray` with relevant metadata is passed to a plotting
    command.
fallback_to_cm : bool, optional
    Whether to replace unavailable glyphs with a glyph from Computer
    Modern or the "¤" dummy character. See `mathtext \
<https://matplotlib.org/3.1.1/tutorials/text/mathtext.html#custom-fonts>`__
    for details.
journal : str, optional
    String name corresponding to an academic journal standard that is used
    to control the figure width (and height, if specified). See below table.

    ===========  ====================  ==========================================================================================================================================================
    Key          Size description      Organization
    ===========  ====================  ==========================================================================================================================================================
    ``'aaas1'``  1-column              `American Association for the Advancement of Science <https://www.sciencemag.org/authors/instructions-preparing-initial-manuscript>`__ (e.g. *Science*)
    ``'aaas2'``  2-column              ”
    ``'agu1'``   1-column              `American Geophysical Union <https://publications.agu.org/author-resource-center/figures-faq/>`__
    ``'agu2'``   2-column              ”
    ``'agu3'``   full height 1-column  ”
    ``'agu4'``   full height 2-column  ”
    ``'ams1'``   1-column              `American Meteorological Society <https://www.ametsoc.org/ams/index.cfm/publications/authors/journal-and-bams-authors/figure-information-for-authors/>`__
    ``'ams2'``   small 2-column        ”
    ``'ams3'``   medium 2-column       ”
    ``'ams4'``   full 2-column         ”
    ``'nat1'``   1-column              `Nature Research <https://www.nature.com/nature/for-authors/formatting-guide>`__
    ``'nat2'``   2-column              ”
    ``'pnas1'``  1-column              `Proceedings of the National Academy of Sciences <http://www.pnas.org/page/authors/submission>`__
    ``'pnas2'``  2-column              ”
    ``'pnas3'``  landscape page        ”
    ===========  ====================  ==========================================================================================================================================================

**kwargs
    Passed to `matplotlib.figure.Figure`.
"""  # noqa
docstring.interpd.update(figure_doc=_figure_doc)
_gridspec_doc = """
Apply the `GridSpec` to the figure or generate a new `GridSpec`
instance with the positional and keyword arguments. For example,
``fig.set_gridspec(GridSpec(1, 1, left=0.1))`` and
``fig.set_gridspec(1, 1, left=0.1)`` are both valid.
"""  # hunk for identically named methods


def close(*args, **kwargs):
    """Pass the input arguments to `matplotlib.pyplot.close`. This is included
    so you don't have to import `~matplotlib.pyplot`."""
    plt.close(*args, **kwargs)


def show():
    """Call `matplotlib.pyplot.show`. This is included so you don't have
    to import `~matplotlib.pyplot`. Note this command should *not be
    necessary* if you are working in an iPython session and :rcraw:`matplotlib`
    is non-empty -- when you create a new figure, it will be automatically
    displayed."""
    plt.show()


class subplot_grid(list):
    """List subclass and pseudo-2d array that is used as a container for the
    list of axes returned by `subplots`. See `~subplot_grid.__getattr__`
    and `~subplot_grid.__getitem__` for details."""
    def __init__(self, objs, n=1, order='C'):
        """
        Parameters
        ----------
        objs : list-like
            1d iterable of `~proplot.axes.Axes` instances.
        n : int, optional
            The length of the fastest-moving dimension, i.e. the number of
            columns when `order` is ``'C'``, and the number of rows when
            `order` is ``'F'``. Used to treat lists as pseudo-2d arrays.
        order : {'C', 'F'}, optional
            Whether 1d indexing returns results in row-major (C-style) or
            column-major (Fortran-style) order, respectively. Used to treat
            lists as pseudo-2d arrays.
        """
        if not all(isinstance(obj, axes.Axes) for obj in objs):
            raise ValueError(
                f'Axes grid must be filled with Axes instances, got {objs!r}.'
            )
        super().__init__(objs)
        self._n = n
        self._order = order
        self._shape = (len(self) // n, n)[::(1 if order == 'C' else -1)]

    def __repr__(self):
        return 'subplot_grid([' + ', '.join(str(ax) for ax in self) + '])'

    def __setitem__(self, key, value):  # noqa: U100
        """Raise an error. This enforces pseudo immutability."""
        raise LookupError('subplot_grid is immutable.')

    def __getitem__(self, key):
        """If an integer is passed, the item is returned. If a slice is passed,
        a `subplot_grid` of the items is returned. You can also use 2D
        indexing, and the corresponding axes in the `subplot_grid` will be
        chosen.

        Example
        -------

        >>> import proplot as plot
        ... f, axs = plot.subplots(nrows=3, ncols=3, colorbars='b', bstack=2)
        ... axs[0] # the subplot in the top-right corner
        ... axs[3] # the first subplot in the second row
        ... axs[1,2] # the subplot in the second row, third from the left
        ... axs[:,0] # the subplots in the first column

        """
        # Allow 2d specification
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        # do not expand single slice to list of integers or we get recursion!
        # len() operator uses __getitem__!
        if not isinstance(key, tuple):
            axlist = isinstance(key, slice)
            objs = list.__getitem__(self, key)
        elif len(key) == 2:
            axlist = any(isinstance(ikey, slice) for ikey in key)
            # Expand keys
            keys = []
            order = self._order
            for i, ikey in enumerate(key):
                if (i == 1 and order == 'C') or (i == 0 and order != 'C'):
                    n = self._n
                else:
                    n = len(self) // self._n
                if isinstance(ikey, slice):
                    start, stop, step = ikey.start, ikey.stop, ikey.step
                    if start is None:
                        start = 0
                    elif start < 0:
                        start = n + start
                    if stop is None:
                        stop = n
                    elif stop < 0:
                        stop = n + stop
                    if step is None:
                        step = 1
                    ikeys = [*range(start, stop, step)]
                else:
                    if ikey < 0:
                        ikey = n + ikey
                    ikeys = [ikey]
                keys.append(ikeys)
            # Get index pairs and get objects
            # Note that in double for loop, right loop varies fastest, so
            # e.g. axs[:,:] delvers (0,0), (0,1), ..., (0,N), (1,0), ...
            # Remember for order == 'F', subplot_grid was sent a list unfurled
            # in column-major order, so we replicate row-major indexing syntax
            # by reversing the order of the keys.
            objs = []
            if self._order == 'C':
                idxs = [key0 * self._n + key1 for key0 in keys[0]
                        for key1 in keys[1]]
            else:
                idxs = [key1 * self._n + key0 for key1 in keys[1]
                        for key0 in keys[0]]
            for idx in idxs:
                objs.append(list.__getitem__(self, idx))
            if not axlist:  # objs will always be length 1
                objs = objs[0]
        else:
            raise IndexError

        # Return
        if axlist:
            return subplot_grid(objs)
        else:
            return objs

    def __getattr__(self, attr):
        """
        If the attribute is *callable*, return a dummy function that loops
        through each identically named method, calls them in succession, and
        returns a tuple of the results. This lets you call arbitrary methods
        on multiple axes at once! If the `subplot_grid` has length ``1``, the
        single result is returned. If the attribute is *not callable*,
        returns a tuple of attributes for every object in the list.

        Example
        -------

        >>> import proplot as plot
        ... f, axs = plot.subplots(nrows=2, ncols=2)
        ... axs.format(...) # calls "format" on all subplots in the list
        ... paxs = axs.panel_axes('r')
        ... paxs.format(...) # calls "format" on all panels

        """
        # TODO: Consider getting rid of __getattr__ override because it is
        # too much of a mind fuck for new users? Could just have bulk format
        # function and colorbar, legend, and text functions for drawing
        # spanning content.
        if not self:
            raise AttributeError(
                f'Invalid attribute {attr!r}, axes grid {self!r} is empty.'
            )
        objs = (*(getattr(ax, attr) for ax in self),)  # may raise error

        # Objects
        if not any(callable(_) for _ in objs):
            if len(self) == 1:
                return objs[0]
            else:
                return objs
        # Methods
        # NOTE: Must manually copy docstring because help() cannot inherit it
        elif all(callable(_) for _ in objs):
            @functools.wraps(objs[0])
            def _iterator(*args, **kwargs):
                ret = []
                for func in objs:
                    ret.append(func(*args, **kwargs))
                ret = (*ret,)
                if len(self) == 1:
                    return ret[0]
                elif all(res is None for res in ret):
                    return None
                elif all(isinstance(res, axes.Axes) for res in ret):
                    return subplot_grid(ret, n=self._n, order=self._order)
                else:
                    return ret
            _iterator.__doc__ = inspect.getdoc(objs[0])
            return _iterator

        # Mixed
        raise AttributeError(f'Found mixed types for attribute {attr!r}.')

    @property
    def shape(self):
        """The "shape" of the subplot grid. For complex subplot grids, where
        subplots may span contiguous rows and columns, this "shape" may be
        incorrect. In such cases, 1d indexing should always be used."""
        return self._shape


class SubplotSpec(mgridspec.SubplotSpec):
    """
    Matplotlib `~matplotlib.gridspec.SubplotSpec` subclass that adds
    a helpful `__repr__` method. Otherwise is identical.
    """
    def __repr__(self):
        nrows, ncols, row1, row2, col1, col2 = self.get_rows_columns()
        return f'SubplotSpec({nrows}, {ncols}; {row1}:{row2}, {col1}:{col2})'


class GridSpec(mgridspec.GridSpec):
    """
    Matplotlib `~matplotlib.gridspec.GridSpec` subclass that allows for grids
    with variable spacing between successive rows and columns of axes.
    """
    def __repr__(self):  # do not show width and height ratios
        nrows, ncols = self.get_geometry()
        return f'GridSpec({nrows}, {ncols})'

    def __init__(
        self, nrows=1, ncols=1,
        left=None, right=None, bottom=None, top=None,
        wspace=None, hspace=None, wratios=None, hratios=None,
        width_ratios=None, height_ratios=None
    ):
        """
        Parameters
        ----------
        nrows, ncols : int, optional
            The number of rows and columns on the subplot grid. This is
            applied automatically when the gridspec is passed.
        left, right, bottom, top : float or str, optional
            Denotes the margin *widths* in physical units. Units are
            interpreted by `~proplot.utils.units`. These are *not* the
            margin coordinates -- for example, ``left=0.1`` and ``right=0.9``
            corresponds to a left-hand margin of 0.1 inches and a right-hand
            margin of 0.9 inches.
        hspace, wspace : float or str or list thereof, optional
            The vertical and horizontal spacing between rows and columns of
            subplots, respectively. Units are interpreted by
            `~proplot.utils.units`.

            If float or string, the spacing is identical between all rows and
            columns. If list, this sets arbitrary spacing between different
            rows and columns, and the lengths must equal ``nrows-1`` and
            ``ncols-1``, respectively.
        hratios, wratios
            Aliases for `height_ratios` and `width_ratios`.
        height_ratios, width_ratios : list of float, optional
            Ratios describing the relative heights and widths of successive
            rows and columns in the gridspec, respectively. For example,
            ``width_ratios=(1,2)`` scales a 2-column gridspec so that the
            second column is twice as wide as the first column.
        """
        # Attributes
        self._figures = set()  # figure tracker
        self._nrows, self._ncols = nrows, ncols
        self.left = self.right = self.bottom = self.top = None
        self.wspace = np.repeat(None, ncols)
        self.hspace = np.repeat(None, nrows)

        # Apply input settings
        hratios = _notNone(
            hratios, height_ratios, 1, names=(
                'hratios', 'height_ratios'))
        wratios = _notNone(
            wratios, width_ratios, 1, names=(
                'wratios', 'width_ratios'))
        hspace = _notNone(
            hspace, np.mean(hratios) * 0.10)  # this is relative to axes
        wspace = _notNone(wspace, np.mean(wratios) * 0.10)
        self.set_height_ratios(hratios)
        self.set_width_ratios(wratios)
        self.set_hspace(hspace)
        self.set_wspace(wspace)
        self.set_margins(left, right, bottom, top)

    def _sanitize_hspace(self, space):
        """Sanitize the hspace vector. This needs to be set apart from
        set_hspace because gridspec params adopted from the figure are often
        scalar and need to be expanded into vectors during
        get_grid_positions."""
        N = self._nrows
        space = np.atleast_1d(units(space))
        if len(space) == 1:
            space = np.repeat(space, (N - 1,))  # note: may be length 0
        if len(space) != N - 1:
            raise ValueError(
                f'GridSpec has {N} rows and accepts {N-1} hspaces, '
                f'but got {len(space)} hspaces.')
        return space

    def _sanitize_wspace(self, space):
        """Sanitize the wspace vector."""
        N = self._ncols
        space = np.atleast_1d(units(space))
        if len(space) == 1:
            space = np.repeat(space, (N - 1,))  # note: may be length 0
        if len(space) != N - 1:
            raise ValueError(
                f'GridSpec has {N} columns and accepts {N-1} wspaces, '
                f'but got {len(space)} wspaces.')
        return space
        filter = (space is not None)
        self.wspace[filter] = space[filter]

    def add_figure(self, figure):
        """Add `~matplotlib.figure.Figure` to the list of figures that are
        using this gridspec. This is done automatically when calling
        `~Figure.add_subplot` with a subplotspec generated by this gridspec."""
        if not isinstance(figure, Figure):
            raise ValueError(
                f'add_figure() accepts only ProPlot Figure instances, '
                f'you passed {type(figure)}.'
            )
        self._figures.add(figure)

    def get_grid_positions(self, figure, raw=False):
        """Calculate grid positions using the input figure and scale
        the width and height ratios to figure relative coordinates."""
        # Retrieve properties
        # TODO: Need to completely rewrite this! Interpret physical parameters
        # and permit variable spacing.
        # TODO: Since we disable interactive gridspec adjustment should also
        # disable interactive figure resizing. See:
        # https://stackoverflow.com/q/21958534/4970632
        # https://stackoverflow.com/q/33881554/4970632
        # NOTE: This gridspec *never* uses figure subplotpars. Matplotlib fills
        # subplotpars with rcParams, then GridSpec values that were *explicitly
        # passed* by user overwrite subplotpars. Instead, we make sure spacing
        # arguments stored on GridSpec are *never* None. Thus we eliminate
        # get_subplot_params and render *subplots_adjust* useless. We *cannot*
        # overwrite subplots_adjust with physical units because various widgets
        # use it with figure-relative units. This approach means interactive
        # subplot adjustment sliders no longer work but for now that is best
        # approach; we would need to make user-specified subplots_adjust
        # instead overwrite default gridspec values, gets complicated.
        nrows, ncols = self.get_geometry()
        if raw:
            left = bottom = 0
            right = top = 1
            wspace = self._sanitize_wspace(0)
            hspace = self._sanitize_hspace(0)
        else:
            if not isinstance(figure, Figure):
                raise ValueError(
                    f'Invalid figure {figure!r}. '
                    f'Must be a proplot.subplots.Figure instance.')
            width, height = figure.get_size_inches()
            left, right, bottom, top, wspace, hspace = figure._gridspecpars
            wspace = self._sanitize_wspace(wspace)
            hspace = self._sanitize_hspace(hspace)
            left = _notNone(self.left, left, 0) / width
            right = 1 - _notNone(self.right, right, 0) / width
            bottom = _notNone(self.bottom, bottom, 0) / height
            top = 1 - _notNone(self.top, top, 0) / height

        # Calculate accumulated heights of columns
        tot_width = right - left
        tot_height = top - bottom
        cell_h = tot_height / (nrows + hspace * (nrows - 1))
        sep_h = hspace * cell_h
        if self._row_height_ratios is not None:
            norm = cell_h * nrows / sum(self._row_height_ratios)
            cell_heights = [r * norm for r in self._row_height_ratios]
        else:
            cell_heights = [cell_h] * nrows
        sep_heights = [0] + ([sep_h] * (nrows - 1))
        cell_hs = np.cumsum(np.column_stack([sep_heights, cell_heights]).flat)

        # Calculate accumulated widths of rows
        cell_w = tot_width / (ncols + wspace * (ncols - 1))
        sep_w = wspace * cell_w
        if self._col_width_ratios is not None:
            norm = cell_w * ncols / sum(self._col_width_ratios)
            cell_widths = [r * norm for r in self._col_width_ratios]
        else:
            cell_widths = [cell_w] * ncols
        sep_widths = [0] + ([sep_w] * (ncols - 1))
        cell_ws = np.cumsum(np.column_stack([sep_widths, cell_widths]).flat)
        fig_tops, fig_bottoms = (top - cell_hs).reshape((-1, 2)).T
        fig_lefts, fig_rights = (left + cell_ws).reshape((-1, 2)).T
        return fig_bottoms, fig_tops, fig_lefts, fig_rights

    def get_subplot_params(self, figure=None):
        """Raise an error. This method is disabled because ProPlot does not
        and cannot use the SubplotParams stored on figures."""
        raise NotImplementedError(
            f'ProPlot GridSpec does not interact with figure SubplotParams.'
        )

    def get_hspace(self):
        """Return the vector of row spaces."""
        return self.hspace

    def get_margins(self):
        """Return the left, bottom, right, top margin spaces."""
        return self.left, self.bottom, self.right, self.top

    def get_wspace(self):
        """Return the vector of column spaces."""
        return self.wspace

    def remove_figure(self, figure):
        """Remove `~matplotlib.figure.Figure` from the list of figures that
        are using this gridspec."""
        self._figures.discard(figure)

    def set_height_ratios(self, ratios):
        """Set the row height ratios. Value must be a vector of length
        ``nrows``."""
        N = self._nrows
        ratios = np.atleast_1d(ratios)
        if len(ratios) == 1:
            ratios = np.repeat(ratios, (N,))
        if len(ratios) != N:
            raise ValueError(
                f'GridSpec has {N} rows, but got {len(ratios)} height ratios.')
        super().set_height_ratios(self)

    def set_hspace(self, space):
        """Set the inter-row spacing in physical units. Units are interpreted
        by `~proplot.utils.units`. Pass a vector of length ``nrows - 1`` to
        implement variable spacing between successive rows."""
        space = self._sanitize_hspace(space)
        filter = (space is not None)
        self.hspace[filter] = space[filter]

    def set_margins(self, left, right, bottom, top):
        """Set the margin values in physical units. Units are interpreted by
        `~proplot.utils.units`."""
        if left is not None:
            self.left = units(left)
        if right is not None:
            self.right = units(right)
        if bottom is not None:
            self.bottom = units(bottom)
        if top is not None:
            self.top = units(top)

    def set_width_ratios(self, ratios):
        """Set the column width ratios. Value must be a vector of length
        ``ncols``."""
        N = self._ncols
        ratios = np.atleast_1d(ratios)
        if len(ratios) == 1:
            ratios = np.repeat(ratios, (N,))
        if len(ratios) != N:
            raise ValueError(
                f'GridSpec has {N} columns, but '
                f'got {len(ratios)} width ratios.')
        super().set_width_ratios(self)

    def set_wspace(self, space):
        """Set the inter-column spacing in physical units. Units are interpreted
        by `~proplot.utils.units`. Pass a vector of length ``ncols - 1`` to
        implement variable spacing between successive columns."""
        space = self._sanitize_wspace(space)
        filter = (space is not None)
        self.wspace[filter] = space[filter]

    def tight_layout(self, *args, **kwargs):
        """Method is disabled because ProPlot has its own tight layout
        algorithm."""
        raise NotImplementedError(
            f'Native matplotlib tight layout is disabled.')

    def update(
        self, left=None, right=None, bottom=None, top=None,
        wspace=None, hspace=None, wratios=None, hratios=None,
        width_ratios=None, height_ratios=None
    ):
        """
        Update the gridspec with arbitrary initialization keyword arguments
        then *apply* those updates to every figure using this gridspec.

        The default `~matplotlib.gridspec.GridSpec.update` tries to update
        positions for axes on all active figures -- but this can fail after
        successive figure edits if it has been removed from the figure
        manager. ProPlot insists one gridspec per figure, tracks the figures
        that are using this gridspec object, and applies updates to those
        tracked figures.

        Parameters
        ----------
        **kwargs
            Valid initialization keyword arguments. See `GridSpec`.
        """
        # Setter methods only apply values if not None
        self.set_margins(left, right, bottom, top)
        self.set_wspace(wspace)
        self.set_hspace(hspace)
        hratios = _notNone(hratios, height_ratios)
        wratios = _notNone(wratios, width_ratios)
        if wratios is not None:
            self.set_width_ratios(wratios)
        if hratios is not None:
            self.set_height_ratios(hratios)
        for figure in self._figures:
            figure._solver._init()  # in case gridspec values changed!
            for ax in figure.axes:
                ax.update_params()
                ax.set_position(ax.figbox)
            figure.stale = True


def _canvas_preprocess(canvas, method):
    """Return a pre-processer that can be used to override instance-level
    canvas draw_idle() and print_figure() methods. This applies tight layout
    and aspect ratio-conserving adjustments and aligns labels. Required so that
    the canvas methods instantiate renderers with the correct dimensions.
    Note that MacOSX currently `cannot be resized \
<https://github.com/matplotlib/matplotlib/issues/15131>`__."""
    # TODO: Update this to use GeometrySolver
    # NOTE: This is by far the most robust approach. Renderer must be (1)
    # initialized with the correct figure size or (2) changed inplace during
    # draw, but vector graphic renderers *cannot* be changed inplace.
    # Options include (1) monkey patch canvas.get_width_height, overriding
    # figure.get_size_inches, and exploit the FigureCanvasAgg.get_renderer()
    # implementation (because FigureCanvasAgg queries the bbox directly
    # rather than using get_width_height() so requires a workaround), or (2)
    # override bbox and bbox_inches as *properties*, but these are really
    # complicated, dangerous, and result in unnecessary extra draws.
    def _preprocess(self, *args, **kwargs):
        fig = self.figure  # update even if not stale! needed after saves
        if method == 'draw_idle' and (
            self._is_idle_drawing  # standard
            or getattr(self, '_draw_pending', None)  # pyqt5
        ):
            # For now we override 'draw' and '_draw' rather than 'draw_idle'
            # but may change mind in the future. This breakout condition is
            # copied from the matplotlib source.
            return
        if method == 'print_figure':
            # When re-generating inline figures, the tight layout algorithm
            # can get figure size *or* spacing wrong unless we force additional
            # draw! Seems to have no adverse effects when calling savefig.
            self.draw()
        if fig._is_preprocessing:
            return
        with fig._context_preprocessing():
            renderer = fig._get_renderer()  # any renderer will do for now
            for ax in fig._iter_axes():
                ax._draw_auto_legends_colorbars()  # may insert panels
            resize = rc['backend'] != 'nbAgg'
            if resize:
                fig._adjust_aspect()  # resizes figure
            if fig._auto_tight:
                fig._adjust_tight_layout(renderer, resize=resize)
            fig._align_axislabels(True)
            fig._align_labels(renderer)
            fallback = _notNone(
                fig._fallback_to_cm, rc['mathtext.fallback_to_cm']
            )
            with rc.context({'mathtext.fallback_to_cm': fallback}):
                return getattr(type(self), method)(self, *args, **kwargs)
    return _preprocess.__get__(canvas)  # ...I don't get it either


def _get_panelargs(
    side, share=None, width=None, space=None,
    filled=False, figure=False
):
    """Return default properties for new axes and figure panels."""
    if side not in ('left', 'right', 'bottom', 'top'):
        raise ValueError(f'Invalid panel location {side!r}.')
    space = space_user = units(space)
    if share is None:
        share = not filled
    if width is None:
        if filled:
            width = rc['colorbar.width']
        else:
            width = rc['subplots.panelwidth']
    width = units(width)
    if space is None:
        key = 'wspace' if side in ('left', 'right') else 'hspace'
        pad = rc['subplots.axpad'] if figure else rc['subplots.panelpad']
        space = _get_space(key, share, pad=pad)
    return share, width, space, space_user


def _get_space(key, share=0, pad=None):
    """Return suitable default spacing given a shared axes setting."""
    if key == 'left':
        space = units(_notNone(pad, rc['subplots.pad'])) + (
            rc['ytick.major.size'] + rc['ytick.labelsize']
            + rc['ytick.major.pad'] + rc['axes.labelsize']) / 72
    elif key == 'right':
        space = units(_notNone(pad, rc['subplots.pad']))
    elif key == 'bottom':
        space = units(_notNone(pad, rc['subplots.pad'])) + (
            rc['xtick.major.size'] + rc['xtick.labelsize']
            + rc['xtick.major.pad'] + rc['axes.labelsize']) / 72
    elif key == 'top':
        space = units(_notNone(pad, rc['subplots.pad'])) + (
            rc['axes.titlepad'] + rc['axes.titlesize']) / 72
    elif key == 'wspace':
        space = (units(_notNone(pad, rc['subplots.axpad']))
                 + rc['ytick.major.size'] / 72)
        if share < 3:
            space += (rc['ytick.labelsize'] + rc['ytick.major.pad']) / 72
        if share < 1:
            space += rc['axes.labelsize'] / 72
    elif key == 'hspace':
        space = units(_notNone(pad, rc['subplots.axpad'])) + (
            rc['axes.titlepad'] + rc['axes.titlesize']
            + rc['xtick.major.size']) / 72
        if share < 3:
            space += (rc['xtick.labelsize'] + rc['xtick.major.pad']) / 72
        if share < 0:
            space += rc['axes.labelsize'] / 72
    else:
        raise KeyError(f'Invalid space key {key!r}.')
    return space


class EdgeStack(object):
    """
    Container for groups of `~matplotlib.artist.Artist` objects stacked
    along the edge of a subplot. Calculates bounding box coordiantes for
    objects in the stack.
    """
    def __init__(self, *args):
        if not all(isinstance(arg, martist.Artst) for arg in args):
            raise ValueError(f'Arguments must be artists.')


class GeometrySolver(object):
    """
    ProPlot's answer to the matplotlib `~matplotlib.figure.SubplotParams`
    class. When `tight` is ``False``, this object is filled with sensible
    default spacing params dependent on the axis sharing settings. When
    `tight` is ``True``, this object is filled with spaces empirically
    determined with the tight layout algorithm.
    """
    def __init__(self, figure):
        """
        Parameters
        ----------
        figure : `Figure`
            The figure instance associated with this geometry configuration.
        """
        if not isinstance(figure, Figure):
            raise ValueError(
                f'GeometrySolver() accepts only proplot.subplots.Figure '
                f'instances, you passed {type(figure)}.')
        self._figure = figure
        self._isinit = False

    def _adjust_aspect(self):
        """Adjust the average aspect ratio used for gridspec calculations.
        This fixes grids with identically fixed aspect ratios, e.g.
        identically zoomed-in cartopy projections and imshow images."""
        # Get aspect ratio
        figure = self.figure
        ax = figure.get_ref_axes()
        if not ax:
            return
        curaspect = ax.get_aspect()
        if isinstance(curaspect, str):
            if curaspect == 'auto':
                return
            elif curaspect != 'equal':
                raise RuntimeError(f'Unknown aspect ratio mode {curaspect!r}.')

        # Compare to current aspect
        xscale, yscale = ax.get_xscale(), ax.get_yscale()
        if not isinstance(curaspect, str):
            aspect = curaspect
        elif xscale == 'linear' and yscale == 'linear':
            aspect = 1.0 / ax.get_data_ratio()
        elif xscale == 'log' and yscale == 'log':
            aspect = 1.0 / ax.get_data_ratio_log()
        else:
            return  # matplotlib should have issued warning
        if np.isclose(aspect, self.aspect):
            return
        self.aspect = aspect
        self.solve()

    def _adjust_tight_layout(self, renderer, resize=True):
        """Apply tight layout scaling that permits flexible figure
        dimensions and preserves panel widths and subplot aspect ratios."""
        # Initial stuff
        axs = self._iter_axes()
        gridspec = self._gridspec
        if not axs or not gridspec:
            return

        # Positions for labels and suptitle
        self._align_axislabels(False)
        self._align_labels(renderer)
        nrows, ncols = gridspec.get_geometry()

        # Boxes and padding
        bbox = self.get_tightbbox(renderer)
        bbox_orig = self.bbox_inches  # original bbox
        pad = self._pad
        axpad = self._axpad
        panelpad = self._panelpad

        # Tight box *around* figure permitting user overrides
        left, right, bottom, top, wspace, hspace = self._gridspecpars
        left = left - bbox.xmin + pad
        right = right - bbox.ymin + pad
        bottom = bottom - (bbox_orig.xmax - bbox.xmax) + pad
        top = top - (bbox_orig.ymax - bbox.ymax) + pad

        # Get new subplot spacings, axes panel spacing, figure panel spacing
        spaces = []
        for (w, x, y, nacross, ispace) in zip(
            'wh', 'xy', 'yx', (nrows, ncols), (wspace, hspace)
        ):
            # Determine which rows and columns correspond to panels
            panels = getattr(self, '_' + w + 'panels')
            jspace = [*ispace]
            ralong = np.array([ax._range_gridspec(x) for ax in axs])
            racross = np.array([ax._range_gridspec(y) for ax in axs])
            for i, space in enumerate(ispace):
                # Figure out whether this is a normal space, or a
                # panel stack space/axes panel space
                if (
                    panels[i] in ('l', 't')
                    and panels[i + 1] in ('l', 't', '')
                    or panels[i] in ('', 'r', 'b')
                    and panels[i + 1] in ('r', 'b')
                    or panels[i] == 'f' and panels[i + 1] == 'f'
                ):
                    pad = panelpad
                else:
                    pad = axpad

                # Find axes that abutt aginst this space on each row
                groups = []
                # i.e. right/bottom edge abutts against this space
                filt1 = ralong[:, 1] == i
                # i.e. left/top edge abutts against this space
                filt2 = ralong[:, 0] == i + 1
                for j in range(nacross):  # e.g. each row
                    # Get indices
                    filt = (racross[:, 0] <= j) & (j <= racross[:, 1])
                    if sum(filt) < 2:  # no interface here
                        continue
                    idx1, = np.where(filt & filt1)
                    idx2, = np.where(filt & filt2)
                    if idx1.size > 1 or idx2.size > 2:
                        _warn_proplot('This should never happen.')
                        continue
                    elif not idx1.size or not idx2.size:
                        continue
                    idx1, idx2 = idx1[0], idx2[0]
                    # Put these axes into unique groups. Store groups as
                    # (left axes, right axes) or (bottom axes, top axes) pairs.
                    ax1, ax2 = axs[idx1], axs[idx2]
                    if x != 'x':  # order bottom-to-top
                        ax1, ax2 = ax2, ax1
                    newgroup = True
                    for (group1, group2) in groups:
                        if ax1 in group1 or ax2 in group2:
                            newgroup = False
                            group1.add(ax1)
                            group2.add(ax2)
                            break
                    if newgroup:
                        groups.append([{ax1}, {ax2}])  # form new group
                # Get spaces
                # Layout is lspace, lspaces[0], rspaces[0], wspace, ...
                # so panels spaces are located where i % 3 is 1 or 2
                jspaces = []
                for (group1, group2) in groups:
                    x1 = max(ax._range_tightbbox(x)[1] for ax in group1)
                    x2 = min(ax._range_tightbbox(x)[0] for ax in group2)
                    jspaces.append((x2 - x1) / self.dpi)
                if jspaces:
                    space = max(0, space - min(jspaces) + pad)
                jspace[i] = space
            spaces.append(jspace)

        # Update gridspec params list
        # NOTE: This is where users can override calculated tight bounds
        self._fill_gridspecpars(left, right, bottom, top, *spaces)
        self.solve()

    def _align_axislabels(self, b=True):
        """Align spanning *x* and *y* axis labels, accounting for figure
        margins and axes and figure panels."""
        # TODO: Ensure this is robust to complex panels and shared axes.
        # NOTE: Aligned labels have to be turned off before
        # _adjust_tight_layout call, so this cannot be inside Axes draw
        tracker = set()
        grpx = getattr(self, '_align_xlabel_grp', None)
        grpy = getattr(self, '_align_ylabel_grp', None)
        spanx, spany = self._spanx, self._spany
        alignx, aligny = self._alignx, self._aligny
        if ((spanx or alignx) and grpx) or ((spany or aligny) and grpy):
            _warn_proplot(
                'Aligning *x* and *y* axis labels requires '
                'matplotlib >=3.1.0'
            )
            return
        for ax in self._mainaxes:
            for x, axis, span, align, grp in zip(
                'xy', (ax.xaxis, ax.yaxis), (spanx, spany),
                (alignx, aligny), (grpx, grpy)
            ):
                # Settings
                if (not span and not align) or not isinstance(
                        ax, axes.XYAxes) or axis in tracker:
                    continue
                # top or bottom, left or right
                side = axis.get_label_position()
                if side not in ('bottom', 'left'):
                    continue
                axs = ax._get_side_axes(side)
                for _ in range(2):
                    axs = [getattr(ax, '_share' + x) or ax for ax in axs]

                # Adjust axis label offsets
                axises = [getattr(ax, x + 'axis') for ax in axs]
                tracker.update(axises)
                if span or align:
                    if grp is not None:
                        for ax in axs[1:]:
                            # copied from source code, add to grouper
                            grp.join(axs[0], ax)
                    elif align:
                        _warn_proplot(
                            f'Aligning *x* and *y* axis labels required '
                            f'matplotlib >=3.1.0'
                        )
                if not span:
                    continue

                # Adjust axis label centering
                c, spanax = self._get_align_coord(side, axs)
                spanaxis = getattr(spanax, x + 'axis')
                spanlabel = spanaxis.label
                if not hasattr(spanlabel, '_orig_transform'):
                    spanlabel._orig_transform = spanlabel.get_transform()
                    spanlabel._orig_position = spanlabel.get_position()
                if not b:  # toggle off, done before tight layout
                    spanlabel.set_transform(spanlabel._orig_transform)
                    spanlabel.set_position(spanlabel._orig_position)
                    for axis in axises:
                        axis.label.set_visible(True)
                else:  # toggle on, done after tight layout
                    if x == 'x':
                        position = (c, 1)
                        transform = mtransforms.blended_transform_factory(
                            self.transFigure, mtransforms.IdentityTransform())
                    else:
                        position = (1, c)
                        transform = mtransforms.blended_transform_factory(
                            mtransforms.IdentityTransform(), self.transFigure)
                    for axis in axises:
                        axis.label.set_visible((axis is spanaxis))
                    spanlabel.update({
                        'position': position, 'transform': transform
                    })

    def _align_labels(self, renderer):
        """Adjust the position of row and column labels, and align figure super
        title accounting for figure margins and axes and figure panels."""
        # Offset using tight bounding boxes
        # TODO: Super labels fail with popup backend!! Fix this
        # NOTE: Must use get_tightbbox so (1) this will work if tight layout
        # mode if off and (2) actually need *two* tight bounding boxes when
        # labels are present: 1 not including the labels, used to position
        # them, and 1 including the labels, used to determine figure borders
        suptitle = self._suptitle
        suptitle_on = suptitle.get_text().strip()
        width, height = self.get_size_inches()
        for side in ('left', 'right', 'bottom', 'top'):
            # Get axes and offset the label to relevant panel
            if side in ('left', 'right'):
                x = 'x'
                iter_panels = ('bottom', 'top')
            else:
                x = 'y'
                iter_panels = ('left', 'right')
            axs = self._get_align_axes(side)
            axs = [ax._reassign_suplabel(side) for ax in axs]
            labels = [getattr(ax, '_' + side + '_label') for ax in axs]
            coords = [None] * len(axs)
            if side == 'top' and suptitle_on:
                supaxs = axs

            # Position labels
            # TODO: No more _hidelabels here! Must determine positions
            # with EdgeStack instead of with axes tightbbox algorithm!
            for i, (ax, label) in enumerate(zip(axs, labels)):
                label_on = label.get_text().strip()
                if not label_on:
                    continue
                # Get coord from tight bounding box
                # Include twin axes and panels along the same side
                icoords = []
                for iax in ax._iter_panels(iter_panels):
                    bbox = iax.get_tightbbox(renderer)
                    if side == 'left':
                        jcoords = (bbox.xmin, 0)
                    elif side == 'right':
                        jcoords = (bbox.xmax, 0)
                    elif side == 'top':
                        jcoords = (0, bbox.ymax)
                    else:
                        jcoords = (0, bbox.ymin)
                    c = self.transFigure.inverted().transform(jcoords)
                    c = c[0] if side in ('left', 'right') else c[1]
                    icoords.append(c)
                # Offset, and offset a bit extra for left/right labels
                # See:
                # https://matplotlib.org/api/text_api.html#matplotlib.text.Text.set_linespacing
                fontsize = label.get_fontsize()
                if side in ('left', 'right'):
                    scale1, scale2 = 0.6, width
                else:
                    scale1, scale2 = 0.3, height
                if side in ('left', 'bottom'):
                    coords[i] = min(icoords) - (
                        scale1 * fontsize / 72) / scale2
                else:
                    coords[i] = max(icoords) + (
                        scale1 * fontsize / 72) / scale2
            # Assign coords
            coords = [i for i in coords if i is not None]
            if coords:
                if side in ('left', 'bottom'):
                    c = min(coords)
                else:
                    c = max(coords)
                for label in labels:
                    label.update({x: c})

        # Update super title position
        # If no axes on the top row are visible, do not try to align!
        if suptitle_on and supaxs:
            ys = []
            for ax in supaxs:
                bbox = ax.get_tightbbox(renderer)
                _, y = self.transFigure.inverted().transform((0, bbox.ymax))
                ys.append(y)
            x, _ = self._get_align_coord('t', supaxs)
            y = max(ys) + (0.3 * suptitle.get_fontsize() / 72) / height
            kw = {'x': x, 'y': y, 'ha': 'center', 'va': 'bottom',
                  'transform': self.transFigure}
            suptitle.update(kw)

    def _init(self):
        """Fill the spacing parameters with defaults."""
        # Get settings
        # NOTE: This gets called (1) when gridspec assigned to a figure and
        # (2) when gridspec.update() is called. This roughly matches the
        # matplotlib behavior -- changes to the gridspec are not applied to
        # figures until update() is explicitly called.
        # NOTE: Kind of redundant to inherit spacing params from gridspec
        # when get_grid_positions() ignores spacing params that were explicitly
        # set on the gridspec anyway. But necessary for tight layout calcs
        # because we need to compare how far apart subplot content *currently*
        # is with the *current* spacing values used to position them, then
        # adjust those spacing values as necessary.
        # NOTE: Think of this as the intersection between figure spacing params
        # and gridspec params, both of which might not have been specified.
        # TODO: In insert_row_column, we can just borrow spacing values
        # directly from the existing gridspec.
        fig = self.figure
        gs = self._gridspec
        if gs is None:
            raise RuntimeError(
                f'GridSpec is not present. Cannot initialize GeometrySolver.')

        # Add spacing params to the object for label alignment calculations
        # Note that gridspec wspace and hspace are already sanitized
        ncols, nrows = gs.get_geometry()
        self.ncols, self.nrows = ncols, nrows
        self.left = _notNone(gs.left, _get_space('left'))
        self.right = _notNone(gs.right, _get_space('right'))
        self.bottom = _notNone(gs.bottom, _get_space('bottom'))
        self.top = _notNone(gs.top, _get_space('top'))
        wspace = np.repeat(_get_space('wspace', fig._sharex), ncols - 1)
        hspace = np.repeat(_get_space('hspace', fig._sharey), nrows - 1)
        self.wspace = _notNone(gs.wspace, wspace)
        self.hspace = _notNone(gs.hspace, hspace)
        self.wratios = gs.get_width_ratios()
        self.hratios = gs.get_height_ratios()

        # Add panel string toggles (contains '' for subplots, 'lrbt' for axes
        # panels, and 'f' for figure panels) and figure panel array trakcers
        self.barray = np.empty((0, ncols), dtype=bool)
        self.tarray = np.empty((0, ncols), dtype=bool)
        self.larray = np.empty((0, nrows), dtype=bool)
        self.rarray = np.empty((0, nrows), dtype=bool)
        self.wpanels = [''] * ncols
        self.hpanels = [''] * nrows

        # Indicate we are initialized
        self._isinit = True

    # def resize(self):
    def solve(self, nrows=None, ncols=None, array=None, **kwargs):
        """Determine the figure size necessary to preserve physical
        gridspec spacing and the widths of special *panel* slots in the
        gridspec object."""
        # Pull out various properties
        nrows, ncols = self.nrows, self.ncols
        width, height = self.width, self.height
        axwidth, axheight = self.axwidth, self.axheight
        left, right = self.left, self.right
        bottom, top = self.bottom, self.top
        wspace, hspace = self.wspace, self.hspace
        wratios, hratios = self.wratios, self.hratios
        wpanels, hpanels = self.wpanels, self.hpanels

        # Horizontal spacing indices
        # map(func, wpanels[idx + 1:]).index(True) - 1]
        wmask = np.array([not s for s in wpanels])
        wratios_main = np.array(wratios)[wmask]
        wratios_panels = np.array(wratios)[~wmask]
        wspace_main = [
            wspace[idx + next(
                i for i, p in enumerate(wpanels[idx + 1:]) if p == 'r'
            )]
            for idx in np.where(wmask)[0][:-1]
        ]
        # Vertical spacing indices
        hmask = np.array([not s for s in hpanels])
        hratios_main = np.array(hratios)[hmask]
        hratios_panels = np.array(hratios)[~hmask]
        hspace_main = [
            hspace[idx + next(
                i for i, p in enumerate(hpanels[idx + 1:]) if p == 'b'
            )]
            for idx in np.where(hmask)[0][:-1]
        ]

        # Try to use ratios and spaces spanned by reference axes
        # NOTE: In old version we automatically resized when figure was
        # created using the reference axes *slot* inferred from the subplots
        # array. In new version we just *try* to use the reference axes, but
        # if it does not exist, use the average width and height for a single
        # cell of the gridspec rather than a particular axes.
        # gs = self._gridspec
        ax = self.get_ref_axes()
        if ax is None:
            rwspace = 0
            rhspace = 0
            rwratio = 1
            rhratio = 1
        else:
            _, _, y1, y2, x1, x2 = ax.get_subplotspec().get_rows_columns()
            dx, dy = x2 - x1 + 1, y2 - y1 + 1
            nrows_main = len(hratios_main)
            ncols_main = len(wratios_main)
            rwspace = sum(wspace_main[x1:x2])
            rhspace = sum(hspace_main[y1:y2])
            rwratio = (
                ncols_main * sum(wratios_main[x1:x2 + 1])
            ) / (dx * sum(wratios_main))
            rhratio = (
                nrows_main * sum(hratios_main[y1:y2 + 1])
            ) / (dy * sum(hratios_main))
            if rwratio == 0 or rhratio == 0:
                raise RuntimeError(
                    f'Something went wrong, got wratio={rwratio!r} '
                    f'and hratio={rhratio!r} for reference axes.')
            aspect = self.aspect

        # Determine figure and axes dims from input in width or height dim.
        # For e.g. common use case [[1,1,2,2],[0,3,3,0]], make sure we still
        # scale the reference axes like square even though takes two columns
        # of gridspec!
        auto_width = (width is None and height is not None)
        auto_height = (height is None and width is not None)
        if width is None and height is None:  # get stuff directly from axes
            if axwidth is None and axheight is None:
                axwidth = units(rc['subplots.axwidth'])
            if axheight is not None:
                auto_width = True
                axheight_all = (
                    nrows_main * (axheight - rhspace)) / (dy * rhratio)
                height = axheight_all + top + bottom + \
                    sum(hspace) + sum(hratios_panels)
            if axwidth is not None:
                auto_height = True
                axwidth_all = (ncols_main * (axwidth - rwspace)
                               ) / (dx * rwratio)
                width = axwidth_all + left + right + \
                    sum(wspace) + sum(wratios_panels)
            if axwidth is not None and axheight is not None:
                auto_width = auto_height = False
        else:
            if height is not None:
                axheight_all = height - top - bottom - \
                    sum(hspace) - sum(hratios_panels)
                axheight = (axheight_all * dy * rhratio) / nrows_main + rhspace
            if width is not None:
                axwidth_all = width - left - right - \
                    sum(wspace) - sum(wratios_panels)
                axwidth = (axwidth_all * dx * rwratio) / ncols_main + rwspace

        # Automatically figure dim that was not specified above
        if auto_height:
            axheight = axwidth / aspect
            axheight_all = (nrows_main * (axheight - rhspace)) / (dy * rhratio)
            height = axheight_all + top + bottom + \
                sum(hspace) + sum(hratios_panels)
        elif auto_width:
            axwidth = axheight * aspect
            axwidth_all = (ncols_main * (axwidth - rwspace)) / (dx * rwratio)
            width = axwidth_all + left + right + \
                sum(wspace) + sum(wratios_panels)
        if axwidth_all < 0:
            raise ValueError(
                'Not enough room for axes '
                f'(would have width {axwidth_all}). '
                'Try using tight=False, increasing figure width, or '
                "decreasing 'left', 'right', or 'wspace' spaces.")
        if axheight_all < 0:
            raise ValueError(
                'Not enough room for axes '
                f'(would have height {axheight_all}). '
                'Try using tight=False, increasing figure height, or '
                "decreasing 'top', 'bottom', or 'hspace' spaces.")

        # Reconstruct the ratios array with physical units for subplot slots
        # The panel slots are unchanged because panels have fixed widths
        wratios_main = axwidth_all * np.array(wratios_main) / sum(wratios_main)
        hratios_main = axheight_all * \
            np.array(hratios_main) / sum(hratios_main)
        for idx, ratio in zip(np.where(hmask)[0], hratios_main):
            hratios[idx] = ratio
        for idx, ratio in zip(np.where(wmask)[0], wratios_main):
            wratios[idx] = ratio

        # Update figure size and gridspec
        self.set_size_inches((width, height), manual=True)
        if self._gridspec:
            self._gridspec.update(
                ncols=ncols, nrows=nrows,
                wspace=wspace, hspace=hspace,
                width_ratios=wratios, height_ratios=hratios,
                left=left, bottom=bottom, right=right, top=top,
            )
        else:
            self._gridspec = GridSpec(
                ncols=ncols, nrows=nrows,
                wspace=wspace, hspace=hspace,
                width_ratios=wratios, height_ratios=hratios,
                left=left, bottom=bottom, right=right, top=top,
            )
        return self._gridspec

    def update(self, renderer=None):
        """Update the default values in case the spacing has changed."""
        # TODO: This will replace the various _adjust functions and
        # carry out what is currently in the _preprocess and draw funcs
        fig = self.figure
        gs = fig.gridspec
        if gs is None:
            raise ValueError(
                'GridSpec has not been initialized yet. '
                'Cannot update GeometrySolver.'
            )

        # Get renderer if not passed
        renderer = fig._get_renderer()  # noqa TODO: finish

    @property
    def figure(self):
        """The `Figure` instance associated with this geometry configuration.
        This cannot be modified."""
        return self._figure


class Figure(mfigure.Figure):
    """The `~matplotlib.figure.Figure` class returned by `subplots`. At
    draw-time, an improved tight layout algorithm is employed, and
    the space around the figure edge, between subplots, and between
    panels is changed to accommodate subplot content. Figure dimensions
    may be automatically scaled to preserve subplot aspect ratios."""
    @docstring.dedent_interpd
    def __init__(
        self,
        figsize=None, width=None, height=None, journal=None,
        axwidth=None, axheight=None, aspect=1,
        tight=None, pad=None, axpad=None, panelpad=None,
        left=None, right=None, bottom=None, top=None,
        wspace=None, hspace=None,
        share=None, sharex=None, sharey=None,
        span=None, spanx=None, spany=None,
        align=None, alignx=None, aligny=None,
        includepanels=False, autoformat=True, ref=1,
        fallback_to_cm=False,
        **kwargs
    ):
        """
        Parameters
        ----------
        %(figure_doc)s
        """
        # Initialize first
        self._is_preprocessing = False
        self._is_resizing = False
        tight_layout = kwargs.pop('tight_layout', None)
        constrained_layout = kwargs.pop('constrained_layout', None)
        if tight_layout or constrained_layout:
            _warn_proplot(
                f'Ignoring tight_layout={tight_layout} and '
                f'contrained_layout={constrained_layout}. ProPlot uses '
                'its own tight layout algorithm, activated by default or '
                'with tight=True.')
        super().__init__(**kwargs)

        # Axes sharing and spanning settings
        sharex = _notNone(sharex, share, rc['share'])
        sharey = _notNone(sharey, share, rc['share'])
        spanx = _notNone(spanx, span, 0 if sharex == 0 else None, rc['span'])
        spany = _notNone(spany, span, 0 if sharey == 0 else None, rc['span'])
        if spanx and (alignx or align):
            _warn_proplot(f'"alignx" has no effect when spanx=True.')
        if spany and (aligny or align):
            _warn_proplot(f'"aligny" has no effect when spany=True.')
        alignx = _notNone(alignx, align, rc['align'])
        aligny = _notNone(aligny, align, rc['align'])
        self.set_alignx(alignx)
        self.set_aligny(aligny)
        self.set_sharex(sharex)
        self.set_sharey(sharey)
        self.set_spanx(spanx)
        self.set_spany(spany)

        # Issue warnings for conflicting dimension specifications
        if journal is not None:
            names = ('axwidth', 'axheight', 'width', 'height')
            values = (axwidth, axheight, width, height)
            figsize = _journal_figsize(journal)
            spec = f'journal={journal!r}'
            width, height = figsize
        elif figsize is not None:
            names = ('axwidth', 'axheight', 'width', 'height')
            values = (axwidth, axheight, width, height)
            spec = f'figsize={figsize!r}'
            width, height = figsize
        elif width is not None or height is not None:
            names = ('axwidth', 'axheight')
            values = (axwidth, axheight)
            spec = []
            if width is not None:
                spec.append(f'width={width!r}')
            if height is not None:
                spec.append(f'height={height!r}')
            spec = ', '.join(spec)
        for name, value in zip(names, values):
            if value is not None:
                _warn_proplot(
                    f'You specified both {spec} and {name}={value!r}. '
                    f'Ignoring {name!r}.')

        # Various constants and hidden settings
        # self._gridspecpars = (None,) * 6
        # self._gridspecpars_user = (
        #   units(left), units(bottom), units(right), units(top),
        #   units(wspace), units(hspace))
        # self._dimensions = (units(width), units(height),
        #   units(axwidth), units(axheight), aspect)
        if np.iterable(aspect):
            aspect = aspect[0] / aspect[1]
        self._solver = GeometrySolver(
            width, height, axwidth, axheight, aspect
        )
        self._gridspecpars = (
            units(left),
            units(bottom),
            units(right),
            units(top),
            units(wspace),
            units(hspace))
        if any(np.iterable(space) for space in self._gridspecpars_user):
            raise ValueError(
                'Invalid spacing parameter. '
                'Must be scalar when passed to Figure().')
        self._pad = units(_notNone(pad, rc['subplots.pad']))
        self._axpad = units(_notNone(axpad, rc['subplots.axpad']))
        self._panelpad = units(_notNone(panelpad, rc['subplots.panelpad']))
        self._auto_format = autoformat
        self._auto_tight_layout = _notNone(tight, rc['tight'])
        self._fallback_to_cm = fallback_to_cm
        self._include_panels = includepanels
        self._mainaxes = []
        self._bpanels = []
        self._tpanels = []
        self._lpanels = []
        self._rpanels = []
        self._gridspec = None
        self._barray = None
        self._tarray = None
        self._larray = None
        self._rarray = None
        self._wpanels = None
        self._hpanels = None
        self.ref = ref
        self.suptitle('')  # add _suptitle attribute

    def _context_resizing(self):
        """Ensure backend calls to `~matplotlib.figure.Figure.set_size_inches`
        during pre-processing are not interpreted as *manual* resizing."""
        return _setstate(self, _is_resizing=True)

    def _context_preprocessing(self):
        """Prevent re-running pre-processing steps due to draws triggered
        by figure resizes during pre-processing."""
        return _setstate(self, _is_preprocessing=True)

    @_counter
    def _add_axes_panel(self, ax, side, filled=False, **kwargs):
        """Add axes panels. This powers `~proplot.axes.panel_axes`."""
        # Interpret args
        # NOTE: Axis sharing not implemented for figure panels, 99% of the
        # time this is just used as construct for adding global colorbars and
        # legends, really not worth implementing axis sharing
        if side not in ('left', 'right', 'top', 'bottom'):
            raise ValueError(f'Invalid side {side!r}.')
        if not self._gridspec:  # needed for wpanels, hpanels, etc.
            raise RuntimeError(f'Gridspec is not set.')
        ax = ax._panel_parent or ax  # redirect to main axes
        share, width, space, space_orig = _get_panelargs(
            side, filled=filled, figure=False, **kwargs
        )

        # Get gridspec and subplotspec indices
        ss = ax.get_subplotspec()
        nrows, ncols, row1, row2, col1, col2 = ss.get_rows_columns()
        pgrid = getattr(ax, '_' + side + '_panels')
        offset = (len(pgrid) * bool(pgrid)) + 1
        if side in ('left', 'right'):
            iratio = col1 - offset if side == 'left' else col2 + offset
            idx1 = slice(row1, row2 + 1)
            idx2 = iratio
        else:
            iratio = row1 - offset if side == 'top' else row2 + offset
            idx1 = iratio
            idx2 = slice(col1, col2 + 1)
        gridspec_prev = self._gridspec
        gs = self._insert_row_column(
            side, iratio, width, space, space_orig, figure=False,
        )
        if gs is not gridspec_prev:
            if side == 'top':
                idx1 += 1
            elif side == 'left':
                idx2 += 1

        # Draw and setup panel
        pax = self.add_subplot(
            gs[idx1, idx2],
            main=False, projection='cartesian'
        )
        pgrid.append(pax)
        pax._panel_side = side
        pax._panel_share = share
        pax._panel_parent = ax
        if not filled:  # axis sharing and setup
            ax._share_setup()
            axis = (pax.yaxis if side in ('left', 'right') else pax.xaxis)
            # sets tick and tick label positions intelligently
            getattr(axis, 'tick_' + side)()
            axis.set_label_position(side)
        return pax

    def _add_figure_panel(
        self, side,
        span=None, row=None, col=None, rows=None, cols=None,
        **kwargs
    ):
        """Add figure panels. This powers `Figure.colorbar` and
        `Figure.legend`."""
        # Interpret args and enforce sensible keyword args
        if side not in ('left', 'right', 'bottom', 'top'):
            raise ValueError(f'Invalid side {side!r}.')
        if not self._gridspec:  # needed for wpanels, hpanels, etc.
            raise RuntimeError(f'Gridspec is not set.')
        _, width, space, space_orig = _get_panelargs(
            side, filled=True, figure=True, **kwargs
        )
        if side in ('left', 'right'):
            for key, value in (('col', col), ('cols', cols)):
                if value is not None:
                    raise ValueError(
                        f'Invalid keyword arg {key!r} '
                        f'for figure panel on side {side!r}.')
            span = _notNone(
                span, row, rows, None, names=('span', 'row', 'rows'))
        else:
            for key, value in (('row', row), ('rows', rows)):
                if value is not None:
                    raise ValueError(
                        f'Invalid keyword arg {key!r} '
                        f'for figure panel on side {side!r}.')
            span = _notNone(
                span, col, cols, None, names=('span', 'col', 'cols'))

        # Get props
        array = getattr(self, '_' + side + '_array')
        if side in ('left', 'right'):
            panels, nacross = self._wpanels, self._ncols
        else:
            panels, nacross = self._hpanels, self._nrows
        npanels, nalong = array.shape

        # Check span array
        span = _notNone(span, (1, nalong))
        if not np.iterable(span) or len(span) == 1:
            span = 2 * np.atleast_1d(span).tolist()
        if len(span) != 2:
            raise ValueError(f'Invalid span {span!r}.')
        if span[0] < 1 or span[1] > nalong:
            raise ValueError(
                f'Invalid coordinates in span={span!r}. '
                f'Coordinates must satisfy 1 <= c <= {nalong}.')
        start, stop = span[0] - 1, span[1]  # zero-indexed

        # See if there is room for panel in current figure panels
        # The 'array' is an array of boolean values, where each row corresponds
        # to another figure panel, moving toward the outside, and boolean
        # True indicates the slot has been filled
        iratio = -1 if side in ('left', 'top') else nacross  # default vals
        for i in range(npanels):
            if not any(array[i, start:stop]):
                array[i, start:stop] = True
                if side in ('left', 'top'):  # descending moves us closer to 0
                    # npanels=1, i=0 --> iratio=0
                    # npanels=2, i=0 --> iratio=1
                    # npanels=2, i=1 --> iratio=0
                    iratio = npanels - 1 - i
                else:  # descending array moves us closer to nacross-1
                    # npanels=1, i=0 --> iratio=nacross-1
                    # npanels=2, i=0 --> iratio=nacross-2
                    # npanels=2, i=1 --> iratio=nacross-1
                    iratio = nacross - (npanels - i)
                break
        if iratio in (-1, nacross):  # add to array
            iarray = np.zeros((1, nalong), dtype=bool)
            iarray[0, start:stop] = True
            array = np.concatenate((array, iarray), axis=0)
            setattr(self, '_' + side + '_array', array)

        # Get gridspec and subplotspec indices
        idxs, = np.where(np.array(panels) == '')
        if len(idxs) != nalong:
            raise RuntimeError('Wut?')
        if side in ('left', 'right'):
            idx1 = slice(idxs[start], idxs[stop - 1] + 1)
            idx2 = max(iratio, 0)
        else:
            idx1 = max(iratio, 0)
            idx2 = slice(idxs[start], idxs[stop - 1] + 1)
        gridspec = self._insert_row_column(
            side, iratio, width, space, space_orig, figure=True)

        # Draw and setup panel
        pax = self.add_subplot(gridspec[idx1, idx2],
                               main=False, projection='cartesian')
        getattr(self, '_' + side + '_panels').append(pax)
        pax._panel_side = side
        pax._panel_share = False
        pax._panel_parent = None
        return pax

    def _get_align_coord(self, side, axs):
        """Return the figure coordinate for spanning labels and super titles.
        """
        # Get position in figure relative coordinates
        if side in ('left', 'right'):
            x = 'y'
            iter_panels = ('top', 'bottom')
        else:
            x = 'x'
            iter_panels = ('left', 'right')
        if self._include_panels:
            axs = [iax for ax in axs for iax in ax._iter_panels(iter_panels)]
        ranges = np.array([ax._range_gridspec(x) for ax in axs])
        min_, max_ = ranges[:, 0].min(), ranges[:, 1].max()
        axlo = axs[np.where(ranges[:, 0] == min_)[0][0]]
        axhi = axs[np.where(ranges[:, 1] == max_)[0][0]]
        lobox = axlo.get_subplotspec().get_position(self)
        hibox = axhi.get_subplotspec().get_position(self)
        if x == 'x':
            pos = (lobox.x0 + hibox.x1) / 2
        else:
            # 'lo' is actually on top, highest up in gridspec
            pos = (lobox.y1 + hibox.y0) / 2
        # Return axis suitable for spanning position
        spanax = axs[(np.argmin(ranges[:, 0]) + np.argmax(ranges[:, 1])) // 2]
        spanax = spanax._panel_parent or spanax
        return pos, spanax

    def _get_align_axes(self, side):
        """Return the main axes along the left, right, bottom, or top sides
        of the figure."""
        # Initial stuff
        idx = 0 if side in ('left', 'top') else 1
        if side in ('left', 'right'):
            x, y = 'x', 'y'
        else:
            x, y = 'y', 'x'
        # Get edge index
        axs = self._mainaxes
        if not axs:
            return []
        ranges = np.array([ax._range_gridspec(x) for ax in axs])
        min_, max_ = ranges[:, 0].min(), ranges[:, 1].max()
        edge = min_ if side in ('left', 'top') else max_
        # Return axes on edge sorted by order of appearance
        axs = [
            ax for ax in self._mainaxes if ax._range_gridspec(x)[idx] == edge
        ]
        ord = [ax._range_gridspec(y)[0] for ax in axs]
        return [ax for _, ax in sorted(zip(ord, axs)) if ax.get_visible()]

    def _insert_row_column(
        self, side, idx,
        ratio, space, space_orig, figure=False,
    ):
        """"Overwrite" the main figure gridspec to make room for a panel. The
        `side` is the panel side, the `idx` is the slot the panel will occupy,
        and the remaining args are the panel widths and spacings."""
        # # Constants and stuff
        # # TODO: This is completely broken, must fix
        # # Insert spaces to the left of right panels or to the right of
        # # left panels. And note that since .insert() pushes everything in
        # # that column to the right, actually must insert 1 slot farther to
        # # the right when inserting left panels/spaces
        # idx_space = idx - 1*bool(side in ('bottom', 'right'))
        # idx_offset = 1*bool(side in ('top', 'left'))
        # if side in ('left', 'right')
        #     sidx, ridx, pidx, ncols = -6, -4, -2, 'ncols'
        # else:
        #     sidx, ridx, pidx, ncols = -5, -3, -1, 'nrows'
        #
        # # Load arrays and test if we need to insert
        # gridspecpars, gridspecpars_user = self._fill_gridspecpars()
        # ratios = gridspecpars[ridx]
        # panels = gridspecpars[pidx]
        # # space = gridspecpars[sidx]
        # # space_user = gridspecpars_user[sidx]
        # # Test if panel slot already exists
        # slot_name = 'f' if figure else side[0]
        # slot_new = (idx in (-1, len(panels)) or panels[idx] == slot_name)
        # if slot_new: # already exists!
        #     idx += idx_offset
        #     idx_space += idx_offset
        #     setattr(self, '_' + ncols, getattr(self, '_' + ncols) + 1)
        #     spaces_orig.insert(idx_space, space_orig)
        #     spaces.insert(idx_space, space)
        #     ratios.insert(idx, ratio)
        #     panels.insert(idx, slot_name)
        # else:
        #     if spaces_orig[idx_space] is None:
        #         spaces_orig[idx_space] = units(space_orig)
        #     spaces[idx_space] = _notNone(spaces_orig[idx_space], space)
        #
        # # Update geometry
        # if slot_new:
        #     self._gridspec = GridSpec(nrows, ncols)
        #     self._gridspec.remove_figure(self)
        # gs = self._solver.solve() # also sets self._gridspec
        #
        # # Reassign subplotspecs to all axes and update positions
        # # May seem inefficient but it literally just assigns a hidden,
        # # attribute, and the creation time for subpltospecs is tiny
        # if slot_new:
        #     axs = [
        #         iax for ax in self._iter_axes() for iax
        #         in (ax, *ax.child_axes)]
        #     for ax in axs:
        #         # Get old index
        #         # NOTE: Endpoints are inclusive, not exclusive!
        #         if not hasattr(ax, 'get_subplotspec'):
        #             continue
        #         if side in ('left', 'right'):
        #             inserts = (None, None, idx, idx)
        #         else:
        #             inserts = (idx, idx, None, None)
        #         ss = ax.get_subplotspec()
        #         igs = ss.get_gridspec()
        #         tss = ss.get_topmost_subplotspec()
        #         # Apply new subplotspec!
        #         nrows, ncols, *coords = tss.get_rows_columns()
        #         for i in range(4):
        #             if inserts[i] is not None and coords[i] >= inserts[i]:
        #                 coords[i] += 1
        #         row1, row2, col1, col2 = coords
        #         ssnew = gs[row1:row2+1, col1:col2+1]
        #         if tss is ss:
        #             ax.set_subplotspec(ssnew)
        #         elif tss is igs._subplot_spec:
        #             igs._subplot_spec = ssnew
        #         else:
        #             raise RuntimeError(
        #                 f'Found unexpected GridSpecFromSubplotSpec nesting.'
        #             )
        #         # Update parent or child position
        #         ax.update_params()
        #         ax.set_position(ax.figbox)
        # return gs, slot_new

    def _iter_axes(self):
        """Return a list of all axes and panels in the figure belonging to the
        `~proplot.axes.Axes` class, excluding inset and twin axes."""
        axs = []
        for ax in (*self._mainaxes, *self._lpanels, *self._rpanels,
                   *self._bpanels, *self._tpanels):
            if not ax or not ax.get_visible():
                continue
            axs.append(ax)
        for ax in axs:
            for side in ('left', 'right', 'bottom', 'top'):
                for iax in getattr(ax, '_' + side + '_panels'):
                    if not iax or not iax.get_visible():
                        continue
                    axs.append(iax)
        return axs

    def _update_axislabels(self, axis=None, **kwargs):
        """Apply axis labels to the relevant shared axis. If spanning
        labels are toggled, keep the labels synced for all subplots in the
        same row or column. Label positions will be adjusted at draw-time
        with _align_axislabels."""
        x = axis.axis_name
        if x not in 'xy':
            return
        # Update label on this axes
        axis.label.update(kwargs)
        kwargs.pop('color', None)

        # Defer to parent (main) axes if possible, then get the axes
        # shared by that parent
        ax = axis.axes
        ax = ax._panel_parent or ax
        ax = getattr(ax, '_share' + x) or ax

        # Apply to spanning axes and their panels
        axs = [ax]
        if getattr(self, '_span' + x):
            side = axis.get_label_position()
            if side in ('left', 'bottom'):
                axs = ax._get_side_axes(side)
        for ax in axs:
            getattr(ax, x + 'axis').label.update(kwargs)  # apply to main axes
            pax = getattr(ax, '_share' + x)
            if pax is not None:  # apply to panel?
                getattr(pax, x + 'axis').label.update(kwargs)

    def _get_renderer(self):
        """Get a renderer at all costs, even if it means generating a brand
        new one! Used for updating the figure bounding box when it is accessed
        and calculating centered-row legend bounding boxes. This is copied
        from tight_layout.py in matplotlib."""
        if self._cachedRenderer:
            renderer = self._cachedRenderer
        else:
            canvas = self.canvas
            if canvas and hasattr(canvas, 'get_renderer'):
                renderer = canvas.get_renderer()
            else:
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                canvas = FigureCanvasAgg(self)
                renderer = canvas.get_renderer()
        return renderer

    def _update_figtitle(self, title, **kwargs):
        """Assign the figure "super title" and update settings."""
        if title is not None and self._suptitle.get_text() != title:
            self._suptitle.set_text(title)
        if kwargs:
            self._suptitle.update(kwargs)

    def _update_labels(self, ax, side, labels, **kwargs):
        """Assign side labels and updates label settings. The labels are
        aligned down the line by geometry_configurator."""
        if side not in ('left', 'right', 'bottom', 'top'):
            raise ValueError(f'Invalid label side {side!r}.')

        # Get main axes on the edge
        axs = self._get_align_axes(side)
        if not axs:
            return  # occurs if called while adding axes

        # Update label text for axes on the edge
        if labels is None or isinstance(labels, str):  # common during testing
            labels = [labels] * len(axs)
        if len(labels) != len(axs):
            raise ValueError(
                f'Got {len(labels)} {side}labels, but there are '
                f'{len(axs)} axes along that side.'
            )
        for ax, label in zip(axs, labels):
            obj = getattr(ax, '_' + side + '_label')
            if label is not None and obj.get_text() != label:
                obj.set_text(label)
            if kwargs:
                obj.update(kwargs)

    def add_gridspec(self, *args, **kwargs):
        """This docstring is replaced below."""
        return self.set_gridspec(*args, **kwargs)

    def add_subplot(
        self, *args,
        proj=None, projection=None, basemap=False,
        proj_kw=None, projection_kw=None, main=True, number=None,
        sharex=None, sharey=None,
        **kwargs
    ):
        """
        Add a subplot to the figure.

        Parameters
        ----------
        *args
            There are three options here. See the matplotlib
            `~matplotlib.figure.add_subplot` documentation for details.

            * A `SubplotSpec` instance. Must be a child of the "main"
              gridspec, and must be a ProPlot `SubplotSpec` instead of a native
              matplotlib `~matplotlib.gridspec.SubplotSpec`.
            * A 3-digit integer, e.g. ``121``. Geometry must be equivalent to
              or divide the "main" gridspec geometry.
            * A tuple indicating (nrows, ncols, index). Geometry must be
              equivalent to or divide the "main" gridspec geometry.
        proj, projection : str, `~mpl_toolkits.basemap.Basemap`, or `~cartopy.crs.CRS`, optional
            A registered matplotlib projection name, a basemap or cartopy
            map projection name, a `~mpl_toolkits.basemap.Basemap` instance, or
            a `~cartpoy.crs.CRS` instance. Passed to `~proplot.projs.Proj`. See
            `~proplot.projs.Proj` for a table of map projection names.
        proj_kw, projection_kw : dict-like, optional
            Dictionary of keyword args for the projection class. Passed to
            `~proplot.projs.Proj`.
        main : bool, optional
            Used internally. Indicates whether this is a "main axes" rather
            than a twin, panel, or inset axes. Default is ``True``.
        number : int, optional
            The subplot number, used for a-b-c labeling. See `~Axes.format`
            for details. Note the first axes is ``1``, not ``0``. Ignored if
            `main` is ``False``.

        Other parameters
        ----------------
        **kwargs
            Passed to `~matplotlib.figure.Figure.add_subplot`. This can also
            include axes properties.
        sharex, sharey
            Ignored. ProPlot toggles axes sharing for the entire figure and
            calculates which axes should be shared based on their gridspec
            positions. See `Figure` for details.
        """  # noqa
        # Copied from matplotlib add_subplot
        if not len(args):
            args = (1, 1, 1)
        if len(args) == 1 and isinstance(args[0], Integral):
            if not 100 <= args[0] <= 999:
                raise ValueError(
                    'Integer subplot specification must be a '
                    'three-digit number, not {args[0]!r}.')
            args = tuple(map(int, str(args[0])))
        if sharex is not None:
            _warn_proplot(
                f'Ignoring sharex={sharex!r}. To toggle axes sharing, '
                'just pass sharex=num to figure() or subplots().')
        if sharey is not None:
            _warn_proplot(
                f'Ignoring sharey={sharey!r}. To toggle axes sharing, '
                'just pass sharey=num to figure() or subplots().')

        # Copied from SubplotBase __init__
        # Interpret positional args
        gs = self._gridspec
        ss = None
        if len(args) == 1:
            if isinstance(args[0], SubplotSpec):
                ss = args[0]
            elif isinstance(args[0], mgridspec.SubplotSpec):
                raise ValueError(
                    f'Invalid subplotspec {args[0]!r}. '
                    'Figure.add_subplot() only accepts SubplotSpecs generated '
                    'by the ProPlot GridSpec class.')
            else:
                try:
                    s = str(int(args[0]))
                    nrows, ncols, num = map(int, s)
                except ValueError:
                    raise ValueError(
                        f'Single argument to subplot must be a 3-digit '
                        'integer, not {args[0]!r}.')
        elif len(args) == 3:
            nrows, ncols, num = args
        else:
            raise ValueError(f'Illegal argument(s) to add_subplot: {args!r}')

        # Initialize gridspec and subplotspec
        # Also enforce constant geometry
        if ss is None:
            nrows, ncols = int(nrows), int(ncols)
            if isinstance(num, tuple) and len(num) == 2:
                num = [int(n) for n in num]
            else:
                if num < 1 or num > nrows * ncols:
                    raise ValueError(
                        f'num must be 1 <= num <= {nrows*ncols}, not {num}')
            if not isinstance(num, tuple):
                num = (num, num)
            if gs is None:
                self._solver.solve(nrows=nrows, ncols=ncols)
                gs = self._gridspec
            elif (nrows, ncols) != gs.get_geometry():
                raise ValueError(
                    f'Input arguments {args!r} conflict with existing '
                    'gridspec geometry of {nrows} rows, {ncols} columns.'
                )
            ss = gs[(num[0] - 1):num[1]]
        else:
            if gs is None:
                nrows, ncols, *_ = ss.get_geometry()
                gs = ss.get_gridspec()
                self._solver.solve(nrows=nrows, ncols=ncols)
            elif ss.get_gridspec() is not gs:  # covers geometry discrepancies
                raise ValueError(
                    f'Invalid subplotspec {args[0]!r}. '
                    'Figure.add_subplot() only accepts SubplotSpec objects '
                    'whose parent is the main gridspec.'
                )
        gs.add_figure(self)

        # Impose projection
        # TODO: Have Proj return all unused keyword args, with a
        # map_projection = obj entry, and maybe hide the Proj constructor as
        # an argument processing utility?
        proj = _notNone(
            proj, projection, 'cartesian',
            names=('proj', 'projection')
        )
        proj_kw = _notNone(
            proj_kw, projection_kw, {},
            names=('proj_kw', 'projection_kw')
        )
        if proj not in ('cartesian', 'polar'):
            map_projection = projs.Proj(proj, basemap=basemap, **proj_kw)
            if 'map_projection' in kwargs:
                _warn_proplot(
                    f'Ignoring input "map_projection" '
                    f'{kwargs["map_projection"]!r}.'
                )
            kwargs['map_projection'] = map_projection
            proj = 'basemap' if basemap else 'cartopy'

        # Return subplot
        ax = super().add_subplot(ss, projection=proj, number=number, **kwargs)
        if main:
            ax.number = _notNone(number, len(self._mainaxes) + 1)
            self._mainaxes.append(ax)
        return ax

    def colorbar(
        self, *args,
        loc='r', width=None, space=None,
        row=None, col=None, rows=None, cols=None, span=None,
        **kwargs
    ):
        """
        Draw a colorbar along the left, right, bottom, or top side
        of the figure, centered between the leftmost and rightmost (or
        topmost and bottommost) main axes.

        Parameters
        ----------
        loc : str, optional
            The colorbar location. Valid location keys are as follows.

            ===========  =====================
            Location     Valid keys
            ===========  =====================
            left edge    ``'l'``, ``'left'``
            right edge   ``'r'``, ``'right'``
            bottom edge  ``'b'``, ``'bottom'``
            top edge     ``'t'``, ``'top'``
            ===========  =====================

        row, rows : optional
            Aliases for `span` for panels on the left or right side.
        col, cols : optional
            Aliases for `span` for panels on the top or bottom side.
        span : int or (int, int), optional
            Describes how the colorbar spans rows and columns of subplots.
            For example, ``fig.colorbar(loc='b', col=1)`` draws a colorbar
            beneath the leftmost column of subplots, and
            ``fig.colorbar(loc='b', cols=(1,2))`` draws a colorbar beneath the
            left two columns of subplots. By default, the colorbar will span
            all rows and columns.
        space : float or str, optional
            The space between the main subplot grid and the colorbar, or the
            space between successively stacked colorbars. Units are interpreted
            by `~proplot.utils.units`. By default, this is determined by
            the "tight layout" algorithm, or is :rc:`subplots.panelpad`
            if "tight layout" is off.
        width : float or str, optional
            The colorbar width. Units are interpreted by
            `~proplot.utils.units`. Default is :rc:`colorbar.width`.
        *args, **kwargs
            Passed to `~proplot.axes.Axes.colorbar`.
        """
        ax = kwargs.pop('ax', None)
        cax = kwargs.pop('cax', None)
        # Fill this axes
        if cax is not None:
            return super().colorbar(*args, cax=cax, **kwargs)
        # Generate axes panel
        elif ax is not None:
            return ax.colorbar(*args, space=space, width=width, **kwargs)
        # Generate figure panel
        loc = self._axes_main[0]._loc_translate(loc, 'panel')
        ax = self._add_figure_panel(
            loc, space=space, width=width, span=span,
            row=row, col=col, rows=rows, cols=cols
        )
        return ax.colorbar(*args, loc='_fill', **kwargs)

    def get_alignx(self):
        """Return the *x* axis label alignment mode."""
        return self._alignx

    def get_aligny(self):
        """Return the *y* axis label alignment mode."""
        return self._aligny

    def get_sharex(self):
        """Return the *x* axis sharing level."""
        return self._sharex

    def get_sharey(self):
        """Return the *y* axis sharing level."""
        return self._sharey

    def get_spanx(self):
        """Return the *x* axis label spanning mode."""
        return self._spanx

    def get_spany(self):
        """Return the *y* axis label spanning mode."""
        return self._spany

    def draw(self, renderer):
        # Certain backends *still* have issues with the tight layout
        # algorithm e.g. due to opening windows in *tabs*. Have not found way
        # to intervene in the FigureCanvas. For this reason we *also* apply
        # the algorithm inside Figure.draw in the same way that matplotlib
        # applies its tight layout algorithm. So far we just do this for Qt*
        # and MacOSX; corrections are generally *small* but notable!
        if not self.get_visible():
            return
        if self._auto_tight and (
            rc['backend'] == 'MacOSX' or rc['backend'][:2] == 'Qt'
        ):
            self._adjust_tight_layout(renderer, resize=False)
            self._align_axislabels(True)  # if spaces changed need to realign
            self._align_labels(renderer)
        return super().draw(renderer)

    def legend(
        self, *args,
        loc='r', width=None, space=None,
        row=None, col=None, rows=None, cols=None, span=None,
        **kwargs
    ):
        """
        Draw a legend along the left, right, bottom, or top side of the
        figure, centered between the leftmost and rightmost (or
        topmost and bottommost) main axes.

        Parameters
        ----------
        loc : str, optional
            The legend location. Valid location keys are as follows.

            ===========  =====================
            Location     Valid keys
            ===========  =====================
            left edge    ``'l'``, ``'left'``
            right edge   ``'r'``, ``'right'``
            bottom edge  ``'b'``, ``'bottom'``
            top edge     ``'t'``, ``'top'``
            ===========  =====================

        row, rows : optional
            Aliases for `span` for panels on the left or right side.
        col, cols : optional
            Aliases for `span` for panels on the top or bottom side.
        span : int or (int, int), optional
            Describes how the legend spans rows and columns of subplots.
            For example, ``fig.legend(loc='b', col=1)`` draws a legend
            beneath the leftmost column of subplots, and
            ``fig.legend(loc='b', cols=(1,2))`` draws a legend beneath the
            left two columns of subplots. By default, the legend will span
            all rows and columns.
        space : float or str, optional
            The space between the main subplot grid and the legend, or the
            space between successively stacked colorbars. Units are interpreted
            by `~proplot.utils.units`. By default, this is adjusted
            automatically in the "tight layout" calculation, or is
            :rc:`subplots.panelpad` if "tight layout" is turned off.
        *args, **kwargs
            Passed to `~proplot.axes.Axes.legend`.
        """
        ax = kwargs.pop('ax', None)
        # Generate axes panel
        if ax is not None:
            return ax.legend(*args, space=space, width=width, **kwargs)
        # Generate figure panel
        loc = self._axes_main[0]._loc_translate(loc, 'panel')
        ax = self._add_figure_panel(
            loc, space=space, width=width, span=span,
            row=row, col=col, rows=rows, cols=cols
        )
        return ax.legend(*args, loc='_fill', **kwargs)

    def save(self, filename, **kwargs):
        # Alias for `~Figure.savefig` because ``fig.savefig`` is redundant.
        return self.savefig(filename, **kwargs)

    def savefig(self, filename, **kwargs):
        # Automatically expand user the user name. Undocumented because we
        # do not want to overwrite the matplotlib docstring.
        super().savefig(os.path.expanduser(filename), **kwargs)

    def set_canvas(self, canvas):
        # Set the canvas and add monkey patches to the instance-level
        # `~matplotlib.backend_bases.FigureCanvasBase.draw_idle` and
        # `~matplotlib.backend_bases.FigureCanvasBase.print_figure`
        # methods. The latter is called by save() and by the inline backend.
        # See `_canvas_preprocess` for details."""
        # NOTE: Cannot use draw_idle() because it causes complications for qt5
        # backend (wrong figure size). Even though usage is less consistent we
        # *must* use draw() and _draw() instead.
        if hasattr(canvas, '_draw'):
            canvas._draw = _canvas_preprocess(canvas, '_draw')
        else:
            canvas.draw = _canvas_preprocess(canvas, 'draw')
        canvas.print_figure = _canvas_preprocess(canvas, 'print_figure')
        super().set_canvas(canvas)

    def set_size_inches(self, w, h=None, forward=True, auto=False):
        # Set the figure size and, if this is being called manually or from
        # an interactive backend, override the geometry tracker so users can
        # use interactive backends. If figure size is unchaged we *do not*
        # update the geometry tracker (figure backends often do this when
        # the figure is being initialized). See #76. Undocumented because this
        # is only relevant internally.
        # NOTE: Bitmap renderers calculate the figure size in inches from
        # int(Figure.bbox.[width|height]) which rounds to whole pixels. When
        # renderer calls set_size_inches, size may be effectively the same, but
        # slightly changed due to roundoff error! Therefore, always compare to
        # *both* get_size_inches() and the truncated bbox dimensions times dpi.
        if h is None:
            width, height = w
        else:
            width, height = w, h
        if not all(np.isfinite(_) for _ in (width, height)):
            raise ValueError(
                'Figure size must be finite, not ({width}, {height}).'
            )
        width_true, height_true = self.get_size_inches()
        width_trunc = int(self.bbox.width) / self.dpi
        height_trunc = int(self.bbox.height) / self.dpi
        if auto:
            with self._context_resizing():
                super().set_size_inches(width, height, forward=forward)
        else:
            if (  # internal resizing not associated with any draws
                (
                    width not in (width_true, width_trunc)
                    or height not in (height_true, height_trunc)
                )
                and not self._is_resizing
                and not self.canvas._is_idle_drawing  # standard
                and not getattr(self.canvas, '_draw_pending', None)  # pyqt5
            ):
                self._subplots_kw.update(width=width, height=height)
            super().set_size_inches(width, height, forward=forward)

    def set_alignx(self, value):
        """Set the *x* axis label alignment mode."""
        self.stale = True
        self._alignx = bool(value)

    def set_aligny(self, value):
        """Set the *y* axis label alignment mode."""
        self.stale = True
        self._aligny = bool(value)

    def set_gridspec(self, *args, **kwargs):
        """This docstring is replaced below."""
        # Create and apply the gridspec
        if self._gridspec is not None:
            raise RuntimeError(
                'The gridspec has already been declared and multiple '
                'GridSpecs are not allowed. Call '
                'Figure.get_gridspec() to retrieve it.'
            )
        if len(args) == 1 and isinstance(args[0], GridSpec):
            gs = args[0]
        elif len(args) == 1 and isinstance(args[0], mgridspec.GridSpec):
            raise ValueError(
                'The gridspec must be a ProPlot GridSpec. Matplotlib '
                'gridspecs are not allowed.'
            )
        else:
            gs = GridSpec(*args, **kwargs)
        gs.add_figure(self)
        ncols, nrows = gs.get_geometry()
        self._gridspec = gs
        self._solver._init()
        self.stale = True
        return gs

    def set_sharex(self, value):
        """Set the *x* axis sharing level."""
        value = int(value)
        if value not in range(4):
            raise ValueError(
                'Invalid sharing level sharex={value!r}. '
                'Axis sharing level can be 0 (share nothing), '
                '1 (hide axis labels), '
                '2 (share limits and hide axis labels), or '
                '3 (share limits and hide axis and tick labels).'
            )
        self.stale = True
        self._sharex = value

    def set_sharey(self, value):
        """Set the *y* axis sharing level."""
        value = int(value)
        if value not in range(4):
            raise ValueError(
                'Invalid sharing level sharey={value!r}. '
                'Axis sharing level can be 0 (share nothing), '
                '1 (hide axis labels), '
                '2 (share limits and hide axis labels), or '
                '3 (share limits and hide axis and tick labels).'
            )
        self.stale = True
        self._sharey = value

    def set_spanx(self, value):
        """Set the *x* axis label spanning mode."""
        self.stale = True
        self._spanx = bool(value)

    def set_spany(self, value):
        """Set the *y* axis label spanning mode."""
        self.stale = True
        self._spany = bool(value)

    @property
    def gridspec(self):
        """The single `GridSpec` instance used for all subplots
        in the figure."""
        return self._gridspec

    @property
    def ref(self):
        """The reference axes number. The `axwidth`, `axheight`, and `aspect`
        `subplots` and `figure` arguments are applied to this axes, and aspect
        ratio is conserved for this axes in tight layout adjustment."""
        return self._ref

    @ref.setter
    def ref(self, ref):
        if not isinstance(ref, Integral) or ref < 1:
            raise ValueError(
                f'Invalid axes number {ref!r}. Must be integer >=1.')
        self.stale = True
        self._ref = ref

    # Add documentation
    add_gridspec.__doc__ = _gridspec_doc
    set_gridspec.__doc__ = _gridspec_doc


def _axes_dict(naxs, value, kw=False, default=None):
    """Return a dictionary that looks like ``{1:value1, 2:value2, ...}`` or
    ``{1:{key1:value1, ...}, 2:{key2:value2, ...}, ...}`` for storing
    standardized axes-specific properties or keyword args."""
    # First build up dictionary
    # 1) 'string' or {1:'string1', (2,3):'string2'}
    if not kw:
        if np.iterable(value) and not isinstance(value, (str, dict)):
            value = {num + 1: item for num, item in enumerate(value)}
        elif not isinstance(value, dict):
            value = {range(1, naxs + 1): value}
    # 2) {'prop':value} or {1:{'prop':value1}, (2,3):{'prop':value2}}
    else:
        nested = [isinstance(value, dict) for value in value.values()]
        if not any(nested):  # any([]) == False
            value = {range(1, naxs + 1): value.copy()}
        elif not all(nested):
            raise ValueError(
                'Pass either of dictionary of key value pairs or '
                'a dictionary of dictionaries of key value pairs.'
            )
    # Then *unfurl* keys that contain multiple axes numbers, i.e. are meant
    # to indicate properties for multiple axes at once
    kwargs = {}
    for nums, item in value.items():
        nums = np.atleast_1d(nums)
        for num in nums.flat:
            if not kw:
                kwargs[num] = item
            else:
                kwargs[num] = item.copy()
    # Fill with default values
    for num in range(1, naxs + 1):
        if num not in kwargs:
            if kw:
                kwargs[num] = {}
            else:
                kwargs[num] = default
    # Verify numbers
    if {*range(1, naxs + 1)} != {*kwargs.keys()}:
        raise ValueError(
            f'Have {naxs} axes, but {value!r} has properties for axes '
            + ', '.join(map(repr, sorted(kwargs))) + '.'
        )
    return kwargs


def _journal_figsize(journal):
    """Return the dimensions associated with this journal string."""
    # Get dimensions for figure from common journals.
    value = JOURNAL_SPECS.get(journal, None)
    if value is None:
        raise ValueError(
            f'Unknown journal figure size specifier {journal!r}. Options are: '
            ', '.join(map(repr, JOURNAL_SPECS.keys())) + '.')
    # Return width, and optionally also the height
    width, height = None, None
    try:
        width, height = value
    except (TypeError, ValueError):
        width = value
    return width, height


# TODO: Figure out how to save subplots keyword args!
@docstring.dedent_interpd
def figure(**kwargs):
    """
    Analogous to `matplotlib.pyplot.figure`, create an empty figure meant
    to be filled with axes using `Figure.add_subplot`.

    Parameters
    ----------
    %(figure_doc)s
    **kwargs
        Passed to `Figure`.
    """
    return plt.figure(FigureClass=Figure, **kwargs)


def subplots(
    array=None, ncols=1, nrows=1, ref=1, order='C',
    left=None, right=None, bottom=None, top=None, wspace=None, hspace=None,
    hratios=None, wratios=None, width_ratios=None, height_ratios=None,
    proj=None, projection=None, proj_kw=None, projection_kw=None,
    basemap=False, **kwargs
):
    """
    Create a figure with a single subplot or arbitrary grids of subplots,
    analogous to `matplotlib.pyplot.subplots`. The subplots can be drawn with
    arbitrary projections.

    Parameters
    ----------
    array : 2d array-like of int, optional
        Array specifying complex grid of subplots. Think of
        this array as a "picture" of your figure. For example, the array
        ``[[1, 1], [2, 3]]`` creates one long subplot in the top row, two
        smaller subplots in the bottom row. Integers must range from 1 to the
        number of plots.

        ``0`` indicates an empty space. For example, ``[[1, 1, 1], [2, 0, 3]]``
        creates one long subplot in the top row with two subplots in the bottom
        row separated by a space.
    ncols, nrows : int, optional
        Number of columns, rows. Ignored if `array` is not ``None``.
        Use these arguments for simpler subplot grids.
    order : {'C', 'F'}, optional
        Whether subplots are numbered in column-major (``'C'``) or row-major
        (``'F'``) order. Analogous to `numpy.array` ordering. This controls
        the order axes appear in the `axs` list, and the order of subplot
        a-b-c labeling (see `~proplot.axes.Axes.format`).
    proj, projection : str or dict-like, optional
        The map projection name(s), passed to `~proplot.projs.Proj`. The
        argument is interpreted as follows.

        * If string, this projection is used for all subplots. See
          `~proplot.projs.Proj` for a table of map projection names.
        * If list of strings, these projections are used for each
          subplot in the order specified by `array` or `order`.
        * If dict-like, the keys are integers or tuples of integers
          corresponding to subplot numbers, and the values are strings
          indicating the projection. If a key is not provided, the subplot
          will be `~proplot.axes.CartesianAxes`.

        For example, with ``ncols=4`` and ``proj={2:'merc', (3,4):'cyl'}``,
        the first subplot is a normal axes, the second is a Mercator
        projection, and the third and fourth are cylindrical projections.
    proj_kw, projection_kw : dict-like, optional
        Dictionary of keyword args for the projection class. Passed to
        `~proplot.projs.Proj`. Can be set for specific subplots just like
        `proj`. For example, with ``ncols=2`` and
        ``proj_kw={1:dict(lon_0=0), 2:dict(lon_0=180)}``, the left subplot is
        centered on the prime meridian and the right subplot is centered on
        the international dateline.
    basemap : bool or dict-like, optional
        Whether to use basemap or cartopy for map projections. Default is
        ``False``. Can be set for specific subplots just like `proj`.
        For example, with ``basemap={1:False, 2:True}``, the left subplot is
        a cartopy projection and the right subplot is a basemap projection.

    Other parameters
    ----------------
    %(figure_doc)s
    hratios, wratios, height_ratios, width_ratios
        Passed to `GridSpec`. These describe the ratios between successive
        rows and columns of the subplot grid.

    Returns
    -------
    f : `Figure`
        The figure instance.
    axs : `subplot_grid`
        A special list of axes instances. See `subplot_grid`.
    """  # noqa
    # Build array
    if order not in ('C', 'F'):  # better error message
        raise ValueError(
            f'Invalid order {order!r}. Choose from "C" (row-major, default) '
            f'and "F" (column-major).'
        )
    if array is None:
        array = np.arange(1, nrows * ncols + 1)[..., None]
        array = array.reshape((nrows, ncols), order=order)
    # Standardize array
    try:
        array = np.array(array, dtype=int)  # enforce array type
        if array.ndim == 1:
            # interpret as single row or column
            array = array[None, :] if order == 'C' else array[:, None]
        elif array.ndim != 2:
            raise ValueError(
                'array must be 1-2 dimensional, but got {array.ndim} dims'
            )
        array[array == None] = 0  # use zero for placeholder  # noqa
    except (TypeError, ValueError):
        raise ValueError(
            f'Invalid subplot array {array!r}. '
            'Must be 1d or 2d array of integers.'
        )
    # Get other props
    nums = np.unique(array[array != 0])
    naxs = len(nums)
    if {*nums.flat} != {*range(1, naxs + 1)}:
        raise ValueError(
            f'Invalid subplot array {array!r}. Numbers must span integers '
            '1 to naxs (i.e. cannot skip over numbers), with 0 representing '
            'empty spaces.'
        )
    if ref not in nums:
        raise ValueError(
            f'Invalid reference number {ref!r}. For array {array!r}, must be '
            'one of {nums}.'
        )
    nrows, ncols = array.shape
    # Get axes ranges from array
    axids = [np.where(array == i) for i in np.sort(np.unique(array)) if i > 0]
    xrange = np.array([[x.min(), x.max()] for _, x in axids])
    yrange = np.array([[y.min(), y.max()] for y, _ in axids])

    # Get basemap.Basemap or cartopy.crs.Projection instances for map
    proj = _notNone(projection, proj, None, names=('projection', 'proj'))
    proj_kw = _notNone(
        projection_kw, proj_kw, {}, names=('projection_kw', 'proj_kw')
    )
    proj = _axes_dict(naxs, proj, kw=False, default='xy')
    proj_kw = _axes_dict(naxs, proj_kw, kw=True)
    basemap = _axes_dict(naxs, basemap, kw=False, default=False)

    # Standardized user input ratios
    wratios = np.atleast_1d(_notNone(
        width_ratios, wratios, 1,
        names=('width_ratios', 'wratios')
    ))
    hratios = np.atleast_1d(_notNone(
        height_ratios, hratios, 1,
        names=('height_ratios', 'hratios')
    ))
    if len(wratios) == 1:
        wratios = np.repeat(wratios, (ncols,))
    if len(hratios) == 1:
        hratios = np.repeat(hratios, (nrows,))
    if len(wratios) != ncols:
        raise ValueError(f'Got {ncols} columns, but {len(wratios)} wratios.')
    if len(hratios) != nrows:
        raise ValueError(f'Got {nrows} rows, but {len(hratios)} hratios.')
    wratios, hratios = wratios.tolist(), hratios.tolist()  # also makes copy

    # Generate figure and gridspec
    # NOTE: This time we initialize the *gridspec* with user input values
    # TODO: Repair solve() so it works!
    fig = plt.figure(FigureClass=Figure, ref=ref, **kwargs)
    gs = fig._solver.solve(
        nrows=nrows, ncols=ncols,
        left=left, right=right, bottom=bottom, top=top,
        wratios=wratios, hratios=hratios
    )

    # Draw main subplots
    axs = naxs * [None]  # list of axes
    for idx in range(naxs):
        # Get figure gridspec ranges
        num = idx + 1
        x0, x1 = xrange[idx, 0], xrange[idx, 1]
        y0, y1 = yrange[idx, 0], yrange[idx, 1]
        # Draw subplot
        ss = gs[y0:y1 + 1, x0:x1 + 1]
        axs[idx] = fig.add_subplot(
            ss, number=num, main=True,
            proj=proj[num], basemap=basemap[num], proj_kw=proj_kw[num]
        )

    # Shared axes setup
    # TODO: Figure out how to defer this to drawtime in #50
    # For some reason just adding _share_setup() to draw() doesn't work
    for ax in axs:
        ax._share_setup()

    # Return figure and axes
    n = (ncols if order == 'C' else nrows)
    return fig, subplot_grid(axs, n=n, order=order)
