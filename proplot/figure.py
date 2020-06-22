#!/usr/bin/env python3
"""
The figure class used for all ProPlot figures.
"""
import os

import matplotlib.figure as mfigure
import matplotlib.transforms as mtransforms
import numpy as np

from . import axes as paxes
from . import gridspec as pgridspec
from .config import rc
from .internals import ic  # noqa: F401
from .internals import _dummy_context, _not_none, _state_context, docstring, warnings
from .utils import units

__all__ = ['Figure']


def _parse_panel_args(
    side, share=None, width=None, space=None,
    filled=False, figure=False
):
    """
    Return default properties for new axes and figure panels.
    """
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
        space = pgridspec._default_space(key, share, pad=pad)

    return share, width, space, space_user


def _canvas_preprocessor(canvas, method):
    """
    Return a pre-processer that can be used to override instance-level
    canvas draw() and print_figure() methods. This applies tight layout
    and aspect ratio-conserving adjustments and aligns labels. Required so that
    the canvas methods instantiate renderers with the correct dimensions.
    """
    # NOTE: Renderer must be (1) initialized with the correct figure size or
    # (2) changed inplace during draw, but vector graphic renderers *cannot*
    # be changed inplace. So options include (1) monkey patch
    # canvas.get_width_height, overriding figure.get_size_inches, and exploit
    # the FigureCanvasAgg.get_renderer() implementation (because FigureCanvasAgg
    # queries the bbox directly rather than using get_width_height() so requires
    # workaround), (2) override bbox and bbox_inches as *properties* (but these
    # are really complicated, dangerous, and result in unnecessary extra draws),
    # or (3) simply override canvas draw methods. Our choice is #3.
    def _preprocess(self, *args, **kwargs):
        fig = self.figure  # update even if not stale! needed after saves
        func = getattr(type(self), method)  # the original method

        # Bail out if we are already pre-processing
        # NOTE: The _is_autoresizing check necessary when inserting new gridspec
        # rows or columns with the qt backend.
        # NOTE: Return value for macosx _draw is the renderer, for qt draw is
        # nothing, and for print_figure is some figure object, but this block
        # has never been invoked when calling print_figure.
        if fig._is_autoresizing or fig._is_preprocessing:
            if method == '_draw':  # macosx backend
                return fig._get_renderer()
            else:
                return

        # Apply formatting
        # NOTE: *Critical* to not add print_figure renderer to the cache when the
        # print method (print_pdf, print_png, etc.) calls Figure.draw(). Otherwise
        # have issues where (1) figure size and/or figure bounds are incorrect after
        # saving figure *then* displaying it in qt or inline notebook backends, and
        # (2) figure fails to update correctly after successively modifying
        # and displaying within inline notebook backend (previously worked around
        # this by forcing additional draw() call in this function before proceeding
        # with print_figure). Solution is to use _state_context with _cachedRenderer.
        fallback = _not_none(fig._fallback_to_cm, rc['mathtext.fallback_to_cm'])
        rc_context = rc.context({'mathtext.fallback_to_cm': fallback})
        fig_context = fig._context_preprocessing(cache=(method != 'print_figure'))
        with rc_context, fig_context:
            fig.auto_layout()
            result = func(self, *args, **kwargs)
        return result

    return _preprocess.__get__(canvas)  # ...I don't get it either


class _hide_artists(object):
    """
    Hide objects temporarily so they are ignored by the tight bounding box
    algorithm.
    """
    # NOTE: This will be removed when labels are implemented with AxesStack!
    def __init__(self, *args):
        self._artists = args

    def __enter__(self):
        for artist in self._artists:
            artist.set_visible(False)

    def __exit__(self, *args):  # noqa: U100
        for artist in self._artists:
            artist.set_visible(True)


class Figure(mfigure.Figure):
    """
    The `~matplotlib.figure.Figure` class returned by
    `~proplot.ui.subplots`. At draw-time, an improved tight layout
    algorithm is employed, and the space around the figure edge, between
    subplots, and between panels is changed to accommodate subplot content.
    Figure dimensions may be automatically scaled to preserve subplot aspect
    ratios.
    """
    # NOTE: If _rename_kwargs argument is an invalid identifier, it is
    # simply used in the warning message.
    @warnings._rename_kwargs('0.6.4', autoformat='plot.rc.autoformat = {}')
    def __init__(
        self, tight=None,
        ref=1, pad=None, axpad=None, panelpad=None, includepanels=False,
        span=None, spanx=None, spany=None,
        align=None, alignx=None, aligny=None,
        share=None, sharex=None, sharey=None,
        gridspec_kw=None, subplots_kw=None, subplots_orig_kw=None,
        fallback_to_cm=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        tight : bool, optional
            Toggles automatic tight layout adjustments. Default is :rc:`subplots.tight`.
            If you manually specified a spacing in the call to `~proplot.ui.subplots`,
            it will be used to override the tight layout spacing. For example, with
            ``left=0.1``, the left margin is set to 0.1 inches wide, while the
            remaining margin widths are calculated automatically.
        ref : int, optional
            The reference subplot number. See `~proplot.ui.subplots` for
            details. Default is ``1``.
        pad : float or str, optional
            Padding around edge of figure. Units are interpreted by
            `~proplot.utils.units`. Default is :rc:`subplots.pad`.
        axpad : float or str, optional
            Padding between subplots in adjacent columns and rows. Units are
            interpreted by `~proplot.utils.units`. Default is
            :rc:`subplots.axpad`.
        panelpad : float or str, optional
            Padding between subplots and axes panels, and between "stacked"
            panels. Units are interpreted by `~proplot.utils.units`. Default is
            :rc:`subplots.panelpad`.
        includepanels : bool, optional
            Whether to include panels when centering *x* axis labels,
            *y* axis labels, and figure "super titles" along the edge of the
            subplot grid. Default is ``False``.
        sharex, sharey, share : {3, 2, 1, 0}, optional
            The "axis sharing level" for the *x* axis, *y* axis, or both axes.
            Default is :rc:`subplots.share`. Options are as follows:

            0. No axis sharing. Also sets the default `spanx` and `spany`
               values to ``False``.
            1. Only draw *axis label* on the leftmost column (*y*) or bottommost row
               (*x*) of subplots. Axis tick labels still appear on every subplot.
            2. As in 1, but forces the axis limits to be identical. Axis
               tick labels still appear on every subplot.
            3. As in 2, but only show the *axis tick labels* on the
               leftmost column (*y*) or bottommost row (*x*) of subplots.

        spanx, spany, span : bool or {0, 1}, optional
            Toggles "spanning" axis labels for the *x* axis, *y* axis, or both
            axes.  Default is ``False`` if `sharex`, `sharey`, or `share` are
            ``0``, :rc:`subplots.span` otherwise. When ``True``, a single, centered
            axis label is used for all axes with bottom and left edges in the same
            row or column.  This can considerably redundancy in your figure.

            "Spanning" labels integrate with "shared" axes. For example,
            for a 3-row, 3-column figure, with ``sharey > 1`` and ``spany=1``,
            your figure will have 1 ylabel instead of 9.
        alignx, aligny, align : bool or {0, 1}, optional
            Whether to `align axis labels \
<https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/align_labels_demo.html>`__
            for the *x* axis, *y* axis, or both axes. Only has an effect when `spanx`,
            `spany`, or `span` are ``False``. Default is :rc:`subplots.align`.
        fallback_to_cm : bool, optional
            Whether to replace unavailable glyphs with a glyph from Computer Modern
            or the "¤" dummy character. See `mathtext \
<https://matplotlib.org/3.1.1/tutorials/text/mathtext.html#custom-fonts>`__
            for details.
        gridspec_kw, subplots_kw, subplots_orig_kw
            Keywords used for initializing the main gridspec, for initializing
            the figure, and original spacing keyword args used for initializing
            the figure that override tight layout spacing.

        Other parameters
        ----------------
        **kwargs
            Passed to `matplotlib.figure.Figure`.
        """
        # Initialize first, because need to provide fully initialized figure
        # as argument to gridspec, because matplotlib tight_layout does that
        tight_layout = kwargs.pop('tight_layout', None)
        constrained_layout = kwargs.pop('constrained_layout', None)
        if tight_layout or constrained_layout:
            warnings._warn_proplot(
                f'Ignoring tight_layout={tight_layout} and '
                f'contrained_layout={constrained_layout}. ProPlot uses its '
                'own tight layout algorithm, activated by default or with '
                'plot.subplots(tight=True).'
            )
        self._authorized_add_subplot = False
        self._is_preprocessing = False
        self._is_autoresizing = False
        super().__init__(**kwargs)

        # Axes sharing and spanning settings
        sharex = _not_none(sharex, share, rc['subplots.share'])
        sharey = _not_none(sharey, share, rc['subplots.share'])
        spanx = _not_none(spanx, span, 0 if sharex == 0 else None, rc['subplots.span'])
        spany = _not_none(spany, span, 0 if sharey == 0 else None, rc['subplots.span'])
        if spanx and (alignx or align):
            warnings._warn_proplot('"alignx" has no effect when spanx=True.')
        if spany and (aligny or align):
            warnings._warn_proplot('"aligny" has no effect when spany=True.')
        alignx = _not_none(alignx, align, rc['subplots.align'])
        aligny = _not_none(aligny, align, rc['subplots.align'])
        if int(sharex) not in range(4):
            raise ValueError(
                'Invalid sharing level sharex={value!r}. '
                'Axis sharing level can be 0 (share nothing), '
                '1 (hide axis labels), '
                '2 (share limits and hide axis labels), or '
                '3 (share limits and hide axis and tick labels).'
            )
        if int(sharey) not in range(4):
            raise ValueError(
                'Invalid sharing level sharey={sharey!r}. '
                'Axis sharing level can be 0 (share nothing), '
                '1 (hide axis labels), '
                '2 (share limits and hide axis labels), or '
                '3 (share limits and hide axis and tick labels).'
            )
        self._alignx = bool(alignx)
        self._aligny = bool(aligny)
        self._sharex = int(sharex)
        self._sharey = int(sharey)
        self._spanx = bool(spanx)
        self._spany = bool(spany)

        # Various other attributes
        gridspec_kw = gridspec_kw or {}
        gridspec = pgridspec.GridSpec(self, **gridspec_kw)
        nrows, ncols = gridspec.get_active_geometry()
        self._pad = units(_not_none(pad, rc['subplots.pad']))
        self._axpad = units(_not_none(axpad, rc['subplots.axpad']))
        self._panelpad = units(_not_none(panelpad, rc['subplots.panelpad']))
        self._auto_tight = _not_none(tight, rc['subplots.tight'])
        self._include_panels = includepanels
        self._fallback_to_cm = fallback_to_cm
        self._ref_num = ref
        self._axes_main = []
        self._subplots_orig_kw = subplots_orig_kw
        self._subplots_kw = subplots_kw
        self._bottom_panels = []
        self._top_panels = []
        self._left_panels = []
        self._right_panels = []
        self._bottom_array = np.empty((0, ncols), dtype=bool)
        self._top_array = np.empty((0, ncols), dtype=bool)
        self._left_array = np.empty((0, nrows), dtype=bool)
        self._right_array = np.empty((0, nrows), dtype=bool)
        self._gridspec_main = gridspec
        self.suptitle('')  # add _suptitle attribute

    def _add_axes_panel(self, ax, side, filled=False, **kwargs):
        """
        Hidden method that powers `~proplot.axes.panel_axes`.
        """
        # Interpret args
        # NOTE: Axis sharing not implemented for figure panels, 99% of the
        # time this is just used as construct for adding global colorbars and
        # legends, really not worth implementing axis sharing
        if side not in ('left', 'right', 'bottom', 'top'):
            raise ValueError(f'Invalid side {side!r}.')
        ax = ax._panel_parent or ax  # redirect to main axes
        share, width, space, space_orig = _parse_panel_args(
            side, filled=filled, figure=False, **kwargs
        )

        # Get gridspec and subplotspec indices
        subplotspec = ax.get_subplotspec()
        *_, row1, row2, col1, col2 = subplotspec.get_active_rows_columns()
        pgrid = getattr(ax, '_' + side + '_panels')
        offset = len(pgrid) * bool(pgrid) + 1
        if side in ('left', 'right'):
            iratio = col1 - offset if side == 'left' else col2 + offset
            idx1 = slice(row1, row2 + 1)
            idx2 = iratio
        else:
            iratio = row1 - offset if side == 'top' else row2 + offset
            idx1 = iratio
            idx2 = slice(col1, col2 + 1)
        gridspec_prev = self._gridspec_main
        gridspec = self._insert_row_column(
            side, iratio, width, space, space_orig, figure=False
        )
        if gridspec is not gridspec_prev:
            if side == 'top':
                idx1 += 1
            elif side == 'left':
                idx2 += 1

        # Draw and setup panel
        with self._context_authorize_add_subplot():
            pax = self.add_subplot(gridspec[idx1, idx2], projection='cartesian')  # noqa: E501
        pgrid.append(pax)
        pax._panel_side = side
        pax._panel_share = share
        pax._panel_parent = ax

        # Axis sharing and axis setup only for non-legend or colorbar axes
        if not filled:
            for ax in self._axes_main:
                ax._auto_share_setup()
            axis = pax.yaxis if side in ('left', 'right') else pax.xaxis
            getattr(axis, 'tick_' + side)()  # set tick and label positions
            axis.set_label_position(side)

        return pax

    def _add_figure_panel(
        self, side, span=None, row=None, col=None, rows=None, cols=None,
        **kwargs
    ):
        """
        Add a figure panel. Also modifies the panel attribute stored
        on the figure to include these panels.
        """
        # Interpret args and enforce sensible keyword args
        if side not in ('left', 'right', 'bottom', 'top'):
            raise ValueError(f'Invalid side {side!r}.')
        _, width, space, space_orig = _parse_panel_args(
            side, filled=True, figure=True, **kwargs
        )
        if side in ('left', 'right'):
            for key, value in (('col', col), ('cols', cols)):
                if value is not None:
                    raise ValueError(
                        f'Invalid keyword arg {key!r} for figure panel '
                        f'on side {side!r}.'
                    )
            span = _not_none(span=span, row=row, rows=rows)
        else:
            for key, value in (('row', row), ('rows', rows)):
                if value is not None:
                    raise ValueError(
                        f'Invalid keyword arg {key!r} for figure panel '
                        f'on side {side!r}.'
                    )
            span = _not_none(span=span, col=col, cols=cols)

        # Get props
        subplots_kw = self._subplots_kw
        if side in ('left', 'right'):
            panels, nacross = subplots_kw['hpanels'], subplots_kw['ncols']
        else:
            panels, nacross = subplots_kw['wpanels'], subplots_kw['nrows']
        array = getattr(self, '_' + side + '_array')
        npanels, nalong = array.shape

        # Check span array
        span = _not_none(span, (1, nalong))
        if not np.iterable(span) or len(span) == 1:
            span = 2 * np.atleast_1d(span).tolist()
        if len(span) != 2:
            raise ValueError(f'Invalid span {span!r}.')
        if span[0] < 1 or span[1] > nalong:
            raise ValueError(
                f'Invalid coordinates in span={span!r}. Coordinates '
                f'must satisfy 1 <= c <= {nalong}.'
            )
        start, stop = span[0] - 1, span[1]  # zero-indexed

        # See if there is room for panel in current figure panels
        # The 'array' is an array of boolean values, where each row corresponds
        # to another figure panel, moving toward the outside, and boolean
        # True indicates the slot has been filled
        iratio = -1 if side in ('left', 'top') else nacross  # default values
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
            raise RuntimeError
        if side in ('left', 'right'):
            idx1 = slice(idxs[start], idxs[stop - 1] + 1)
            idx2 = max(iratio, 0)
        else:
            idx1 = max(iratio, 0)
            idx2 = slice(idxs[start], idxs[stop - 1] + 1)
        gridspec = self._insert_row_column(
            side, iratio, width, space, space_orig, figure=True
        )

        # Draw and setup panel
        with self._context_authorize_add_subplot():
            pax = self.add_subplot(gridspec[idx1, idx2], projection='cartesian')  # noqa: E501
        pgrid = getattr(self, '_' + side + '_panels')
        pgrid.append(pax)
        pax._panel_side = side
        pax._panel_share = False
        pax._panel_parent = None
        return pax

    def _align_axis_labels(self, b=True):
        """
        Align spanning *x* and *y* axis labels in the perpendicular
        direction and, if `b` is ``True``, the parallel direction.
        """
        # TODO: Ensure this is robust to complex panels and shared axes
        # NOTE: Need to turn off aligned labels before
        # _update_geometry_from_spacing
        # call, so cannot put this inside Axes draw
        xaxs_updated = set()
        for ax in self._axes_main:
            if not isinstance(ax, paxes.CartesianAxes):
                continue
            for x, axis in zip('xy', (ax.xaxis, ax.yaxis)):
                side = axis.get_label_position()
                span = getattr(self, '_span' + x)
                align = getattr(self, '_align' + x)
                if side not in ('bottom', 'left') or axis in xaxs_updated:
                    continue

                # Get panels for axes on each side (2 levels deep is maximum)
                axs = ax._get_side_axes(side, panels=False)
                axs = [getattr(ax, '_share' + x) or ax for ax in axs]
                axs = [getattr(ax, '_share' + x) or ax for ax in axs]

                # Align axis label offsets
                xaxs = [getattr(ax, x + 'axis') for ax in axs]
                xaxs_updated.update(xaxs)
                if span or align:
                    group = getattr(self, '_align_' + x + 'label_grp', None)
                    if group is not None:
                        for ax in axs[1:]:
                            group.join(axs[0], ax)  # add to grouper
                    elif align:
                        warnings._warn_proplot(
                            'Aligning *x* and *y* axis labels requires '
                            'matplotlib >=3.1.0'
                        )
                if not span:
                    continue

                # Get spanning label position
                c, ax_span = self._get_align_coord(side, axs)
                ax_span = getattr(ax_span, '_share' + x) or ax_span
                ax_span = getattr(ax_span, '_share' + x) or ax_span
                axis_span = getattr(ax_span, x + 'axis')
                label_span = axis_span.label
                if not hasattr(label_span, '_orig_transform'):
                    label_span._orig_transform = label_span.get_transform()
                    label_span._orig_position = label_span.get_position()
                if not b:  # toggle off, done before tight layout
                    label_span.set_transform(label_span._orig_transform)
                    label_span.set_position(label_span._orig_position)
                    for axis in xaxs:
                        axis.label.set_visible(True)
                else:  # toggle on, done after tight layout
                    if x == 'x':
                        position = (c, 1)
                        transform = mtransforms.blended_transform_factory(
                            self.transFigure, mtransforms.IdentityTransform()
                        )
                    else:
                        position = (1, c)
                        transform = mtransforms.blended_transform_factory(
                            mtransforms.IdentityTransform(), self.transFigure
                        )
                    for axis in xaxs:
                        axis.label.set_visible((axis is axis_span))
                    label_span.update({'position': position, 'transform': transform})

    def _align_subplot_super_labels(self, renderer):
        """
        Adjust the position of row and column labels, and align figure super
        title accounting for figure margins and axes and figure panels.
        """
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
                panels = ('bottom', 'top')
            else:
                x = 'y'
                panels = ('left', 'right')
            axs = self._get_align_axes(side)
            axs = [ax._reassign_subplot_label(side) for ax in axs]
            labels = [getattr(ax, '_' + side + '_label') for ax in axs]
            coords = [None] * len(axs)
            if side == 'top' and suptitle_on:
                supaxs = axs

            # Adjust the labels
            with _hide_artists(*labels):
                for i, (ax, label) in enumerate(zip(axs, labels)):
                    label_on = label.get_text().strip()
                    if not label_on:
                        continue

                    # Get coord from tight bounding box
                    # Include twin axes and panels along the same side
                    icoords = []
                    for iax in ax._iter_axes(panels=panels, children=False):
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
                    # https://matplotlib.org/api/text_api.html#matplotlib.text.Text.set_linespacing
                    fontsize = label.get_fontsize()
                    if side in ('left', 'right'):
                        scale1, scale2 = 0.6, width
                    else:
                        scale1, scale2 = 0.3, height
                    if side in ('left', 'bottom'):
                        coords[i] = min(icoords) - (
                            scale1 * fontsize / 72
                        ) / scale2
                    else:
                        coords[i] = max(icoords) + (
                            scale1 * fontsize / 72
                        ) / scale2

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
            x, _ = self._get_align_coord('top', supaxs)
            y = max(ys) + (0.3 * suptitle.get_fontsize() / 72) / height
            kw = {
                'x': x, 'y': y,
                'ha': 'center', 'va': 'bottom',
                'transform': self.transFigure
            }
            suptitle.update(kw)

    def _context_authorize_add_subplot(self):
        """
        Prevent warning message when adding subplots one-by-one. Used
        internally.
        """
        return _state_context(self, _authorized_add_subplot=True)

    def _context_autoresizing(self):
        """
        Ensure backend calls to `~matplotlib.figure.Figure.set_size_inches`
        during pre-processing are not interpreted as *manual* resizing.
        """
        return _state_context(self, _is_autoresizing=True)

    def _context_preprocessing(self, cache=True):
        """
        Prevent re-running pre-processing steps due to draws triggered
        by figure resizes during pre-processing. `cache` controls whether the
        renderer passed to draw should be cached.
        """
        kwargs = {}
        if not cache:
            kwargs['_cachedRenderer'] = None  # __exit__ will restore previous value
        return _state_context(self, _is_preprocessing=True, **kwargs)

    def _draw_auto_legends_colorbars(self):
        """
        Draw legends and colorbars requested via plotting commands. Drawing is
        deferred so that successive calls to the plotting commands can successively
        add entries to legends and colorbars in particular locations.
        """
        for ax in self._iter_axes(hidden=False, children=True):
            if isinstance(ax, paxes.Axes):
                ax._draw_auto_legends_colorbars()  # may insert panels

    def _get_align_coord(self, side, axs):
        """
        Return the figure coordinate for spanning labels or super titles.
        The `x` can be ``'x'`` or ``'y'``.
        """
        # Get position in figure relative coordinates
        if side in ('left', 'right'):
            x = 'y'
            panels = ('top', 'bottom')
        else:
            x = 'x'
            panels = ('left', 'right')
        if self._include_panels:
            axs = [
                iax for ax in axs
                for iax in ax._iter_axes(panels=panels, children=False)
            ]

        # Get coordinates
        ranges = np.array([ax._range_gridspec(x) for ax in axs])
        min_, max_ = ranges[:, 0].min(), ranges[:, 1].max()
        ax_lo = axs[np.where(ranges[:, 0] == min_)[0][0]]
        ax_hi = axs[np.where(ranges[:, 1] == max_)[0][0]]
        box_lo = ax_lo.get_subplotspec().get_position(self)
        box_hi = ax_hi.get_subplotspec().get_position(self)
        if x == 'x':
            pos = 0.5 * (box_lo.x0 + box_hi.x1)
        else:
            pos = 0.5 * (box_lo.y1 + box_hi.y0)  # 'lo' is actually on top of figure

        # Return axis suitable for spanning position
        ax_span = axs[(np.argmin(ranges[:, 0]) + np.argmax(ranges[:, 1])) // 2]
        ax_span = ax_span._panel_parent or ax_span
        return pos, ax_span

    def _get_align_axes(self, side):
        """
        Return the main axes along the left, right, bottom, or top sides
        of the figure.
        """
        # Initial stuff
        idx = 0 if side in ('left', 'top') else 1
        if side in ('left', 'right'):
            x, y = 'x', 'y'
        else:
            x, y = 'y', 'x'

        # Get edge index
        axs = self._axes_main
        if not axs:
            return []
        ranges = np.array([ax._range_gridspec(x) for ax in axs])
        min_, max_ = ranges[:, 0].min(), ranges[:, 1].max()
        edge = min_ if side in ('left', 'top') else max_

        # Return axes on edge sorted by order of appearance
        axs = [
            ax for ax in self._axes_main if ax._range_gridspec(x)[idx] == edge
        ]
        ranges = [ax._range_gridspec(y)[0] for ax in axs]
        return [ax for _, ax in sorted(zip(ranges, axs)) if ax.get_visible()]

    def _get_renderer(self):
        """
        Get a renderer at all costs, even if it means generating a brand new one!
        Used for updating the figure bounding box when it is accessed and calculating
        centered-row legend bounding boxes. This is copied from tight_layout.py in
        matplotlib.
        """
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

    def _insert_row_column(
        self, side, idx,
        ratio, space, space_orig, figure=False,
    ):
        """
        "Overwrite" the main figure gridspec to make room for a panel. The
        `side` is the panel side, the `idx` is the slot you want the panel
        to occupy, and the remaining args are the panel widths and spacings.
        """
        # Constants and stuff
        # Insert spaces to the left of right panels or to the right of
        # left panels. And note that since .insert() pushes everything in
        # that column to the right, actually must insert 1 slot farther to
        # the right when inserting left panels/spaces
        if side not in ('left', 'right', 'bottom', 'top'):
            raise ValueError(f'Invalid side {side}.')
        idx_space = idx - 1 * bool(side in ('bottom', 'right'))
        idx_offset = 1 * bool(side in ('top', 'left'))
        if side in ('left', 'right'):
            w, ncols = 'w', 'ncols'
        else:
            w, ncols = 'h', 'nrows'

        # Load arrays and test if we need to insert
        subplots_kw = self._subplots_kw
        subplots_orig_kw = self._subplots_orig_kw
        panels = subplots_kw[w + 'panels']
        ratios = subplots_kw[w + 'ratios']
        spaces = subplots_kw[w + 'space']
        spaces_orig = subplots_orig_kw[w + 'space']

        # Adjust space, ratio, and panel indicator arrays
        slot_type = 'f' if figure else side[0]
        slot_exists = idx not in (-1, len(panels)) and panels[idx] == slot_type
        if slot_exists:
            # Slot already exists
            if spaces_orig[idx_space] is None:
                spaces_orig[idx_space] = units(space_orig)
            spaces[idx_space] = _not_none(spaces_orig[idx_space], space)

        else:
            # Modify basic geometry and insert new slot
            idx += idx_offset
            idx_space += idx_offset
            subplots_kw[ncols] += 1
            spaces_orig.insert(idx_space, space_orig)
            spaces.insert(idx_space, space)
            ratios.insert(idx, ratio)
            panels.insert(idx, slot_type)

        # Update figure
        figsize, gridspec_kw, _ = pgridspec._calc_geometry(**subplots_kw)
        if slot_exists:
            gridspec = self._gridspec_main
            gridspec.update(**gridspec_kw)

        else:
            # Make new gridspec
            gridspec = pgridspec.GridSpec(self, **gridspec_kw)
            self._gridspec_main.figure = None
            self._gridspec_main = gridspec

            # Reassign subplotspecs to all axes and update positions
            for ax in self._iter_axes(hidden=True, children=True):
                # Get old index
                # NOTE: Endpoints are inclusive, not exclusive!
                if not hasattr(ax, 'get_subplotspec'):
                    continue
                if side in ('left', 'right'):
                    inserts = (None, None, idx, idx)
                else:
                    inserts = (idx, idx, None, None)
                subplotspec = ax.get_subplotspec()
                gridspec_ss = subplotspec.get_gridspec()
                subplotspec_top = subplotspec.get_topmost_subplotspec()

                # Apply new subplotspec
                _, _, *coords = subplotspec_top.get_active_rows_columns()
                for i in range(4):
                    if inserts[i] is not None and coords[i] >= inserts[i]:
                        coords[i] += 1
                row1, row2, col1, col2 = coords
                subplotspec_new = gridspec[row1:row2 + 1, col1:col2 + 1]
                if subplotspec_top is subplotspec:
                    ax.set_subplotspec(subplotspec_new)
                elif subplotspec_top is gridspec_ss._subplot_spec:
                    gridspec_ss._subplot_spec = subplotspec_new
                else:
                    raise ValueError(
                        'Unexpected GridSpecFromSubplotSpec nesting.'
                    )
                ax.update_params()
                ax.set_position(ax.figbox)

        # Adjust figure size *after* gridspecs are fixed
        self.set_size_inches(figsize, auto=True)

        return gridspec

    def _update_super_title(self, title, **kwargs):
        """
        Assign the figure "super title" and update settings.
        """
        if title is not None and self._suptitle.get_text() != title:
            self._suptitle.set_text(title)
        if kwargs:
            self._suptitle.update(kwargs)

    def _update_subplot_labels(self, ax, side, labels, **kwargs):
        """
        Assign the side labels and update settings.
        """
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

    def _update_geometry(self, resize=True):
        """
        Update the figure geometry based on the subplots keyword arguments.
        """
        subplots_kw = self._subplots_kw
        width, height = self.get_size_inches()
        if not resize:
            subplots_kw = subplots_kw.copy()
            subplots_kw.update(width=width, height=height)
        figsize, gridspec_kw, _ = pgridspec._calc_geometry(**subplots_kw)
        self._gridspec_main.update(**gridspec_kw)
        # Resize the figure
        # NOTE: Critical for qt backend to only do this if the figure size
        # has changed. Otherwise triggers recursive draws (???). Tried avoiding
        # this with additional context blocks in set_size_inches but could not
        # find consistent solution. For some reason *this* works consistently.
        fixed = all(subplots_kw[key] is not None for key in ('width', 'height'))
        if not fixed:
            self.set_size_inches(figsize, auto=True)

    def _update_geometry_from_aspect(self, resize=True):
        """
        Adjust the average aspect ratio used for gridspec calculations.
        This fixes grids with identically fixed aspect ratios, e.g.
        identically zoomed-in cartopy projections and imshow images.
        """
        # Get aspect ratio
        if not self._axes_main:
            return
        ax = self._axes_main[self._ref_num - 1]
        curaspect = ax.get_aspect()
        if isinstance(curaspect, str):
            if curaspect == 'auto':
                return
            elif curaspect != 'equal':
                raise RuntimeError(f'Unknown aspect ratio mode {curaspect!r}.')

        # Compare to current aspect
        subplots_kw = self._subplots_kw
        xscale, yscale = ax.get_xscale(), ax.get_yscale()
        if not isinstance(curaspect, str):
            aspect = curaspect
        elif xscale == 'linear' and yscale == 'linear':
            aspect = 1.0 / ax.get_data_ratio()
        elif xscale == 'log' and yscale == 'log':
            aspect = 1.0 / ax.get_data_ratio_log()
        else:
            return  # matplotlib should have issued warning
        if np.isclose(aspect, subplots_kw['aspect']):
            return

        # Apply new aspect
        subplots_kw['aspect'] = aspect
        self._update_geometry(resize=resize)

    def _update_geometry_from_spacing(self, renderer, resize=True):
        """
        Apply tight layout scaling that permits flexible figure
        dimensions and preserves panel widths and subplot aspect ratios.
        """
        # Initial stuff
        axs = list(self._iter_axes(hidden=True, children=False))
        subplots_kw = self._subplots_kw
        subplots_orig_kw = self._subplots_orig_kw  # tight layout overrides
        if not axs or not subplots_kw or not subplots_orig_kw:
            return

        # Temporarily disable spanning labels and get correct
        # positions for labels and suptitle
        self._align_axis_labels(False)
        self._align_subplot_super_labels(renderer)

        # Tight box *around* figure
        # Get bounds from old bounding box
        pad = self._pad
        obox = self.bbox_inches  # original bbox
        bbox = self.get_tightbbox(renderer)
        left = bbox.xmin
        bottom = bbox.ymin
        right = obox.xmax - bbox.xmax
        top = obox.ymax - bbox.ymax

        # Apply new bounds, permitting user overrides
        # TODO: Account for bounding box NaNs?
        for key, offset in zip(
            ('left', 'right', 'top', 'bottom'),
            (left, right, top, bottom)
        ):
            previous = subplots_orig_kw[key]
            current = subplots_kw[key]
            subplots_kw[key] = _not_none(previous, current - offset + pad)

        # Get arrays storing gridspec spacing args
        axpad = self._axpad
        panelpad = self._panelpad
        gridspec = self._gridspec_main
        nrows, ncols = gridspec.get_active_geometry()
        wspace = subplots_kw['wspace']
        hspace = subplots_kw['hspace']
        wspace_orig = subplots_orig_kw['wspace']
        hspace_orig = subplots_orig_kw['hspace']

        # Get new subplot spacings, axes panel spacing, figure panel spacing
        spaces = []
        for (w, x, y, nacross, ispace, ispace_orig) in zip(
            'wh', 'xy', 'yx', (nrows, ncols),
            (wspace, hspace), (wspace_orig, hspace_orig),
        ):
            # Determine which rows and columns correspond to panels
            panels = subplots_kw[w + 'panels']
            jspace = [*ispace]
            ralong = np.array([ax._range_gridspec(x) for ax in axs])
            racross = np.array([ax._range_gridspec(y) for ax in axs])
            for i, (space, space_orig) in enumerate(zip(ispace, ispace_orig)):
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
                filt1 = ralong[:, 1] == i  # i.e. right/bottom edge abutts against this
                filt2 = ralong[:, 0] == i + 1  # i.e. left/top edge abutts against this
                for j in range(nacross):  # e.g. each row

                    # Get indices
                    filt = (racross[:, 0] <= j) & (j <= racross[:, 1])
                    if sum(filt) < 2:  # no interface here
                        continue
                    idx1, = np.where(filt & filt1)
                    idx2, = np.where(filt & filt2)
                    if idx1.size > 1 or idx2.size > 2:
                        warnings._warn_proplot('This should never happen.')
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
                    space = _not_none(space_orig, space)  # overwritten by user
                jspace[i] = space
            spaces.append(jspace)

        # Apply new spacing
        subplots_kw.update({'wspace': spaces[0], 'hspace': spaces[1]})
        self._update_geometry(resize=resize)

    def add_subplot(self, *args, **kwargs):
        # Issue warning for new users that try to call
        # `~matplotlib.figure.Figure.add_subplot` manually.
        if not self._authorized_add_subplot:
            warnings._warn_proplot(
                'Using "fig.add_subplot()" with ProPlot figures may result in '
                'unexpected behavior. Please use "proplot.subplots()" instead.'
            )
        return super().add_subplot(*args, **kwargs)

    def auto_layout(self, renderer=None, resize=None, aspect=None, tight=None):
        """
        Trigger proplot's automatic figure size and tight layout algorithm. This
        is triggered automatically when the figure is drawn but is also documented
        here for pedagogical purposes.

        Parameters
        ----------
        renderer : `~matplotlib.backend_bases.RendererBase`, optional
            The renderer. If ``None`` a default renderer will be produced.
        resize : bool, optional
            If ``False``, the current figure dimensions are fixed and automatic
            figure resizing is disabled. This is set to ``False`` if the current
            backend is the `interactive ipython notebook backend \
<https://ipython.readthedocs.io/en/stable/interactive/plotting.html#id1>`__,
            which cannot handle automatic resizing. By default, the figure size may
            change unless both `width` and `height` or `figsize` were passed
            to `~proplot.ui.subplots`, `~matplotlib.figure.Figure.set_size_inches`
            was called by the user at any point, or the user manually resized the
            figure window with an interactive backend.
        aspect : bool, optional
            Whether to make figure size adjustments based on the aspect ratios of
            subplots. By default, this is ``True``.
        tight : bool, optional
            Whether to make figure size adjustments and gridspec spacing adjustments
            to produce a "tight layout". By default, this takes on the value of
            `tight` passed to `Figure`.
        """
        # *Impossible* to get notebook backend to work with auto resizing so we
        # just do the tight layout adjustments and skip resizing.
        renderer = self._get_renderer()
        if aspect is None:
            aspect = True
        if tight is None:
            tight = self._auto_tight
        if resize is None:
            resize = rc['backend'] != 'nbAgg'

        # Draw objects that will affect tight layout
        self._draw_auto_legends_colorbars()

        # Aspect ratio adjustment
        if aspect:
            self._update_geometry_from_aspect(resize=resize)  # resizes figure

        # Tight layout
        if tight:
            self._update_geometry_from_spacing(renderer, resize=resize)

        # Align labels after all is said and done
        self._align_axis_labels(True)
        self._align_subplot_super_labels(renderer)

    @docstring.add_snippets
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
        %(plot.colorbar_args)s
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
        %(plot.colorbar_kwargs)s

        Other parameters
        ----------------
        **kwargs
            Passed to `~matplotlib.axes.Axes.colorbar`.
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
        return ax.colorbar(*args, loc='fill', **kwargs)

    @docstring.add_snippets
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
        %(plot.legend_args)s
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
        %(plot.legend_kwargs)s

        Other parameters
        ----------------
        **kwargs
            Passed to `~matplotlib.axes.Axes.legend`.
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
        return ax.legend(*args, loc='fill', **kwargs)

    def save(self, filename, **kwargs):
        """
        Alias for `~matplotlib.figure.Figure.savefig`. User paths are
        expanded with `os.path.expanduser`.
        """
        return self.savefig(filename, **kwargs)

    def savefig(self, filename, **kwargs):
        # Automatically expand user the user name. Undocumented because we
        # do not want to overwrite the matplotlib docstring.
        # TODO: Concatenate docstrings.
        if isinstance(filename, str):
            filename = os.path.expanduser(filename)
        super().savefig(filename, **kwargs)

    def set_canvas(self, canvas):
        # Set the canvas and add monkey patches to the instance-level
        # `~matplotlib.backend_bases.FigureCanvasBase.draw_idle` and
        # `~matplotlib.backend_bases.FigureCanvasBase.print_figure`
        # methods. The latter is called by save() and by the inline backend.
        # See `_canvas_preprocessor` for details.
        # TODO: Concatenate docstrings.
        # NOTE: Cannot use draw_idle() because it causes complications for qt5
        # backend (wrong figure size).
        if callable(getattr(canvas, '_draw', None)):  # for macos backend
            canvas._draw = _canvas_preprocessor(canvas, '_draw')
        else:
            canvas.draw = _canvas_preprocessor(canvas, 'draw')
        canvas.print_figure = _canvas_preprocessor(canvas, 'print_figure')
        super().set_canvas(canvas)

    def set_size_inches(self, w, h=None, forward=True, auto=False):
        # Set the figure size and, if this is being called manually or from
        # an interactive backend, override the geometry tracker so users can
        # use interactive backends. If figure size is unchaged we *do not*
        # update the geometry tracker (figure backends often do this when
        # the figure is being initialized). See #76. Undocumented because this
        # is only relevant internally.
        # TODO: Concatenate docstrings.
        # NOTE: Bitmap renderers calculate the figure size in inches from
        # int(Figure.bbox.[width|height]) which rounds to whole pixels. When
        # renderer calls set_size_inches, size may be effectively the same, but
        # slightly changed due to roundoff error! Therefore, always compare to
        # *both* get_size_inches() and the truncated bbox dimensions times dpi.
        # NOTE: If we fail to detect 'manual' resize as manual, not only will
        # result be incorrect, but qt backend will crash because it detects a
        # recursive size change, since preprocessor size will differ.
        if h is None:
            width, height = w
        else:
            width, height = w, h
        if not all(np.isfinite(_) for _ in (width, height)):
            raise ValueError(f'Figure size must be finite, not ({width}, {height}).')
        width_true, height_true = self.get_size_inches()
        width_trunc = int(self.bbox.width) / self.dpi
        height_trunc = int(self.bbox.height) / self.dpi
        user = (  # detect user resize
            (  # sometimes get (width_trunc, height_true) or (width_true, height_trunc)
                width not in (width_true, width_trunc)
                or height not in (height_true, height_trunc)
            )
            and not auto
            and not self._is_autoresizing
            and not self._is_preprocessing
            and not getattr(self.canvas, '_is_idle_drawing', None)
            and not getattr(self.canvas, '_is_drawing', None)
            and not getattr(self.canvas, '_draw_pending', None)
        )
        if user:
            self._subplots_kw.update(width=width, height=height)
        context = self._context_autoresizing if auto or not user else _dummy_context
        with context():
            super().set_size_inches(width, height, forward=forward)

    @property
    def alignx(self):
        """
        The *x* axis label alignment mode.
        """
        return self._alignx

    @property
    def aligny(self):
        """
        The *y* axis label alignment mode.
        """
        return self._aligny

    @property
    def sharex(self):
        """
        The *x* axis sharing level.
        """
        return self._sharex

    @property
    def sharey(self):
        """
        The *y* axis sharing level.
        """
        return self._sharey

    @property
    def spanx(self):
        """
        The *x* axis label spanning mode.
        """
        return self._spanx

    @property
    def spany(self):
        """
        The *y* axis label spanning mode.
        """
        return self._spany

    @property
    def tight(self):
        """
        Whether subplot spacing is determined automatically for spaces that
        were not explicitly fixed by the user.
        """
        return self._auto_tight

    @property
    def gridspec(self):
        """
        The single `~proplot.gridspec.GridSpec` instance used for all subplots
        in the figure.
        """
        return self._gridspec_main

    @property
    def ref(self):
        """
        The reference axes number. The `axwidth`, `axheight`, and `aspect`
        arguments passed to `~proplot.ui.subplots` and
        `~proplot.ui.figure` arguments are applied to this axes, and
        aspect ratio is conserved for this axes in tight layout adjustment.
        """
        return self._ref_num

    def _iter_axes(self, hidden=False, children=False):
        """
        Iterate over all axes and panels in the figure belonging to the
        `~proplot.axes.Axes` class. Exclude inset and twin axes.

        Parameters
        ----------
        hidden : bool, optional
            Include hidden panels? This is useful for tight layout
            calculations.
        children : bool, optional
            Include child axes? This is useful for tight layout calculations.
            Includes inset axes and, due to proplot change, "twin" axes.
        """
        for ax in (
            *self._axes_main,
            *self._left_panels, *self._right_panels,
            *self._bottom_panels, *self._top_panels
        ):
            if not hidden and ax._panel_hidden:
                continue  # ignore hidden panel and its colorbar/legend child
            yield from ax._iter_axes(hidden=hidden, children=children)

    # Deprecations
    # NOTE: None of these even *worked* after drawing the figure. And not sure
    # what value (if any) they add even if we do get them to work.
    get_alignx, set_alignx = warnings._deprecate_getter_setter('0.6', 'alignx')
    get_aligny, set_aligny = warnings._deprecate_getter_setter('0.6', 'aligny')
    get_sharex, set_sharex = warnings._deprecate_getter_setter('0.6', 'sharex')
    get_sharey, set_sharey = warnings._deprecate_getter_setter('0.6', 'sharey')
    get_spanx, set_spanx = warnings._deprecate_getter_setter('0.6', 'spanx')
    get_spany, set_spany = warnings._deprecate_getter_setter('0.6', 'spany')
