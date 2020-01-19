#!/usr/bin/env python3
"""
Various axis `~matplotlib.ticker.Formatter` and `~matplotlib.scale.ScaleBase`
classes. Includes constructor functions so that these classes can be selected
with a shorthand syntax.
"""
import re
from .cbook import _notNone, _warn_proplot
from .rctools import rc
from numbers import Number
from fractions import Fraction
import copy
import numpy as np
import numpy.ma as ma
import matplotlib.dates as mdates
import matplotlib.projections.polar as mpolar
import matplotlib.ticker as mticker
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
try:  # use this for debugging instead of print()!
    from icecream import ic
except ImportError:  # graceful fallback if IceCream isn't installed
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

__all__ = [
    'formatters', 'locators', 'scales',
    'Formatter', 'Locator', 'Scale',
    'AutoFormatter', 'CutoffScale', 'ExpScale',
    'FracFormatter', 'FuncScale',
    'InverseScale',
    'LinearScale',
    'LogitScale',
    'LogScale',
    'MercatorLatitudeScale', 'PowerScale', 'SimpleFormatter',
    'SineLatitudeScale',
    'SymmetricalLogScale',
]

MAX_DIGITS = 32  # do not draw 1000 digits when LogScale limits include zero!
SCALE_PRESETS = {
    'quadratic': ('power', 2,),
    'cubic': ('power', 3,),
    'quartic': ('power', 4,),
    'height': ('exp', np.e, -1 / 7, 1013.25, True),
    'pressure': ('exp', np.e, -1 / 7, 1013.25, False),
    'db': ('exp', 10, 1, 0.1, True),
    'idb': ('exp', 10, 1, 0.1, False),
    'np': ('exp', np.e, 1, 1, True),
    'inp': ('exp', np.e, 1, 1, False),
}


def Locator(locator, *args, **kwargs):
    """
    Return a `~matplotlib.ticker.Locator` instance. This function is used to
    interpret the `xlocator`, `xlocator_kw`, `ylocator`, `ylocator_kw`, `xminorlocator`,
    `xminorlocator_kw`, `yminorlocator`, and `yminorlocator_kw` arguments when
    passed to `~proplot.axes.XYAxes.format`, and the `locator`, `locator_kw`
    `minorlocator`, and `minorlocator_kw` arguments when passed to colorbar
    methods wrapped by `~proplot.wrappers.colorbar_wrapper`.

    Parameters
    ----------
    locator : `~matplotlib.ticker.Locator`, str, float, or list of float
        If `~matplotlib.ticker.Locator`, the object is returned.

        If number, specifies the *multiple* used to define tick separation.
        Returns a `~matplotlib.ticker.MultipleLocator` instance.

        If list of numbers, these points are ticked. Returns a
        `~matplotlib.ticker.FixedLocator` instance.

        If string, a dictionary lookup is performed (see below table).

        ======================  =================================================  ================================================================================================
        Key                     Class                                              Description
        ======================  =================================================  ================================================================================================
        ``'null'``, ``'none'``  `~matplotlib.ticker.NullLocator`                   No ticks
        ``'auto'``              `~matplotlib.ticker.AutoLocator`                   Major ticks at sensible locations
        ``'minor'``             `~matplotlib.ticker.AutoMinorLocator`              Minor ticks at sensible locations
        ``'date'``              `~matplotlib.dates.AutoDateLocator`                Default tick locations for datetime axes
        ``'fixed'``             `~matplotlib.ticker.FixedLocator`                  Ticks at these exact locations
        ``'index'``             `~matplotlib.ticker.IndexLocator`                  Ticks on the non-negative integers
        ``'linear'``            `~matplotlib.ticker.LinearLocator`                 Exactly ``N`` ticks encompassing the axis limits, spaced as ``numpy.linspace(lo, hi, N)``
        ``'log'``               `~matplotlib.ticker.LogLocator`                    Ticks for log-scale axes
        ``'logminor'``          `~matplotlib.ticker.LogLocator` preset             Ticks for log-scale axes on the 1st through 9th multiples of each power of the base
        ``'logit'``             `~matplotlib.ticker.LogitLocator`                  Ticks for logit-scale axes
        ``'logitminor'``        `~matplotlib.ticker.LogitLocator` preset           Ticks for logit-scale axes with ``minor=True`` passed to `~matplotlib.ticker.LogitLocator`
        ``'maxn'``              `~matplotlib.ticker.MaxNLocator`                   No more than ``N`` ticks at sensible locations
        ``'multiple'``          `~matplotlib.ticker.MultipleLocator`               Ticks every ``N`` step away from zero
        ``'symlog'``            `~matplotlib.ticker.SymmetricalLogLocator`         Ticks for symmetrical log-scale axes
        ``'symlogminor'``       `~matplotlib.ticker.SymmetricalLogLocator` preset  Ticks for symmetrical log-scale axes on the 1st through 9th multiples of each power of the base
        ``'theta'``             `~matplotlib.projections.polar.ThetaLocator`       Like the base locator but default locations are every `numpy.pi`/8 radians
        ``'year'``              `~matplotlib.dates.YearLocator`                    Ticks every ``N`` years
        ``'month'``             `~matplotlib.dates.MonthLocator`                   Ticks every ``N`` months
        ``'weekday'``           `~matplotlib.dates.WeekdayLocator`                 Ticks every ``N`` weekdays
        ``'day'``               `~matplotlib.dates.DayLocator`                     Ticks every ``N`` days
        ``'hour'``              `~matplotlib.dates.HourLocator`                    Ticks every ``N`` hours
        ``'minute'``            `~matplotlib.dates.MinuteLocator`                  Ticks every ``N`` minutes
        ``'second'``            `~matplotlib.dates.SecondLocator`                  Ticks every ``N`` seconds
        ``'microsecond'``       `~matplotlib.dates.MicrosecondLocator`             Ticks every ``N`` microseconds
        ======================  =================================================  ================================================================================================

    *args, **kwargs
        Passed to the `~matplotlib.ticker.Locator` class.

    Returns
    -------
    `~matplotlib.ticker.Locator`
        A `~matplotlib.ticker.Locator` instance.
    """  # noqa
    if isinstance(locator, mticker.Locator):
        return locator
    # Pull out extra args
    if np.iterable(locator) and not isinstance(locator, str) and not all(
            isinstance(num, Number) for num in locator):
        locator, args = locator[0], (*locator[1:], *args)
    # Get the locator
    if isinstance(locator, str):  # dictionary lookup
        # Shorthands and defaults
        if locator in ('logminor', 'logitminor', 'symlogminor'):
            locator, _ = locator.split('minor')
            if locator == 'logit':
                kwargs.setdefault('minor', True)
            else:
                kwargs.setdefault('subs', np.arange(1, 10))
        elif locator == 'index':
            args = args or (1,)
            if len(args) == 1:
                args = (*args, 0)
        # Lookup
        if locator not in locators:
            raise ValueError(
                f'Unknown locator {locator!r}. Options are '
                + ', '.join(map(repr, locators.keys())) + '.'
            )
        locator = locators[locator](*args, **kwargs)
    elif isinstance(locator, Number):  # scalar variable
        locator = mticker.MultipleLocator(locator, *args, **kwargs)
    elif np.iterable(locator):
        locator = mticker.FixedLocator(
            np.sort(locator), *args, **kwargs)  # not necessary
    else:
        raise ValueError(f'Invalid locator {locator!r}.')
    return locator


def Formatter(formatter, *args, date=False, index=False, **kwargs):
    """
    Return a `~matplotlib.ticker.Formatter` instance. This function is used to
    interpret the `xformatter`, `xformatter_kw`, `yformatter`, and
    `yformatter_kw` arguments when passed to
    `~proplot.axes.XYAxes.format`, and the `formatter`
    and `formatter_kw` arguments when passed to colorbar methods wrapped by
    `~proplot.wrappers.colorbar_wrapper`.

    Parameters
    ----------
    formatter : `~matplotlib.ticker.Formatter`, str, list of str, or function
        If `~matplotlib.ticker.Formatter`, the object is returned.

        If list of strings, ticks are labeled with these strings. Returns a
        `~matplotlib.ticker.FixedFormatter` instance when `index` is ``False``
        and an `~matplotlib.ticker.IndexFormatter` instance when `index` is
        ``True``.

        If function, labels will be generated using this function. Returns a
        `~matplotlib.ticker.FuncFormatter` instance.

        If string, there are 4 possibilities:

        1. If string contains ``'%'`` and `date` is ``False``, ticks will be
           formatted using the C-notation ``string % number`` method. See
           `this page \
<https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting>`__
           for a review.
        2. If string contains ``'%'`` and `date` is ``True``, datetime
           ``string % number`` formatting is used. See
           `this page \
<https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`__
           for a review.
        3. If string contains ``{x}`` or ``{x:...}``, ticks will be
           formatted by calling ``string.format(x=number)``.
        4. In all other cases, a dictionary lookup is performed
           (see below table).

        ======================  ==============================================  ===================================================================================================================================
        Key                     Class                                           Description
        ======================  ==============================================  ===================================================================================================================================
        ``'null'``, ``'none'``  `~matplotlib.ticker.NullFormatter`              No tick labels
        ``'auto'``              `AutoFormatter`                                 New default tick labels for axes
        ``'simple'``            `SimpleFormatter`                               New default tick labels for e.g. contour labels
        ``'frac'``              `FracFormatter`                                 Rational fractions
        ``'date'``              `~matplotlib.dates.AutoDateFormatter`           Default tick labels for datetime axes
        ``'concise'``           `~matplotlib.dates.ConciseDateFormatter`        More concise date labels introduced in `matplotlib 3.1 <https://matplotlib.org/3.1.0/users/whats_new.html#concisedateformatter>`__
        ``'datestr'``           `~matplotlib.dates.DateFormatter`               Date formatting with C-style ``string % format`` notation
        ``'eng'``               `~matplotlib.ticker.EngFormatter`               Engineering notation
        ``'fixed'``             `~matplotlib.ticker.FixedFormatter`             List of strings
        ``'formatstr'``         `~matplotlib.ticker.FormatStrFormatter`         From C-style ``string % format`` notation
        ``'index'``             `~matplotlib.ticker.IndexFormatter`             List of strings corresponding to non-negative integer positions along the axis
        ``'log'``, ``'sci'``    `~matplotlib.ticker.LogFormatterSciNotation`    For log-scale axes with scientific notation
        ``'logit'``             `~matplotlib.ticker.LogitFormatter`             For logistic-scale axes
        ``'math'``              `~matplotlib.ticker.LogFormatterMathtext`       For log-scale axes with math text
        ``'percent'``           `~matplotlib.ticker.PercentFormatter`           Trailing percent sign
        ``'scalar'``            `~matplotlib.ticker.ScalarFormatter`            Old default tick labels for axes
        ``'strmethod'``         `~matplotlib.ticker.StrMethodFormatter`         From the ``string.format`` method
        ``'theta'``             `~matplotlib.projections.polar.ThetaFormatter`  Formats radians as degrees, with a degree symbol
        ``'e'``                 `FracFormatter` preset                          Fractions of *e*
        ``'pi'``                `FracFormatter` preset                          Fractions of :math:`\\pi`
        ``'deg'``               `AutoFormatter` preset                          Trailing degree symbol
        ``'deglat'``            `AutoFormatter` preset                          Trailing degree symbol and cardinal "SN" indicator
        ``'deglon'``            `AutoFormatter` preset                          Trailing degree symbol and cardinal "WE" indicator
        ``'lat'``               `AutoFormatter` preset                          Cardinal "SN" indicator
        ``'lon'``               `AutoFormatter` preset                          Cardinal "WE" indicator
        ======================  ==============================================  ===================================================================================================================================

    date : bool, optional
        Toggles the behavior when `formatter` contains a ``'%'`` sign (see
        above).
    index : bool, optional
        Controls the behavior when `formatter` is a list of strings (see
        above).
    *args, **kwargs
        Passed to the `~matplotlib.ticker.Formatter` class.

    Returns
    -------
    `~matplotlib.ticker.Formatter`
        A `~matplotlib.ticker.Formatter` instance.
    """  # noqa
    if isinstance(formatter, mticker.Formatter):  # formatter object
        return formatter
    # Pull out extra args
    if np.iterable(formatter) and not isinstance(formatter, str) and not all(
            isinstance(item, str) for item in formatter):
        formatter, args = formatter[0], (*formatter[1:], *args)
    # Get the formatter
    if isinstance(formatter, str):  # assumption is list of strings
        # Format strings
        if re.search(r'{x?(:.+)?}', formatter):
            formatter = mticker.StrMethodFormatter(
                formatter, *args, **kwargs)  # new-style .format() form
        elif '%' in formatter:
            if date:
                formatter = mdates.DateFormatter(
                    formatter, *args, **kwargs)  # %-style, dates
            else:
                formatter = mticker.FormatStrFormatter(
                    formatter, *args, **kwargs)  # %-style, numbers
        else:
            # Fraction shorthands
            if formatter in ('pi', 'e'):
                if formatter == 'pi':
                    symbol, number = r'$\pi$', np.pi
                else:
                    symbol, number = '$e$', np.e
                kwargs.setdefault('symbol', symbol)
                kwargs.setdefault('number', number)
                formatter = 'frac'
            # Cartographic shorthands
            if formatter in ('deg', 'deglon', 'deglat', 'lon', 'lat'):
                negpos, suffix = None, None
                if 'deg' in formatter:
                    suffix = '\N{DEGREE SIGN}'
                if 'lat' in formatter:
                    negpos = 'SN'
                if 'lon' in formatter:
                    negpos = 'WE'
                kwargs.setdefault('suffix', suffix)
                kwargs.setdefault('negpos', negpos)
                formatter = 'auto'
            # Lookup
            if formatter not in formatters:
                raise ValueError(
                    f'Unknown formatter {formatter!r}. Options are '
                    + ', '.join(map(repr, formatters.keys())) + '.'
                )
            formatter = formatters[formatter](*args, **kwargs)
    elif callable(formatter):
        formatter = mticker.FuncFormatter(formatter, *args, **kwargs)
    elif np.iterable(formatter):  # list of strings on the major ticks
        if index:
            formatter = mticker.IndexFormatter(formatter)
        else:
            formatter = mticker.FixedFormatter(formatter)
    else:
        raise ValueError(f'Invalid formatter {formatter!r}.')
    return formatter


def Scale(scale, *args, **kwargs):
    """
    Return a `~matplotlib.scale.ScaleBase` instance. This function is used to
    interpret the `xscale`, `xscale_kw`, `yscale`, and `yscale_kw` arguments
    when passed to `~proplot.axes.CartesianAxes.format`.

    Parameters
    ----------
    scale : `~matplotlib.scale.ScaleBase`, str, (str, ...), or class
        If `~matplotlib.scale.ScaleBase`, the object is returned.

        If string, this is the registered scale name or scale "preset" (see
        below table).

        If list or tuple and the first element is a string, the subsequent
        items are passed to the scale class as positional arguments. For
        example, ``ax.format(xscale=('power', 2))`` applies the ``'quadratic'``
        scale to the *x* axis.

        =================  =======================  =======================================================================
        Key                Class                    Description
        =================  =======================  =======================================================================
        ``'linear'``       `LinearScale`            Linear
        ``'log'``          `LogScale`               Logarithmic
        ``'symlog'``       `SymmetricalLogScale`    Logarithmic beyond finite space around zero
        ``'logit'``        `LogitScale`             Logistic
        ``'inverse'``      `InverseScale`           Inverse
        ``'function'``     `FuncScale`              Scale from arbitrary forward and backwards functions
        ``'sine'``         `SineLatitudeScale`      Sine function (in degrees)
        ``'mercator'``     `MercatorLatitudeScale`  Mercator latitude function (in degrees)
        ``'exp'``          `ExpScale`               Arbitrary exponential function
        ``'power'``        `PowerScale`             Arbitrary power function
        ``'cutoff'``       `CutoffScale`            Arbitrary piecewise linear transformations
        ``'quadratic'``    `PowerScale` (preset)    Quadratic function
        ``'cubic'``        `PowerScale` (preset)    Cubic function
        ``'quartic'``      `PowerScale` (preset)    Cubic function
        ``'db'``           `ExpScale` (preset)      Ratio expressed as `decibels <https://en.wikipedia.org/wiki/Decibel>`__
        ``'np'``           `ExpScale` (preset)      Ratio expressed as `nepers <https://en.wikipedia.org/wiki/Neper>`__
        ``'idb'``          `ExpScale` (preset)      `Decibels <https://en.wikipedia.org/wiki/Decibel>`__ expressed as ratio
        ``'inp'``          `ExpScale` (preset)      `Nepers <https://en.wikipedia.org/wiki/Neper>`__ expressed as ratio
        ``'pressure'``     `ExpScale` (preset)      Height (in km) expressed linear in pressure
        ``'height'``       `ExpScale` (preset)      Pressure (in hPa) expressed linear in height
        =================  =======================  =======================================================================

    *args, **kwargs
        Passed to the `~matplotlib.scale.ScaleBase` class.

    Returns
    -------
    `~matplotlib.scale.ScaleBase`
        The scale instance.
    """  # noqa
    # NOTE: Why not try to interpret FuncScale arguments, like when lists
    # of numbers are passed to Locator? Because FuncScale *itself* accepts
    # ScaleBase classes as arguments... but constructor functions cannot
    # do anything but return the class instance upon receiving one.
    if isinstance(scale, mscale.ScaleBase):
        return scale
    # Pull out extra args
    if np.iterable(scale) and not isinstance(scale, str):
        scale, args = scale[0], (*scale[1:], *args)
    if not isinstance(scale, str):
        raise ValueError(f'Invalid scale name {scale!r}. Must be string.')
    # Get scale preset
    if scale in SCALE_PRESETS:
        if args or kwargs:
            _warn_proplot(
                f'Scale {scale!r} is a scale *preset*. Ignoring positional '
                'argument(s): {args} and keyword argument(s): {kwargs}. '
            )
        scale, *args = SCALE_PRESETS[scale]
    # Get scale
    scale = scale.lower()
    if scale in scales:
        scale = scales[scale]
    else:
        raise ValueError(
            f'Unknown scale or preset {scale!r}. Options are '
            + ', '.join(map(repr, list(scales) + list(SCALE_PRESETS))) + '.'
        )
    return scale(*args, **kwargs)


def _zerofix(x, string, precision=6):
    """
    Try to fix non-zero tick labels formatted as ``'0'``.
    """
    if string.rstrip('0').rstrip('.') == '0' and x != 0:
        string = ('{:.%df}' % precision).format(x)
    return string


class AutoFormatter(mticker.ScalarFormatter):
    """
    The new default formatter, a simple wrapper around
    `~matplotlib.ticker.ScalarFormatter`. Differs from
    `~matplotlib.ticker.ScalarFormatter` in the following ways:

    1. Trims trailing zeros if any exist.
    2. Allows user to specify *range* within which major tick marks
       are labelled.
    3. Allows user to add arbitrary prefix or suffix to every
       tick label string.
    """
    def __init__(
        self, *args,
        zerotrim=None, precision=None, tickrange=None,
        prefix=None, suffix=None, negpos=None, **kwargs
    ):
        """
        Parameters
        ----------
        zerotrim : bool, optional
            Whether to trim trailing zeros.
            Default is :rc:`axes.formatter.zerotrim`.
        tickrange : (float, float), optional
            Range within which major tick marks are labelled.
        prefix, suffix : str, optional
            Prefix and suffix for all strings.
        negpos : str, optional
            Length-2 string indicating the suffix for "negative" and "positive"
            numbers, meant to replace the minus sign. This is useful for
            indicating cardinal geographic coordinates.
        *args, **kwargs
            Passed to `~matplotlib.ticker.ScalarFormatter`.

        Warning
        -------
        The matplotlib `~matplotlib.ticker.ScalarFormatter` determines the
        number of significant digits based on the axis limits, and therefore
        may *truncate* digits while formatting ticks on highly non-linear
        axis scales like `~proplot.axistools.LogScale`. We try to correct
        this behavior with a patch.
        """
        tickrange = tickrange or (-np.inf, np.inf)
        super().__init__(*args, **kwargs)
        zerotrim = _notNone(zerotrim, rc['axes.formatter.zerotrim'])
        self._zerotrim = zerotrim
        self._tickrange = tickrange
        self._prefix = prefix or ''
        self._suffix = suffix or ''
        self._negpos = negpos or ''

    def __call__(self, x, pos=None):
        """
        Convert number to a string.

        Parameters
        ----------
        x : float
            The value.
        pos : float, optional
            The position.
        """
        # Tick range limitation
        eps = abs(x) / 1000
        tickrange = self._tickrange
        if (x + eps) < tickrange[0] or (x - eps) > tickrange[1]:
            return ''  # avoid some ticks
        # Negative positive handling
        if not self._negpos or x == 0:
            tail = ''
        elif x > 0:
            tail = self._negpos[1]
        else:
            x *= -1
            tail = self._negpos[0]
        # Format the string
        string = super().__call__(x, pos)
        for i in range(2):
            # Try to fix non-zero values formatted as zero
            if self._zerotrim and '.' in string:
                string = string.rstrip('0').rstrip('.')
            string = string.replace('-', '\N{MINUS SIGN}')
            if string == '\N{MINUS SIGN}0':
                string = '0'
            if i == 0 and string == '0' and x != 0:
                # Hard limit of MAX_DIGITS sigfigs
                string = ('{:.%df}' % min(
                    abs(np.log10(x) // 1), MAX_DIGITS)).format(x)
                continue
            break
        # Prefix and suffix
        sign = ''
        if string and string[0] == '\N{MINUS SIGN}':
            sign, string = string[0], string[1:]
        return sign + self._prefix + string + self._suffix + tail


def SimpleFormatter(*args, precision=6, zerotrim=True, **kwargs):
    """
    Return a `~matplotlib.ticker.FuncFormatter` instance that replicates the
    `zerotrim` feature from `AutoFormatter`. This is more suitable for
    arbitrary number formatting not necessarily associated with any
    `~matplotlib.axis.Axis` instance, e.g. labeling contours.

    Parameters
    ----------
    precision : int, optional
        The maximum number of digits after the decimal point.
    zerotrim : bool, optional
        Whether to trim trailing zeros.
        Default is :rc:`axes.formatter.zerotrim`.
    """
    zerotrim = _notNone(zerotrim, rc['axes.formatter.zerotrim'])

    def f(x, pos):
        string = ('{:.%df}' % precision).format(x)
        if zerotrim and '.' in string:
            string = string.rstrip('0').rstrip('.')
        if string == '-0' or string == '\N{MINUS SIGN}0':
            string = '0'
        return string.replace('-', '\N{MINUS SIGN}')
    return mticker.FuncFormatter(f)


def FracFormatter(symbol='', number=1):
    r"""
    Return a `~matplotlib.ticker.FuncFormatter` that formats numbers as
    fractions or multiples of some arbitrary value.
    This is powered by the builtin `~fractions.Fraction` class
    and the `~fractions.Fraction.limit_denominator` method.

    Parameters
    ----------
    symbol : str
        The symbol, e.g. ``r'$\pi$'``. Default is ``''``.
    number : float
        The value, e.g. `numpy.pi`. Default is ``1``.
    """
    def f(x, pos):  # must accept location argument
        frac = Fraction(x / number).limit_denominator()
        if x == 0:
            string = '0'
        elif frac.denominator == 1:  # denominator is one
            if frac.numerator == 1 and symbol:
                string = f'{symbol:s}'
            elif frac.numerator == -1 and symbol:
                string = f'-{symbol:s}'
            else:
                string = f'{frac.numerator:d}{symbol:s}'
        else:
            if frac.numerator == 1 and symbol:  # numerator is +/-1
                string = f'{symbol:s}/{frac.denominator:d}'
            elif frac.numerator == -1 and symbol:
                string = f'-{symbol:s}/{frac.denominator:d}'
            else:  # and again make sure we use unicode minus!
                string = f'{frac.numerator:d}{symbol:s}/{frac.denominator:d}'
        return string.replace('-', '\N{MINUS SIGN}')
    return mticker.FuncFormatter(f)


def _scale_factory(scale, axis, *args, **kwargs):
    """If `scale` is a `~matplotlib.scale.ScaleBase` instance, do nothing. If
    it is a registered scale name, look up and instantiate that scale."""
    if isinstance(scale, mscale.ScaleBase):
        if args or kwargs:
            _warn_proplot(f'Ignoring args {args} and keyword args {kwargs}.')
        return scale  # do nothing
    else:
        scale = scale.lower()
        if scale not in scales:
            raise ValueError(
                f'Unknown scale {scale!r}. Options are '
                + ', '.join(map(repr, scales.keys())) + '.'
            )
        return scales[scale](*args, **kwargs)


def _parse_logscale_args(kwargs, *keys):
    """Parse arguments for `LogScale` and `SymmetricalLogScale` that
    inexplicably require ``x`` and ``y`` suffixes by default."""
    for key in keys:
        value = _notNone(  # issues warning when multiple args passed!
            kwargs.pop(key, None),
            kwargs.pop(key + 'x', None),
            kwargs.pop(key + 'y', None),
            None, names=(key, key + 'x', key + 'y'),
        )
        if key == 'linthresh' and value is None:
            # NOTE: If linthresh is *exactly* on a power of the base, can
            # end up with additional log-locator step inside the threshold,
            # e.g. major ticks on -10, -1, -0.1, 0.1, 1, 10 for linthresh of
            # 1. Adding slight offset to *desired* linthresh prevents this.
            value = 1 + 1e-10
        if key == 'subs' and value is None:
            value = np.arange(1, 10)
        if value is not None:  # dummy axis_name is 'x'
            kwargs[key + 'x'] = value
    return kwargs


class _ScaleBase(object):
    """Mixin scale class that standardizes the
    `~matplotlib.scale.ScaleBase.set_default_locators_and_formatters`
    and `~matplotlib.scale.ScaleBase.get_transform` methods.
    Also overrides `__init__` so you no longer have to instantiate scales
    with an `~matplotlib.axis.Axis` instance."""
    def __init__(self, *args, **kwargs):
        # Pass a dummy axis to the superclass
        axis = type('Axis', (object,), {'axis_name': 'x'})()
        super().__init__(axis, *args, **kwargs)
        self._default_smart_bounds = None
        self._default_major_locator = None
        self._default_minor_locator = None
        self._default_major_formatter = None
        self._default_minor_formatter = None

    def set_default_locators_and_formatters(self, axis, only_if_default=False):
        """
        Apply all locators and formatters defined as attributes on
        initialization and define defaults for all scales.

        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`
            The axis.
        only_if_default : bool, optional
            Whether to refrain from updating the locators and formatters if
            the axis is currently using non-default versions. Useful if we
            want to avoid overwriting user customization when the scale
            is changed.
        """
        # Apply isDefault because matplotlib does this in axis._set_scale
        # but sometimes we need to bypass this method! Minor locator can be
        # "non default" even when user has not changed it, due to "turning
        # minor ticks" on and off, so set as 'default' if AutoMinorLocator.
        if self._default_smart_bounds is not None:
            axis.set_smart_bounds(self._default_smart_bounds)
        if not only_if_default or axis.isDefault_majloc:
            axis.set_major_locator(
                self._default_major_locator or Locator('auto')
            )
            axis.isDefault_majloc = True
        if not only_if_default or axis.isDefault_majfmt:
            axis.set_major_formatter(
                self._default_major_formatter or Formatter('auto')
            )
            axis.isDefault_majfmt = True
        if not only_if_default or axis.isDefault_minloc:
            name = axis.axis_name if axis.axis_name in 'xy' else 'x'
            axis.set_minor_locator(
                self._default_minor_locator or Locator(
                    'minor' if rc[name + 'tick.minor.visible'] else 'null'
                )
            )
            axis.isDefault_minloc = True
        if not only_if_default or axis.isDefault_minfmt:
            axis.set_minor_formatter(
                self._default_minor_formatter or Formatter('null')
            )
            axis.isDefault_minfmt = True

    def get_transform(self):
        """Return the scale transform."""
        return self._transform


class LinearScale(_ScaleBase, mscale.LinearScale):
    """
    As with `~matplotlib.scale.LinearScale` but with `AutoFormatter` as the
    default major formatter.
    """
    name = 'linear'
    """The registered scale name."""
    def __init__(self, **kwargs):
        """
        """
        super().__init__(**kwargs)
        self._transform = mtransforms.IdentityTransform()


class LogitScale(_ScaleBase, mscale.LogitScale):
    """
    As with `~matplotlib.scale.LogitScale` but with `AutoFormatter` as the
    default major formatter.
    """
    name = 'logit'
    """The registered scale name."""

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        nonpos : {'mask', 'clip'}
          Values outside of (0, 1) can be masked as invalid, or clipped to a
          number very close to 0 or 1.
        """
        super().__init__(**kwargs)
        # self._default_major_formatter = Formatter('logit')
        self._default_major_locator = Locator('logit')
        self._default_minor_locator = Locator('logit', minor=True)


class LogScale(_ScaleBase, mscale.LogScale):
    """
    As with `~matplotlib.scale.LogScale` but with `AutoFormatter` as the
    default major formatter. Also, "``x``" and "``y``" versions of each
    keyword argument are no longer required.
    """
    name = 'log'
    """The registered scale name."""

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        base : float, optional
            The base of the logarithm. Default is ``10``.
        nonpos : {'mask', 'clip'}, optional
            Non-positive values in *x* or *y* can be masked as
            invalid, or clipped to a very small positive number.
        subs : list of int, optional
            Default *minor* tick locations are on these multiples of each power
            of the base. For example, ``subs=(1,2,5)`` draws ticks on 1, 2, 5,
            10, 20, 50, etc. The default is ``subs=numpy.arange(1, 10)``.
        basex, basey, nonposx, nonposy, subsx, subsy
            Aliases for the above keywords. These used to be conditional
            on the *name* of the axis.
        """
        kwargs = _parse_logscale_args(kwargs, 'base', 'nonpos', 'subs')
        super().__init__(**kwargs)
        # self._default_major_formatter = Formatter('log')
        self._default_major_locator = Locator(
            'log', base=self.base)
        self._default_minor_locator = Locator(
            'log', base=self.base, subs=self.subs)


class SymmetricalLogScale(_ScaleBase, mscale.SymmetricalLogScale):
    """
    As with `~matplotlib.scale.SymmetricLogScale`. `AutoFormatter` is the new
    default major formatter. Also, "``x``" and "``y``" versions of each
    keyword argument are no longer required.
    """
    name = 'symlog'
    """The registered scale name."""

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        base : float, optional
            The base of the logarithm. Default is ``10``.
        linthresh : float, optional
            Defines the range ``(-linthresh, linthresh)``, within which the
            plot is linear.  This avoids having the plot go to infinity around
            zero. Defaults to 2.
        linscale : float, optional
            This allows the linear range ``(-linthresh, linthresh)`` to be
            stretched relative to the logarithmic range. Its value is the
            number of decades to use for each half of the linear range. For
            example, when `linscale` is ``1`` (the default), the space used
            for the positive and negative halves of the linear range will be
            equal to one decade in the logarithmic range.
        subs : sequence of int, optional
            Default *minor* tick locations are on these multiples of each power
            of the base. For example, ``subs=(1, 2, 5)`` draws ticks on 1, 2,
            5, 10, 20, 50, 100, 200, 500, etc. The default is
            ``subs=numpy.arange(1, 10)``.
        basex, basey, linthreshx, linthreshy, linscalex, linscaley, \
subsx, subsy
            Aliases for the above keywords. These used to be conditional
            on the *name* of the axis.
        """
        # Note the symlog locator gets base and linthresh from the transform
        kwargs = _parse_logscale_args(
            kwargs, 'base', 'linthresh', 'linscale', 'subs')
        super().__init__(**kwargs)
        # self._default_major_formatter = Formatter('symlog'))
        self._default_major_locator = Locator(
            'symlog', transform=self.get_transform())
        self._default_minor_locator = Locator(
            'symlog', transform=self.get_transform(), subs=self.subs)


class FuncScale(_ScaleBase, mscale.ScaleBase):
    """
    An axis scale comprised of arbitrary forward and inverse transformations.
    """
    name = 'function'
    """The registered scale name."""

    def __init__(
        self, arg, invert=False, parent_scale=None,
        major_locator=None, minor_locator=None,
        major_formatter=None, minor_formatter=None,
        smart_bounds=None,
    ):
        """
        Parameters
        ----------
        arg : function, (function, function), or \
`~matplotlib.scale.ScaleBase`
            The transform used to translate units from the parent axis to
            the secondary axis. Input can be as follows:

            * A single function that accepts a number and returns some
              transformation of that number. If you do not provide the
              inverse, the function must be
              `linear <https://en.wikipedia.org/wiki/Linear_function>`__ or \
`involutory <https://en.wikipedia.org/wiki/Involution_(mathematics)>`__.
              For example, to convert Kelvin to Celsius, use
              ``ax.dual%(x)s(lambda x: x - 273.15)``. To convert kilometers
              to meters, use ``ax.dual%(x)s(lambda x: x*1e3)``.
            * A 2-tuple of such functions. The second function must be the
              *inverse* of the first. For example, to apply the square, use
              ``ax.dual%(x)s((lambda x: x**2, lambda x: x**0.5))``.
              Again, if the first function is linear or involutory, you do
              not need to provide the second!
            * A `~matplotlib.scale.ScaleBase` instance, e.g. a scale returned
              by the `~proplot.axistools.Scale` constructor function. The
              forward transformation, inverse transformation, and default axis
              locators and formatters are borrowed from the resulting scale
              class.  For example, to apply the inverse, use
              ``ax.dual%(x)s(plot.Scale('inverse'))``.
              To apply the base-10 exponential function, use
              ``ax.dual%(x)s(plot.Scale('exp', 10))``.

        invert : bool, optional
            If ``True``, the forward and inverse functions are *swapped*.
            Used when drawing dual axes.
        parent_scale : `~matplotlib.scale.ScaleBase`
            The axis scale of the "parent" axis. Its forward transform is
            applied to the `FuncTransform`. Used when drawing dual axes.
        major_locator, minor_locator : `~matplotlib.ticker.Locator`, optional
            The default major and minor locator. By default these are
            borrowed from `transform`. If `transform` is not an axis scale,
            they are the same as `~matplotlib.scale.LinearScale`.
        major_formatter, minor_formatter : `~matplotlib.ticker.Formatter`, \
optional
            The default major and minor formatter. By default these are
            borrowed from `transform`. If `transform` is not an axis scale,
            they are the same as `~matplotlib.scale.LinearScale`.
        smart_bounds : bool, optional
            Whether "smart bounds" are enabled by default. If not ``None``,
            this is passed to `~matplotlib.axis.Axis.set_smart_bounds` when
            `~matplotlib.scale.ScaleBase.set_default_locators_and_formatters`
            is called. By default these are borrowed from `transform`.
        """
        # NOTE: We permit *arbitrary* parent axis scales. If the parent is
        # non-linear, we use *its* default locators and formatters. Assumption
        # is this is a log scale and the child is maybe some multiple or offset
        # of that scale. If the parent axis scale is linear, use the funcscale
        # defaults, which can inherit defaults.
        super().__init__()
        if callable(arg):
            forward = inverse = arg
        elif np.iterable(arg) and len(arg) == 2 and all(map(callable, arg)):
            forward, inverse = arg
        elif isinstance(arg, mscale.ScaleBase):
            trans = arg.get_transform()
            forward = trans.transform
            inverse = trans.inverted().transform
        else:
            raise ValueError(
                'Input should be a function, 2-tuple of forward and '
                'and inverse functions, or a matplotlib.scale.ScaleBase '
                f'instance, not {arg!r}.'
            )

        # Create the FuncTransform or composite transform used for this class
        # May need to invert functions for dualx() and dualy()
        if invert:
            forward, inverse = inverse, forward
        functransform = FuncTransform(forward, inverse)

        # Manage the "parent" axis scale
        # NOTE: Makes sense to use the "inverse" function here because this is
        # a transformation from some *other* axis to this one, not vice versa.
        if isinstance(parent_scale, mscale.ScaleBase):
            if isinstance(parent_scale, mscale.SymmetricalLogScale):
                kwargs = {
                    key: getattr(parent_scale, key)
                    for key in ('base', 'linthresh', 'linscale', 'subs')
                }
                kwargs['linthresh'] = inverse(kwargs['linthresh'])
                parent_scale = SymmetricalLogScale(**kwargs)
            elif isinstance(parent_scale, CutoffScale):
                args = list(parent_scale.args)  # copy
                for i in range(0, len(args), 2):
                    args[i] = inverse(args[i])
                parent_scale = CutoffScale(*args)
            functransform = parent_scale.get_transform() + functransform
        elif parent_scale is not None:
            raise ValueError(
                f'parent_scale {parent_scale!r} must be a ScaleBase instance, '
                f'not {type(parent_scale)!r}.'
            )

        # Transform and default stuff
        self.functions = (forward, inverse)
        self._transform = functransform
        self._default_smart_bounds = smart_bounds
        self._default_major_locator = major_locator
        self._default_minor_locator = minor_locator
        self._default_major_formatter = major_formatter
        self._default_minor_formatter = minor_formatter

        # Try to borrow locators and formatters
        # WARNING: Using the same locator on multiple axes can evidently
        # have unintended side effects! Matplotlib bug. So we make copies.
        for scale in (arg, parent_scale):
            if not isinstance(scale, _ScaleBase):
                continue
            if isinstance(scale, mscale.LinearScale):
                continue
            for key in (
                'smart_bounds', 'major_locator', 'minor_locator',
                'major_formatter', 'minor_formatter'
            ):
                key = '_default_' + key
                attr = getattr(scale, key)
                if getattr(self, key) is None and attr is not None:
                    setattr(self, key, copy.copy(attr))


class FuncTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, forward, inverse):
        super().__init__()
        if callable(forward) and callable(inverse):
            self._forward = forward
            self._inverse = inverse
        else:
            raise ValueError('arguments to FuncTransform must be functions')

    def inverted(self):
        return FuncTransform(self._inverse, self._forward)

    def transform_non_affine(self, values):
        return self._forward(values)


class PowerScale(_ScaleBase, mscale.ScaleBase):
    r"""
    "Power scale" that performs the transformation

    .. math::

        x^{c}

    """
    name = 'power'
    """The registered scale name."""

    def __init__(self, power=1, inverse=False, *, minpos=1e-300, **kwargs):
        """
        Parameters
        ----------
        power : float, optional
            The power :math:`c` to which :math:`x` is raised.
        inverse : bool, optional
            If ``True``, the "forward" direction performs
            the inverse operation :math:`x^{1/c}`.
        minpos : float, optional
            The minimum permissible value, used to truncate negative values.
        """
        super().__init__()
        if not inverse:
            self._transform = PowerTransform(power, minpos)
        else:
            self._transform = InvertedPowerTransform(power, minpos)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """Returns the range *vmin* and *vmax* limited to positive numbers."""
        return max(vmin, minpos), max(vmax, minpos)


class PowerTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    has_inverse = True
    is_separable = True

    def __init__(self, power, minpos):
        super().__init__()
        self.minpos = minpos
        self._power = power

    def inverted(self):
        return InvertedPowerTransform(self._power, self.minpos)

    def transform_non_affine(self, a):
        aa = np.array(a)
        aa[aa <= self.minpos] = self.minpos  # necessary
        return np.power(np.array(a), self._power)


class InvertedPowerTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    has_inverse = True
    is_separable = True

    def __init__(self, power, minpos):
        super().__init__()
        self.minpos = minpos
        self._power = power

    def inverted(self):
        return PowerTransform(self._power, self.minpos)

    def transform_non_affine(self, a):
        aa = np.array(a)
        aa[aa <= self.minpos] = self.minpos  # necessary
        return np.power(np.array(a), 1 / self._power)


class ExpScale(_ScaleBase, mscale.ScaleBase):
    r"""
    "Exponential scale" that performs either of two transformations. When
    `inverse` is ``False`` (the default), performs the transformation

    .. math::

        Ca^{bx}

    where the constants :math:`a`, :math:`b`, and :math:`C` are set by the
    input (see below). When `inverse` is ``True``, this performs the inverse
    transformation

    .. math::

        (\log_a(x) - \log_a(C))/b

    which in appearence is equivalent to `LogScale` since it is just a linear
    transformation of the logarithm.
    """
    name = 'exp'
    """The registered scale name."""

    def __init__(
        self, a=np.e, b=1, c=1, inverse=False, minpos=1e-300, **kwargs
    ):
        """
        Parameters
        ----------
        a : float, optional
            The base of the exponential, i.e. the :math:`a` in :math:`Ca^{bx}`.
        b : float, optional
            The scale for the exponent, i.e. the :math:`b` in :math:`Ca^{bx}`.
        c : float, optional
            The coefficient of the exponential, i.e. the :math:`C`
            in :math:`Ca^{bx}`.
        minpos : float, optional
            The minimum permissible value, used to truncate negative values.
        inverse : bool, optional
            If ``True``, the "forward" direction performs the inverse
            operation.
        """
        super().__init__()
        if not inverse:
            self._transform = ExpTransform(a, b, c, minpos)
        else:
            self._transform = InvertedExpTransform(a, b, c, minpos)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """Return *vmin* and *vmax* limited to positive numbers."""
        return max(vmin, minpos), max(vmax, minpos)


class ExpTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    has_inverse = True
    is_separable = True

    def __init__(self, a, b, c, minpos):
        super().__init__()
        self.minpos = minpos
        self._a = a
        self._b = b
        self._c = c

    def inverted(self):
        return InvertedExpTransform(self._a, self._b, self._c, self.minpos)

    def transform_non_affine(self, a):
        return self._c * np.power(self._a, self._b * np.array(a))


class InvertedExpTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    has_inverse = True
    is_separable = True

    def __init__(self, a, b, c, minpos):
        super().__init__()
        self.minpos = minpos
        self._a = a
        self._b = b
        self._c = c

    def inverted(self):
        return ExpTransform(self._a, self._b, self._c, self.minpos)

    def transform_non_affine(self, a):
        aa = np.array(a)
        aa[aa <= self.minpos] = self.minpos  # necessary
        return np.log(aa / self._c) / (self._b * np.log(self._a))


class MercatorLatitudeScale(_ScaleBase, mscale.ScaleBase):
    """
    Axis scale that transforms coordinates as with latitude in the `Mercator \
projection <http://en.wikipedia.org/wiki/Mercator_projection>`__.
    Adapted from `this matplotlib example \
<https://matplotlib.org/examples/api/custom_scale_example.html>`__.
    """r""""The scale function is as follows:

    .. math::

        y = \\ln(\\tan(\\pi x/180) + \\sec(\\pi x/180))

    The inverse scale function is as follows:

    .. math::

        x = 180\\arctan(\\sinh(y))/\\pi

    """
    name = 'mercator'
    """The registered scale name."""

    def __init__(self, thresh=85.0):
        """
        Parameters
        ----------
        thresh : float, optional
            Threshold between 0 and 90, used to constrain axis limits between
            ``-thresh`` and ``+thresh``.
        """
        super().__init__()
        if thresh >= 90.0:
            raise ValueError('Threshold "thresh" must be <=90.')
        self._thresh = thresh
        self._transform = MercatorLatitudeTransform(thresh)
        self._default_major_formatter = Formatter('deg')
        self._default_smart_bounds = True

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """Return *vmin* and *vmax* limited to some range within
        +/-90 degrees (exclusive)."""
        return max(vmin, -self._thresh), min(vmax, self._thresh)


class MercatorLatitudeTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, thresh):
        super().__init__()
        self._thresh = thresh

    def inverted(self):
        return InvertedMercatorLatitudeTransform(self._thresh)

    def transform_non_affine(self, a):
        # With safeguards
        # TODO: Can improve this?
        a = np.deg2rad(a)  # convert to radians
        m = ma.masked_where((a < -self._thresh) | (a > self._thresh), a)
        if m.mask.any():
            return ma.log(np.abs(ma.tan(m) + 1 / ma.cos(m)))
        else:
            return np.log(np.abs(np.tan(a) + 1 / np.cos(a)))


class InvertedMercatorLatitudeTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, thresh):
        super().__init__()
        self._thresh = thresh

    def inverted(self):
        return MercatorLatitudeTransform(self._thresh)

    def transform_non_affine(self, a):
        # m = ma.masked_where((a < -self._thresh) | (a > self._thresh), a)
        # always assume in first/fourth quadrant, i.e. go from -pi/2 to pi/2
        return np.rad2deg(np.arctan2(1, np.sinh(a)))


class SineLatitudeScale(_ScaleBase, mscale.ScaleBase):
    r"""
    Axis scale that is linear in the *sine* of *x*. The axis limits are
    constrained to fall between ``-90`` and ``+90`` degrees. The scale
    function is as follows:

    .. math::

        y = \sin(\pi x/180)

    The inverse scale function is as follows:

    .. math::

        x = 180\arcsin(y)/\pi
    """
    name = 'sine'
    """The registered scale name."""

    def __init__(self):
        super().__init__()
        self._transform = SineLatitudeTransform()
        self._default_major_formatter = Formatter('deg')
        self._default_smart_bounds = True

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """Return *vmin* and *vmax* limited to some range within
        +/-90 degrees (inclusive)."""
        return max(vmin, -90), min(vmax, 90)


class SineLatitudeTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self):
        super().__init__()

    def inverted(self):
        return InvertedSineLatitudeTransform()

    def transform_non_affine(self, a):
        # With safeguards
        # TODO: Can improve this?
        with np.errstate(invalid='ignore'):  # NaNs will always be False
            m = (a >= -90) & (a <= 90)
        if not m.all():
            aa = ma.masked_where(~m, a)
            return ma.sin(np.deg2rad(aa))
        else:
            return np.sin(np.deg2rad(a))


class InvertedSineLatitudeTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self):
        super().__init__()

    def inverted(self):
        return SineLatitudeTransform()

    def transform_non_affine(self, a):
        # Clipping, instead of setting invalid
        # NOTE: Using ma.arcsin below caused super weird errors, dun do that
        aa = a.copy()
        return np.rad2deg(np.arcsin(aa))


class CutoffScale(_ScaleBase, mscale.ScaleBase):
    """
    Axis scale composed of arbitrary piecewise linear transformations.
    The axis can undergo discrete jumps, "accelerations", or "decelerations"
    between successive thresholds. Adapted from
    `this stackoverflow post <https://stackoverflow.com/a/5669301/4970632>`__.
    """
    name = 'cutoff'
    """The registered scale name."""

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args : (thresh_1, scale_1, ..., thresh_N, [scale_N]), optional
            Sequence of "thresholds" and "scales". If the final scale is
            omitted (i.e. you passed an odd number of arguments) it is set
            to ``1``. Each ``scale_i`` in the sequence can be interpreted
            as follows:

            * If ``scale_i < 1``, the axis is decelerated from ``thresh_i`` to
              ``thresh_i+1``. For ``scale_N``, the axis is decelerated
              everywhere above ``thresh_N``.
            * If ``scale_i > 1``, the axis is accelerated from ``thresh_i`` to
              ``thresh_i+1``. For ``scale_N``, the axis is accelerated
              everywhere above ``thresh_N``.
            * If ``scale_i == numpy.inf``, the axis *discretely jumps* from
              ``thresh_i`` to ``thresh_i+1``. The final scale ``scale_N``
              *cannot* be ``numpy.inf``.

        Example
        -------

        >>> import proplot as plot
        ... import numpy as np
        ... scale = plot.CutoffScale(10, 0.5)  # move slower above 10
        ... scale = plot.CutoffScale(10, 2, 20)  # zoom out between 10 and 20
        ... scale = plot.CutoffScale(10, np.inf, 20)  # jump from 10 to 20

        """
        super().__init__()
        args = list(args)
        if len(args) % 2 == 1:
            args.append(1)
        self.args = args
        self.threshs = args[::2]
        self.scales = args[1::2]
        self._transform = CutoffTransform(self.threshs, self.scales)


class CutoffTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    has_inverse = True
    is_separable = True

    def __init__(self, threshs, scales, zero_dists=None):
        # The zero_dists array is used to fill in distances where scales and
        # threshold steps are zero. Used for inverting discrete transorms.
        super().__init__()
        dists = np.diff(threshs)
        scales = np.asarray(scales)
        threshs = np.asarray(threshs)
        if len(scales) != len(threshs):
            raise ValueError(f'Got {len(threshs)} but {len(scales)} scales.')
        if any(scales < 0):
            raise ValueError('Scales must be non negative.')
        if scales[-1] in (0, np.inf):
            raise ValueError('Final scale must be finite.')
        if any(dists < 0):
            raise ValueError('Thresholds must be monotonically increasing.')
        if any((dists == 0) | (scales == 0)) and (
                any((dists == 0) != (scales == 0)) or zero_dists is None):
            raise ValueError(
                'Got zero scales and distances in different places or '
                'zero_dists is None.'
            )
        self._scales = scales
        self._threshs = threshs
        with np.errstate(divide='ignore', invalid='ignore'):
            dists = np.concatenate((threshs[:1], dists / scales[:-1]))
            if zero_dists is not None:
                dists[scales[:-1] == 0] = zero_dists
            self._dists = dists

    def inverted(self):
        # Use same algorithm for inversion!
        threshs = np.cumsum(self._dists)  # thresholds in transformed space
        with np.errstate(divide='ignore', invalid='ignore'):
            scales = 1 / self._scales  # new scales are inverse
        zero_dists = np.diff(self._threshs)[scales[:-1] == 0]
        return CutoffTransform(threshs, scales, zero_dists=zero_dists)

    def transform_non_affine(self, a):
        # Cannot do list comprehension because this method sometimes
        # received non-1d arrays
        dists = self._dists
        scales = self._scales
        threshs = self._threshs
        aa = np.array(a)  # copy
        with np.errstate(divide='ignore', invalid='ignore'):
            for i, ai in np.ndenumerate(a):
                j = np.searchsorted(threshs, ai)
                if j > 0:
                    aa[i] = dists[:j].sum() + (
                        ai - threshs[j - 1]) / scales[j - 1]
        return aa


class InverseScale(_ScaleBase, mscale.ScaleBase):
    r"""
    Axis scale that is linear in the *inverse* of *x*. The forward and inverse
    scale functions are as follows:

    .. math::

        y = x^{-1}

    """
    # Unlike log-scale, we can't just warp the space between
    # the axis limits -- have to actually change axis limits. Also this
    # scale will invert and swap the limits you provide. Weird!
    name = 'inverse'
    """The registered scale name."""

    def __init__(self, **kwargs):
        super().__init__()
        self._transform = InverseTransform()
        # self._default_major_formatter = Fromatter('log')
        self._default_major_locator = Locator(
            'log', base=10)
        self._default_minor_locator = Locator(
            'log', base=10, subs=np.arange(1, 10))
        self._default_smart_bounds = True

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """Return *vmin* and *vmax* limited to positive numbers."""
        return max(vmin, minpos), max(vmax, minpos)


class InverseTransform(mtransforms.Transform):
    # Create transform object
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self):
        super().__init__()

    def inverted(self):
        return InverseTransform()

    def transform_non_affine(self, a):
        a = np.array(a)
        # f = np.abs(a) <= self.minpos # attempt for negative-friendly
        # aa[f] = np.sign(a[f])*self.minpos
        with np.errstate(divide='ignore', invalid='ignore'):
            return 1.0 / a


#: The registered scale names and their associated
#: `~matplotlib.scale.ScaleBase` classes. See `Scale` for a table.
scales = mscale._scale_mapping

#: Mapping of strings to `~matplotlib.ticker.Locator` classes. See
#: `Locator` for a table."""
locators = {
    'none': mticker.NullLocator,
    'null': mticker.NullLocator,
    'auto': mticker.AutoLocator,
    'log': mticker.LogLocator,
    'maxn': mticker.MaxNLocator,
    'linear': mticker.LinearLocator,
    'multiple': mticker.MultipleLocator,
    'fixed': mticker.FixedLocator,
    'index': mticker.IndexLocator,
    'symlog': mticker.SymmetricalLogLocator,
    'logit': mticker.LogitLocator,
    'minor': mticker.AutoMinorLocator,
    'date': mdates.AutoDateLocator,
    'microsecond': mdates.MicrosecondLocator,
    'second': mdates.SecondLocator,
    'minute': mdates.MinuteLocator,
    'hour': mdates.HourLocator,
    'day': mdates.DayLocator,
    'weekday': mdates.WeekdayLocator,
    'month': mdates.MonthLocator,
    'year': mdates.YearLocator,
}
if hasattr(mpolar, 'ThetaLocator'):
    locators['theta'] = mpolar.ThetaLocator

#: Mapping of strings to `~matplotlib.ticker.Formatter` classes. See
#: `Formatter` for a table.
formatters = {  # note default LogFormatter uses ugly e+00 notation
    'auto': AutoFormatter,
    'frac': FracFormatter,
    'simple': SimpleFormatter,
    'date': mdates.AutoDateFormatter,
    'datestr': mdates.DateFormatter,
    'scalar': mticker.ScalarFormatter,
    'none': mticker.NullFormatter,
    'null': mticker.NullFormatter,
    'strmethod': mticker.StrMethodFormatter,
    'formatstr': mticker.FormatStrFormatter,
    'log': mticker.LogFormatterSciNotation,
    'sci': mticker.LogFormatterSciNotation,
    'math': mticker.LogFormatterMathtext,
    'logit': mticker.LogitFormatter,
    'eng': mticker.EngFormatter,
    'percent': mticker.PercentFormatter,
    'index': mticker.IndexFormatter,
}
if hasattr(mdates, 'ConciseDateFormatter'):
    formatters['concise'] = mdates.ConciseDateFormatter
if hasattr(mpolar, 'ThetaFormatter'):
    formatters['theta'] = mpolar.ThetaFormatter

# Monkey patch. Force scale_factory to accept ScaleBase instances, so that
# set_xscale and set_yscale can accept scales returned by the Scale constructor
if mscale.scale_factory is not _scale_factory:
    mscale.scale_factory = _scale_factory

# Custom scales and overrides
mscale.register_scale(CutoffScale)
mscale.register_scale(ExpScale)
mscale.register_scale(LogScale)
mscale.register_scale(LinearScale)
mscale.register_scale(LogitScale)
mscale.register_scale(FuncScale)
mscale.register_scale(PowerScale)
mscale.register_scale(SymmetricalLogScale)
mscale.register_scale(InverseScale)
mscale.register_scale(SineLatitudeScale)
mscale.register_scale(MercatorLatitudeScale)
