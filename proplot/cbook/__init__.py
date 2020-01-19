#!/usr/bin/env python3
"""
Various utilities used internally.
"""
import time
import functools
import warnings

# Change this constant to turn on benchmarking
BENCHMARK = False


class _benchmark(object):
    """Context object for timing arbitrary blocks of code."""
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        if BENCHMARK:
            self.time = time.perf_counter()

    def __exit__(self, *args):
        if BENCHMARK:
            print(f'{self.message}: {time.perf_counter() - self.time}s')


class _setstate(object):
    """Temporarily modify attribute(s) for an arbitrary object."""
    def __init__(self, obj, **kwargs):
        self._obj = obj
        self._kwargs = kwargs
        self._kwargs_orig = {
            key: getattr(obj, key) for key in kwargs if hasattr(obj, key)
        }

    def __enter__(self):
        for key, value in self._kwargs.items():
            setattr(self._obj, key, value)

    def __exit__(self, *args):
        for key in self._kwargs.keys():
            if key in self._kwargs_orig:
                setattr(self._obj, key, self._kwargs_orig[key])
            else:
                delattr(self._obj, key)


def _counter(func):
    """A decorator that counts and prints the cumulative time a function
    has benn running. See `this link \
<https://stackoverflow.com/a/1594484/4970632>`__."""
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if BENCHMARK:
            t = time.perf_counter()
        res = func(*args, **kwargs)
        if BENCHMARK:
            decorator.time += (time.perf_counter() - t)
            decorator.count += 1
            print(f'{func.__name__}() cumulative time: {decorator.time}s '
                  f'({decorator.count} calls)')
        return res
    decorator.time = 0
    decorator.count = 0  # initialize
    return decorator


def _timer(func):
    """Decorator that prints the time a function takes to execute.
    See: https://stackoverflow.com/a/1594484/4970632"""
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if BENCHMARK:
            t = time.perf_counter()
        res = func(*args, **kwargs)
        if BENCHMARK:
            print(f'{func.__name__}() time: {time.perf_counter()-t}s')
        return res
    return decorator


def _warn_format(message, category, filename, lineno, line=None):
    """Simple format for warnings issued by ProPlot. See the
    `internal warning call signature \
<https://docs.python.org/3/library/warnings.html#warnings.showwarning>`__
    and the `default warning source code \
<https://github.com/python/cpython/blob/master/Lib/warnings.py>`__."""
    return f'{filename}:{lineno}: ProPlotWarning: {message}\n'  # needs newline


def _warn_proplot(message):
    """*Temporarily* apply the `_warn_format` monkey patch and emit the
    warning. Do not want to affect warnings emitted by other modules."""
    with _setstate(warnings, formatwarning=_warn_format):
        warnings.warn(message)


def _notNone(*args, names=None):
    """Return the first non-``None`` value. This is used with keyword arg
    aliases and for setting default values. Ugly name but clear purpose. Pass
    the `names` keyword arg to issue warning if multiple args were passed. Must
    be list of non-empty strings."""
    if names is None:
        for arg in args:
            if arg is not None:
                return arg
        return arg  # last one
    else:
        first = None
        kwargs = {}
        if len(names) != len(args) - 1:
            raise ValueError(
                f'Need {len(args)+1} names for {len(args)} args, '
                f'but got {len(names)} names.'
            )
        names = [*names, '']
        for name, arg in zip(names, args):
            if arg is not None:
                if first is None:
                    first = arg
                if name:
                    kwargs[name] = arg
        if len(kwargs) > 1:
            warnings.warn(
                f'Got conflicting or duplicate keyword args: {kwargs}. '
                'Using the first one.'
            )
        return first
