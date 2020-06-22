#!/usr/bin/env python3
"""
Utilities for modifying proplot docstrings.
"""
import inspect

#: Dictionary of docstring snippets.
snippets = {}


def add_snippets(arg):
    """
    Add un-indented snippets from the global `snippets` dictionary to the input
    string or function docstring. Also *dedent* the docstring with `inspect.getdoc`
    before adding snippets if the input is a function. This function uses C-style
    ``%(name)s`` substitution rather than `str.format` substitution so that the
    `snippets` keys do not have to be valid identifiers.
    """
    snippets_stripped = {key: value.strip() for key, value in snippets.items()}
    if isinstance(arg, str):
        arg %= snippets_stripped
    else:
        arg.__doc__ = inspect.getdoc(arg)
        if arg.__doc__:
            arg.__doc__ %= snippets_stripped
    return arg
