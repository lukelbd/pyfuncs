#!/usr/bin/env python3
"""
Create a metaclass
"""
import matplotlib.artist as martist

__all__ = []

PROPS_PROTECTED = [
    'figure', 'axes', 'pickradius',
]


def _gen_subclasses(cls):
    """
    Generate all of a class's sublcasses with recursion.
    """
    try:
        for subclass in cls.__subclasses__():
            yield subclass
            for subclass in _gen_subclasses(subclass):
                yield subclass
    except TypeError:
        return


def _gen_properties(cls):
    """
    Generate property definitions for every artist getter.
    """
    for attr in dir(cls):
        obj = getattr(cls, attr)
        if callable(obj) and attr[:4] == 'get_':
            getter = obj
            name = attr[4:]
            if hasattr(cls, name):
                continue
            elif name in PROPS_PROTECTED:
                continue
            args = [getter]  # property args
            setter = getattr(cls, 'set_' + name, None)
            if callable(setter):
                args.append(setter)
            yield name, property(*args, doc=getter.__doc__)


for cls in _gen_subclasses(martist.Artist):
    for name, prop in _gen_properties(cls):
        setattr(cls, name, prop)
    print(f'Added properties to {cls.__name__}.')
