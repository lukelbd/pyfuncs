#!/usr/bin/env python3
# Import everything into the top-level module namespace
# Make sure to load styletools early so we can try to update TTFPATH before
# the fontManager is loaded by other modules (requiring a rebuild)
import os as _os
import pkg_resources as _pkg
from .cbook import _benchmark
with _benchmark('total time'):
    with _benchmark('utils'):
        from .utils import *  # noqa: F401 F403
    with _benchmark('styletools'):
        from .styletools import *  # noqa: F401 F403
    with _benchmark('rctools'):
        from .rctools import *  # noqa: F401 F403
    with _benchmark('axistools'):
        from .axistools import *  # noqa: F401 F403
    with _benchmark('wrappers'):
        from .wrappers import *  # noqa: F401 F403
    with _benchmark('projs'):
        from .projs import *  # noqa: F401 F403
    with _benchmark('axes'):
        from .axes import *  # noqa: F401 F403
    with _benchmark('subplots'):
        from .subplots import *  # noqa: F401 F403


# Initialize customization folders
_rc_folder = _os.path.join(_os.path.expanduser('~'), '.proplot')
if not _os.path.isdir(_rc_folder):
    _os.mkdir(_rc_folder)
for _rc_sub in ('cmaps', 'cycles', 'colors', 'fonts'):
    _rc_sub = _os.path.join(_rc_folder, _rc_sub)
    if not _os.path.isdir(_rc_sub):
        _os.mkdir(_rc_sub)

# SCM versioning
name = 'proplot'
try:
    version = __version__ = _pkg.get_distribution(__name__).version
except _pkg.DistributionNotFound:
    version = __version__ = 'unknown'
