# -*- coding: utf-8 -*-
"""
Initialization file for the vectorized PVMismatch (v_PVMismatch) Package.

This package contains the basic library modules, methods, classes and
attributes to model PV system mismatch.
    >>> from v_pvmismatch import vpvcell  # imports the PVcell methods
    >>> # import vpvcell, vpvmodule, vpvstring and vpvsystem
    >>> # from v_pvmismatch.pvmismatch import * # Imports original PVMismatch
    >>> from v_pvmismatch import *
"""
from v_pvmismatch.version import __version__

from v_pvmismatch import (
    utils,
    circuit_comb,
    cell_curr,
    plotting,
    vpvcell,
    vpvmodule,
    vpvstring,
    vpvsystem,
    pvmismatch
)
