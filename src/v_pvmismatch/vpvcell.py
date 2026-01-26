# -*- coding: utf-8 -*-
"""Vectorized two diode model."""

import os

import numpy as np
import pandas as pd
from scipy import constants

from .pvmismatch import pvconstants, pvcell
from .utils import save_pickle

# ----------------------------------------------------------------------------
# TWO DIODE MODEL ------------------------------------------------------------


# PVConstants
k = constants.k  #: [kJ/mole/K] Boltzmann constant
q = constants.e  #: [Coloumbs] elementary charge
E0 = 1000.  #: [W/m^2] irradiance of 1 sun
T0 = 298.15  #: [K] reference temperature
# Other Constants
EPS = np.finfo(np.float64).eps
VBYPASS = np.float64(-0.5)  # [V] trigger voltage of bypass diode
CELLAREA = np.float64(153.33)  # [cm^2] cell area


def two_diode_model(pvcs, Ee, u_cell_type, Tcell, NPTS=200, NPTS_cell=100,
                    use_cell_NPT=False, fname_pre='cell_data',
                    res_path=None):
    """
    Estimate IV curve using the two diode model native to pvmismatch.

    Parameters
    ----------
    pvcs : list
        List of different pvmismatch pvcell objects.
        Most modules have the same cells. Some might not.
    Ee : numpy.ndarray
        1-D array containing irradiances in suns.
    u_cell_type : list
        List of cell types at each irradiance setting.
    Tcell : numpy.ndarray
        1-D array containing cell temperatures in Kelvin.
    NPTS : int, optional
        Number of points in IV curve. The default is 200.
    NPTS_cell : int, optional
        Number of points in cell IV curve. The default is 100.
    use_cell_NPT : bool, optional
        Turn on this if cell IV curves use 'NPTS_cell'. The default is False.
    fname_pre : str, optional
        Pre-pend string for file name. The default is 'cell_data'.
    res_path : str, optional
        Folder where data will be stored. The default is None.
        If None, store data in cwd.

    Returns
    -------
    None.

    """
    # If res_path is None, store results in cwd
    if res_path is None:
        res_path = os.getcwd()
    # Generate NPTs
    pts, negpts, Imod_pts, Imod_negpts = NPTS_f(
        NPTS, np.repeat(Ee[:, np.newaxis], NPTS, axis=1).shape)
    if not use_cell_NPT:
        NPTS_cell = NPTS

    for ie, ee in enumerate(Ee):
        fname = '_'.join([fname_pre, str(ie)]) + '.pickle'
        fpath = os.path.join(res_path, fname)
        subcell_data = {}
        uct = int(u_cell_type[ie])
        pvc = pvcs[uct]
        # Define cell parameters
        Rs = pvc.Rs
        Rsh = pvc.Rsh
        Isat1_T0 = pvc.Isat1_T0
        Isat2_T0 = pvc.Isat2_T0
        Isc0_T0 = pvc.Isc0_T0
        aRBD = pvc.aRBD
        bRBD = pvc.bRBD
        VRBD = pvc.VRBD
        nRBD = pvc.nRBD
        Eg = pvc.Eg
        alpha_Isc = pvc.alpha_Isc
        # PVMismatch
        pvconst = pvconstants.PVconstants(npts=NPTS_cell)
        # create pvcell
        pvc = pvcell.PVcell(pvconst=pvconst, Rs=Rs,
                            Rsh=Rsh,
                            Isat1_T0=Isat1_T0,
                            Isat2_T0=Isat2_T0,
                            Isc0_T0=Isc0_T0,
                            aRBD=aRBD,
                            bRBD=bRBD,
                            VRBD=VRBD,
                            nRBD=nRBD,
                            Eg=Eg,
                            alpha_Isc=alpha_Isc)
        pvc.Ee = ee
        pvc.Tcell = Tcell[ie]

        # Store in a dict
        subcell_data['Icell'] = pvc.Icell.flatten()
        subcell_data['Vcell'] = pvc.Vcell.flatten()
        subcell_data['Pcell'] = pvc.Pcell.flatten()
        subcell_data['VRBD'] = pvc.VRBD
        subcell_data['Voc'] = pvc.Voc
        subcell_data['Isc'] = pvc.Isc
        # Save results
        save_pickle(fpath, subcell_data)
    NPT_fname = 'NPT_dict.pickle'
    NPT_fpath = os.path.join(res_path, NPT_fname)
    NPT_dict = dict()
    NPT_dict['pts'] = pts
    NPT_dict['negpts'] = negpts
    NPT_dict['Imod_pts'] = Imod_pts
    NPT_dict['Imod_negpts'] = Imod_negpts
    NPT_dict['Npts'] = NPTS
    # Save NPTS data
    save_pickle(NPT_fpath, NPT_dict)


def NPTS_f(Npts=200, vec_shape=(1, 1, 1, 1)):
    """
    Generate point spacing for IV curves for both cells and module.

    Parameters
    ----------
    Npts : int, optional
        Number of points in IV curve. The default is 200. The default is 200.
    vec_shape : tuple, optional
        tuple with system shape (num_str, str_len, row_cells, col_cells).
        The default is (1, 1, 1, 1).

    Returns
    -------
    pts_vec : numpy.ndarray
        Positive point spacing for cell IV curves.
    negpts_vec : numpy.ndarray
        Negative point spacing for cell IV curves.
    Imod_pts_vec : numpy.ndarray
        Positive point spacing for module IV curves.
    Imod_negpts_vec : numpy.ndarray
        Negative point spacing for module IV curves.

    """
    # set number of points in IV curve(s)
    # point spacing from 0 to 1, used for Vcell, Vmod, Vsys and Istring
    # decrease point spacing as voltage approaches Voc by using logspace
    pts_vec = np.zeros(vec_shape)
    negpts_vec = pts_vec.copy()
    Imod_pts_vec = pts_vec.copy()
    Imod_negpts_vec = pts_vec.copy()
    pts = (11. - np.logspace(np.log10(11.), 0., Npts)) / 10.
    pts[0] = 0.  # first point must be exactly zero
    pts_vec[:, :] = pts
    negpts = (11. - np.logspace(np.log10(11. - 1. / float(Npts)),
                                0., Npts)) / 10.
    Imod_negpts = 1 + 1. / float(Npts) / 10. - negpts
    Imod_negpts_vec[:, :] = Imod_negpts
    negpts = np.flipud(negpts)  # reverse the order
    negpts_vec[:, :] = negpts
    # shift and concatenate pvconst.negpts and pvconst.pts
    # so that tight spacing is around MPP and RBD
    Imod_pts = 1 - np.flipud(pts)
    Imod_pts_vec[:, :] = Imod_pts

    return pts_vec, negpts_vec, Imod_pts_vec, Imod_negpts_vec
