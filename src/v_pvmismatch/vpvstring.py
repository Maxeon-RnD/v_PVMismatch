# -*- coding: utf-8 -*-
"""Vectorized pvstring."""

import copy
import numpy as np
from .utils import isin_nd
from .circuit_comb import calcSeries_with_bypass


def calcStrings(Ee_str, Ee_mod, mod_data, NPT_dict,
                run_bpact=True, run_annual=False, run_cellcurr=True):
    """
    Generate all string IV curves and store results in a dictionary.

    Parameters
    ----------
    Ee_str : numpy.ndarray
        4-D Irradiance array at the cell level for all modules in each string.
    Ee_mod : numpy.ndarray
        3-D array containing the Irradiance at the cell level for all modules.
    mod_data : dict
        Dictionary containing module IV curves.
    NPT_dict : numpy.ndarray
        NPTs dictionary from the cell data dictionary generated by pvcell.
    run_bpact : bool, optional
        Flag to run bypass diode activation logic. The default is True.
    run_annual : bool, optional
        Flag to delete large BPD activation array for an annual simulation.
        The default is False.
    run_cellcurr : bool, optional
        Flag to run cell current estimation logic. The default is True.

    Returns
    -------
    str_data : dict
        Dictionary containing string IV curves.

    """
    I_str_curves = []
    V_str_curves = []
    P_str_curves = []
    Bypass_str_curves = []
    if run_cellcurr:
        full_data = []
    for idx_str in range(Ee_str.shape[0]):
        if run_cellcurr:
            sing_str = {}

        # 1 String
        Ee_str1 = Ee_str[idx_str]
        # Extract mod IV curves
        str_in_mod = isin_nd(Ee_mod, Ee_str1)
        Imod_red = mod_data['Imod'][str_in_mod, :]
        Vmod_red = mod_data['Vmod'][str_in_mod, :]
        meanIsc_red = mod_data['mean_Isc'][str_in_mod]
        if run_bpact:
            Bypassed_substr_red = mod_data['Bypassed_substr'][str_in_mod, :, :]
        else:
            Bypassed_substr_red = np.nan
        u, inverse, counts = np.unique(Ee_str1, axis=0, return_inverse=True,
                                       return_counts=True)
        # Expand for Str curves
        Imod = Imod_red[inverse, :]
        Vmod = Vmod_red[inverse, :]
        meanIsc = meanIsc_red[inverse]
        if run_cellcurr:
            sing_str['Imods'] = Imod.copy()
            sing_str['Vmods'] = Vmod.copy()
            sing_str['Mod_idxs'] = str_in_mod[inverse].copy()
        if run_bpact:
            Bypassed_substr = Bypassed_substr_red[inverse, :, :]
        else:
            Bypassed_substr = np.nan
        Imod_pts = NPT_dict['Imod_pts'][0, :].reshape(
            NPT_dict['Imod_pts'].shape[1], 1)
        Imod_negpts = NPT_dict['Imod_negpts'][0, :].reshape(
            NPT_dict['Imod_negpts'].shape[1], 1)
        Npts = NPT_dict['Npts']
        # Run String Circuit model
        Istring, Vstring, bypassed_str = calcSeries_with_bypass(
            Imod, Vmod, meanIsc.mean(), Imod.max(), Imod_pts, Imod_negpts,
            Npts, Bypassed_substr, run_bpact=run_bpact)
        Pstring = Istring * Vstring
        # If it is an annual simulation, delete the bypassed related variables
        if run_annual:
            del Bypassed_substr_red
            del Bypassed_substr

        I_str_curves.append(np.reshape(Istring, (1, len(Istring))))
        V_str_curves.append(np.reshape(Vstring, (1, len(Vstring))))
        P_str_curves.append(np.reshape(Pstring, (1, len(Pstring))))
        if run_bpact:
            Bypass_str_curves.append(np.reshape(bypassed_str,
                                                (1,
                                                 bypassed_str.shape[0],
                                                 bypassed_str.shape[1],
                                                 bypassed_str.shape[2])))
        else:
            Bypass_str_curves.append(bypassed_str)
        if run_cellcurr:
            full_data.append(sing_str)

    I_str_curves = np.concatenate(I_str_curves, axis=0)
    V_str_curves = np.concatenate(V_str_curves, axis=0)
    P_str_curves = np.concatenate(P_str_curves, axis=0)
    if run_bpact:
        Bypass_str_curves = np.concatenate(Bypass_str_curves, axis=0)
    else:
        Bypass_str_curves = np.array(Bypass_str_curves)
    # If it is an annual simulation, delete the bypassed related variables
    if run_annual:
        del mod_data['Bypassed_substr']

    # Store results in a dict
    str_data = dict()
    str_data['Istring'] = I_str_curves
    str_data['Vstring'] = V_str_curves
    str_data['Pstring'] = P_str_curves
    str_data['Bypassed_substr'] = Bypass_str_curves
    if run_cellcurr:
        str_data['full_data'] = copy.deepcopy(full_data)

    return str_data
