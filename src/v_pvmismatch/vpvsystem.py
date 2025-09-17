# -*- coding: utf-8 -*-
"""Vectorized pvsystem."""

import copy
import numpy as np
from .utils import (reshape_ndarray, isin_nd, calcMPP_IscVocFFBPD,
                    calcMPP_IscVocFF)
from .vpvstring import calcStrings
from .circuit_comb import calcParallel_with_bypass

# ------------------------------------------------------------------------------
# BUILD SYSTEM IV-PV CURVES----------------------------------------------------


def gen_sys_Ee_Tcell_array(sim_len, Num_mod_X, Num_mod_Y,
                           Num_cell_X, Num_cell_Y,
                           Ee=1., Tcell=298.15):
    """
    Generate system level irradiance and cell temperature arrays.

    Parameters
    ----------
    sim_len : int
        Length of the simulation or number of combinations.
    Num_mod_X : int
        Number of modules in a string.
    Num_mod_Y : int
        Number of strings in a system.
    Num_cell_X : int
        Number of columns in a module.
    Num_cell_Y : int
        Number of rows in a module.
    Ee : float or numpy.ndarray, optional
        Input irradiance value or array. The default is 1..
    Tcell : float or numpy.ndarray, optional
        Input cell temperature value or array. The default is 298.15.

    Returns
    -------
    Ee : numpy.ndarray
        System level irradiance array.
    Tcell : numpy.ndarray
        System level cell temperature array.

    """
    # Check if Ee is scalar
    if not (isinstance(Ee, np.ndarray)):
        Ee = Ee*np.ones((sim_len, Num_mod_X, Num_mod_Y,
                        Num_cell_X, Num_cell_Y))
    # Check if Ee is a 1d array with different Ee for each sim index
    elif len(Ee.shape) == 1:
        Ee = reshape_ndarray(
            Ee, (Num_mod_X, Num_mod_Y, Num_cell_X, Num_cell_Y))
    # Check if Tcell is scalar
    if not (isinstance(Tcell, np.ndarray)):
        Tcell = Tcell * \
            np.ones((sim_len, Num_mod_X, Num_mod_Y, Num_cell_X, Num_cell_Y))
    # Check if Tcell is a 1d array with different Ee for each sim index
    elif len(Tcell.shape) == 1:
        Tcell = reshape_ndarray(
            Tcell, (Num_mod_X, Num_mod_Y, Num_cell_X, Num_cell_Y))

    return Ee, Tcell


def get_unique_Ee(Ee, search_type='cell', cell_type=None):
    """
    Generate unique irradiance at the cell, module, string or system level.

    Parameters
    ----------
    Ee : numpy.ndarray
        System level irradiance array.
    search_type : str, optional
        Which type of unique irrad: 'cell', 'module', 'string', or 'system'.
        The default is 'cell'.
    cell_type : numpy.ndarray or None, optional
        Array containing the different cell types. The default is None.

    Returns
    -------
    u : numpy.ndarray
        Unique irradiance values.
    u_cell_type : numpy.ndarray or None
        Unique cell types sorted based on u but only for cell level search.

    """
    # Get the unique Ee for cell level
    if search_type == 'cell':
        if isinstance(cell_type, np.ndarray):
            u_ctype = np.unique(cell_type)
            u = []
            u_cell_type = []
            for uct in u_ctype:
                idx_ct = np.where(cell_type == uct)
                Ee_sub = Ee[:, :, :, idx_ct[0], idx_ct[1]]
                usub = np.unique(Ee_sub, axis=None)
                ctype = uct * np.ones(usub.shape)
                u.append(usub)
                u_cell_type.append(ctype)
            u = np.concatenate(u)
            u_cell_type = np.concatenate(u_cell_type)
        else:
            u = np.unique(Ee, axis=None)
            u_cell_type = 0 * np.ones(u.shape)
        cts = None
    elif search_type == 'module':
        Ee_shp = Ee.shape
        u_list = []
        for idx_0 in range(Ee_shp[0]):
            for idx_1 in range(Ee_shp[1]):
                u_list.append(np.unique(Ee[idx_0, idx_1, :], axis=0))
        u = np.concatenate(u_list)
        u, cts = np.unique(u, axis=0, return_counts=True)
        u_cell_type = None
    elif search_type == 'string':
        Ee_shp = Ee.shape
        u_list = []
        for idx_0 in range(Ee_shp[0]):
            u_list.append(np.unique(Ee[idx_0, :], axis=0))
        u = np.concatenate(u_list)
        u = np.unique(u, axis=0)
        u_cell_type = None
        cts = None
    elif search_type == 'system':
        u = np.unique(Ee, axis=0)
        u_cell_type = None
        cts = None
    else:
        print('Incorrect search_type. Allowed search_type: cell, module, string, system.')

    return u, u_cell_type, cts


def calcTimeSeries(Ee_vec, sys_data):
    """
    Generate IV curves for the entire simulation or all combinations.

    Parameters
    ----------
    Ee_vec : numpy.ndarray
        System level irradiance array.
    sys_data : dict
        System level IV curves.

    Returns
    -------
    time_data : dict
        Simulation level IV curves.

    """
    u, inverse, counts = np.unique(Ee_vec, axis=0, return_inverse=True,
                                   return_counts=True)
    time_data = dict()
    time_data['Isys'] = sys_data['Isys'][inverse]
    time_data['Vsys'] = sys_data['Vsys'][inverse]
    time_data['Psys'] = sys_data['Psys'][inverse]
    time_data['Imp'] = sys_data['Imp'][inverse]
    time_data['Vmp'] = sys_data['Vmp'][inverse]
    time_data['Pmp'] = sys_data['Pmp'][inverse]
    time_data['Isc'] = sys_data['Isc'][inverse]
    time_data['Voc'] = sys_data['Voc'][inverse]
    time_data['FF'] = sys_data['FF'][inverse]

    return time_data


def calcsubModuleSystem(Ee_vec, Ee_mod, mod_data, NPT_dict,
                        run_bpact=True, run_annual=False,
                        save_bpact_freq=False):
    """
    Generate the system-level IV curves for the unique systems in a simulation.

    This function is for Sub-module level MPPT systems.

    Parameters
    ----------
    Ee_vec : numpy.ndarray
        System level irradiance array.
    Ee_mod : numpy.ndarray
        3-D array containing the Irradiance at the cell level for all modules.
    mod_data : dict
        Dictionary containing module IV curves.
    NPT_dict : dict
        NPTs dictionary from the cell data dictionary generated by pvcell.
    run_bpact : bool, optional
        Flag to run bypass diode activation logic. The default is True.
    run_annual : bool, optional
        Flag to delete large BPD activation array for an annual simulation.
        The default is False.
    save_bpact_freq : bool, optional
        Flag to turn on saving bypass diode activation frequency input.
        The default is False.
    run_cellcurr : bool, optional
        Flag to run cell current estimation logic. The default is True.

    Returns
    -------
    sys_data : dict
        Dictionary containing system IV curves.

    """
    Ee_sys, _ = get_unique_Ee(Ee_vec, search_type='system')
    Ee_shp = Ee_sys.shape
    Pmp = np.zeros(Ee_shp[0])
    for idx_0 in range(Ee_shp[0]):
        for idx_1 in range(Ee_shp[1]):
            # 1 String
            Ee_str1 = Ee_sys[idx_0, idx_1, :, :, :]
            # Extract mod IV curves
            str_in_mod = isin_nd(Ee_mod, Ee_str1)
            u, inverse, counts = np.unique(Ee_str1, axis=0,
                                           return_inverse=True,
                                           return_counts=True)
            Pmp_red = np.zeros((len(str_in_mod),))
            for idx_red in range(str_in_mod.shape[0]):
                Imod_red = mod_data['Isubstr'][str_in_mod[idx_red], :, :]
                Vmod_red = mod_data['Vsubstr'][str_in_mod[idx_red], :, :]
                Pmod_red = mod_data['Psubstr'][str_in_mod[idx_red], :, :]
                Imp_s, Vmp_s, Pmp_s, Isc_s, Voc_s, FF_s = calcMPP_IscVocFF(
                    Imod_red, Vmod_red, Pmod_red)
                Pmp_red[idx_red] = Pmp_s.sum()
            Pmp_str = Pmp_red[inverse]
            Pmp[idx_0] += Pmp_str.sum()
    # Store results in a dict
    u, inverse, counts = np.unique(Ee_vec, axis=0, return_inverse=True,
                                   return_counts=True)
    Pmp_ts = Pmp[inverse]
    num_bpd_active_ts = 0
    BpDmp_full = np.zeros(Ee_vec.shape)
    sys_data = dict()
    sys_data['Isys'] = np.zeros(Pmp_ts.shape)
    sys_data['Vsys'] = np.zeros(Pmp_ts.shape)
    sys_data['Psys'] = np.zeros(Pmp_ts.shape)
    sys_data['Bypass_activation'] = np.zeros(Pmp_ts.shape)
    sys_data['Imp'] = np.zeros(Pmp_ts.shape)
    sys_data['Vmp'] = np.zeros(Pmp_ts.shape)
    sys_data['Pmp'] = Pmp_ts
    sys_data['Isc'] = np.zeros(Pmp_ts.shape)
    sys_data['Voc'] = np.zeros(Pmp_ts.shape)
    sys_data['FF'] = np.zeros(Pmp_ts.shape)
    sys_data['Bypass_Active_MPP'] = BpDmp_full
    sys_data['num_active_bpd'] = num_bpd_active_ts
    return sys_data


def calcACSystem(Ee_vec, Ee_mod, mod_data, NPT_dict,
                 run_bpact=True, run_annual=False, save_bpact_freq=False,
                 run_cellcurr=True):
    """
    Generate the system-level IV curves for the unique systems in a simulation.

    This function is for AC or MLPE systems.

    Parameters
    ----------
    Ee_vec : numpy.ndarray
        System level irradiance array.
    Ee_mod : numpy.ndarray
        3-D array containing the Irradiance at the cell level for all modules.
    mod_data : dict
        Dictionary containing module IV curves.
    NPT_dict : dict
        NPTs dictionary from the cell data dictionary generated by pvcell.
    run_bpact : bool, optional
        Flag to run bypass diode activation logic. The default is True.
    run_annual : bool, optional
        Flag to delete large BPD activation array for an annual simulation.
        The default is False.
    save_bpact_freq : bool, optional
        Flag to turn on saving bypass diode activation frequency input.
        The default is False.
    run_cellcurr : bool, optional
        Flag to run cell current estimation logic. The default is True.

    Returns
    -------
    sys_data : dict
        Dictionary containing system IV curves.

    """
    if run_cellcurr:
        full_data = [0] * Ee_vec.shape[0]
    Ee_sys, _ = get_unique_Ee(Ee_vec, search_type='system')
    Ee_shp = Ee_sys.shape
    if run_bpact:
        BpDmp = np.zeros((Ee_shp[0], Ee_shp[1], Ee_shp[2],
                          mod_data['BPDiode_Active_MPP'].shape[1]))
    Pmp = np.zeros(Ee_shp[0])
    if run_bpact:
        num_bpd_active = np.zeros(Ee_shp[0])
    else:
        num_bpd_active = 0
    if run_cellcurr:
        full_sys_data = []
    for idx_0 in range(Ee_shp[0]):
        if run_cellcurr:
            sim_data = []
        for idx_1 in range(Ee_shp[1]):
            # 1 String
            Ee_str1 = Ee_sys[idx_0, idx_1, :, :, :]
            if run_cellcurr:
                string_data = [0] * Ee_str1.shape[0]
            # Extract mod IV curves
            str_in_mod = isin_nd(Ee_mod, Ee_str1)
            u, inverse, counts = np.unique(Ee_str1, axis=0,
                                           return_inverse=True,
                                           return_counts=True)
            if run_cellcurr:
                str_data_red = [0] * len(str_in_mod)
            Pmp_red = np.zeros((len(str_in_mod),))
            if run_bpact:
                num_bpd_active_red = np.zeros((len(str_in_mod),))
                BpDmp_red = np.zeros((len(str_in_mod), BpDmp.shape[-1]))
            for idx_red in range(str_in_mod.shape[0]):
                if run_cellcurr:
                    module_data = {}
                    module_data['full_data'] = mod_data['full_data'][str_in_mod[idx_red]]
                Ee_mod_red = Ee_mod[str_in_mod[idx_red], :, :]
                Ee_str_red = np.stack([Ee_mod_red], axis=0)
                Ee_str_red = np.stack([Ee_str_red], axis=0)
                Ee_sys_red = np.stack([Ee_str_red], axis=0)
                # String calculation
                str_data = calcStrings(Ee_str_red, Ee_mod, mod_data, NPT_dict,
                                       run_bpact=run_bpact,
                                       run_cellcurr=run_cellcurr)
                # System calculation
                sys_data1 = calcSystem(
                    Ee_sys_red, Ee_str_red, str_data, NPT_dict,
                    run_bpact=run_bpact, run_cellcurr=run_cellcurr)
                if run_cellcurr:
                    module_data['BPDMPP'] = sys_data1['Bypass_Active_MPP'][0, 0, :, :].copy(
                    )
                    module_data['Imp'] = sys_data1['Imp'].copy()
                    module_data['Vmp'] = sys_data1['Vmp'].copy()
                    module_data['Isc'] = sys_data1['Isc'].copy()
                    str_data_red[idx_red] = copy.deepcopy(module_data)
                Pmp_red[idx_red] = sys_data1['Pmp'][0]
                if run_bpact:
                    num_bpd_active_red[idx_red] = sys_data1['num_active_bpd'][0]
                    if save_bpact_freq:
                        BpDmp_red[idx_red,
                                  :] = sys_data1['Bypass_Active_MPP'][0, 0, :, :]
            Pmp_str = Pmp_red[inverse]
            if run_cellcurr:
                for idx_mod, idx_inv in enumerate(inverse.tolist()):
                    string_data[idx_mod] = str_data_red[idx_inv]
                sim_data.append(string_data)
            Pmp[idx_0] += Pmp_str.sum()
            if run_bpact:
                num_bpd_active_str = num_bpd_active_red[inverse]
                num_bpd_active[idx_0] += num_bpd_active_str.sum()
                if save_bpact_freq:
                    BpDmp[idx_0, idx_1, :, :] = BpDmp_red[inverse, :]
        if run_cellcurr:
            full_sys_data.append(sim_data)
    # Store results in a dict
    u, inverse, counts = np.unique(Ee_vec, axis=0, return_inverse=True,
                                   return_counts=True)
    Pmp_ts = Pmp[inverse]
    if run_bpact:
        num_bpd_active_ts = num_bpd_active[inverse]
        if save_bpact_freq:
            BpDmp_full = BpDmp[inverse, :, :, :]
        else:
            BpDmp_full = np.zeros(Ee_vec.shape)
    else:
        num_bpd_active_ts = 0
        BpDmp_full = np.zeros(Ee_vec.shape)
    if run_cellcurr:
        for idx_sim, idx_inv in enumerate(inverse.tolist()):
            full_data[idx_sim] = full_sys_data[idx_inv]
    sys_data = dict()
    sys_data['Isys'] = np.zeros(Pmp_ts.shape)
    sys_data['Vsys'] = np.zeros(Pmp_ts.shape)
    sys_data['Psys'] = np.zeros(Pmp_ts.shape)
    sys_data['Bypass_activation'] = np.zeros(Pmp_ts.shape)
    sys_data['Imp'] = np.zeros(Pmp_ts.shape)
    sys_data['Vmp'] = np.zeros(Pmp_ts.shape)
    sys_data['Pmp'] = Pmp_ts
    sys_data['Isc'] = np.zeros(Pmp_ts.shape)
    sys_data['Voc'] = np.zeros(Pmp_ts.shape)
    sys_data['FF'] = np.zeros(Pmp_ts.shape)
    sys_data['Bypass_Active_MPP'] = BpDmp_full
    sys_data['num_active_bpd'] = num_bpd_active_ts
    if run_cellcurr:
        sys_data['full_data'] = copy.deepcopy(full_data)

    return sys_data


def calcSystem(Ee_sys, Ee_str, str_data, NPT_dict,
               run_bpact=True, run_annual=False, run_cellcurr=False):
    """
    Generate the system-level IV curves for the unique systems in a simulation.

    This function is for DC string systems.

    Parameters
    ----------
    Ee_sys : numpy.ndarray
        System level irradiance array.
    Ee_str : numpy.ndarray
        4-D Irrad array at the cell level for all modules in each string.
    str_data : dict
        Dictionary containing string IV curves.
    NPT_dict : dict
        NPTs dictionary from the cell data dictionary generated by pvcell.
    run_bpact : bool, optional
        Flag to run bypass diode activation logic. The default is True.
    run_annual : bool, optional
        Flag to delete BPD activation array for an annual simulation.
        The default is False.
    run_cellcurr : bool, optional
        Flag to run cell current estimation logic. The default is True.

    Returns
    -------
    sys_data : dict
        Dictionary containing system IV curves.

    """
    I_sys_curves = []
    V_sys_curves = []
    P_sys_curves = []
    Bypass_str_curves = []
    if run_cellcurr:
        full_data = []
    for idx_sys in range(Ee_sys.shape[0]):
        if run_cellcurr:
            sing_sys = {}

        # 1 String
        Ee_sys1 = Ee_sys[idx_sys]
        # Extract mod IV curves
        sys_in_str = isin_nd(Ee_str, Ee_sys1)
        Istr_red = str_data['Istring'][sys_in_str, :]
        Vstr_red = str_data['Vstring'][sys_in_str, :]
        if run_bpact:
            Bypass_substr_red = str_data['Bypassed_substr'][sys_in_str, :, :]
        else:
            Bypass_substr_red = np.nan
        u, inverse, counts = np.unique(Ee_sys1, axis=0, return_inverse=True,
                                       return_counts=True)
        # Expand for Str curves
        Istring = Istr_red[inverse, :]
        Vstring = Vstr_red[inverse, :]
        if run_cellcurr:
            sing_sys['Istrings'] = Istring.copy()
            sing_sys['Vstrings'] = Vstring.copy()
            sing_sys['Str_idxs'] = sys_in_str[inverse]
        if run_bpact:
            Bypass_substr = Bypass_substr_red[inverse, :, :]
        else:
            Bypass_substr = np.nan
        pts = NPT_dict['pts'][0, :].reshape(NPT_dict['pts'].shape[1], 1)
        negpts = NPT_dict['negpts'][0, :].reshape(
            NPT_dict['negpts'].shape[1], 1)
        Npts = NPT_dict['Npts']
        # Run System Circuit model
        Isys, Vsys, bypassed_str = calcParallel_with_bypass(
            Istring, Vstring, Vstring.max(), Vstring.min(), negpts, pts, Npts,
            Bypass_substr, run_bpact=run_bpact)
        Psys = Isys * Vsys

        I_sys_curves.append(np.reshape(Isys, (1, len(Isys))))
        V_sys_curves.append(np.reshape(Vsys, (1, len(Vsys))))
        P_sys_curves.append(np.reshape(Psys, (1, len(Psys))))
        if run_bpact:
            Bypass_str_curves.append(np.reshape(bypassed_str, (1,
                                                bypassed_str.shape[0],
                                                bypassed_str.shape[1],
                                                bypassed_str.shape[2],
                                                bypassed_str.shape[3])))
        else:
            Bypass_str_curves.append(bypassed_str)
        if run_cellcurr:
            full_data.append(sing_sys)

    I_sys_curves = np.concatenate(I_sys_curves, axis=0)
    V_sys_curves = np.concatenate(V_sys_curves, axis=0)
    P_sys_curves = np.concatenate(P_sys_curves, axis=0)
    if run_bpact:
        Bypass_str_curves = np.concatenate(Bypass_str_curves, axis=0)
    else:
        Bypass_str_curves = np.array(Bypass_str_curves)
    # If it is an annual simulation, delete the bypassed related variables
    if run_annual:
        del str_data['Bypassed_substr']

    Imp, Vmp, Pmp, Isc, Voc, FF, BpDmp, num_bpd_active = calcMPP_IscVocFFBPD(
        I_sys_curves, V_sys_curves, P_sys_curves, Bypass_str_curves,
        run_bpact=run_bpact, run_annual=run_annual)

    # Store results in a dict
    sys_data = dict()
    sys_data['Isys'] = I_sys_curves
    sys_data['Vsys'] = V_sys_curves
    sys_data['Psys'] = P_sys_curves
    if run_annual:
        sys_data['Bypass_activation'] = np.nan
        del Bypass_str_curves
    else:
        sys_data['Bypass_activation'] = Bypass_str_curves
    sys_data['Imp'] = Imp
    sys_data['Vmp'] = Vmp
    sys_data['Pmp'] = Pmp
    sys_data['Isc'] = Isc
    sys_data['Voc'] = Voc
    sys_data['FF'] = FF
    sys_data['Bypass_Active_MPP'] = BpDmp
    sys_data['num_active_bpd'] = num_bpd_active
    if run_cellcurr:
        sys_data['full_data'] = copy.deepcopy(full_data)

    return sys_data
