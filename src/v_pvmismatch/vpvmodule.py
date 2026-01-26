# -*- coding: utf-8 -*-
"""Vectorized pvmodule."""

import os
import pickle
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import numpy as np
import pandas as pd
from .pvmismatch import pvconstants
from .utils import (save_pickle, calcMPP_IscVocFFBPD)
from .circuit_comb import calcSeries, calcParallel
from .circuit_comb import combine_parallel_circuits, parse_diode_config
from .circuit_comb import calcSeries_with_bypass, calcParallel_with_bypass
from .circuit_comb import DEFAULT_BYPASS, MODULE_BYPASS, CUSTOM_SUBSTR_BYPASS
# ----------------------------------------------------------------------------
# CALCULATE MODULE IV-PV CURVES-----------------------------------------------


def calcMods(cell_pos, maxmod, cell_index_map, Ee_mod, Tcell_mod, Ee_cell,
             Tcell_cell, u_cell_type, cell_type,
             outer_circuit, run_bpact=True, run_cellcurr=False,
             cfname_pre='cell_data', res_path=None, mfname_pre='mod_data'):
    """
    Generate all module IV curves and store results in a dictionary.

    Parameters
    ----------
    cell_pos : dict
        cell position pattern from pvmismatch package.
    maxmod : pvmodule object
        pvmodule class from pvmismatch package.
    cell_index_map : numpy.ndarray
        2-D array specifying the physical cell positions in the module.
    Ee_mod : numpy.ndarray
        3-D array containing the Irradiance at the cell level for all modules.
    Ee_cell : numpy.ndarray
        1-D array containing irradiances in suns.
    u_cell_type : list
        List of cell types at each irradiance setting.
    cell_type : numpy.ndarray
        2-D array of cell types for each cell in each module.
    outer_circuit : str
        series or parallel.
    run_bpact : bool, optional
        Flag to run bypass diode activation logic. The default is True.
    run_cellcurr : bool, optional
        Flag to run cell current estimation logic. The default is True.
    cfname_pre : str, optional
        Pre-pend string for cell data file name. The default is 'cell_data'.
    res_path : str, optional
        Folder where data will be stored. The default is None.
        If None, store data in cwd.
    mfname_pre : str, optional
        Pre-pend string for module data file name. The default is 'mod_data'.

    Returns
    -------
    NPT_dict : dict
        Dictionary containing NPTS curves.

    """
    # If res_path is None, store results in cwd
    if res_path is None:
        res_path = os.getcwd()
    Vbypass = maxmod.Vbypass
    # --- Define cell position pattern and sort indices ---
    cell_ids = cell_index_map.flatten()
    idx_sort = np.argsort(cell_ids)
    pre_idx = prepare_global_cell_key_index(
        Ee_cell=Ee_cell,        # 1-D unique Ee from get_unique_Ee(..., 'cell')
        Tcell_cell=Tcell_cell,  # 1-D unique Tcell aligned with Ee_cell
        u_cell_type=u_cell_type,  # 1-D unique cell types aligned
        round_decimals=None
    )

    for idx_mod in range(Ee_mod.shape[0]):
        # 1 Module
        # --- Flatten and sort module-level data for this pattern ---
        Ee_mod1 = Ee_mod[idx_mod].flatten()[idx_sort]
        Tcell_mod1 = Tcell_mod[idx_mod].flatten()[idx_sort]   # include Tcell
        cell_type1 = cell_type.flatten()[idx_sort]

        # Extract cell IV curves
        # Expand for Mod curves
        Icell, Vcell, VRBD, Voc, Isc, counts_mod, inverse_mod, NPT_dict = \
            extract_module_cell_curves_vectorized_preindexed_from_pickles(
                Ee_mod1=Ee_mod1,
                Tcell_mod1=Tcell_mod1,
                cell_type1=cell_type1,
                idx_sort=idx_sort,
                idx_mod=idx_mod,
                res_path=res_path,
                cfname_pre=cfname_pre,
                pre_idx=pre_idx
            )
        # Run Module Circuit model
        sing_mod = calcMod(Ee_mod1, Icell, Vcell, VRBD, Voc, Isc,
                           cell_pos, Vbypass, NPT_dict,
                           outer=outer_circuit,
                           run_bpact=run_bpact,
                           run_cellcurr=run_cellcurr)
        # Calc MPP
        if run_bpact:
            bypassed_mod = sing_mod['bypassed_mod'][np.newaxis, :, :]
        else:
            bypassed_mod = sing_mod['bypassed_mod']
        Imp, Vmp, Pmp, Isc, Voc, FF, BpDmp, num_bpd_active = calcMPP_IscVocFFBPD(
            sing_mod['Imod'][np.newaxis, :], sing_mod['Vmod'][np.newaxis, :],
            sing_mod['Pmod'][np.newaxis, :],
            bypassed_mod,
            run_bpact=run_bpact, run_annual=False)
        sing_mod['Imp'] = Imp
        sing_mod['Vmp'] = Vmp
        sing_mod['Pmp'] = Pmp
        sing_mod['Isc'] = Isc
        sing_mod['Voc'] = Voc
        sing_mod['FF'] = FF
        sing_mod['BpDmp'] = BpDmp
        sing_mod['num_bpd_active'] = num_bpd_active
        # Save results to pickle
        fname = '_'.join([mfname_pre, str(idx_mod)]) + '.pickle'
        fpath = os.path.join(res_path, fname)
        save_pickle(fpath, sing_mod)
    return NPT_dict


def calcMod(Ee_mod, Icell, Vcell, VRBD, Voc, Isc, cell_pos, Vbypass,
            NPT_dict, outer='series', run_bpact=True, run_cellcurr=True,
            ):
    """
    Calculate module I-V curves.

    Returns module currents [A], voltages [V] and powers [W]
    """
    # Extract Npt data
    pts = NPT_dict['pts'][0, :].reshape(NPT_dict['pts'].shape[1], 1)
    negpts = NPT_dict['negpts'][0, :].reshape(NPT_dict['negpts'].shape[1], 1)
    Imod_pts = NPT_dict['Imod_pts'][0, :].reshape(
        NPT_dict['Imod_pts'].shape[1], 1)
    Imod_negpts = NPT_dict['Imod_negpts'][0, :].reshape(
        NPT_dict['Imod_negpts'].shape[1], 1)
    Npts = NPT_dict['Npts']
    sing_mod = {}
    # iterate over substrings
    Isubstr, Vsubstr, Isc_substr, Imax_substr = [], [], [], []
    Isubstr_pre_bypass, Vsubstr_pre_bypass = [], []
    substr_bypass = []
    for substr_idx, substr in enumerate(cell_pos):
        if run_cellcurr:
            sing_mod[substr_idx] = {}
        # check if cells are in series or any crosstied circuits
        if all(r['crosstie'] == False for c in substr for r in c):
            if run_cellcurr:
                ss_s_ct = 0
                ss_p_ct = 0
                sing_mod[substr_idx][ss_s_ct] = {}
                sing_mod[substr_idx][ss_s_ct][ss_p_ct] = {}
            idxs = [r['idx'] for c in substr for r in c]
            IatVrbd = np.asarray(
                [np.interp(vrbd, v, i) for vrbd, v, i in
                 zip(VRBD[idxs], Vcell[idxs], Icell[idxs])]
            )
            Isub, Vsub = calcSeries(
                Icell[idxs], Vcell[idxs], Isc[idxs].mean(),
                IatVrbd.max(), Imod_pts, Imod_negpts, Npts
            )
            if run_cellcurr:
                sing_mod[substr_idx][ss_s_ct]['Isubstr'] = Isub.copy()
                sing_mod[substr_idx][ss_s_ct]['Vsubstr'] = Vsub.copy()
                sing_mod[substr_idx][ss_s_ct][
                    ss_p_ct]['cell_currents'] = Icell[idxs].copy()
                sing_mod[substr_idx][ss_s_ct][
                    ss_p_ct]['cell_voltages'] = Vcell[idxs].copy()
                sing_mod[substr_idx][ss_s_ct][
                    ss_p_ct]['cell_idxs'] = copy.deepcopy(
                    idxs)
                sing_mod[substr_idx][ss_s_ct][
                    ss_p_ct]['Isubstr'] = Isub.copy()
                sing_mod[substr_idx][ss_s_ct][
                    ss_p_ct]['Vsubstr'] = Vsub.copy()
        elif all(r['crosstie'] == True for c in substr for r in c):
            Irows, Vrows = [], []
            Isc_rows, Imax_rows = [], []
            for row in zip(*substr):
                idxs = [c['idx'] for c in row]
                Irow, Vrow = calcParallel(
                    Icell[idxs], Vcell[idxs],
                    Voc[idxs].max(), VRBD.min(), negpts, pts, Npts
                )
                Irows.append(Irow)
                Vrows.append(Vrow)
                Isc_rows.append(np.interp(np.float64(0), Vrow, Irow))
                Imax_rows.append(Irow.max())
            Irows, Vrows = np.asarray(Irows), np.asarray(Vrows)
            Isc_rows = np.asarray(Isc_rows)
            Imax_rows = np.asarray(Imax_rows)
            Isub, Vsub = calcSeries(
                Irows, Vrows, Isc_rows.mean(), Imax_rows.max(),
                Imod_pts, Imod_negpts, Npts
            )
        else:
            IVall_cols = []
            prev_col = None
            IVprev_cols = []
            idxsprev_cols = []
            ss_s_ct = 0
            for col in substr:
                IVcols = []
                IV_idxs = []
                is_first = True
                ss_p_ct = 0
                # combine series between crossties
                for idxs in pvconstants.get_series_cells(col, prev_col):
                    if not idxs:
                        # first row should always be empty since it must be
                        # crosstied
                        is_first = False
                        continue
                    elif is_first:
                        raise Exception(
                            "First row and last rows must be crosstied."
                        )
                    elif len(idxs) > 1:
                        IatVrbd = np.asarray(
                            [np.interp(vrbd, v, i) for vrbd, v, i in
                             zip(VRBD[idxs], Vcell[idxs],
                                 Icell[idxs])]
                        )
                        Icol, Vcol = calcSeries(
                            Icell[idxs], Vcell[idxs],
                            Isc[idxs].mean(), IatVrbd.max(),
                            Imod_pts, Imod_negpts, Npts
                        )
                    else:
                        Icol = Icell[idxs]
                        Vcol = Vcell[idxs]
                    IVcols.append([Icol, Vcol])
                    IV_idxs.append(np.array(idxs))
                # append IVcols and continue
                IVprev_cols.append(IVcols)
                idxsprev_cols.append(IV_idxs)
                if prev_col:
                    # if circuits are same in both columns then continue
                    if not all(icol['crosstie'] == jcol['crosstie']
                               for icol, jcol in zip(prev_col, col)):
                        # combine crosstied circuits
                        Iparallel, Vparallel, sub_str_data = combine_parallel_circuits(
                            IVprev_cols, pvconstants,
                            negpts, pts, Imod_pts, Imod_negpts, Npts,
                            idxsprev_cols
                        )
                        IVall_cols.append([Iparallel, Vparallel])
                        # reset prev_col
                        prev_col = None
                        IVprev_cols = []
                        continue
                # set prev_col and continue
                prev_col = col
            # combine any remaining crosstied circuits in substring
            if not IVall_cols:
                # combine crosstied circuits
                Isub, Vsub, sub_str_data = combine_parallel_circuits(
                    IVprev_cols, pvconstants,
                    negpts, pts, Imod_pts, Imod_negpts, Npts, idxsprev_cols
                )
                if run_cellcurr:
                    for ss_s_ct in range(sub_str_data['Irows'].shape[0]):
                        sing_mod[substr_idx][ss_s_ct] = {}
                        sing_mod[substr_idx][ss_s_ct]['Isubstr'] = sub_str_data['Irows'][ss_s_ct, :].copy(
                        )
                        sing_mod[substr_idx][ss_s_ct]['Vsubstr'] = sub_str_data['Vrows'][ss_s_ct, :].copy(
                        )
                        for ss_p_ct in range(
                                sub_str_data['Iparallels'][ss_s_ct].shape[0]):
                            sing_mod[substr_idx][ss_s_ct][ss_p_ct] = {}
                            sing_mod[substr_idx][ss_s_ct][ss_p_ct]['Isubstr'] = sub_str_data['Iparallels'][ss_s_ct][ss_p_ct, :].copy(
                            )
                            sing_mod[substr_idx][ss_s_ct][ss_p_ct]['Vsubstr'] = sub_str_data['Vparallels'][ss_s_ct][ss_p_ct, :].copy(
                            )
                            idxs = sub_str_data['idxparallels'][ss_s_ct][ss_p_ct, :].tolist(
                            )
                            sing_mod[substr_idx][ss_s_ct][ss_p_ct]['cell_idxs'] = copy.deepcopy(
                                idxs)
                            sing_mod[substr_idx][ss_s_ct][ss_p_ct]['cell_currents'] = Icell[idxs]
                            sing_mod[substr_idx][ss_s_ct][ss_p_ct]['cell_voltages'] = Vcell[idxs]
            else:
                Iparallel, Vparallel = zip(*IVall_cols)
                Iparallel = np.asarray(Iparallel)
                Vparallel = np.asarray(Vparallel)
                Isub, Vsub = calcParallel(
                    Iparallel, Vparallel, Vparallel.max(), Vparallel.min(),
                    negpts, pts, Npts
                )

        if run_cellcurr:
            sing_mod[substr_idx]['Idiode_pre'] = Isub.copy()
            sing_mod[substr_idx]['Vdiode_pre'] = Vsub.copy()
        Isubstr_pre_bypass.append(Isub.copy())
        Vsubstr_pre_bypass.append(Vsub.copy())

        Vbypass_config = parse_diode_config(Vbypass, cell_pos)
        if Vbypass_config == DEFAULT_BYPASS:
            bypassed = Vsub < Vbypass
            Vsub[bypassed] = Vbypass
        elif Vbypass_config == CUSTOM_SUBSTR_BYPASS:
            if Vbypass[substr_idx] is None:
                # no bypass for this substring
                bypassed = np.zeros(Vsub.shape, dtype=bool)
                pass
            else:
                # bypass the substring
                bypassed = Vsub < Vbypass[substr_idx]
                Vsub[bypassed] = Vbypass[substr_idx]
        elif Vbypass_config == MODULE_BYPASS:
            # module bypass value assigned after loop for substrings is over.
            bypassed = np.zeros(Vsub.shape, dtype=bool)
            pass

        if run_cellcurr:
            sing_mod[substr_idx]['Idiode'] = Isub.copy()
            sing_mod[substr_idx]['Vdiode'] = Vsub.copy()
        Isubstr.append(Isub)
        Vsubstr.append(Vsub)
        Isc_substr.append(np.interp(np.float64(0), Vsub, Isub))
        Imax_substr.append(Isub.max())
        substr_bypass.append(bypassed)

    Isubstr, Vsubstr = np.asarray(Isubstr), np.asarray(Vsubstr)
    substr_bypass = np.asarray(substr_bypass)
    Isubstr_pre_bypass = np.asarray(Isubstr_pre_bypass)
    Vsubstr_pre_bypass = np.asarray(Vsubstr_pre_bypass)
    Isc_substr = np.asarray(Isc_substr)
    Imax_substr = np.asarray(Imax_substr)
    if outer == 'series':
        Imod, Vmod, bypassed_mod = calcSeries_with_bypass(
            Isubstr, Vsubstr, Isc_substr.mean(), Imax_substr.max(),
            Imod_pts, Imod_negpts, Npts, substr_bypass, run_bpact=run_bpact
        )
    else:
        Imod, Vmod, bypassed_mod = calcParallel_with_bypass(
            Isubstr, Vsubstr, Vsubstr.max(), Vsubstr.min(),
            Imod_negpts, Imod_pts, Npts, substr_bypass, run_bpact=run_bpact)

    # if entire module has only one bypass diode
    if Vbypass_config == MODULE_BYPASS:
        if run_cellcurr:
            sing_mod[substr_idx]['Idiode_pre'] = Imod.copy()
            sing_mod[substr_idx]['Vdiode_pre'] = Vmod.copy()
        bypassed = Vmod < Vbypass[0]
        Vmod[bypassed] = Vbypass[0]
        if run_cellcurr:
            sing_mod[substr_idx]['Idiode'] = Imod.copy()
            sing_mod[substr_idx]['Vdiode'] = Vmod.copy()
        bypassed_mod = bypassed[np.newaxis, ...]
    else:
        pass

    Pmod = Imod * Vmod
    sing_mod['Imod'] = Imod.copy()
    sing_mod['Vmod'] = Vmod.copy()
    sing_mod['Pmod'] = Pmod.copy()
    sing_mod['Isubstr'] = Isubstr.copy()
    sing_mod['Vsubstr'] = Vsubstr.copy()
    sing_mod['Isc'] = Isc.mean()
    sing_mod['Isubstr_pre_bypass'] = Isubstr_pre_bypass.copy()
    sing_mod['Vsubstr_pre_bypass'] = Vsubstr_pre_bypass.copy()
    if run_bpact:
        sing_mod['bypassed_mod'] = bypassed_mod.copy()
    else:
        sing_mod['bypassed_mod'] = bypassed_mod

    return sing_mod


def calcsubMods(cell_pos, maxmod, cell_index_map, Ee_mod, Tcell_mod,
                Ee_cell, u_cell_type, Tcell_cell, cell_type,
                cfname_pre='cell_data', res_path=None, mfname_pre='mod_data'):
    """
    Generate all sub-module IV curves and store results in a dictionary.

    Parameters
    ----------
    cell_pos : dict
        cell position pattern from pvmismatch package.
    maxmod : pvmodule object
        pvmodule class from pvmismatch package.
    cell_index_map : numpy.ndarray
        2-D array specifying the physical cell positions in the module.
    Ee_mod : numpy.ndarray
        3-D array containing the Irradiance at the cell level for all modules.
    Ee_cell : numpy.ndarray
        1-D array containing irradiances in suns.
    u_cell_type : list
        List of cell types at each irradiance setting.
    cell_type : numpy.ndarray
        2-D array of cell types for each cell in each module.
    cell_data : dict
        Dictionary containing cell IV curves.

    Returns
    -------
    mod_data : dict
        Dictionary containing module IV curves.

    """
    Vbypass = maxmod.Vbypass
    # --- Define cell position pattern and sort indices ---
    cell_ids = cell_index_map.flatten()
    idx_sort = np.argsort(cell_ids)
    pre_idx = prepare_global_cell_key_index(
        Ee_cell=Ee_cell,        # 1-D unique Ee from get_unique_Ee(..., 'cell')
        Tcell_cell=Tcell_cell,  # 1-D unique Tcell aligned with Ee_cell
        u_cell_type=u_cell_type,  # 1-D unique cell types aligned
        round_decimals=None
    )
    for idx_mod in range(Ee_mod.shape[0]):
        # 1 Module
        Ee_mod1 = Ee_mod[idx_mod].flatten()[idx_sort]
        Tcell_mod1 = Tcell_mod[idx_mod].flatten()[idx_sort]   # include Tcell
        cell_type1 = cell_type.flatten()[idx_sort]
        # Extract cell IV curves
        # Expand for Mod curves
        Icell, Vcell, VRBD, Voc, Isc, counts_mod, inverse_mod, NPT_dict = \
            extract_module_cell_curves_vectorized_preindexed_from_pickles(
                Ee_mod1=Ee_mod1,
                Tcell_mod1=Tcell_mod1,
                cell_type1=cell_type1,
                idx_sort=idx_sort,
                idx_mod=idx_mod,
                res_path=res_path,
                cfname_pre=cfname_pre,
                pre_idx=pre_idx
            )
        # Run Module Circuit model
        Isubstr, Vsubstr, Psubstr, mean_Isc = calcsubMod(Icell, Vcell, VRBD,
                                                         Voc, Isc, cell_pos,
                                                         Vbypass, NPT_dict)
        sing_mod = {}
        sing_mod['Isubstr'] = Isubstr
        sing_mod['Vsubstr'] = Vsubstr
        sing_mod['Psubstr'] = Psubstr
        sing_mod['mean_Isc'] = mean_Isc
        # Save results to pickle
        fname = '_'.join([mfname_pre, str(idx_mod)]) + '.pickle'
        fpath = os.path.join(res_path, fname)
        save_pickle(fpath, sing_mod)
    return NPT_dict


def calcsubMod(Icell, Vcell, VRBD, Voc, Isc, cell_pos, Vbypass, NPT_dict):
    """
    Calculate module I-V curves.

    Returns module currents [A], voltages [V] and powers [W]
    """
    # Extract Npt data
    pts = NPT_dict['pts'][0, :].reshape(NPT_dict['pts'].shape[1], 1)
    negpts = NPT_dict['negpts'][0, :].reshape(NPT_dict['negpts'].shape[1], 1)
    Imod_pts = NPT_dict['Imod_pts'][0, :].reshape(
        NPT_dict['Imod_pts'].shape[1], 1)
    Imod_negpts = NPT_dict['Imod_negpts'][0, :].reshape(
        NPT_dict['Imod_negpts'].shape[1], 1)
    Npts = NPT_dict['Npts']
    # iterate over substrings
    Isubstr, Vsubstr, Isc_substr, Imax_substr = [], [], [], []
    for substr_idx, substr in enumerate(cell_pos):
        # check if cells are in series or any crosstied circuits
        if all(r['crosstie'] == False for c in substr for r in c):
            idxs = [r['idx'] for c in substr for r in c]
            # t0 = time.time()
            IatVrbd = np.asarray(
                [np.interp(vrbd, v, i) for vrbd, v, i in
                 zip(VRBD[idxs], Vcell[idxs], Icell[idxs])]
            )
            Isub, Vsub = calcSeries(
                Icell[idxs], Vcell[idxs], Isc[idxs].mean(),
                IatVrbd.max(), Imod_pts, Imod_negpts, Npts
            )
        elif all(r['crosstie'] == True for c in substr for r in c):
            Irows, Vrows = [], []
            Isc_rows, Imax_rows = [], []
            for row in zip(*substr):
                idxs = [c['idx'] for c in row]
                Irow, Vrow = calcParallel(
                    Icell[idxs], Vcell[idxs],
                    Voc[idxs].max(), VRBD.min(), negpts, pts, Npts
                )
                Irows.append(Irow)
                Vrows.append(Vrow)
                Isc_rows.append(np.interp(np.float64(0), Vrow, Irow))
                Imax_rows.append(Irow.max())
            Irows, Vrows = np.asarray(Irows), np.asarray(Vrows)
            Isc_rows = np.asarray(Isc_rows)
            Imax_rows = np.asarray(Imax_rows)
            Isub, Vsub = calcSeries(
                Irows, Vrows, Isc_rows.mean(), Imax_rows.max(),
                Imod_pts, Imod_negpts, Npts
            )
        else:
            IVall_cols = []
            prev_col = None
            IVprev_cols = []
            for col in substr:
                IVcols = []
                is_first = True
                # combine series between crossties
                for idxs in pvconstants.get_series_cells(col, prev_col):
                    if not idxs:
                        # first row should always be empty since it must be
                        # crosstied
                        is_first = False
                        continue
                    elif is_first:
                        raise Exception(
                            "First row and last rows must be crosstied."
                        )
                    elif len(idxs) > 1:
                        IatVrbd = np.asarray(
                            [np.interp(vrbd, v, i) for vrbd, v, i in
                             zip(VRBD[idxs], Vcell[idxs],
                                 Icell[idxs])]
                        )
                        Icol, Vcol = calcSeries(
                            Icell[idxs], Vcell[idxs],
                            Isc[idxs].mean(), IatVrbd.max(),
                            Imod_pts, Imod_negpts, Npts
                        )
                    else:
                        Icol, Vcol = Icell[idxs], Vcell[idxs]
                    IVcols.append([Icol, Vcol])
                # append IVcols and continue
                IVprev_cols.append(IVcols)
                if prev_col:
                    # if circuits are same in both columns then continue
                    if not all(icol['crosstie'] == jcol['crosstie']
                               for icol, jcol in zip(prev_col, col)):
                        # combine crosstied circuits
                        Iparallel, Vparallel, _ = combine_parallel_circuits(
                            IVprev_cols, pvconstants,
                            negpts, pts, Imod_pts, Imod_negpts, Npts
                        )
                        IVall_cols.append([Iparallel, Vparallel])
                        # reset prev_col
                        prev_col = None
                        IVprev_cols = []
                        continue
                # set prev_col and continue
                prev_col = col
            # combine any remaining crosstied circuits in substring
            if not IVall_cols:
                # combine crosstied circuits
                Isub, Vsub, _ = combine_parallel_circuits(
                    IVprev_cols, pvconstants,
                    negpts, pts, Imod_pts, Imod_negpts, Npts
                )
            else:
                Iparallel, Vparallel = zip(*IVall_cols)
                Iparallel = np.asarray(Iparallel)
                Vparallel = np.asarray(Vparallel)
                Isub, Vsub = calcParallel(
                    Iparallel, Vparallel, Vparallel.max(), Vparallel.min(),
                    negpts, pts, Npts
                )

        Isubstr.append(Isub)
        Vsubstr.append(Vsub)
        Isc_substr.append(np.interp(np.float64(0), Vsub, Isub))
        Imax_substr.append(Isub.max())

    Isubstr, Vsubstr = np.asarray(Isubstr), np.asarray(Vsubstr)
    Isc_substr = np.asarray(Isc_substr)
    Imax_substr = np.asarray(Imax_substr)

    Psubstr = Isubstr * Vsubstr
    return Isubstr, Vsubstr, Psubstr, Isc.mean()


def extract_module_cell_curves_vectorized_preindexed_from_pickles(
    Ee_mod1: np.ndarray,         # 3-D: (Num_mod, Num_cell_X, Num_cell_Y)
    Tcell_mod1: np.ndarray,      # 3-D: (Num_mod, Num_cell_X, Num_cell_Y)
    cell_type1: np.ndarray,      # 2-D: (Num_cell_X, Num_cell_Y)
    # 2-D: (Num_cell_X, Num_cell_Y)  (not used here, preserved for API compatibility)
    idx_sort: np.ndarray,
    # int: module index             (not used here, preserved for API compatibility)
    idx_mod: int,
    # str: directory containing the pickled rows and NPT_dict.pickle
    res_path: str,
    # str: filename prefix for each row pickle => f"{cfname_pre}_{row_idx}.pickle"
    cfname_pre: str,
    # dict: output of prepare_global_cell_key_index(...)
    pre_idx: Dict[str, Any],
    # optional cache: row_id -> row dict
    row_cache: Optional[Dict[int, Dict[str, Any]]] = None,
    # optional: use threads to load rows in parallel (0 or 1 => sequential)
    num_workers: int = 0,
    strict_shapes: bool = True   # validate row shapes strictly
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Vectorized extraction using precomputed global key index, reading per-row
    pickle shards from disk.

    This function mirrors the original logic, but instead of indexing into
    in-memory arrays stored in `cell_data`, it deduplicates the required row
    indices, loads those rows from individual pickle files, assembles reduced
    arrays, and expands the results back to the flattened module.

    Parameters
    ----------
    Ee_mod1, Tcell_mod1, cell_type1, idx_sort, idx_mod : same meaning as
    original
    res_path : str
        Directory containing pickled shards. Must contain:
           - One file per row: f"{cfname_pre}_{row_idx}.pickle"
           - One file for NPT: "NPT_dict.pickle"
    cfname_pre : str
        Prefix for the row pickle filenames.
    pre_idx : dict
        Must contain fields 'dtype_key', 'u_keys_sorted', 'order',
        'round_decimals', 'u_ctypes_dtype'.
        'order' is used to map the per-module unique keys to global row indices
        (== row file suffix).
    row_cache : dict or None
        If provided, used as an in-memory cache mapping `row_idx -> row_dict`
        to avoid repeated IO.
    num_workers : int
        If > 1, will load row files in parallel using a thread pool of this
        size.
    strict_shapes : bool
        If True, performs additional consistency checks on shapes.

    Returns
    -------
    Icell : np.ndarray     # shape: (Num_cells_flat, Npts)
    Vcell : np.ndarray     # shape: (Num_cells_flat, Npts)
    VRBD  : np.ndarray     # shape: (Num_cells_flat,)
    Voc   : np.ndarray     # shape: (Num_cells_flat,)
    Isc   : np.ndarray     # shape: (Num_cells_flat,)
    counts: np.ndarray     # counts from np.unique over module keys
    inverse: np.ndarray    # inverse indices from np.unique over module keys
    NPT_dict: dict         # loaded from NPT_dict.pickle
    """

    # ------------------------------
    # 1) Unpack precomputed global index
    # ------------------------------
    dtype_key = pre_idx['dtype_key']
    u_keys_sorted = pre_idx['u_keys_sorted']
    # maps positions in global unique sorted keys -> global row index
    order = pre_idx['order']
    round_decimals = pre_idx.get('round_decimals', None)
    u_ctypes_dtype = pre_idx.get('u_ctypes_dtype', None)

    # ------------------------------
    # 2) Optional quantization (must match global rounding settings if used)
    # ------------------------------
    if round_decimals is not None:
        Ee_mod1_q = np.round(Ee_mod1,    round_decimals)
        Tcell_mod1_q = np.round(Tcell_mod1, round_decimals)
    else:
        Ee_mod1_q, Tcell_mod1_q = Ee_mod1, Tcell_mod1

    # ------------------------------
    # 3) Build structured module keys (Ee, Tcell, ctype)
    #    Use flattened views to avoid shape pitfalls with structured arrays.
    # ------------------------------
    num_cells_flat = Ee_mod1_q.size
    mod_keys = np.empty(num_cells_flat, dtype=dtype_key)

    # Flatten all fields
    mod_keys['Ee'] = Ee_mod1_q.ravel()
    mod_keys['T'] = Tcell_mod1_q.ravel()
    if u_ctypes_dtype is not None:
        mod_keys['ctype'] = cell_type1.ravel().astype(
            u_ctypes_dtype, copy=False)
    else:
        # Fall back to direct assignment if dtype already matches
        mod_keys['ctype'] = cell_type1.ravel()

    # ------------------------------
    # 4) Unique keys present in this module, with inverse for expansion
    # ------------------------------
    u_mod_keys, inverse, counts = np.unique(
        mod_keys, return_inverse=True, return_counts=True)

    # ------------------------------
    # 5) Vectorized mapping: binary search precomputed global sorted keys
    # ------------------------------
    pos = np.searchsorted(u_keys_sorted, u_mod_keys)

    # Validate found positions
    valid = (pos >= 0) & (pos < u_keys_sorted.size) & (
        u_keys_sorted[pos] == u_mod_keys)
    if not np.all(valid):
        bad = np.where(~valid)[0]
        sample = bad[:10].tolist()
        raise ValueError(
            f"Module keys not found in global unique keys (example indices: {sample}). "
            f"Ensure Ee/Tcell rounding matches between global and module keys "
            f"(round_decimals={round_decimals})."
        )

    # ------------------------------
    # 6) Map to original indices inside the *global* arrays => now they are
    # row ids (file suffixes)
    # ------------------------------
    # 1-D array of row indices (== per-row pickle filenames’ suffix)
    mod_in_cell = order[pos]

    # ------------------------------
    # 7) Load only the unique rows from disk (with optional caching and
    # parallelism)
    # ------------------------------
    unique_rows = np.unique(mod_in_cell)
    cache = row_cache if row_cache is not None else {}

    def _row_file_path(row_idx: int) -> str:
        return os.path.join(res_path, f"{cfname_pre}_{int(row_idx)}.pickle")

    def _load_one_row(row_idx: int) -> Dict[str, Any]:
        if row_idx in cache:
            return cache[row_idx]

        fpath = _row_file_path(row_idx)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Missing row pickle: '{fpath}'. "
                f"Expected one file per global row id using prefix '{cfname_pre}_'."
            )
        with open(fpath, "rb") as f:
            row_obj = pickle.load(f)

        # Basic sanity checks
        required_keys = ('Icell', 'Vcell', 'VRBD', 'Voc', 'Isc')
        for k in required_keys:
            if k not in row_obj:
                raise KeyError(
                    f"Row file '{fpath}' missing required key '{k}'.")

        # Normalize shapes: Icell/Vcell should be 1-D (Npts,), scalars for
        # others.
        # If they are stored as shape (1, Npts) or (Npts, 1), change to (Npts,)
        def _to_1d(arr: np.ndarray) -> np.ndarray:
            if isinstance(arr, np.ndarray):
                arr2 = np.asarray(arr)
                if arr2.ndim == 1:
                    return arr2
                if arr2.ndim == 2 and 1 in arr2.shape:
                    return arr2.reshape(-1)
                if strict_shapes:
                    raise ValueError(
                        f"Inconsistent shape for IV array in '{fpath}': {arr2.shape}. "
                        f"Expected 1-D (Npts,) or a singleton-2D."
                    )
                return arr2.reshape(-1)
            else:
                raise TypeError(
                    f"Expected numpy array for IV data in '{fpath}', got {type(arr)}.")

        row_obj['Icell'] = _to_1d(row_obj['Icell'])
        row_obj['Vcell'] = _to_1d(row_obj['Vcell'])

        # Cast VRBD/Voc/Isc to 0-D ndarray or scalar
        for k in ('VRBD', 'Voc', 'Isc'):
            val = row_obj[k]
            if isinstance(val, np.ndarray):
                if val.size != 1:
                    if strict_shapes:
                        raise ValueError(
                            f"Expected scalar for '{k}' in '{fpath}', got shape {val.shape}."
                        )
                    val = np.asarray(val).reshape(-1)[0]
                else:
                    val = val.reshape(()).item()
            elif not np.isscalar(val):
                if strict_shapes:
                    raise ValueError(
                        f"Expected scalar for '{k}' in '{fpath}', got type {type(val)}."
                    )
            row_obj[k] = val

        cache[row_idx] = row_obj
        return row_obj

    # Parallel or sequential loading
    if num_workers and num_workers > 1 and unique_rows.size > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_load_one_row, int(r)): int(r)
                       for r in unique_rows}
            for fut in as_completed(futures):
                _ = fut.result()
    else:
        for r in unique_rows:
            _load_one_row(int(r))

    # ------------------------------
    # 8) Assemble reduced arrays for u_mod_keys (in mod_in_cell order)
    # ------------------------------
    # Use the first row (already loaded) to determine Npts and dtypes
    sample_row = cache[int(unique_rows[0])]
    Npts = int(sample_row['Icell'].shape[0])

    Icell_red = np.empty(
        (mod_in_cell.shape[0], Npts), dtype=sample_row['Icell'].dtype)
    Vcell_red = np.empty(
        (mod_in_cell.shape[0], Npts), dtype=sample_row['Vcell'].dtype)
    Vrbd_red = np.empty(
        (mod_in_cell.shape[0],), dtype=np.asarray(sample_row['VRBD']).dtype)
    Voc_red = np.empty(
        (mod_in_cell.shape[0],), dtype=np.asarray(sample_row['Voc']).dtype)
    Isc_red = np.empty(
        (mod_in_cell.shape[0],), dtype=np.asarray(sample_row['Isc']).dtype)

    # Fill in order (vectorized gather over cached rows)
    # (We loop because each row is different; this is IO-bound and already
    # deduped above.)
    for i, row_idx in enumerate(mod_in_cell):
        r = int(row_idx)
        row = cache[r]

        # Optional extra checks
        if strict_shapes and row['Icell'].shape[0] != Npts:
            raise ValueError(
                f"Inconsistent Npts in row {r}: got {row['Icell'].shape[0]}, expected {Npts}."
            )

        Icell_red[i, :] = row['Icell']
        Vcell_red[i, :] = row['Vcell']
        Vrbd_red[i] = row['VRBD']
        Voc_red[i] = row['Voc']
        Isc_red[i] = row['Isc']

    # ------------------------------
    # 9) Expand back to the full flattened module using 'inverse'
    # ------------------------------
    Icell = Icell_red[inverse, :]
    Vcell = Vcell_red[inverse, :]
    VRBD = Vrbd_red[inverse]
    Voc = Voc_red[inverse]
    Isc = Isc_red[inverse]

    # ------------------------------
    # 10) Load NPT dictionary (once per call)
    # ------------------------------
    npt_path = os.path.join(res_path, "NPT_dict.pickle")
    if not os.path.exists(npt_path):
        raise FileNotFoundError(
            f"Missing NPT dictionary pickle at '{npt_path}'. "
            f"Expected file name: 'NPT_dict.pickle'."
        )
    with open(npt_path, "rb") as f:
        NPT_dict = pickle.load(f)
    return Icell, Vcell, VRBD, Voc, Isc, counts, inverse, NPT_dict


def prepare_global_cell_key_index(
    Ee_cell,        # 1-D (K,)
    Tcell_cell,     # 1-D (K,)
    u_cell_type,    # 1-D (K,)
    round_decimals=None
):
    """
    Precompute global structured keys and sorting order for fast binary-search
    mapping.

    Returns
    -------
    pre_idx : dict
        {
          'dtype_key': structured dtype,
          'u_keys_sorted': sorted structured array of global keys,
          'order': argsort indices mapping sorted -> original,
          'round_decimals': int or None,
          'u_ctypes_dtype': dtype used for ctype field
        }
    """
    # Optional quantization to mitigate floating-point mismatches
    if round_decimals is not None:
        Ee_cell_q = np.round(Ee_cell, round_decimals)
        Tcell_cell_q = np.round(Tcell_cell, round_decimals)
    else:
        Ee_cell_q, Tcell_cell_q = Ee_cell, Tcell_cell

    # Ensure dtype consistency for cell types
    u_ctypes_dtype = u_cell_type.dtype
    u_ctypes_cast = u_cell_type.astype(u_ctypes_dtype, copy=False)

    # Structured dtype for keys
    dtype_key = np.dtype([
        ('Ee',   Ee_cell_q.dtype),
        ('T',    Tcell_cell_q.dtype),
        ('ctype', u_ctypes_dtype)
    ])

    # Build structured global keys once
    u_keys = np.empty(Ee_cell_q.size, dtype=dtype_key)
    u_keys['Ee'] = Ee_cell_q
    u_keys['T'] = Tcell_cell_q
    u_keys['ctype'] = u_ctypes_cast

    # Sort once and cache the result
    order = np.argsort(u_keys)
    u_keys_sorted = u_keys[order]

    return {
        'dtype_key': dtype_key,
        'u_keys_sorted': u_keys_sorted,
        'order': order,
        'round_decimals': round_decimals,
        'u_ctypes_dtype': u_ctypes_dtype
    }
