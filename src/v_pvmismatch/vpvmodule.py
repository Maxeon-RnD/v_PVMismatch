# -*- coding: utf-8 -*-
"""Vectorized pvmodule."""

import copy
import numpy as np
import pandas as pd
from .pvmismatch import pvconstants
from .utils import (calcMPP_IscVocFFBPD, save_pickle, load_pickle,
                    round_to_dec, find_row_index)
from .circuit_comb import calcSeries, calcParallel
from .circuit_comb import combine_parallel_circuits, parse_diode_config
from .circuit_comb import calcSeries_with_bypass, calcParallel_with_bypass
from .circuit_comb import DEFAULT_BYPASS, MODULE_BYPASS, CUSTOM_SUBSTR_BYPASS
# ----------------------------------------------------------------------------
# CALCULATE MODULE IV-PV CURVES-----------------------------------------------


def calcMods(cell_pos, maxmod, cell_index_map, Ee_mod, Ee_cell,
             u_cell_type, cell_type, cell_data,
             outer_circuit, run_bpact=True, run_cellcurr=False,
             mod_DBs=None, ss_DBs=None, Ee_round=2, IV_trk_ct=True):
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
    cell_data : dict
        Dictionary containing cell IV curves.
    outer_circuit : str
        series or parallel.
    run_bpact : bool, optional
        Flag to run bypass diode activation logic. The default is True.
    run_cellcurr : bool, optional
        Flag to run cell current estimation logic. The default is True.

    Returns
    -------
    mod_data : dict
        Dictionary containing module IV curves.

    """
    if mod_DBs:
        # Load databases
        mod_irr_db_path = mod_DBs[0]
        mod_isc_db_path = mod_DBs[1]
        mod_I_db_path = mod_DBs[2]
        mod_V_db_path = mod_DBs[3]
        mod_Isubstr_db_path = mod_DBs[4]
        mod_Vsubstr_db_path = mod_DBs[5]
        mod_Isubstr_pbp_db_path = mod_DBs[6]
        mod_Vsubstr_pbp_db_path = mod_DBs[7]
        mod_bypassed_db_path = mod_DBs[8]
        mod_counts_db_path = mod_DBs[9]
        IV_res = mod_DBs[10]
        try:
            mod_irr_db = load_pickle(mod_irr_db_path)
            mod_isc_db = load_pickle(mod_isc_db_path)
            mod_I_db = load_pickle(mod_I_db_path)
            mod_V_db = load_pickle(mod_V_db_path)
            mod_Isubstr_db = load_pickle(mod_Isubstr_db_path)
            mod_Vsubstr_db = load_pickle(mod_Vsubstr_db_path)
            mod_Isubstr_pbp_db = load_pickle(mod_Isubstr_pbp_db_path)
            mod_Vsubstr_pbp_db = load_pickle(mod_Vsubstr_pbp_db_path)
            mod_bypassed_db = load_pickle(mod_bypassed_db_path)
            if IV_trk_ct:
                mod_counts_db = load_pickle(mod_counts_db_path)
            last_index_df = mod_irr_db.shape[0]
        except FileNotFoundError:
            mod_irr_db = []
            mod_isc_db = []
            mod_I_db = []
            mod_V_db = []
            mod_Isubstr_db = {}
            mod_Vsubstr_db = {}
            mod_Isubstr_pbp_db = {}
            mod_Vsubstr_pbp_db = {}
            mod_bypassed_db = {}
            if IV_trk_ct:
                mod_counts_db = []
            last_index_df = 0
        # Round the Ee
        # Ee_mod = round_to_dec(Ee_mod, IV_res)
        # Ee_mod = Ee_mod.round(Ee_round)
    if ss_DBs:
        # Load databases
        ss_irr_db_path = ss_DBs[0]
        ss_Id_pre_db_path = ss_DBs[1]
        ss_Vd_pre_db_path = ss_DBs[2]
        ss_cts_pre_db_path = ss_DBs[3]
        ss_irr_ss_db_path = ss_DBs[4]
        ss_Iss_db_path = ss_DBs[5]
        ss_Vss_db_path = ss_DBs[6]
        ss_cts_ss_db_path = ss_DBs[7]
        IV_res = ss_DBs[8]
        try:
            ss_irr_db = load_pickle(ss_irr_db_path)
            ss_Id_pre_db = load_pickle(ss_Id_pre_db_path)
            ss_Vd_pre_db = load_pickle(ss_Vd_pre_db_path)
            ss_cts_pre_db = load_pickle(ss_cts_pre_db_path)
            last_index_df = ss_irr_db.shape[0]
        except FileNotFoundError:
            ss_irr_db = []
            ss_Id_pre_db = []
            ss_Vd_pre_db = []
            ss_cts_pre_db = []
            last_index_df = 0
        try:
            ss_irr_ss_db = load_pickle(ss_irr_ss_db_path)
            ss_Iss_db = load_pickle(ss_Iss_db_path)
            ss_Vss_db = load_pickle(ss_Vss_db_path)
            ss_cts_ss_db = load_pickle(ss_cts_ss_db_path)
            last_index_ss_df = ss_irr_ss_db.shape[0]
        except FileNotFoundError:
            ss_irr_ss_db = []
            ss_Iss_db = []
            ss_Vss_db = []
            ss_cts_ss_db = []
            last_index_ss_df = 0
        ss_pDBs = [ss_irr_db, ss_Id_pre_db, ss_Vd_pre_db, ss_cts_pre_db,
                   ss_irr_ss_db, ss_Iss_db, ss_Vss_db, ss_cts_ss_db,
                   last_index_df, last_index_ss_df, IV_res]
    else:
        ss_pDBs = None
    Vbypass = maxmod.Vbypass
    I_mod_curves = []
    V_mod_curves = []
    P_mod_curves = []
    Isubstr_curves = []
    Vsubstr_curves = []
    Isubstr_pre_bypass_curves = []
    Vsubstr_pre_bypass_curves = []
    mean_Iscs = []
    bypassed_mod_arr = []
    if run_cellcurr:
        full_data = []
    for idx_mod in range(Ee_mod.shape[0]):
        # 1 Module
        cell_ids = cell_index_map.flatten()
        idx_sort = np.argsort(cell_ids)
        Ee_mod1 = Ee_mod[idx_mod].flatten()[idx_sort]
        # Ee_mod1 = Ee_mod1.round(Ee_round)
        if mod_DBs:
            # If a database is used
            if len(mod_irr_db) > 0:
                idx_row = find_row_index(mod_irr_db, Ee_mod1)
            else:
                idx_row = None
            if idx_row is not None:
                use_DB = True
                mean_Isc = float(mod_isc_db[idx_row, 0])
                Imod = mod_I_db[idx_row, :].astype(float)
                Vmod = mod_V_db[idx_row, :].astype(float)
                Pmod = Imod * Vmod
                Isubstr = mod_Isubstr_db[idx_row].copy()
                Vsubstr = mod_Vsubstr_db[idx_row].copy()
                Isubstr_pre_bypass = mod_Isubstr_pbp_db[idx_row].copy()
                Vsubstr_pre_bypass = mod_Vsubstr_pbp_db[idx_row].copy()
                bypassed_mod = mod_bypassed_db[idx_row].copy()
                if IV_trk_ct:
                    mod_ct = mod_counts_db[idx_row] + 1
            else:
                use_DB = False
                if IV_trk_ct:
                    mod_ct = 1
            if IV_trk_ct:
                if len(mod_counts_db) == 0:
                    mod_counts_db = np.array([mod_ct])
                else:
                    if idx_row is not None:
                        mod_counts_db[idx_row] = mod_ct
                    else:
                        mod_counts_db = np.vstack([mod_counts_db, mod_ct])
        else:
            use_DB = False
        if not use_DB:
            cell_type1 = cell_type.flatten()[idx_sort]
            # Extract cell IV curves
            mod_in_cell = np.where(np.in1d(Ee_cell+u_cell_type,
                                           Ee_mod1+cell_type1))[0]
            Icell_red = cell_data['Icell'][mod_in_cell, :]
            Vcell_red = cell_data['Vcell'][mod_in_cell, :]
            Vrbd_red = cell_data['VRBD'][mod_in_cell]
            Voc_red = cell_data['Voc'][mod_in_cell]
            Isc_red = cell_data['Isc'][mod_in_cell]
            u, inverse, counts = np.unique(Ee_mod1+cell_type1,
                                           return_inverse=True,
                                           return_counts=True)
            # Expand for Mod curves
            Icell = Icell_red[inverse, :]
            Vcell = Vcell_red[inverse, :]
            VRBD = Vrbd_red[inverse]
            Voc = Voc_red[inverse]
            Isc = Isc_red[inverse]
            NPT_dict = cell_data['NPT']
            # Run Module Circuit model
            sing_mod, ss_pDBs = calcMod(Ee_mod1, Icell, Vcell, VRBD, Voc, Isc,
                                        cell_pos, Vbypass, NPT_dict,
                                        outer=outer_circuit,
                                        run_bpact=run_bpact,
                                        run_cellcurr=run_cellcurr,
                                        ss_DBs=ss_pDBs, IV_trk_ct=IV_trk_ct)
            if run_cellcurr:
                full_data.append(sing_mod)
            Imod = sing_mod['Imod'].copy()
            Vmod = sing_mod['Vmod'].copy()
            Pmod = sing_mod['Pmod'].copy()
            Isubstr = sing_mod['Isubstr'].copy()
            Vsubstr = sing_mod['Vsubstr'].copy()
            mean_Isc = sing_mod['Isc']
            Isubstr_pre_bypass = sing_mod['Isubstr_pre_bypass'].copy()
            Vsubstr_pre_bypass = sing_mod['Vsubstr_pre_bypass'].copy()
            if run_bpact:
                bypassed_mod = sing_mod['bypassed_mod'].copy()
            else:
                bypassed_mod = sing_mod['bypassed_mod']
            if mod_DBs:
                # Store results in IV database
                # First the prms
                if len(mod_irr_db) == 0:
                    mod_irr_db = np.array([Ee_mod1.tolist()])
                    mod_isc_db = np.array([[mean_Isc]])
                    mod_I_db = np.array([Imod.tolist()])
                    mod_V_db = np.array([Vmod.tolist()])
                else:
                    mod_isc_db = np.vstack([mod_isc_db, mean_Isc])
                    mod_irr_db = np.vstack([mod_irr_db, Ee_mod1])
                    mod_I_db = np.vstack([mod_I_db, Imod])
                    mod_V_db = np.vstack([mod_V_db, Vmod])
                mod_Isubstr_db[last_index_df] = Isubstr.copy()
                mod_Vsubstr_db[last_index_df] = Vsubstr.copy()
                mod_Isubstr_pbp_db[last_index_df] = Isubstr_pre_bypass.copy()
                mod_Vsubstr_pbp_db[last_index_df] = Vsubstr_pre_bypass.copy()
                mod_bypassed_db[last_index_df] = bypassed_mod.copy()
                last_index_df += 1
        I_mod_curves.append(np.reshape(Imod, (1, len(Imod))))
        V_mod_curves.append(np.reshape(Vmod, (1, len(Vmod))))
        P_mod_curves.append(np.reshape(Pmod, (1, len(Pmod))))
        # Isubstr_curves.append(np.reshape(
        #     Isubstr, (1, Isubstr.shape[0], Isubstr.shape[1])))
        # Vsubstr_curves.append(np.reshape(
        #     Vsubstr, (1, Vsubstr.shape[0], Vsubstr.shape[1])))
        # Isubstr_pre_bypass_curves.append(np.reshape(
        #     Isubstr_pre_bypass,
        #     (1, Isubstr_pre_bypass.shape[0],
        #      Isubstr_pre_bypass.shape[1])))
        # Vsubstr_pre_bypass_curves.append(np.reshape(
        #     Vsubstr_pre_bypass,
        #     (1, Vsubstr_pre_bypass.shape[0],
        #      Vsubstr_pre_bypass.shape[1])))
        mean_Iscs.append(mean_Isc)
        if run_bpact:
            bypassed_mod_arr.append(np.reshape(
                bypassed_mod, (1, bypassed_mod.shape[0],
                               bypassed_mod.shape[1])))
        else:
            bypassed_mod_arr.append(bypassed_mod)
    I_mod_curves = np.concatenate(I_mod_curves, axis=0)
    V_mod_curves = np.concatenate(V_mod_curves, axis=0)
    P_mod_curves = np.concatenate(P_mod_curves, axis=0)
    # Isubstr_curves = np.concatenate(Isubstr_curves, axis=0)
    # Vsubstr_curves = np.concatenate(Vsubstr_curves, axis=0)
    # Isubstr_pre_bypass_curves = np.concatenate(Isubstr_pre_bypass_curves,
    #                                            axis=0)
    # Vsubstr_pre_bypass_curves = np.concatenate(Vsubstr_pre_bypass_curves,
    #                                            axis=0)
    mean_Iscs = np.array(mean_Iscs)
    if run_bpact:
        bypassed_mod_arr = np.concatenate(bypassed_mod_arr, axis=0)
    else:
        bypassed_mod_arr = np.array(bypassed_mod_arr)

    Imp, Vmp, Pmp, Isc, Voc, FF, BpDmp, num_bpd_active = calcMPP_IscVocFFBPD(
        I_mod_curves, V_mod_curves, P_mod_curves, bypassed_mod_arr,
        run_bpact=run_bpact)

    # Save the updated databases
    if mod_DBs:
        save_pickle(mod_irr_db_path, mod_irr_db)
        save_pickle(mod_isc_db_path, mod_isc_db)
        save_pickle(mod_I_db_path, mod_I_db)
        save_pickle(mod_V_db_path, mod_V_db)
        save_pickle(mod_Isubstr_db_path, mod_Isubstr_db)
        save_pickle(mod_Vsubstr_db_path, mod_Vsubstr_db)
        save_pickle(mod_Isubstr_pbp_db_path, mod_Isubstr_pbp_db)
        save_pickle(mod_Vsubstr_pbp_db_path, mod_Vsubstr_pbp_db)
        save_pickle(mod_bypassed_db_path, mod_bypassed_db)
        if IV_trk_ct:
            save_pickle(mod_counts_db_path, mod_counts_db)
    # Save the updated databases
    if ss_pDBs:
        ss_irr_db = ss_pDBs[0]
        ss_Id_pre_db = ss_pDBs[1]
        ss_Vd_pre_db = ss_pDBs[2]
        ss_cts_pre_db = ss_pDBs[3]
        ss_irr_ss_db = ss_pDBs[4]
        ss_Iss_db = ss_pDBs[5]
        ss_Vss_db = ss_pDBs[6]
        ss_cts_ss_db = ss_pDBs[7]
        last_index_df = ss_pDBs[8]
        last_index_ss_df = ss_pDBs[9]
        IV_res = ss_pDBs[10]
    if ss_DBs:
        if len(ss_irr_db) > 0:
            save_pickle(ss_irr_db_path, ss_irr_db)
            save_pickle(ss_Id_pre_db_path, ss_Id_pre_db)
            save_pickle(ss_Vd_pre_db_path, ss_Vd_pre_db)
            if IV_trk_ct:
                save_pickle(ss_cts_pre_db_path, ss_cts_pre_db)
        if len(ss_irr_ss_db) > 0:
            save_pickle(ss_irr_ss_db_path, ss_irr_ss_db)
            save_pickle(ss_Iss_db_path, ss_Iss_db)
            save_pickle(ss_Vss_db_path, ss_Vss_db)
            if IV_trk_ct:
                save_pickle(ss_cts_ss_db_path, ss_cts_ss_db)

    # Store results in a dict
    mod_data = dict()
    mod_data['Imod'] = I_mod_curves
    mod_data['Vmod'] = V_mod_curves
    mod_data['Pmod'] = P_mod_curves
    mod_data['Isubstr'] = Isubstr_curves
    mod_data['Vsubstr'] = Vsubstr_curves
    mod_data['Isubstr_pre_bypass'] = Isubstr_pre_bypass_curves
    mod_data['Vsubstr_pre_bypass'] = Vsubstr_pre_bypass_curves
    mod_data['Bypassed_substr'] = bypassed_mod_arr
    mod_data['mean_Isc'] = np.reshape(mean_Iscs, (len(mean_Iscs), 1))
    mod_data['Imp'] = Imp
    mod_data['Vmp'] = Vmp
    mod_data['Pmp'] = Pmp
    mod_data['Isc'] = Isc
    mod_data['Voc'] = Voc
    mod_data['FF'] = FF
    mod_data['BPDiode_Active_MPP'] = BpDmp
    mod_data['num_bpd_active'] = num_bpd_active
    if run_cellcurr:
        mod_data['full_data'] = copy.deepcopy(full_data)

    return mod_data


def calcMod(Ee_mod, Icell, Vcell, VRBD, Voc, Isc, cell_pos, Vbypass,
            NPT_dict, outer='series', run_bpact=True, run_cellcurr=True,
            ss_DBs=None, IV_trk_ct=True):
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
    if ss_DBs:
        ss_irr_db = ss_DBs[0]
        ss_Id_pre_db = ss_DBs[1]
        ss_Vd_pre_db = ss_DBs[2]
        ss_cts_pre_db = ss_DBs[3]
        ss_irr_ss_db = ss_DBs[4]
        ss_Iss_db = ss_DBs[5]
        ss_Vss_db = ss_DBs[6]
        ss_cts_ss_db = ss_DBs[7]
        last_index_df = ss_DBs[8]
        last_index_ss_df = ss_DBs[9]
        IV_res = ss_DBs[8]
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
            if ss_DBs:
                E_ss = Ee_mod[idxs]
                E_ss.sort()
                # If a database is used
                if len(ss_irr_db) > 0:
                    idx_row = find_row_index(ss_irr_db, E_ss)
                else:
                    idx_row = None
                if idx_row is not None:
                    Isub = ss_Id_pre_db[idx_row, :].astype(float)
                    Vsub = ss_Vd_pre_db[idx_row, :].astype(float)
                    use_DB = True
                    if IV_trk_ct:
                        p_ct = ss_cts_pre_db[idx_row] + 1
                else:
                    use_DB = False
                    if IV_trk_ct:
                        p_ct = 1
                if IV_trk_ct:
                    if len(ss_cts_pre_db) == 0:
                        ss_cts_pre_db = np.array([p_ct])
                    else:
                        if idx_row is not None:
                            ss_cts_pre_db[idx_row] = p_ct
                        else:
                            ss_cts_pre_db = np.vstack([ss_cts_pre_db, p_ct])
            else:
                use_DB = False
            if not use_DB:
                # t0 = time.time()
                IatVrbd = np.asarray(
                    [np.interp(vrbd, v, i) for vrbd, v, i in
                     zip(VRBD[idxs], Vcell[idxs], Icell[idxs])]
                )
                Isub, Vsub = calcSeries(
                    Icell[idxs], Vcell[idxs], Isc[idxs].mean(),
                    IatVrbd.max(), Imod_pts, Imod_negpts, Npts
                )
                if ss_DBs:
                    if len(ss_irr_db) == 0:
                        ss_irr_db = np.array([E_ss.tolist()])
                        ss_Id_pre_db = np.array([Isub.tolist()])
                        ss_Vd_pre_db = np.array([Vsub.tolist()])
                    else:
                        if ss_irr_db.shape[1] < len(E_ss):
                            zeros_column = np.zeros(
                                (ss_irr_db.shape[0],
                                 len(E_ss) - ss_irr_db.shape[1]
                                 ))
                            ss_irr_db = np.hstack((
                                ss_irr_db, zeros_column))
                        elif ss_irr_db.shape[1] > len(E_ss):
                            E_ss = np.pad(E_ss,
                                          (0,
                                           ss_irr_db.shape[1] - len(E_ss)),
                                          'constant')
                        ss_irr_db = np.vstack([ss_irr_db, E_ss])
                        ss_Id_pre_db = np.vstack([ss_Id_pre_db, Isub])
                        ss_Vd_pre_db = np.vstack([ss_Vd_pre_db, Vsub])
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
                        if ss_DBs:
                            E_ss = Ee_mod[idxs]
                            E_ss.sort()
                            # If a database is used
                            if len(ss_irr_ss_db) > 0:
                                idx_row = find_row_index(ss_irr_ss_db, E_ss)
                            else:
                                idx_row = None
                            if idx_row is not None:
                                Icol = ss_Iss_db[idx_row, :].astype(float)
                                Vcol = ss_Vss_db[idx_row, :].astype(float)
                                use_DB = True
                                if IV_trk_ct:
                                    ss_ct = ss_cts_ss_db[idx_row] + 1
                            else:
                                use_DB = False
                                if IV_trk_ct:
                                    ss_ct = 1
                            if IV_trk_ct:
                                if len(ss_cts_ss_db) == 0:
                                    ss_cts_ss_db = np.array([ss_ct])
                                else:
                                    if idx_row is not None:
                                        ss_cts_ss_db[idx_row] = ss_ct
                                    else:
                                        ss_cts_ss_db = np.vstack(
                                            [ss_cts_ss_db, ss_ct])
                        else:
                            use_DB = False
                        if not use_DB:
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
                            if ss_DBs:
                                if len(ss_irr_ss_db) == 0:
                                    ss_irr_ss_db = np.array([E_ss.tolist()])
                                    ss_Iss_db = np.array([Icol.tolist()])
                                    ss_Vss_db = np.array([Vcol.tolist()])
                                else:
                                    if ss_irr_ss_db.shape[1] < len(E_ss):
                                        zeros_column = np.zeros(
                                            (ss_irr_ss_db.shape[0],
                                             len(E_ss) - ss_irr_ss_db.shape[1]
                                             ))
                                        ss_irr_ss_db = np.hstack((
                                            ss_irr_ss_db, zeros_column))
                                    elif ss_irr_ss_db.shape[1] > len(E_ss):
                                        E_ss = np.pad(E_ss,
                                                      (0,
                                                       ss_irr_ss_db.shape[1] - len(E_ss)),
                                                      'constant')
                                    ss_irr_ss_db = np.vstack([ss_irr_ss_db,
                                                              E_ss])
                                    ss_Iss_db = np.vstack([ss_Iss_db, Icol])
                                    ss_Vss_db = np.vstack([ss_Vss_db, Vcol])
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

    if ss_DBs:
        ss_DBs = [ss_irr_db, ss_Id_pre_db, ss_Vd_pre_db, ss_cts_pre_db,
                  ss_irr_ss_db, ss_Iss_db, ss_Vss_db, ss_cts_ss_db,
                  last_index_df, last_index_ss_df, IV_res]
    return sing_mod, ss_DBs


def calcsubMods(cell_pos, maxmod, cell_index_map, Ee_mod,
                Ee_cell, u_cell_type, cell_type, cell_data):
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
    Isubstr_curves = []
    Vsubstr_curves = []
    mean_Iscs = []
    for idx_mod in range(Ee_mod.shape[0]):
        # 1 Module
        cell_ids = cell_index_map.flatten()
        idx_sort = np.argsort(cell_ids)
        Ee_mod1 = Ee_mod[idx_mod].flatten()[idx_sort]
        cell_type1 = cell_type.flatten()[idx_sort]
        # Extract cell IV curves
        mod_in_cell = np.where(
            np.in1d(Ee_cell+u_cell_type, Ee_mod1+cell_type1))[0]
        Icell_red = cell_data['Icell'][mod_in_cell, :]
        Vcell_red = cell_data['Vcell'][mod_in_cell, :]
        Vrbd_red = cell_data['VRBD'][mod_in_cell]
        Voc_red = cell_data['Voc'][mod_in_cell]
        Isc_red = cell_data['Isc'][mod_in_cell]
        u, inverse, counts = np.unique(Ee_mod1+cell_type1, return_inverse=True,
                                       return_counts=True)
        # Expand for Mod curves
        Icell = Icell_red[inverse, :]
        Vcell = Vcell_red[inverse, :]
        VRBD = Vrbd_red[inverse]
        Voc = Voc_red[inverse]
        Isc = Isc_red[inverse]
        NPT_dict = cell_data['NPT']
        # Run Module Circuit model
        Isubstr, Vsubstr, Psubstr, mean_Isc = calcsubMod(Icell, Vcell, VRBD,
                                                         Voc, Isc, cell_pos,
                                                         Vbypass, NPT_dict)
        Isubstr_curves.append(np.reshape(
            Isubstr, (1, Isubstr.shape[0], Isubstr.shape[1])))
        Vsubstr_curves.append(np.reshape(
            Vsubstr, (1, Vsubstr.shape[0], Vsubstr.shape[1])))
        mean_Iscs.append(mean_Isc)
    Isubstr_curves = np.concatenate(Isubstr_curves, axis=0)
    Vsubstr_curves = np.concatenate(Vsubstr_curves, axis=0)
    Psubstr_curves = Vsubstr_curves * Isubstr_curves
    mean_Iscs = np.array(mean_Iscs)

    # Store results in a dict
    mod_data = dict()
    mod_data['Isubstr'] = Isubstr_curves
    mod_data['Vsubstr'] = Vsubstr_curves
    mod_data['Psubstr'] = Psubstr_curves
    mod_data['mean_Isc'] = np.reshape(mean_Iscs, (len(mean_Iscs), 1))
    mod_data['Imp'] = np.zeros((Isubstr_curves.shape[0], 1))
    mod_data['Vmp'] = np.zeros((Isubstr_curves.shape[0], 1))
    mod_data['Pmp'] = np.zeros((Isubstr_curves.shape[0], 1))
    mod_data['Isc'] = np.zeros((Isubstr_curves.shape[0], 1))
    mod_data['Voc'] = np.zeros((Isubstr_curves.shape[0], 1))
    mod_data['FF'] = np.zeros((Isubstr_curves.shape[0], 1))

    return mod_data


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
