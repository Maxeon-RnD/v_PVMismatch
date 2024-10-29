# -*- coding: utf-8 -*-
"""Vectorized pvmodule."""

import copy
import numpy as np
from .pvmismatch import pvconstants
from .utils import calcMPP_IscVocFFBPD
from .circuit_comb import calcSeries, calcParallel
from .circuit_comb import combine_parallel_circuits, parse_diode_config
from .circuit_comb import calcSeries_with_bypass, calcParallel_with_bypass
from .circuit_comb import DEFAULT_BYPASS, MODULE_BYPASS, CUSTOM_SUBSTR_BYPASS
# ------------------------------------------------------------------------------
# CALCULATE MODULE IV-PV CURVES------------------------------------------------


def calcMods(cell_pos, maxmod, cell_index_map, Ee_mod, Ee_cell,
             u_cell_type, cell_type, cell_data,
             outer_circuit, run_bpact=True, run_cellcurr=True):
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
        cell_type1 = cell_type.flatten()[idx_sort]
        # Extract cell IV curves
        mod_in_cell = np.where(np.in1d(Ee_cell+u_cell_type,
                                       Ee_mod1+cell_type1))[0]
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
        sing_mod = calcMod(Icell, Vcell, VRBD, Voc, Isc,
                           cell_pos, Vbypass, NPT_dict,
                           outer=outer_circuit, run_bpact=run_bpact,
                           run_cellcurr=run_cellcurr)
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
        bypassed_mod = sing_mod['bypassed_mod'].copy()
        I_mod_curves.append(np.reshape(Imod, (1, len(Imod))))
        V_mod_curves.append(np.reshape(Vmod, (1, len(Vmod))))
        P_mod_curves.append(np.reshape(Pmod, (1, len(Pmod))))
        Isubstr_curves.append(np.reshape(
            Isubstr, (1, Isubstr.shape[0], Isubstr.shape[1])))
        Vsubstr_curves.append(np.reshape(
            Vsubstr, (1, Vsubstr.shape[0], Vsubstr.shape[1])))
        Isubstr_pre_bypass_curves.append(np.reshape(
            Isubstr_pre_bypass,
            (1, Isubstr_pre_bypass.shape[0], Isubstr_pre_bypass.shape[1])))
        Vsubstr_pre_bypass_curves.append(np.reshape(
            Vsubstr_pre_bypass,
            (1, Vsubstr_pre_bypass.shape[0], Vsubstr_pre_bypass.shape[1])))
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
    Isubstr_curves = np.concatenate(Isubstr_curves, axis=0)
    Vsubstr_curves = np.concatenate(Vsubstr_curves, axis=0)
    Isubstr_pre_bypass_curves = np.concatenate(Isubstr_pre_bypass_curves,
                                               axis=0)
    Vsubstr_pre_bypass_curves = np.concatenate(Vsubstr_pre_bypass_curves,
                                               axis=0)
    mean_Iscs = np.array(mean_Iscs)
    if run_bpact:
        bypassed_mod_arr = np.concatenate(bypassed_mod_arr, axis=0)
    else:
        bypassed_mod_arr = np.array(bypassed_mod_arr)

    Imp, Vmp, Pmp, Isc, Voc, FF, BpDmp, num_bpd_active = calcMPP_IscVocFFBPD(
        I_mod_curves, V_mod_curves, P_mod_curves, bypassed_mod_arr,
        run_bpact=run_bpact)

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


def calcMod(Icell, Vcell, VRBD, Voc, Isc, cell_pos, Vbypass, NPT_dict,
            outer='series', run_bpact=True, run_cellcurr=True):
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
            # t0 = time.time()
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
                sing_mod[substr_idx][ss_s_ct][ss_p_ct]['cell_currents'] = Icell[idxs].copy()
                sing_mod[substr_idx][ss_s_ct][ss_p_ct]['cell_voltages'] = Vcell[idxs].copy()
                sing_mod[substr_idx][ss_s_ct][ss_p_ct]['cell_idxs'] = copy.deepcopy(
                    idxs)
                sing_mod[substr_idx][ss_s_ct][ss_p_ct]['Isubstr'] = Isub.copy()
                sing_mod[substr_idx][ss_s_ct][ss_p_ct]['Vsubstr'] = Vsub.copy()
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
                        for ss_p_ct in range(sub_str_data['Iparallels'][ss_s_ct].shape[0]):
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
    sing_mod['bypassed_mod'] = bypassed_mod.copy()
    return sing_mod
