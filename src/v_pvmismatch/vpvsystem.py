# -*- coding: utf-8 -*-
"""Vectorized pvsystem."""

import os
from typing import Dict, Any, Optional, Tuple
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from .utils import (reshape_ndarray, calcMPP_IscVocFFBPD,
                    calcMPP_IscVocFF, save_pickle)
from .vpvstring import (prepare_global_module_key_index,
                        extract_string_module_curves_pair_aware_from_pickles)
from .circuit_comb import calcParallel_with_bypass
from .utils import row_void_view_1d

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


def get_unique_Ee_Tcell(Ee, Tcell, search_type='cell', cell_type=None):
    """
    Generate unique (irradiance, cell temperature) pairs at the cell, module,
    string or system level.

    Parameters
    ----------
    Ee : numpy.ndarray
        5-D system level irradiance array with shape
        (sim, Num_mod_X, Num_mod_Y, Num_cell_X, Num_cell_Y).
    Tcell : numpy.ndarray
        5-D system level cell temperature array with the same shape as `Ee`.
    search_type : str, optional
        Which type of unique: 'cell', 'module', 'string', or 'system'.
        The default is 'cell'.
    cell_type : numpy.ndarray or None, optional
        Array containing the different cell types (typically shape
        (Num_cell_X, Num_cell_Y)). Only used when search_type == 'cell'.
        The default is None.

    Returns
    -------
    uEe : numpy.ndarray
        Unique irradiance arrays (shape depends on search_type):
        - 'cell'    : (K,)
        - 'module'  : (K, Num_cell_X, Num_cell_Y)
        - 'string'  : (K, Num_mod_Y, Num_cell_X, Num_cell_Y)
        - 'system'  : (K, Num_mod_X, Num_mod_Y, Num_cell_X, Num_cell_Y)
    uTcell : numpy.ndarray
        Unique temperature arrays (aligned one-to-one with `uEe` shapes above).
    u_cell_type : numpy.ndarray or None
        For 'cell' search:
            - If `cell_type` is provided, returns a vector mapping each unique
              (Ee, Tcell) pair to its cell type (same length as `uEe`).
            - If `cell_type` is None, returns zeros of length len(`uEe`).
        For other search types: None.
    cts : numpy.ndarray or None
        Counts of unique occurrences:
        - 'module'  : counts per unique module pair.
        - 'cell', 'string', 'system' : None (preserving original behavior).
    """

    # --- Basic validation ---
    if not (isinstance(Ee, np.ndarray) and isinstance(Tcell, np.ndarray)):
        raise TypeError("Ee and Tcell must be numpy arrays.")
    if Ee.shape != Tcell.shape:
        raise ValueError(
            f"Ee and Tcell must have the same shape. Got Ee{Ee.shape} and Tcell{Tcell.shape}"
        )
    if Ee.ndim != 5:
        raise ValueError(f"Expected 5-D arrays. Got Ee.ndim={Ee.ndim}")

    # Helper: unique pairs where we dedupe across the first axis
    def _unique_pairs_across_first_axis(a, b):
        """
        Return unique rows (across axis=0) for pair arrays a,b.
        a,b have shape (N, ...). We flatten '...' to a vector and concatenate
        a_flat and b_flat horizontally, then unique along axis=0.
        Returns: u_a, u_b, counts
        """
        N = a.shape[0]
        rest = a.shape[1:]
        a_flat = a.reshape(N, int(np.prod(rest)))
        b_flat = b.reshape(N, int(np.prod(rest)))
        combo = np.concatenate([a_flat, b_flat], axis=1)
        u_combo, _, counts = np.unique(combo, axis=0, return_inverse=True,
                                       return_counts=True)
        # Split back
        M = a_flat.shape[1]
        u_a_flat = u_combo[:, :M]
        u_b_flat = u_combo[:, M:]
        u_a = u_a_flat.reshape((-1,) + rest)
        u_b = u_b_flat.reshape((-1,) + rest)
        return u_a, u_b, counts

    # Helper: unique pairs for flattened vectors
    def _unique_pairs_flat(a, b):
        a1 = a.ravel()
        b1 = b.ravel()
        combo = np.stack([a1, b1], axis=1)
        u_combo, _counts = np.unique(combo, axis=0, return_counts=True)
        u_a = u_combo[:, 0]
        u_b = u_combo[:, 1]
        return u_a, u_b

    u_cell_type = None
    cts = None

    if search_type == 'cell':
        if isinstance(cell_type, np.ndarray):
            # Group by each unique cell type (same approach as the original)
            u_ctypes = np.unique(cell_type)
            uEe_list, uT_list, uctype_list = [], [], []
            for uct in u_ctypes:
                idx_ct = np.where(cell_type == uct)  # (row_idx, col_idx)
                Ee_sub = Ee[:, :, :, idx_ct[0], idx_ct[1]]
                T_sub = Tcell[:, :, :, idx_ct[0], idx_ct[1]]
                # Uniqueness over all selected elements (flatten)
                uEe_sub, uT_sub = _unique_pairs_flat(Ee_sub, T_sub)
                uEe_list.append(uEe_sub)
                uT_list.append(uT_sub)
                # Map each unique pair to this cell type
                uctype_list.append(uct * np.ones(uEe_sub.shape))
            uEe = np.concatenate(
                uEe_list) if len(uEe_list) > 0 else np.array([])
            uTcell = np.concatenate(
                uT_list) if len(uT_list) > 0 else np.array([])
            u_cell_type = np.concatenate(
                uctype_list) if len(uctype_list) > 0 else np.array([])
            cts = None  # keep behavior consistent with prior impl
            return uEe, uTcell, u_cell_type, cts

        else:
            uEe, uTcell = _unique_pairs_flat(Ee, Tcell)
            u_cell_type = np.zeros(uEe.shape)
            cts = None  # counts not returned previously for 'cell'
            return uEe, uTcell, u_cell_type, cts

    elif search_type == 'module':
        # Unique module-level (Num_cell_X x Num_cell_Y) pairs across all sims
        # and modX
        Ee_shp = Ee.shape
        uEe_list, uT_list = [], []
        for idx_0 in range(Ee_shp[0]):  # sim
            for idx_1 in range(Ee_shp[1]):  # modX
                # Slice across strings (Num_mod_Y) => shape
                # (Num_mod_Y, Num_cell_X, Num_cell_Y)
                Ee_sub = Ee[idx_0, idx_1, :]
                T_sub = Tcell[idx_0, idx_1, :]
                # Unique across the first axis (Num_mod_Y) to get unique
                # modules
                uEe_sub, uT_sub, _ = _unique_pairs_across_first_axis(
                    Ee_sub, T_sub)
                uEe_list.append(uEe_sub)
                uT_list.append(uT_sub)
        # Concatenate all unique modules collected per (sim, modX), then dedupe
        # again globally
        if len(uEe_list) > 0:
            uEe_all = np.concatenate(uEe_list)
            uT_all = np.concatenate(uT_list)
            uEe, uTcell, cts = _unique_pairs_across_first_axis(uEe_all, uT_all)
        else:
            uEe = np.empty((0,) + Ee.shape[3:])
            uTcell = np.empty((0,) + Tcell.shape[3:])
            cts = None
        u_cell_type = None
        return uEe, uTcell, u_cell_type, cts

    elif search_type == 'string':
        # Unique string-level (Num_mod_Y x Num_cell_X x Num_cell_Y) pairs
        Ee_shp = Ee.shape
        uEe_list, uT_list = [], []
        for idx_0 in range(Ee_shp[0]):  # sim
            # Slice across modX => shape (Num_mod_X, Num_mod_Y, Num_cell_X,
            # Num_cell_Y)
            Ee_sub = Ee[idx_0, :]
            T_sub = Tcell[idx_0, :]
            # Unique across first axis (Num_mod_X) to get unique strings
            uEe_sub, uT_sub, _ = _unique_pairs_across_first_axis(Ee_sub, T_sub)
            uEe_list.append(uEe_sub)
            uT_list.append(uT_sub)
        if len(uEe_list) > 0:
            uEe_all = np.concatenate(uEe_list)
            uT_all = np.concatenate(uT_list)
            # Final dedupe across all sims
            uEe, uTcell, _ = _unique_pairs_across_first_axis(uEe_all, uT_all)
        else:
            uEe = np.empty((0,) + Ee.shape[2:])
            uTcell = np.empty((0,) + Tcell.shape[2:])
        u_cell_type = None
        cts = None
        return uEe, uTcell, u_cell_type, cts

    elif search_type == 'system':
        # Unique system-level (Num_mod_X x Num_mod_Y x Num_cell_X x Num_cell_Y)
        # pairs
        uEe, uTcell, _ = _unique_pairs_across_first_axis(Ee, Tcell)
        u_cell_type = None
        cts = None
        return uEe, uTcell, u_cell_type, cts

    else:
        raise ValueError(
            'Incorrect search_type. Allowed search_type: cell, module, string, system.')


def calcTimeSeries(Ee_vec, Tcell_vec, sys_data):
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
    inverse, counts_ts, _unique_count = pairwise_inverse_timeseries(
        Ee_vec, Tcell_vec, round_decimals=None, use_void_view=True
    )
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


def calcsubModuleSystem(Ee_vec, Tcell_vec,
                        Ee_mod, Tcell_mod, NPT_dict,
                        run_bpact=True, run_annual=False,
                        save_bpact_freq=False, round_decimals=8,
                        pre_mod_idx=None, mfname_pre: str = 'mod_data',
                        res_path: Optional[str] = None):
    """
    Generate the system-level IV curves for the unique systems in a simulation.
    Sub-module level MPPT systems (each module's MPPT aggregates sub-modules).

    Pair-aware: all uniqueness/mapping use (Ee, Tcell).

    Parameters
    ----------
    Ee_vec : numpy.ndarray
        5-D system level irradiance array.
    Tcell_vec : numpy.ndarray
        5-D system level cell temperature array (same shape as Ee_vec).
    Ee_mod : numpy.ndarray
        (K_mod, Num_cell_X, Num_cell_Y) unique module Ee maps.
    Tcell_mod : numpy.ndarray
        (K_mod, Num_cell_X, Num_cell_Y) unique module Tcell maps
        (aligned with Ee_mod).
    mod_data : dict
        Dictionary containing sub-module curves:
            'Isubstr' : (K_mod, Nsubstr, Npts)
            'Vsubstr' : (K_mod, Nsubstr, Npts)
            'Psubstr' : (K_mod, Nsubstr, Npts)
        (Other keys not required for sub-module aggregation, but can exist.)
    NPT_dict : dict
        NPTs dictionary from pvcell (not directly used here).
    run_bpact, run_annual, save_bpact_freq : bool
        Kept for signature consistency; bypass arrays are not computed in this
        function.
    round_decimals : int
        Rounding precision for pair-aware matching (recommended: 6–8).
    pre_mod_idx : dict or None
        Optional precomputed global module key index from
        prepare_global_module_key_index.
        If None, it will be built internally.

    Returns
    -------
    sys_data : dict
        Dictionary containing system-level curves and metrics:
            'Isys', 'Vsys', 'Psys' (zeros),
            'Bypass_activation' (zeros),
            'Imp', 'Vmp', 'Pmp' (Pmp aggregated),
            'Isc', 'Voc', 'FF' (zeros),
            'Bypass_Active_MPP' (zeros),
            'num_active_bpd' (0).
    """
    # Build or reuse global module key index (pair-aware)
    if pre_mod_idx is None:
        pre_mod_idx = prepare_global_module_key_index(
            Ee_mod, Tcell_mod, round_decimals=round_decimals)

    # Unique systems (pair-aware)
    Ee_sys, Tcell_sys, _u_ct, _cts = get_unique_Ee_Tcell(Ee_vec, Tcell_vec,
                                                         search_type='system')
    Ee_shp = Ee_sys.shape

    # Aggregated Pmp per unique system
    Pmp = np.zeros(Ee_shp[0])

    # Iterate over unique systems and their strings
    for idx_0 in range(Ee_shp[0]):
        for idx_1 in range(Ee_shp[1]):
            Ee_str1 = Ee_sys[idx_0, idx_1, :, :, :]
            Tcell_str1 = Tcell_sys[idx_0, idx_1, :, :, :]

            # Vectorized per-string sub-module aggregation (pair-aware)
            Pmp_exp, counts, inverse, Mod_idxs = extract_string_submodule_pmp_pair_aware_from_pickles(
                Ee_str1=Ee_str1,
                Tcell_str1=Tcell_str1,
                mfname_pre=mfname_pre,
                res_path=res_path,
                pre_mod_idx=pre_mod_idx,
                calcMPP_func=calcMPP_IscVocFF,
            )

            # Sum module Pmp across the string for this unique system
            Pmp[idx_0] += Pmp_exp.sum()

    # Pair-aware expansion to full time-series order
    inverse_ts, counts_ts, _unique_ts = pairwise_inverse_timeseries(
        Ee_vec, Tcell_vec, round_decimals=round_decimals, use_void_view=True
    )

    Pmp_ts = Pmp[inverse_ts]

    # No bypass metrics computed here; set to zero/empty placeholders
    num_bpd_active_ts = 0
    BpDmp_full = np.zeros(Ee_vec.shape)

    # Build output dictionary (same fields as your original function)
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


def calcACSystem(
    Ee_vec: np.ndarray,
    Tcell_vec: np.ndarray,
    Ee_mod: np.ndarray,
    Tcell_mod: np.ndarray,
    NPT_dict: Dict[str, Any],
    run_bpact: bool = True,
    run_annual: bool = False,
    save_bpact_freq: bool = False,
    run_cellcurr: bool = True,
    round_decimals: int = 8,
    mfname_pre: str = 'mod_data',
    res_path: Optional[str] = None,
    sfname_pre: str = 'sys_data',
) -> Dict[str, Any]:
    """
    Generate system-level IV curves for an AC/MLPE system where modules operate
    independently.
    This version loads `mod_data` per-row from sharded pickle files in
    `res_path` using the filename pattern f"{mfname_pre}_{row_idx}.pickle".

    Parameters
    ----------
    Ee_vec : np.ndarray
        System-level irradiance array (5-D), e.g., (T, S, R, C, ?).
    Tcell_vec : np.ndarray
        System-level cell temperature array (5-D), same shape as Ee_vec.
    Ee_mod : np.ndarray
        3-D (K_mod, Num_cell_X, Num_cell_Y) irradiance maps for unique modules.
    Tcell_mod : np.ndarray
        3-D (K_mod, Num_cell_X, Num_cell_Y) Tcell maps for unique modules.
    NPT_dict : dict
        Points dictionary (not used directly here; kept for signature
                           compatibility).
    run_bpact : bool
        Calculate bypass diode metrics and (optionally) frequency maps.
    run_annual : bool
        If True, avoid materializing the large bypass frequency tensor in the
        output.
    save_bpact_freq : bool
        If True and run_bpact, save per-module bypass maps into output
        (memory heavy).
    run_cellcurr : bool
        If True, attach per-module scalars (Imp, Vmp, Isc, Pmp, num_bpd_active)
        per string.
    round_decimals : int
        Rounding precision for key building/matching.
    mfname_pre : str
        Filename prefix for per-row `mod_data` pickles.
    res_path : str or None
        Folder containing the pickles; if None, uses cwd.
    sfname_pre : str
        Prefix for system data naming (not used here; kept for compatibility).

    Returns
    -------
    sys_data : dict
        Aggregated AC/MLPE system timeseries with:
        - 'Pmp' (T,), main output
        - 'num_active_bpd' (T,), if run_bpact
        - 'Bypass_Active_MPP' (T, S, R, C, ...) or None
        (see run_annual/save_bpact_freq)
        - Optional 'full_data' containing per-module scalars by
        time-string-module indexing
          if run_cellcurr=True.
        Remaining keys are placeholders (zeros) for API compatibility.
    """

    Ee_shp = Ee_vec.shape  # Expect (U_sys, N_str, ...)

    # ------------------------------
    # 3) Prepare global module key index (pair-aware)
    # ------------------------------
    pre_mod_idx = prepare_global_module_key_index(
        Ee_mod, Tcell_mod, round_decimals=round_decimals
    )

    # ------------------------------
    # 4) Optional bypass frequency tensor (lazy allocate when we know shape)
    # ------------------------------
    BpDmp = None  # shape will be decided after first BpDmp_exp is available

    # ------------------------------
    # 5) Aggregators per unique system
    # ------------------------------
    Pmp = np.zeros(Ee_shp[0], dtype=float)
    if run_bpact:
        num_bpd_active = np.zeros(Ee_shp[0], dtype=float)
    else:
        num_bpd_active = 0.0

    # ------------------------------
    # 6) Iterate over unique systems
    # ------------------------------
    for idx_0 in range(Ee_shp[0]):
        if run_cellcurr:
            sim_data = []

        # Iterate over strings inside the system
        for idx_1 in range(Ee_shp[1]):
            # Extract one string (pair-aware Ee/Tcell)
            Ee_str1 = Ee_vec[idx_0, idx_1, :, :, :]
            Tcell_str1 = Tcell_vec[idx_0, idx_1, :, :, :]

            # Vectorized extraction of module curves and per-row scalars for
            # this string
            (
                # curves and meanIsc (not used for AC aggregation)
                Imod, Vmod, meanIsc,
                # (S,) per-module scalars (expanded)
                Pmp_exp, Imp_exp, Vmp_exp, Isc_exp,
                num_bpd_active_exp,         # (S,) if run_bpact else None
                BpDmp_exp,                 # optional bypass map per module row
                # unique row counts and inverse (available if needed)
                counts, inverse,
                # (S,) global row ids for this string
                Mod_idxs,
            ) = extract_string_module_curves_pair_aware_from_pickles(
                Ee_str1=Ee_str1,
                Tcell_str1=Tcell_str1,
                pre_mod_idx=pre_mod_idx,
                res_path=res_path,
                mfname_pre=mfname_pre,
                run_bpact=run_bpact,
                row_cache=None,
                num_workers=0,
                strict_shapes=True
            )

            # 6.a) Aggregate Pmp for this unique system
            # (AC/MLPE => sum of per-module MPPs)
            Pmp[idx_0] += Pmp_exp.sum()

            # 6.b) Aggregate bypass metrics
            if run_bpact and (num_bpd_active_exp is not None):
                if isinstance(num_bpd_active, np.ndarray):
                    num_bpd_active[idx_0] += num_bpd_active_exp.sum()

                # Save bypass frequency maps if requested
                # (and not doing run_annual)
                if save_bpact_freq and (BpDmp_exp is not None) and (
                        not run_annual):
                    # Lazily allocate BpDmp once we know the per-string shape
                    if BpDmp is None:
                        # BpDmp_exp is per-module in the string, commonly
                        # (S, L) or (S, A, B)
                        # We store a tensor indexed by (U_sys, N_str, ...)
                        # where ... == BpDmp_exp.shape
                        BpDmp = np.zeros(
                            (Ee_shp[0], Ee_shp[1]) + BpDmp_exp.shape,
                            dtype=BpDmp_exp.dtype
                        )
                    BpDmp[idx_0, idx_1, ...] = BpDmp_exp

            # 6.c) Per-module scalars collection (if requested)
            if run_cellcurr:
                S_modules = Imp_exp.shape[0]
                string_data = [None] * S_modules
                for mpos in range(S_modules):
                    module_data = {
                        'Mod_idx': int(Mod_idxs[mpos]),
                        'Pmp': float(Pmp_exp[mpos]),
                        'Imp': float(Imp_exp[mpos]),
                        'Vmp': float(Vmp_exp[mpos]),
                        'Isc': float(Isc_exp[mpos]),
                    }
                    if run_bpact and (num_bpd_active_exp is not None):
                        module_data['num_bpd_active'] = float(
                            num_bpd_active_exp[mpos])
                    if save_bpact_freq and (BpDmp_exp is not None) and (
                            not run_annual):
                        # Store the per-module bypass map for this module pos
                        module_data['Bypassed_substr'] = np.array(
                            BpDmp_exp[mpos, ...])
                    string_data[mpos] = module_data
                sim_data.append(string_data)

                # Save results to pickle
                fname = '_'.join([sfname_pre, str(idx_0)]) + '.pickle'
                fpath = os.path.join(res_path, fname)
                save_pickle(fpath, sim_data)

    # ------------------------------
    # 8) Build output dictionary (AC/MLPE)
    # ------------------------------
    sys_data: Dict[str, Any] = dict()
    # Placeholders for compatibility with original API
    sys_data['Isys'] = np.zeros(Pmp.shape)
    sys_data['Vsys'] = np.zeros(Pmp.shape)
    sys_data['Psys'] = np.zeros(Pmp.shape)
    sys_data['Bypass_activation'] = np.zeros(Pmp.shape)
    sys_data['Imp'] = np.zeros(Pmp.shape)
    sys_data['Vmp'] = np.zeros(Pmp.shape)
    sys_data['Isc'] = np.zeros(Pmp.shape)
    sys_data['Voc'] = np.zeros(Pmp.shape)
    sys_data['FF'] = np.zeros(Pmp.shape)

    # Primary AC results
    sys_data['Pmp'] = Pmp
    sys_data['Bypass_Active_MPP'] = BpDmp         # None unless saved
    sys_data['num_active_bpd'] = num_bpd_active     # (T,) or 0.0

    fname = '_'.join([sfname_pre, 'BypassMPP']) + '.pickle'
    fpath = os.path.join(res_path, fname)
    save_pickle(fpath, BpDmp)
    return sys_data


def _row_void_view_1d(arr_2d: np.ndarray) -> np.ndarray:
    """
    Return a 1-D void view for lexicographic comparisons/searchsorted on 2-D
    rows.
    Equivalent to what you used elsewhere for pair-aware keying.
    """
    arr_2d = np.ascontiguousarray(arr_2d)
    return arr_2d.view(np.dtype((
        np.void, arr_2d.dtype.itemsize * arr_2d.shape[1]))).ravel()


def extract_system_string_curves_pair_aware_from_pickles(
    Ee_sys1: np.ndarray,
    Tcell_sys1: np.ndarray,
    pre_str_idx: Dict[str, Any],
    res_path: str,
    stfname_pre: str,
    run_bpact: bool = True,
    return_pstring: bool = True,
    row_cache: Optional[Dict[int, Dict[str, Any]]] = {},
    num_workers: int = 0,
    strict_shapes: bool = True,
) -> Tuple[
    np.ndarray,                # Istring: (N_str, Npts)
    np.ndarray,                # Vstring: (N_str, Npts)
    # Pstring: (N_str, Npts) or None if return_pstring=False
    Optional[np.ndarray],
    Optional[np.ndarray],      # Bypassed_substr: (N_str, *B_shape) or None
    np.ndarray,                # counts: (U,), unique-local counts
    # inverse: (N_str,), map to expand back to string order
    np.ndarray,
    # Str_idxs: (N_str,), global row ids aligned with string order
    np.ndarray,
]:
    """
    Pair-aware extraction of per-string IV curves using precomputed string key index,
    loading sharded `str_data` per-row pickle files.

    Each required unique string row is loaded from:
        os.path.join(res_path, f"{stfname_pre}_{row_idx}.pickle")

    Required keys in each row pickle:
        - 'Istring' : ndarray, shape (Npts,)
        - 'Vstring' : ndarray, shape (Npts,)
        - optional 'Pstring' : ndarray, shape (Npts,), computed if missing
        - optional 'Bypassed_substr' if run_bpact=True (must have consistent shape across rows)
    """
    # 1) Unpack index & flatten pair-aware rows (strings per system)
    M = pre_str_idx['M']  # number of (pair-)entries per string row
    order = pre_str_idx['order']
    rows_void_sorted = pre_str_idx['rows_void_sorted']
    round_decimals = pre_str_idx.get('round_decimals', None)

    # Expect Ee_sys1/Tcell_sys1 flattenable to (N_str, M)
    N_str = Ee_sys1.shape[0]
    Ee_flat = Ee_sys1.reshape(N_str, M)
    T_flat = Tcell_sys1.reshape(N_str, M)

    if round_decimals is not None:
        Ee_flat = np.round(Ee_flat, round_decimals)
        T_flat = np.round(T_flat, round_decimals)

    # Build (N_str, 2*M) signature and get unique + inverse
    str_rows = np.concatenate([Ee_flat, T_flat], axis=1)
    u_str_rows, inverse, counts = np.unique(
        np.ascontiguousarray(str_rows), axis=0, return_inverse=True, return_counts=True
    )

    # 2) Map unique local rows to global row indices using void view + searchsorted
    u_void = _row_void_view_1d(u_str_rows)
    pos = np.searchsorted(rows_void_sorted, u_void)
    valid = (pos >= 0) & (pos < rows_void_sorted.size) & (
        rows_void_sorted[pos] == u_void)
    if not np.all(valid):
        bad = np.where(~valid)[0].tolist()
        raise ValueError(
            f"String keys not found in global unique strings (example indices: {bad[:10]}). "
            f"Check rounding policy (round_decimals={round_decimals})."
        )

    str_idxs_unique = order[pos]         # (U,)
    Str_idxs = str_idxs_unique[inverse]  # (N_str,)

    # 3) Load required unique rows from disk with optional cache/parallel I/O
    unique_rows = np.unique(str_idxs_unique)
    cache = row_cache if row_cache is not None else {}

    def _row_file_path(row_idx: int) -> str:
        return os.path.join(res_path, f"{stfname_pre}_{int(row_idx)}.pickle")

    def _to_1d(a: Any, what: str, fpath: str) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 1:
            return a
        if a.ndim == 2 and 1 in a.shape:
            return a.reshape(-1)
        if strict_shapes:
            raise ValueError(
                f"Expected 1-D for {what} in '{fpath}', got shape {a.shape}."
            )
        return a.reshape(-1)

    def _load_one_row(row_idx: int) -> Dict[str, Any]:
        if row_idx in cache:
            return cache[row_idx]
        fpath = _row_file_path(row_idx)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Missing str_data row pickle: '{fpath}'. Expected '{stfname_pre}_<row_idx>.pickle'."
            )
        with open(fpath, "rb") as f:
            row = pickle.load(f)

        # Required IV curves
        if 'Istring' not in row or 'Vstring' not in row:
            raise KeyError(
                f"Row file '{fpath}' must contain 'Istring' and 'Vstring'.")

        Istr = _to_1d(row['Istring'], 'Istring', fpath)
        Vstr = _to_1d(row['Vstring'], 'Vstring', fpath)

        # Optional Pstring
        if return_pstring:
            if 'Pstring' in row:
                Pstr = _to_1d(row['Pstring'], 'Pstring', fpath)
            else:
                Pstr = Istr * Vstr  # compute if absent
        else:
            Pstr = None

        # Optional bypass map
        if run_bpact:
            if 'Bypassed_substr' not in row:
                raise KeyError(
                    f"Row file '{fpath}' missing 'Bypassed_substr' required when run_bpact=True."
                )
            B = np.asarray(row['Bypassed_substr'])
        else:
            B = None

        cache[row_idx] = {'Istring': Istr, 'Vstring': Vstr,
                          'Pstring': Pstr, 'Bypassed_substr': B}
        return cache[row_idx]

    if num_workers and num_workers > 1 and unique_rows.size > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_load_one_row, int(r)): int(r)
                       for r in unique_rows}
            for fut in as_completed(futures):
                _ = fut.result()
    else:
        for r in unique_rows:
            _load_one_row(int(r))

    # 4) Assemble reduced arrays in unique order
    sample = cache[int(unique_rows[0])]
    Npts = int(sample['Istring'].shape[0])

    U = str_idxs_unique.shape[0]
    Istr_red = np.empty((U, Npts), dtype=sample['Istring'].dtype)
    Vstr_red = np.empty((U, Npts), dtype=sample['Vstring'].dtype)
    if return_pstring:
        Pstr_red = np.empty((U, Npts), dtype=sample['Pstring'].dtype)

    if run_bpact:
        B_shape = None if sample['Bypassed_substr'] is None else np.asarray(
            sample['Bypassed_substr']).shape
        if strict_shapes and B_shape is not None and len(unique_rows) > 1:
            for r in unique_rows[1:]:
                if np.asarray(cache[int(r)]['Bypassed_substr']).shape != B_shape:
                    raise ValueError(
                        f"Inconsistent Bypassed_substr shapes across rows. "
                        f"Row {int(unique_rows[0])}: {B_shape}, Row {int(r)}: "
                        f"{np.asarray(cache[int(r)]['Bypassed_substr']).shape}"
                    )
        Bstr_red = None if B_shape is None else np.empty((U,) + B_shape,
                                                         dtype=np.asarray(sample['Bypassed_substr']).dtype)
    else:
        Bstr_red = None

    for i, ridx in enumerate(str_idxs_unique):
        row = cache[int(ridx)]
        if strict_shapes and row['Istring'].shape[0] != Npts:
            raise ValueError(
                f"Inconsistent Npts in string row {int(ridx)}: got {row['Istring'].shape[0]}, expected {Npts}."
            )
        Istr_red[i, :] = row['Istring']
        Vstr_red[i, :] = row['Vstring']
        if return_pstring:
            Pstr_red[i, :] = row['Pstring']
        if run_bpact:
            B = row['Bypassed_substr']
            if Bstr_red is not None:
                Bstr_red[i, ...] = B

    # 5) Expand to full system string order
    Istring = Istr_red[inverse, :]
    Vstring = Vstr_red[inverse, :]
    Pstring = Pstr_red[inverse, :] if return_pstring else None
    Bypassed_substr = Bstr_red[inverse, ...] if (
        run_bpact and Bstr_red is not None) else None

    return Istring, Vstring, Pstring, Bypassed_substr, counts, inverse, Str_idxs


def calcSystem(
    Ee_sys: np.ndarray,
    Tcell_sys: np.ndarray,
    Ee_str: np.ndarray,
    Tcell_str: np.ndarray,
    # kept for API compatibility; NOT used in pickle mode
    NPT_dict: Dict[str, Any],
    run_bpact: bool = True,
    run_annual: bool = False,
    run_cellcurr: bool = False,
    res_path: Optional[str] = None,
    stfname_pre: str = 'str_data',
    sfname_pre: str = 'sys_data',
) -> Dict[str, Any]:
    """
    Generate the system-level IV curves for the unique systems in a DC string
    simulation.

    This version assumes `str_data` is sharded into per-row pickle files stored
    in `res_path`, named f"{stfname_pre}_{row_idx}.pickle", where `row_idx`
    corresponds to the global unique string index produced by
    `prepare_global_string_key_index(...)`.

    Required keys per row pickle:
        - 'Istring' : (Npts,)
        - 'Vstring' : (Npts,)
        - optional 'Pstring' : (Npts,) (computed as I*V if absent)
        - if run_bpact=True: 'Bypassed_substr' : ndarray with consistent shape
        across rows

    Returns
    -------
    sys_data : dict
        Dictionary containing system IV curves and optional bypass activation
        arrays.
    """
    # Containers
    I_sys_curves = []
    V_sys_curves = []
    P_sys_curves = []
    Bypass_str_curves = []

    # NPTs
    pts = NPT_dict['pts'][0, :].reshape(NPT_dict['pts'].shape[1], 1)
    negpts = NPT_dict['negpts'][0, :].reshape(NPT_dict['negpts'].shape[1], 1)
    Npts = int(NPT_dict['Npts'])

    # Precompute global string index ONCE (hoisted)
    pre_str_idx = prepare_global_string_key_index(
        Ee_str, Tcell_str, round_decimals=None
    )

    # Iterate over unique systems
    for idx_sys in range(Ee_sys.shape[0]):
        if run_cellcurr:
            sing_sys = {}

        # Extract one system's string-level (pair-aware) Ee/T
        Ee_sys1 = Ee_sys[idx_sys]
        Tcell_sys1 = Tcell_sys[idx_sys]

        # Load and map per-string IV curves for this system from sharded
        # str_data
        (
            Istring, Vstring, Pstring, Bypass_substr,
            counts, inverse, Str_idxs
        ) = extract_system_string_curves_pair_aware_from_pickles(
            Ee_sys1=Ee_sys1,
            Tcell_sys1=Tcell_sys1,
            pre_str_idx=pre_str_idx,
            res_path=(res_path or os.getcwd()),
            stfname_pre=stfname_pre,
            run_bpact=run_bpact,
            return_pstring=True,
            row_cache=None,   # set to a persistent dict across calls for speed
            num_workers=0,        # >0 to parallelize I/O if storage benefits
            strict_shapes=True
        )

        # Optional: capture per-system string data
        if run_cellcurr:
            sing_sys['Istrings'] = Istring.copy()
            sing_sys['Vstrings'] = Vstring.copy()
            sing_sys['Str_idxs'] = Str_idxs.copy()

        # System circuit model (strings in PARALLEL with bypass handling)
        Isys, Vsys, bypassed_str = calcParallel_with_bypass(
            Istring, Vstring, Vstring.max(), Vstring.min(), negpts, pts, Npts,
            Bypass_substr, run_bpact=run_bpact
        )
        Psys = Isys * Vsys

        I_sys_curves.append(np.reshape(Isys, (1, len(Isys))))
        V_sys_curves.append(np.reshape(Vsys, (1, len(Vsys))))
        P_sys_curves.append(np.reshape(Psys, (1, len(Psys))))

        if run_bpact:
            Bypass_str_curves.append(
                np.reshape(
                    bypassed_str,
                    (1, bypassed_str.shape[0], bypassed_str.shape[1],
                     bypassed_str.shape[2], bypassed_str.shape[3])
                )
            )
        else:
            Bypass_str_curves.append(bypassed_str)

        if run_cellcurr:
            # Save results to pickle
            fname = '_'.join([sfname_pre, str(idx_sys)]) + '.pickle'
            fpath = os.path.join(res_path, fname)
            save_pickle(fpath, sing_sys)

    # Concatenate per-system results
    I_sys_curves = np.concatenate(I_sys_curves, axis=0)
    V_sys_curves = np.concatenate(V_sys_curves, axis=0)
    P_sys_curves = np.concatenate(P_sys_curves, axis=0)
    if run_bpact:
        Bypass_str_curves = np.concatenate(Bypass_str_curves, axis=0)
    else:
        Bypass_str_curves = np.array(Bypass_str_curves)

    # No in-memory str_data to delete in annual mode (keep interface parity)
    # if run_annual: (nothing to delete from str_data here)

    # Compute MPP, Isc/Voc/FF and bypass-at-MPP
    Imp, Vmp, Pmp, Isc, Voc, FF, BpDmp, num_bpd_active = calcMPP_IscVocFFBPD(
        I_sys_curves, V_sys_curves, P_sys_curves, Bypass_str_curves,
        run_bpact=run_bpact, run_annual=run_annual
    )

    # Build output
    sys_data: Dict[str, Any] = dict()
    sys_data['Isys'] = I_sys_curves
    sys_data['Vsys'] = V_sys_curves
    sys_data['Psys'] = P_sys_curves

    if run_annual:
        sys_data['Bypass_activation'] = np.nan
        # Explicitly drop big arrays to help GC
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

    return sys_data


def prepare_global_string_key_index(Ee_str, Tcell_str, round_decimals=None):
    K_str, modY, cellX, cellY = Ee_str.shape
    M = modY * cellX * cellY

    Ee_flat = Ee_str.reshape(K_str, M)
    T_flat = Tcell_str.reshape(K_str, M)

    if round_decimals is not None:
        Ee_flat = np.round(Ee_flat, round_decimals)
        T_flat = np.round(T_flat, round_decimals)

    rows = np.concatenate([Ee_flat, T_flat], axis=1)  # (K_str, 2*M)

    row_void_1d = row_void_view_1d(rows)             # shape: (K_str,)
    order = np.argsort(row_void_1d)
    rows_void_sorted = row_void_1d[order]

    return {
        'M': M,
        'order': order,
        'rows_void_sorted': rows_void_sorted,
        'round_decimals': round_decimals,
    }


def extract_system_string_curves_pair_aware(
    Ee_sys1,
    Tcell_sys1,
    str_data,
    pre_str_idx,
    run_bpact=True,
    return_pstring=True,  # if True, also return Pstring from str_data
):
    """
    Vectorized extraction of string IV curves for a single system using (Ee, Tcell) pairs.
    """

    M = pre_str_idx['M']
    order = pre_str_idx['order']            # (K_str,)
    rows_void_sorted = pre_str_idx['rows_void_sorted']  # (K_str,)
    round_decimals = pre_str_idx['round_decimals']

    # --- Flatten system strings (row per string) ---
    S = Ee_sys1.shape[0]  # Num_strings_in_system
    Ee_flat = Ee_sys1.reshape(S, M)
    T_flat = Tcell_sys1.reshape(S, M)

    if round_decimals is not None:
        Ee_flat = np.round(Ee_flat, round_decimals)
        T_flat = np.round(T_flat, round_decimals)

    # Build row-wise keys for strings in this system
    sys_rows = np.concatenate([Ee_flat, T_flat], axis=1)  # (S, 2*M)

    # --- System-local unique string patterns, with inverse for expansion ---
    u_sys_rows, inverse, counts = np.unique(
        sys_rows, axis=0, return_inverse=True, return_counts=True
    )

    # --- Create a 1-D void view for searchsorted ---
    u_void = row_void_view_1d(u_sys_rows)            # (U,)

    # Sanity: ensure our global sorted keys are 1-D
    if rows_void_sorted.ndim != 1:
        rows_void_sorted = rows_void_sorted.reshape(-1)

    # --- Map unique system string keys to global unique strings via binary search ---
    pos = np.searchsorted(rows_void_sorted, u_void)  # (U,)
    valid = (pos >= 0) & (pos < rows_void_sorted.size) & (
        rows_void_sorted[pos] == u_void)
    if not np.all(valid):
        bad = np.where(~valid)[0]
        sample = bad[:10].tolist()
        raise ValueError(
            f"System string keys not found in global unique strings (example indices: {sample}). "
            f"Ensure identical rounding policy (round_decimals={round_decimals}) was used."
        )

    # Indices into the global unique string list (aligns with str_data arrays)
    # (U,) ← 1-D by construction
    str_idxs_unique = order[pos]
    Str_idxs = str_idxs_unique[inverse]               # (S,)

    # --- Extract reduced arrays from str_data using the mapped indices ---
    Istr_red = str_data['Istring'][str_idxs_unique, :]      # (U, Npts)
    Vstr_red = str_data['Vstring'][str_idxs_unique, :]      # (U, Npts)

    if return_pstring and ('Pstring' in str_data):
        Pstr_red = str_data['Pstring'][str_idxs_unique, :]  # (U, Npts)
    else:
        Pstr_red = None

    if run_bpact:
        # NOTE: typical shape is (K_str, S1, S2); index with 3 dims
        # (U, S1, S2)
        Bypass_substr_red = str_data['Bypassed_substr'][str_idxs_unique, :, :, :]
    else:
        Bypass_substr_red = np.nan

    # --- Expand back to system order using inverse ---
    Istring = Istr_red[inverse, :]                         # (S, Npts)
    Vstring = Vstr_red[inverse, :]                         # (S, Npts)
    Pstring = Pstr_red[inverse, :] if Pstr_red is not None else None
    Bypassed_substr = Bypass_substr_red[inverse,
                                        :, :, :] if run_bpact else np.nan

    return Istring, Vstring, Pstring, Bypassed_substr, counts, inverse, Str_idxs


def pairwise_inverse_timeseries(Ee_vec, Tcell_vec, round_decimals=None,
                                use_void_view=True):
    """
    Compute a pair-aware inverse mapping for expanding aggregated arrays back
    to full time-series order, using (Ee, Tcell) together to define uniqueness.

    Parameters
    ----------
    Ee_vec : np.ndarray
        System-level irradiance array (expected 5-D, but any array with time
                                       axis on axis=0 is supported).
        Shape: (sim, ...).
    Tcell_vec : np.ndarray
        System-level cell temperature array with the same shape as Ee_vec.
    round_decimals : int or None, optional
        If set, rounds both Ee_vec and Tcell_vec before computing uniqueness.
        Use the same value across your pipeline (e.g., 6–8) to mitigate tiny
        floating-point mismatches.
    use_void_view : bool, optional
        If True (default), uses a row-wise void view trick for efficient
        uniqueness. If False, uses `np.unique` directly on concatenated rows.

    Returns
    -------
    inverse_ts : np.ndarray
        1-D array of length `sim`. For each time step, this is the index of the
        corresponding unique (Ee, Tcell) row.
    counts_ts : np.ndarray
        Counts per unique row (same order as the unique row set).
    unique_count : int
        Number of unique (Ee, Tcell) rows across the time series.

    Notes
    -----
    - This function mirrors the pair-aware uniqueness used elsewhere
      (system/string/module matching) so reconstructed arrays align perfectly.
    - It is safe for very large runs; the void-view path performs row-wise
      equality/ordering efficiently on concatenated flattened rows.
    """

    # --- Basic validation ---
    if not (isinstance(Ee_vec, np.ndarray) and isinstance(
            Tcell_vec, np.ndarray)):
        raise TypeError("Ee_vec and Tcell_vec must be numpy arrays.")
    if Ee_vec.shape != Tcell_vec.shape:
        raise ValueError(
            f"Ee_vec and Tcell_vec shapes must match. Got Ee{Ee_vec.shape} vs Tcell{Tcell_vec.shape}.")
    if Ee_vec.ndim < 1:
        raise ValueError(
            f"Expected time axis on axis=0. Got Ee_vec.ndim={Ee_vec.ndim}")

    # --- Flatten rows: each time step becomes a single row ---
    sim = Ee_vec.shape[0]
    M = int(np.prod(Ee_vec.shape[1:]))
    Ee_flat = Ee_vec.reshape(sim, M)
    Tcell_flat = Tcell_vec.reshape(sim, M)

    # Optional quantization for robust equality on floats
    if round_decimals is not None:
        Ee_flat = np.round(Ee_flat, round_decimals)
        Tcell_flat = np.round(Tcell_flat, round_decimals)

    # Build concatenated row matrix: [Ee | Tcell]
    rows = np.concatenate([Ee_flat, Tcell_flat], axis=1)  # shape: (sim, 2*M)
    rows = np.ascontiguousarray(rows)

    if use_void_view:
        # Row-wise void view (treat each row as a single binary blob)
        rows_void = rows.view(np.dtype((np.void,
                                        rows.dtype.itemsize * rows.shape[1])))
        # Unique via void-view; inverse maps each time step to its unique index
        _u_void_ts, inverse_ts, counts_ts = np.unique(rows_void,
                                                      return_inverse=True,
                                                      return_counts=True)
        unique_count = _u_void_ts.shape[0]
    else:
        # Direct unique on 2D rows
        # (slightly heavier but fine for smaller/mid cases)
        _u_rows_ts, inverse_ts, counts_ts = np.unique(rows, axis=0,
                                                      return_inverse=True,
                                                      return_counts=True)
        unique_count = _u_rows_ts.shape[0]

    return inverse_ts, counts_ts, unique_count


def extract_string_submodule_pmp_pair_aware_from_pickles(
    Ee_str1: np.ndarray,
    Tcell_str1: np.ndarray,
    pre_mod_idx: Dict[str, Any],
    calcMPP_func,
    res_path: str,
    mfname_pre: str,
    row_cache: Optional[Dict[int, Dict[str, Any]]] = None,
    num_workers: int = 0,
    strict_shapes: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized per-string mapping of modules via (Ee, Tcell) pairs and
    computation of module-level Pmp by summing sub-module MPPs, with mod_data
    read from per-row pickle shards.

    Each required unique module row is loaded from:
        os.path.join(res_path, f"{mfname_pre}_{row_idx}.pickle")

    Expected keys in each row pickle:
        - 'Isubstr': ndarray, shape (Nsubstr, Npts)
        - 'Vsubstr': ndarray, shape (Nsubstr, Npts)
        - 'Psubstr': ndarray, shape (Nsubstr, Npts)

    The provided `calcMPP_func(Isubstr, Vsubstr, Psubstr)` must return:
        (Imp_s, Vmp_s, Pmp_s, Isc_s, Voc_s, FF_s)
    where `Pmp_s` is per-substring (shape: (Nsubstr,)) or (Nsubstr, 1).

    Parameters
    ----------
    Ee_str1, Tcell_str1 : np.ndarray
        Pair-aware irradiance and temperature for modules in a string.
        Each flattens to shape (S, M) where S = Num_mod_Y and M = modules/pairs
        per row.
    pre_mod_idx : dict
        Must include:
            - 'M' : int
            - 'order' : np.ndarray
            (maps global sorted unique rows -> global row indices)
            - 'rows_void_sorted' : np.ndarray
            (1-D void view used by searchsorted)
            - 'round_decimals' : Optional[int]
    calcMPP_func : callable
        Function that computes per-substring MPP quantities given
        (Isubstr, Vsubstr, Psubstr).
    res_path : str
        Directory where row pickles are stored.
    mfname_pre : str
        Filename prefix for the per-row mod_data pickles (e.g., "mod_data").
    row_cache : dict, optional
        Optional cache mapping row_idx -> row_dict to avoid repeated I/O across
        calls.
    num_workers : int, optional
        If > 1, loads unique rows in parallel via a thread pool.
    strict_shapes : bool, optional
        If True, enforces shape checks
        (Nsubstr and Npts consistency per row, Pmp_s 1-D coercion).

    Returns
    -------
    Pmp_exp : np.ndarray
        (S,) module-level Pmp per position in this string.
    counts : np.ndarray
        (U,) counts per unique module pattern present in this string
        (string-local).
    inverse : np.ndarray
        (S,) inverse index mapping from per-position modules to unique module
        patterns.
    Mod_idxs : np.ndarray
        (S,) global unique module indices aligned with string order.
    """
    # ------------------------------------------------------------
    # 1) Unpack precomputed index and flatten pair-aware string rows
    # ------------------------------------------------------------
    M = pre_mod_idx['M']
    order = pre_mod_idx['order']
    rows_void_sorted = pre_mod_idx['rows_void_sorted']
    round_decimals = pre_mod_idx.get('round_decimals', None)

    S = Ee_str1.shape[0]  # number of module positions along the string
    Ee_flat = Ee_str1.reshape(S, M)
    T_flat = Tcell_str1.reshape(S, M)

    if round_decimals is not None:
        Ee_flat = np.round(Ee_flat, round_decimals)
        T_flat = np.round(T_flat, round_decimals)

    # Build (S, 2*M) signature rows; ensure contiguous for void view
    str_rows = np.ascontiguousarray(np.concatenate([Ee_flat, T_flat], axis=1))

    # ------------------------------------------------------------
    # 2) Unique local rows + inverse for expansion
    # ------------------------------------------------------------
    u_str_rows, inverse, counts = np.unique(
        str_rows, axis=0, return_inverse=True, return_counts=True
    )

    # Create 1-D void view for vectorized binary search
    # (compatibly with how rows_void_sorted was constructed)
    void_dtype = np.dtype(
        (np.void, u_str_rows.dtype.itemsize * u_str_rows.shape[1]))
    u_void = u_str_rows.view(void_dtype).ravel()

    pos = np.searchsorted(rows_void_sorted, u_void)
    valid = (pos >= 0) & (pos < rows_void_sorted.size) & (
        rows_void_sorted[pos] == u_void)
    if not np.all(valid):
        bad = np.where(~valid)[0].tolist()
        sample = bad[:10]
        raise ValueError(
            f"String module keys not found in global unique modules (example indices: {sample}). "
            f"Check rounding policy (round_decimals={round_decimals})."
        )

    # Map to global row indices (these match the row pickle suffixes)
    mod_idxs_unique = order[pos]       # (U,)
    Mod_idxs = mod_idxs_unique[inverse]  # (S,)

    # ------------------------------------------------------------
    # 3) Load unique rows (Isubstr, Vsubstr, Psubstr) from disk
    # ------------------------------------------------------------
    unique_rows = np.unique(mod_idxs_unique)
    cache = row_cache if row_cache is not None else {}

    def _row_file_path(row_idx: int) -> str:
        return os.path.join(res_path, f"{mfname_pre}_{int(row_idx)}.pickle")

    def _load_one_row(row_idx: int) -> Dict[str, Any]:
        if row_idx in cache:
            return cache[row_idx]

        fpath = _row_file_path(row_idx)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Missing mod_data row pickle: '{fpath}'. "
                f"Expected file named '{mfname_pre}_<row_idx>.pickle'."
            )

        with open(fpath, "rb") as f:
            row = pickle.load(f)

        # Required arrays
        for k in ('Isubstr', 'Vsubstr', 'Psubstr'):
            if k not in row:
                raise KeyError(
                    f"Row file '{fpath}' missing required key '{k}'.")

        # Normalize to ndarray
        I = np.asarray(row['Isubstr'])
        V = np.asarray(row['Vsubstr'])
        P = np.asarray(row['Psubstr'])

        # Validate basic shapes: expect 2-D (Nsubstr, Npts)
        if strict_shapes:
            for name, arr in (('Isubstr', I), ('Vsubstr', V), ('Psubstr', P)):
                if arr.ndim != 2:
                    raise ValueError(
                        f"Expected 2-D array for '{name}' in '{fpath}', got shape {arr.shape}."
                    )
            if not (I.shape == V.shape == P.shape):
                raise ValueError(
                    f"Mismatched substring shapes in '{fpath}': "
                    f"I{I.shape}, V{V.shape}, P{P.shape}."
                )

        row['Isubstr'] = I
        row['Vsubstr'] = V
        row['Psubstr'] = P

        cache[row_idx] = row
        return row

    # Parallel/sequential load
    if num_workers and num_workers > 1 and unique_rows.size > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_load_one_row, int(r)): int(r)
                       for r in unique_rows}
            for fut in as_completed(futures):
                _ = fut.result()
    else:
        for r in unique_rows:
            _load_one_row(int(r))

    # ------------------------------------------------------------
    # 4) Compute Pmp per unique module by summing substring MPPs
    # ------------------------------------------------------------
    U = mod_idxs_unique.shape[0]
    Pmp_red = np.zeros((U,), dtype=float)

    for i, midx in enumerate(mod_idxs_unique):
        r = int(midx)
        row = cache[r]
        I_sub = row['Isubstr']  # (Nsubstr, Npts)
        V_sub = row['Vsubstr']  # (Nsubstr, Npts)
        P_sub = row['Psubstr']  # (Nsubstr, Npts)

        # calcMPP_func should vectorize over substrings along axis=0
        Imp_s, Vmp_s, Pmp_s, Isc_s, Voc_s, FF_s = calcMPP_func(
            I_sub, V_sub, P_sub)

        # Coerce Pmp_s to 1-D and sum across substrings
        Pmp_s = np.asarray(Pmp_s)
        if Pmp_s.ndim == 2 and 1 in Pmp_s.shape:
            Pmp_s = Pmp_s.reshape(-1)
        elif Pmp_s.ndim != 1 and strict_shapes:
            raise ValueError(
                f"calcMPP_func returned Pmp_s with unexpected shape {Pmp_s.shape} "
                f"for row {r}; expected (Nsubstr,) or singleton 2-D."
            )
        Pmp_red[i] = float(Pmp_s.sum())

    # ------------------------------------------------------------
    # 5) Expand to full string order and return
    # ------------------------------------------------------------
    Pmp_exp = Pmp_red[inverse]  # (S,)

    return Pmp_exp, counts, inverse, Mod_idxs
