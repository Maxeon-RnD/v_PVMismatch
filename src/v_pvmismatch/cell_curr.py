# -*- coding: utf-8 -*-
"""Estimate the current, voltage, power, cells in reverse during mismatch.

The functions also calculate the diode currents.
The methodology is solving the circuit (similar to LTSpice).
"""

import os

import numpy as np
from typing import Dict, Any
import pickle
from functools import lru_cache
from typing import Iterable, Tuple, Optional

from .utils import load_pickle


def _last_min_index(a: np.ndarray) -> int:
    """
    Return the last index where a equals its minimum.
    Equivalent to np.where(a == a.min())[0][-1], but safe when min repeats.
    """
    # np.argwhere returns 2-D for general arrays; we need the last index along the first axis.
    # For 1-D arrays, the first column contains the indices.
    idxs = np.where(a == np.min(a))[0]
    return int(idxs[-1]) if idxs.size > 0 else 0


def _interp_rows_scalar_npinterp_equiv(x: float,
                                       # (R, P) monotonically non-decreasing along axis=1
                                       x_rows: np.ndarray,
                                       # (R, P) aligned with x_rows
                                       y_rows: np.ndarray
                                       ) -> np.ndarray:
    """
    Exact row-wise equivalent of np.interp(x, x_rows[r,:], y_rows[r,:]) for a single scalar x, across many rows.

    Semantics (matches np.interp):
      - If x < x_rows[r, 0], returns y_rows[r, 0]  (left clamp).
      - If x > x_rows[r, -1], returns y_rows[r, -1] (right clamp).
      - Else linear interpolation between the bracketing samples:
            y = y_l + (x - x_l) * (y_u - y_l) / (x_u - x_l)
        If x equals a sample value (x == x_l == x_u), returns y_l.

    Handles duplicates (weakly increasing x_rows) and produces identical results
    to per-row np.interp for well-formed inputs.
    """
    x_rows = np.asarray(x_rows)
    y_rows = np.asarray(y_rows)

    if x_rows.ndim != 2 or y_rows.ndim != 2:
        raise ValueError(
            "x_rows and y_rows must be 2-D arrays of shape (R, P).")
    if x_rows.shape != y_rows.shape:
        raise ValueError("x_rows and y_rows must have identical shapes.")

    R, P = x_rows.shape
    y_out = np.empty(R, dtype=y_rows.dtype)

    # Endpoint clamps — identical to np.interp
    left_mask = (x < x_rows[:, 0])
    right_mask = (x > x_rows[:, -1])

    y_out[left_mask] = y_rows[left_mask,  0]
    y_out[right_mask] = y_rows[right_mask, -1]

    # Interior rows
    interior_mask = ~(left_mask | right_mask)
    if np.any(interior_mask):
        X = x_rows[interior_mask]  # (R_int, P)
        Y = y_rows[interior_mask]  # (R_int, P)

        # Lower bracket index: last j with X[j] <= x
        mask = (X <= x)                           # (R_int, P)
        rev = mask[:, ::-1]                      # reverse columns
        # (R_int,) position from the right
        idx_from_right = np.argmax(rev, axis=1)
        # (R_int,) absolute lower indices
        idx_l = P - 1 - idx_from_right

        # Upper bracket index
        idx_u = np.clip(idx_l + 1, 0, P - 1)

        # Gather brackets (vectorized)
        idx_l_exp = idx_l[:, None]
        idx_u_exp = idx_u[:, None]
        x_l = np.take_along_axis(X, idx_l_exp, axis=1).squeeze(1)
        x_u = np.take_along_axis(X, idx_u_exp, axis=1).squeeze(1)
        y_l = np.take_along_axis(Y, idx_l_exp, axis=1).squeeze(1)
        y_u = np.take_along_axis(Y, idx_u_exp, axis=1).squeeze(1)

        # Linear interpolation; exact np.interp semantics at boundaries and duplicates
        denom = (x_u - x_l)
        with np.errstate(divide='ignore', invalid='ignore'):
            weight = np.where(denom != 0.0, (x - x_l) / denom, 0.0)

        y_out[interior_mask] = y_l + weight * (y_u - y_l)

    return y_out


def est_cell_current_DC(
    sim_data: Dict[str, Any],
    res_path: str,
    sfname_pre: str,
    stfname_pre: str,
    mfname_pre: str,
    cell_index: np.ndarray,
    run_isc: bool = False,
    # cache tuning knobs:
    max_cache_items: int = 2048,     # total files cached by path
    clear_cache_on_exit: bool = True
) -> Dict[str, np.ndarray]:
    """
    Estimate per-cell DC quantities at MPP and SC points, with bypass logic.
    This version uses internal LRU caches for all on-disk loads and keeps the
    original algorithmic semantics intact (np.interp-equivalent behavior).

    Parameters
    ----------
    sim_data : dict
        Must contain keys:
          - 'Bypass_Active_MPP' : ndarray [Nsim, Nstr, Nmod, Ndio] (booleans/ints)
          - 'Vmp'               : ndarray [Nsim,] MPP voltages per simulation
          - (unchanged from your baseline)
    res_path : str
        Root folder containing all pickles.
    sfname_pre : str
        Prefix for per-simulation files: f"{sfname_pre}_{idx_sim}.pickle"
        Each file contains:
          - 'Istrings': (Nstr, npts)
          - 'Vstrings': (Nstr, npts)
          - 'Str_idxs': (Nstr,)
    stfname_pre : str
        Prefix for per-string full-data files: f"{stfname_pre}_{str_idx}.pickle"
        Each file contains a dict with key 'full_data' (string-level structure).
    mfname_pre : str
        Prefix for per-module full-data files: f"{mfname_pre}_{mod_idx}.pickle"
        Each file contains per-module IV breakdown used below ('Vsubstr', 'Isubstr', ...).
    cell_index : 2-D ndarray
        Map of cell IDs to (row, col) positions; IDs are unique per (row, col).
    run_isc : bool (default False)
        If True, also compute SC (short-circuit) values.
    max_cache_items : int
        Maximum number of file objects kept in the LRU cache (all file types combined).
    clear_cache_on_exit : bool
        If True, clears the LRU cache when the function returns.

    Returns
    -------
    cell_currs : dict of ndarrays
        Keys:
          - 'cell_Imps', 'cell_Vmps', 'cell_Pmps', 'cell_isRev_mp', 'diode_Imps'
          - plus if run_isc:
              'cell_Iscs', 'cell_Vscs', 'cell_Pscs', 'cell_isRev_sc', 'diode_Iscs'
        Shapes:
          - cell_*: (Nsim, Nstr, Nmod, rows, cols) where (rows, cols) = cell_index.shape
          - diode_*: (Nsim, Nstr, Nmod, Ndio)
    """
    # ------------------------------
    # 0) Inputs
    # ------------------------------
    bpd_mpp = sim_data['Bypass_Active_MPP']  # (Nsim, Nstr, Nmod, Ndio)
    Vmp = sim_data['Vmp']                    # (Nsim,)

    if bpd_mpp.ndim != 4:
        raise ValueError(
            "sim_data['Bypass_Active_MPP'] must be 4-D: (Nsim, Nstr, Nmod, Ndio).")
    if Vmp.ndim != 1 or Vmp.shape[0] != bpd_mpp.shape[0]:
        raise ValueError(
            "sim_data['Vmp'] must be 1-D with length Nsim equal to bpd_mpp.shape[0].")

    Nsim, Nstr, Nmod, Ndio = bpd_mpp.shape
    rows, cols = cell_index.shape

    # ------------------------------
    # 1) Caches (LRU by file path)
    # ------------------------------
    # We bind max_cache_items at runtime by creating the cached loader closure
    def _make_cached_loader(max_items: int):
        @lru_cache(maxsize=max_items)
        def _load_pickle_cached(path: str):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required pickle not found: {path}")
            with open(path, "rb") as f:
                return pickle.load(f)
        return _load_pickle_cached

    _load_pickle_cached = _make_cached_loader(max_cache_items)

    def _sim_row_path(idx_sim: int) -> str:
        # f"{sfname_pre}_{idx_sim}.pickle"
        return os.path.join(res_path, f"{sfname_pre}_{int(idx_sim)}.pickle")

    def _string_full_path(str_idx: int) -> str:
        # f"{stfname_pre}_{str_idx}.pickle"
        return os.path.join(res_path, f"{stfname_pre}_{int(str_idx)}.pickle")

    def _module_full_path(mod_idx: int) -> str:
        # f"{mfname_pre}_{mod_idx}.pickle"
        return os.path.join(res_path, f"{mfname_pre}_{int(mod_idx)}.pickle")

    # ------------------------------
    # 2) Precompute cell ID -> (r,c) map once
    # ------------------------------
    ids_flat = cell_index.ravel()
    rr, cc = np.indices(cell_index.shape).reshape(2, -1)
    id_to_rc = {int(val): (int(r), int(c))
                for val, r, c in zip(ids_flat, rr, cc)}

    # ------------------------------
    # 3) Allocate outputs
    # ------------------------------
    out_shape = (Nsim, Nstr, Nmod, rows, cols)
    cell_Imps = np.zeros(out_shape, dtype=float)
    cell_Vmps = np.zeros(out_shape, dtype=float)
    cell_isRev_mp = np.zeros(out_shape, dtype=bool)
    diode_Imps = np.zeros((Nsim, Nstr, Nmod, Ndio), dtype=float)

    if run_isc:
        cell_Iscs = np.zeros(out_shape, dtype=float)
        cell_Vscs = np.zeros(out_shape, dtype=float)
        cell_isRev_sc = np.zeros(out_shape, dtype=bool)
        diode_Iscs = np.zeros((Nsim, Nstr, Nmod, Ndio), dtype=float)

    # ------------------------------
    # 4) Main loops (with cached loads)
    # ------------------------------
    for idx_sim, Vsim in enumerate(Vmp):
        # Per-simulation row file -> Istrings, Vstrings, Str_idxs
        sim_row = _load_pickle_cached(_sim_row_path(idx_sim))
        # Expect keys per your baseline
        Istrings = sim_row['Istrings']   # (Nstr, npts)
        Vstrings = sim_row['Vstrings']   # (Nstr, npts)
        idxs_str = sim_row['Str_idxs']   # (Nstr,)

        # Iterate strings
        for idx_str in range(Nstr):
            Istr = Istrings[idx_str, :]
            Vstr = Vstrings[idx_str, :]

            # String current at MPP voltage and SC (np.interp semantics)
            I_str_mp = float(np.interp(Vsim, Vstr, Istr))
            if run_isc:
                I_str_sc = float(np.interp(0.0, Vstr, Istr))  # SC at V=0

            # Load string-level full data once (cached across modules)
            str_full_data = _load_pickle_cached(
                _string_full_path(int(idxs_str[idx_str])))

            # Iterate modules within this string (module positions)
            num_mod_positions = int(
                str_full_data['full_data']['Imods'].shape[0])
            # map position->global module id
            Mod_idxs_arr = str_full_data['full_data']['Mod_idxs']
            for idx_mod in range(num_mod_positions):
                mod_idx = int(Mod_idxs_arr[idx_mod])

                # Load module full data (cached across all callers)
                mod_full_data = _load_pickle_cached(_module_full_path(mod_idx))

                # Shortcut: per-module arrays for pre/post bypass
                Vsubstr = np.asarray(mod_full_data['Vsubstr'])
                Isubstr = np.asarray(mod_full_data['Isubstr'])
                Vsubstr_pre = np.asarray(mod_full_data['Vsubstr_pre_bypass'])
                Isubstr_pre = np.asarray(mod_full_data['Isubstr_pre_bypass'])

                # Iterate diodes/substrates
                for idx_dio in range(Vsubstr.shape[0]):
                    diode_act = bool(
                        bpd_mpp[idx_sim, idx_str, idx_mod, idx_dio])

                    if diode_act:
                        # Post-bypass IV (clamp region)
                        Vd = Vsubstr[idx_dio, :].copy()
                        Id = Isubstr[idx_dio, :].copy()

                        # Slice from the last minimum to ensure consistent clamp range
                        last_index = _last_min_index(Vd)
                        Vd = Vd[last_index:]
                        Id = Id[last_index:]

                        # Diode current at MP / SC
                        if I_str_mp > Id[0]:
                            Idiode = I_str_mp - Id[0]
                            Issmp = Id[0]
                            diode_Imps[idx_sim, idx_str,
                                       idx_mod, idx_dio] = Idiode
                        else:
                            Issmp = I_str_mp

                        if run_isc:
                            if I_str_sc > Id[0]:
                                Idiode = I_str_sc - Id[0]
                                Isssc = Id[0]
                                diode_Iscs[idx_sim, idx_str,
                                           idx_mod, idx_dio] = Idiode
                            else:
                                Isssc = I_str_sc
                    else:
                        # No diode active → substrate tracks string current at MP/SC
                        Issmp = I_str_mp
                        if run_isc:
                            Isssc = I_str_sc

                    # Series crossties are stored under integer keys in the per-diode dict
                    if idx_dio not in mod_full_data:
                        # Defensive guard: you rely on this structure by design
                        raise KeyError(
                            f"Module full_data must contain a nested dict at key {idx_dio} (per-diode)."
                        )

                    per_diode_dict = mod_full_data[idx_dio]
                    sct_keys = [k for k in per_diode_dict if isinstance(
                        k, int)]  # series nodes

                    # Common bypass clamp voltage (if active)
                    if diode_act:
                        byp_v = float(np.min(Vd))

                    for idx_sct in sct_keys:
                        Vsct = np.array(
                            per_diode_dict[idx_sct]['Vsubstr'], copy=True)
                        Isct = np.array(
                            per_diode_dict[idx_sct]['Isubstr'], copy=True)

                        if diode_act:
                            # Clamp and slice for monotonic ascending interp range
                            Vsct[Vsct < byp_v] = byp_v
                            last_index = _last_min_index(Vsct)
                            Vsct = Vsct[last_index:]
                            Isct = Isct[last_index:]

                        # Sector voltage at Issubsection (np.interp semantics; invert I)
                        V_sct_mp = float(
                            np.interp(Issmp, np.flipud(Isct), np.flipud(Vsct)))
                        if run_isc:
                            V_sct_sc = float(
                                np.interp(Isssc, np.flipud(Isct), np.flipud(Vsct)))

                        # Parallel crossties (tiles) under this sector are also integer keys
                        pct_keys = [
                            k for k in per_diode_dict[idx_sct] if isinstance(k, int)]
                        for idx_pct in pct_keys:
                            tile = per_diode_dict[idx_sct][idx_pct]
                            Vpct = np.array(tile['Vsubstr'], copy=True)
                            Ipct = np.array(tile['Isubstr'], copy=True)

                            if diode_act:
                                Vpct[Vpct < byp_v] = byp_v
                                last_index = _last_min_index(Vpct)
                                Vpct = Vpct[last_index:]
                                Ipct = Ipct[last_index:]

                            # Tile current at sector voltages (np.interp semantics)
                            Imp_pct = float(np.interp(V_sct_mp, Vpct, Ipct))
                            if run_isc:
                                Isc_pct = float(
                                    np.interp(V_sct_sc, Vpct, Ipct))

                            # --- Vectorized per-cell interpolation (exact np.interp semantics) ---
                            # Gather per-cell curves for this tile; stored descending in I
                            # (R,)
                            cell_ids = tile['cell_idxs']
                            # (R, P), descending I
                            cell_I_desc = tile['cell_currents']
                            # (R, P), aligned with currents
                            cell_V_aligned_desc = tile['cell_voltages']

                            # Flip once to ASCENDING I for np.interp semantics
                            Iasc = cell_I_desc[:, ::-1]
                            Vasc = cell_V_aligned_desc[:, ::-1]

                            V_cell_mp_all = _interp_rows_scalar_npinterp_equiv(
                                Imp_pct, Iasc, Vasc)  # (R,)
                            if run_isc:
                                V_cell_sc_all = _interp_rows_scalar_npinterp_equiv(
                                    Isc_pct, Iasc, Vasc)  # (R,)

                            # Assign to outputs; map cell_id -> (r,c) via prebuilt lookup
                            for row_idx_in_tile, cid in enumerate(cell_ids):
                                r, c = id_to_rc[int(cid)]
                                cell_Imps[idx_sim, idx_str,
                                          idx_mod, r, c] = Imp_pct
                                cell_Vmps[idx_sim, idx_str, idx_mod, r,
                                          c] = V_cell_mp_all[row_idx_in_tile]
                                if run_isc:
                                    cell_Iscs[idx_sim, idx_str,
                                              idx_mod, r, c] = Isc_pct
                                    # preserve your rounding convention for Vsc:
                                    cell_Vscs[idx_sim, idx_str, idx_mod, r, c] = np.round(
                                        V_cell_sc_all[row_idx_in_tile], 2
                                    )
                        # end tiles
                    # end sectors
                # end diodes
            # end modules
        # end strings
    # end simulations

    # ------------------------------
    # 5) Reverse flags and powers
    # ------------------------------
    isRev_mp = (cell_Vmps < 0.0)
    cell_isRev_mp[isRev_mp] = True

    if run_isc:
        isRev_sc = (cell_Vscs < 0.0)
        cell_isRev_sc[isRev_sc] = True

    cell_Pmps = cell_Imps * cell_Vmps
    if run_isc:
        cell_Pscs = cell_Iscs * cell_Vscs

    # Optionally clear the caches to free memory for very long runs
    if clear_cache_on_exit:
        _load_pickle_cached.cache_clear()

    # ------------------------------
    # 6) Package outputs
    # ------------------------------
    cell_currs: Dict[str, np.ndarray] = {
        'cell_Imps': cell_Imps,
        'cell_Vmps': cell_Vmps,
        'cell_Pmps': cell_Pmps,
        'cell_isRev_mp': cell_isRev_mp,
        'diode_Imps': diode_Imps
    }
    if run_isc:
        cell_currs.update({
            'cell_Iscs': cell_Iscs,
            'cell_Vscs': cell_Vscs,
            'cell_Pscs': cell_Pscs,
            'cell_isRev_sc': cell_isRev_sc,
            'diode_Iscs': diode_Iscs
        })

    return cell_currs


def est_cell_current_AC(
    cell_index: np.ndarray,
    res_path: str,
    sfname_pre: str,
    run_isc: bool = False,
    *,
    max_cache_items: int = 1024,       # tune cache size for per-sim files
    clear_cache_on_exit: bool = True  # free cache RAM after completion
) -> Dict[str, np.ndarray]:
    """
    Estimate per-cell AC (module-level) quantities at MPP (and SC if run_isc=True),
    with bypass logic. Vectorized per-cell interpolation matches np.interp semantics.

    Storage layout
    --------------
    - Global bypass activation at MPP (all simulations):
        path: os.path.join(res_path, f"{sfname_pre}_BypassMPP.pickle")
        content: ndarray of shape (Nsim, Nstr, Nmod, Ndio)
    - Per-simulation nested module data (for each idx_sim):
        path: os.path.join(res_path, f"{sfname_pre}_{idx_sim}.pickle")
        content: nested list: sim_data_i[idx_str][idx_mod] -> module dict with:
            'Imp' : scalar-like
            'Isc' : scalar-like (if run_isc=True)
            'full_data' : dict with module data:
                - 'Vsubstr', 'Isubstr', 'Vsubstr_pre_bypass', 'Isubstr_pre_bypass'
                - and nested per-diode -> per-sector -> per-tile dicts containing
                  'Vsubstr', 'Isubstr', 'cell_idxs', 'cell_currents', 'cell_voltages'

    Notes
    -----
    - This function assumes the helpers `_last_min_index(...)` and
      `_interp_rows_scalar_npinterp_equiv(...)` are available in scope.
    - No shared helpers are defined here to keep AC/DC implementations independent.

    Returns
    -------
    cell_currs : dict of numpy.ndarray
        Keys present (depending on run_isc):
          - 'cell_Imps', 'cell_Vmps', 'cell_Pmps', 'cell_isRev_mp', 'diode_Imps'
          - plus if run_isc:
                'cell_Iscs', 'cell_Vscs', 'cell_Pscs', 'cell_isRev_sc', 'diode_Iscs'
        Shapes:
          - cell_*: (Nsim, Nstr, Nmod, rows, cols) where (rows, cols) = cell_index.shape
          - diode_*: (Nsim, Nstr, Nmod, Ndio)
    """

    # ------------------------------
    # 0) Load global bypass tensor
    # ------------------------------
    bpd_path = os.path.join(res_path, f"{sfname_pre}_BypassMPP.pickle")
    if not os.path.exists(bpd_path):
        raise FileNotFoundError(f"Bypass MPP file not found: {bpd_path}")
    with open(bpd_path, "rb") as f:
        bpd_mpp = pickle.load(f)

    if not isinstance(bpd_mpp, np.ndarray) or bpd_mpp.ndim != 4:
        raise ValueError(
            f"'{bpd_path}' must contain an ndarray of shape (Nsim, Nstr, Nmod, Ndio)."
        )
    Nsim, Nstr, Nmod, Ndio = bpd_mpp.shape

    # ------------------------------
    # 1) Cached loader for per-sim files
    # ------------------------------
    @lru_cache(maxsize=max_cache_items)
    def _load_sim_pickle_cached(path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required per-simulation pickle not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def _sim_path(idx_sim: int) -> str:
        return os.path.join(res_path, f"{sfname_pre}_{int(idx_sim)}.pickle")

    # ------------------------------
    # 2) Cell ID -> (row, col) map
    # ------------------------------
    ids_flat = cell_index.ravel()
    rr, cc = np.indices(cell_index.shape).reshape(2, -1)
    id_to_rc = {int(val): (int(r), int(c))
                for val, r, c in zip(ids_flat, rr, cc)}
    rows, cols = cell_index.shape

    # ------------------------------
    # 3) Allocate outputs
    # ------------------------------
    np_shape = (Nsim, Nstr, Nmod, rows, cols)

    cell_Imps = np.zeros(np_shape, dtype=float)
    cell_Vmps = np.zeros(np_shape, dtype=float)
    cell_isRev_mp = np.zeros(np_shape, dtype=bool)
    diode_Imps = np.zeros((Nsim, Nstr, Nmod, Ndio), dtype=float)

    if run_isc:
        cell_Iscs = np.zeros(np_shape, dtype=float)
        cell_Vscs = np.zeros(np_shape, dtype=float)
        cell_isRev_sc = np.zeros(np_shape, dtype=bool)
        diode_Iscs = np.zeros((Nsim, Nstr, Nmod, Ndio), dtype=float)

    # Small local to coerce scalar-like to float
    def _to_float(x) -> float:
        if np.isscalar(x):
            return float(x)
        if isinstance(x, (list, tuple)) and len(x) == 1:
            return float(x[0])
        if isinstance(x, np.ndarray) and x.size == 1:
            return float(x.reshape(()).item())
        return float(x)

    # ------------------------------
    # 4) Main traversal (with cached per-sim loads)
    # ------------------------------
    for idx_sim in range(Nsim):
        sim_path = _sim_path(idx_sim)
        sim_data_i = _load_sim_pickle_cached(sim_path)

        # Expect nested list: sim_data_i[idx_str][idx_mod] = module dict
        if not isinstance(sim_data_i, (list, tuple)) or len(sim_data_i) != Nstr:
            raise ValueError(
                f"'{sim_path}' must be a list (length Nstr={Nstr}) of per-string module lists."
            )
        if not isinstance(sim_data_i[0], (list, tuple)) or len(sim_data_i[0]) != Nmod:
            raise ValueError(
                f"'{sim_path}': sim_data_i[0] must be a list (length Nmod={Nmod}) of module dicts."
            )

        for idx_str in range(Nstr):
            str_data_i = sim_data_i[idx_str]

            for idx_mod in range(Nmod):
                mod_data_i = str_data_i[idx_mod]

                # Per-module currents at MP / SC (scalars)
                Imp = _to_float(mod_data_i['Imp'])
                if run_isc:
                    Isc = _to_float(mod_data_i['Isc'])

                # Bypass activation flags from the GLOBAL tensor (not from the module dict)
                bpd_act_mod = bpd_mpp[idx_sim, idx_str, idx_mod, :]  # (Ndio,)

                mod_full_data = mod_data_i['full_data']

                # Basic presence checks
                if 'Vsubstr' not in mod_full_data or 'Isubstr' not in mod_full_data:
                    raise KeyError(
                        "Module 'full_data' must include 'Vsubstr' and 'Isubstr'.")
                if 'Vsubstr_pre_bypass' not in mod_full_data or 'Isubstr_pre_bypass' not in mod_full_data:
                    raise KeyError(
                        "Module 'full_data' must include 'Vsubstr_pre_bypass' and 'Isubstr_pre_bypass'.")

                Vsubstr = np.asarray(mod_full_data['Vsubstr'])
                Isubstr = np.asarray(mod_full_data['Isubstr'])

                if Vsubstr.ndim < 2 or Vsubstr.shape[0] != Ndio:
                    raise ValueError(
                        f"Inconsistent diodes: full_data['Vsubstr'].shape={Vsubstr.shape}, expected leading dim Ndio={Ndio}"
                    )

                # Iterate diodes (substrates)
                for idx_dio in range(Ndio):
                    diode_act = bool(bpd_act_mod[idx_dio])

                    if diode_act:
                        Vd = np.array(Vsubstr[idx_dio, :], copy=True)
                        Id = np.array(Isubstr[idx_dio, :], copy=True)

                        # Slice from last minimum
                        last_index = _last_min_index(Vd)
                        Vd = Vd[last_index:]
                        Id = Id[last_index:]

                        # Diode current at MP
                        if Imp > Id[0]:
                            Idiode = Imp - Id[0]
                            Issmp = Id[0]
                            diode_Imps[idx_sim, idx_str,
                                       idx_mod, idx_dio] = Idiode
                        else:
                            Issmp = Imp

                        # Diode current at SC
                        if run_isc:
                            if Isc > Id[0]:
                                Idiode = Isc - Id[0]
                                Isssc = Id[0]
                                diode_Iscs[idx_sim, idx_str,
                                           idx_mod, idx_dio] = Idiode
                            else:
                                Isssc = Isc
                    else:
                        # No diode active → substrate current tracks module current
                        Issmp = Imp
                        if run_isc:
                            Isssc = Isc

                    # Series crossties under per-diode dict use integer keys
                    if idx_dio not in mod_full_data:
                        raise KeyError(
                            f"Module 'full_data' must contain a per-diode nested dict at key {idx_dio}."
                        )
                    per_diode_dict = mod_full_data[idx_dio]
                    sct_keys = [
                        k for k in per_diode_dict if isinstance(k, int)]

                    if diode_act:
                        byp_v = float(np.min(Vd))

                    for idx_sct in sct_keys:
                        Vsct = np.array(
                            per_diode_dict[idx_sct]['Vsubstr'], copy=True)
                        Isct = np.array(
                            per_diode_dict[idx_sct]['Isubstr'], copy=True)

                        if diode_act:
                            Vsct[Vsct < byp_v] = byp_v
                            last_index = _last_min_index(Vsct)
                            Vsct = Vsct[last_index:]
                            Isct = Isct[last_index:]

                        # Sector voltage at Issubsection (flip to ascending I for np.interp)
                        V_sct_mp = float(
                            np.interp(Issmp, np.flipud(Isct), np.flipud(Vsct)))
                        if run_isc:
                            V_sct_sc = float(
                                np.interp(Isssc, np.flipud(Isct), np.flipud(Vsct)))

                        # Parallel crossties (tiles) also under integer keys
                        pct_keys = [
                            k for k in per_diode_dict[idx_sct] if isinstance(k, int)]
                        for idx_pct in pct_keys:
                            tile = per_diode_dict[idx_sct][idx_pct]
                            Vpct = np.array(tile['Vsubstr'], copy=True)
                            Ipct = np.array(tile['Isubstr'], copy=True)

                            if diode_act:
                                Vpct[Vpct < byp_v] = byp_v
                                last_index = _last_min_index(Vpct)
                                Vpct = Vpct[last_index:]
                                Ipct = Ipct[last_index:]

                            # Tile current at sector voltages
                            Imp_pct = float(np.interp(V_sct_mp, Vpct, Ipct))
                            if run_isc:
                                Isc_pct = float(
                                    np.interp(V_sct_sc, Vpct, Ipct))

                            # --- Vectorized per-cell interpolation (exact np.interp semantics) ---
                            cell_ids = tile['cell_idxs']               # (R,)
                            # (R, P) descending I
                            I_desc = tile['cell_currents']
                            # (R, P) aligned with currents
                            V_desc = tile['cell_voltages']

                            Iasc = I_desc[:, ::-1]
                            Vasc = V_desc[:, ::-1]

                            V_cell_mp_all = _interp_rows_scalar_npinterp_equiv(
                                Imp_pct, Iasc, Vasc)  # (R,)
                            if run_isc:
                                V_cell_sc_all = _interp_rows_scalar_npinterp_equiv(
                                    Isc_pct, Iasc, Vasc)  # (R,)

                            # Assign
                            for row_idx_in_tile, cid in enumerate(cell_ids):
                                r, c = id_to_rc[int(cid)]
                                cell_Imps[idx_sim, idx_str,
                                          idx_mod, r, c] = Imp_pct
                                cell_Vmps[idx_sim, idx_str, idx_mod, r,
                                          c] = V_cell_mp_all[row_idx_in_tile]
                                if run_isc:
                                    cell_Iscs[idx_sim, idx_str,
                                              idx_mod, r, c] = Isc_pct
                                    cell_Vscs[idx_sim, idx_str, idx_mod, r, c] = np.round(
                                        V_cell_sc_all[row_idx_in_tile], 2
                                    )
                        # end tiles
                    # end sectors
                # end diodes
            # end modules
        # end strings
    # end simulations

    # ------------------------------
    # 5) Reverse flags and powers
    # ------------------------------
    isRev_mp = (cell_Vmps < 0.0)
    cell_isRev_mp[isRev_mp] = True

    if run_isc:
        isRev_sc = (cell_Vscs < 0.0)
        cell_isRev_sc[isRev_sc] = True

    cell_Pmps = cell_Imps * cell_Vmps
    if run_isc:
        cell_Pscs = cell_Iscs * cell_Vscs

    # Optional: clear the cache to free RAM
    if clear_cache_on_exit:
        _load_sim_pickle_cached.cache_clear()

    # ------------------------------
    # 6) Package outputs
    # ------------------------------
    cell_currs: Dict[str, np.ndarray] = {
        'cell_Imps': cell_Imps,
        'cell_Vmps': cell_Vmps,
        'cell_Pmps': cell_Pmps,
        'cell_isRev_mp': cell_isRev_mp,
        'diode_Imps': diode_Imps
    }
    if run_isc:
        cell_currs.update({
            'cell_Iscs': cell_Iscs,
            'cell_Vscs': cell_Vscs,
            'cell_Pscs': cell_Pscs,
            'cell_isRev_sc': cell_isRev_sc,
            'diode_Iscs': diode_Iscs
        })

    return cell_currs
