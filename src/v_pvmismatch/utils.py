# -*- coding: utf-8 -*-
"""Helper functions that are vectorized and numerically consistent with 1-D
interpolation.

This module provides:
 - reshape_ndarray: build higher-dim arrays from 1-D inputs.
 - view1D / isin_nd: set membership helpers across N-D views.
 - interp (vectorized): consistent row-wise interpolation with 'linear',
 'previous', 'next'.
 - calcMPP_IscVocFFBPD: MPP & bypass-at-MPP computation.
"""

import os
from typing import Iterable, List
import numpy as np
from scipy.interpolate import interp1d
import pickle

=======
# from functools import partial
from typing import Tuple
import pickle
from typing import Dict, Optional
import hashlib


def _interp_rows_at_scalar(x0_rows: np.ndarray, y0_rows: np.ndarray,
                           xq_scalar: float) -> np.ndarray:
    """
    Row-wise linear interpolation at a single scalar query xq for each row.
    Assumes x0_rows is strictly increasing per row.

    Parameters
    ----------
    x0_rows : (R, N) float64
        Per-row abscissas (increasing).
    y0_rows : (R, N) float64
        Per-row ordinates aligned with x0_rows.
    xq_scalar : float
        The scalar query (same for all rows).

    Returns
    -------
    yq : (R,) float64
        Interpolated values per row at xq_scalar with linear extrapolation.
    """
    R, N = x0_rows.shape
    yq = np.empty(R, dtype=np.float64)

    # For each row, find segment using searchsorted; do all math in NumPy.
    # This tight loop only calls NumPy C functions and avoids Python object
    # creation per row.
    for r in range(R):
        x0 = x0_rows[r]
        y0 = y0_rows[r]
        hi = np.searchsorted(x0, xq_scalar, side='right')
        if hi <= 0:
            # Left extrapolation
            x1, x2 = x0[0], x0[1] if N > 1 else x0[0]
            y1, y2 = y0[0], y0[1] if N > 1 else y0[0]
            slope = (y2 - y1) / (x2 - x1 + np.finfo(float).eps)
            yq[r] = y1 + (xq_scalar - x1) * slope
        elif hi >= N:
            # Right extrapolation
            x1, x2 = x0[N - 2] if N > 1 else x0[N - 1], x0[N - 1]
            y1, y2 = y0[N - 2] if N > 1 else y0[N - 1], y0[N - 1]
            slope = (y2 - y1) / (x2 - x1 + np.finfo(float).eps)
            yq[r] = y2 + (xq_scalar - x2) * slope
        else:
            lo = hi - 1
            x_lo, x_hi = x0[lo], x0[hi]
            y_lo, y_hi = y0[lo], y0[hi]
            denom = x_hi - x_lo
            if denom != 0.0:
                t = (x_hi - xq_scalar) / denom
                yq[r] = t * y_lo + (1.0 - t) * y_hi
            else:
                yq[r] = y_hi  # degenerate (identical x)
    return yq


def _prevstep_nd_rows(P_rows: np.ndarray, B_rows: np.ndarray,
                      Pmp: np.ndarray) -> np.ndarray:
    """
    Previous-step interpolation at MPP across N-D bypass arrays (axis-robust).
    Evaluates a step function defined by (P_rows[r, :], B_rows[r, ..., :]) at
    Pmp[r].

    Parameters
    ----------
    P_rows : (R, K) float64
        Per-row power samples (e.g., K=3 around MPP index).
    B_rows : (R, d1, d2, ..., K?) bool/uint8
        Per-row bypass activation samples; the samples axis may NOT be last due
        to NumPy advanced indexing. This function finds it and moves it to the
        last axis.
    Pmp : (R,) float64
        Per-row MPP power to evaluate.

    Returns
    -------
    B_at_mpp : (R, d1, d2, ...) bool
        Bypass activation at MPP for each row and all inner dims.
    """
    # ---- shape validation ----
    if P_rows.ndim != 2:
        raise ValueError(f"P_rows must be 2-D (R,K); got {P_rows.shape}")
    if B_rows.ndim < 2:
        raise ValueError(f"B_rows must be >=2-D; got {B_rows.shape}")
    R, K = P_rows.shape

    # B_rows needs to have the same row count R on axis 0
    if B_rows.shape[0] != R:
        raise ValueError(
            f"Row count mismatch: P_rows.shape[0]={R} != B_rows.shape[0]={B_rows.shape[0]}"
        )

    # ---- find the samples axis in B_rows (length == K), then move it to last
    # search axes 1..ndim-1 to avoid the row axis
    candidate_axes = [ax for ax in range(1,
                                         B_rows.ndim) if B_rows.shape[ax] == K]
    if not candidate_axes:
        raise ValueError(
            f"No axis of length K={K} found in B_rows.shape={B_rows.shape}. "
            "Cannot align bypass samples with power samples."
        )
    # Use the FIRST match (typical when advanced indexing pushes K to axis 1)
    s_axis = candidate_axes[0]
    if s_axis != (B_rows.ndim - 1):
        B_rows = np.moveaxis(B_rows, s_axis,
                             -1)  # now B_rows[..., K] is last axis

    # -- sort K samples per row and permute B_rows accordingly on the last axis
    order = np.argsort(P_rows, axis=1)                            # (R, K)
    P_sorted = np.take_along_axis(P_rows, order, axis=1)          # (R, K)

    # Expand 'order' to match B_rows inner dims on its last axis
    expand = (slice(None),) + (None,) * (B_rows.ndim - 2) + (slice(None),)
    order_exp = order[expand]  # (R, 1, 1, ..., K)
    B_sorted = np.take_along_axis(B_rows.astype(np.uint8), order_exp, axis=-1)

    # ---- row-wise searchsorted on 1-D arrays ----
    idx = np.empty(R, dtype=np.int64)
    for r in range(R):
        idx[r] = np.searchsorted(P_sorted[r], Pmp[r], side='right')

    # previous sample with constant clamps
    idx = np.clip(idx - 1, 0, K - 1)                              # (R,)

    # ---- robust gather: flatten inner dims to avoid broadcast quirks ----
    inner_shape = B_sorted.shape[1:-1]
    M = int(np.prod(inner_shape)) if inner_shape else 1
    B_flat = B_sorted.reshape(R, M, K)                            # (R, M, K)

    # index array replicated across M for each row
    idx2 = idx[:, None]                                          # (R, 1)
    if M > 1:
        idx2 = np.repeat(idx2, M, axis=1)                        # (R, M)
    idx2 = idx2[..., None]                                       # (R, M, 1)

    # Gather previous samples and restore original inner shape
    gathered = np.take_along_axis(B_flat, idx2, axis=-1)         # (R, M, 1)
    B_at_mpp = gathered.squeeze(-1).reshape((R,) + inner_shape).astype(bool)
    return B_at_mpp


# ----------------------------------------------------------------------------
# HELPER FUNCTIONS THAT ARE VECTORIZED ---------------------------------------


def reshape_ndarray(x, arr_shape):
    """
    Reshape an N dimensional array given the new array shape.

    Parameters
    ----------
    x : numpy.ndarray
        Array of N-dimensions.
    arr_shape : tuple
        Tuple specifying the shape of the input array.

    Returns
    -------
    xout : numpy.ndarray
        Output array reshaped to input shape.

    """
    xout = x.copy()
    num_new_axis = len(arr_shape)
    # Run for loop over the new axis to create
    for idx_ax in range(num_new_axis):
        xout = np.repeat(xout[..., np.newaxis],
                         arr_shape[idx_ax], axis=idx_ax+1)

    return xout


def view1D(a, b):
    """
    Create 1d views of a, b.

    Parameters
    ----------
    a : numpy.ndarray
        Array of N-dimensions.
    b : numpy.ndarray
        Array of N-dimensions.

    Returns
    -------
    numpy.ndarray
        View of a.
    numpy.ndarray
        View of b.

    """
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    return a.view(void_dt).ravel(),  b.view(void_dt).ravel()


def isin_nd(a, b):
    """
    Reproduce isin in N-dimensions.

    Parameters
    ----------
    a : numpy.ndarray
        3D input array.
    b : numpy.ndarray
        3D input array.

    Returns
    -------
    numpy.ndarray
        DESCRIPTION.

    """
    # a,b are 3D input arrays to give us "isin-like" functionality across them
    A, B = view1D(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1))
    return np.nonzero(np.isin(A, B))[0]


def isin_nd_searchsorted(a, b):
    """
    Perform isin and search sorting in N-dimensions.

    Parameters
    ----------
    a : numpy.ndarray
        3D input array.
    b : numpy.ndarray
        3D input array.

    Returns
    -------
    numpy.ndarray
        Boolean array.

    """
    # a,b are the 3D input arrays
    A, B = view1D(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1))
    sidx = A.argsort()
    sorted_index = np.searchsorted(A, B, sorter=sidx)
    sorted_index[sorted_index == len(A)] = len(A)-1
    idx = sidx[sorted_index]
    return A[idx] == B


# def searchsorted2d(a, b):
# # https://stackoverflow.com/questions/40588403/vectorized-searchsorted-numpy
#     m, n = a.shape
#     max_num = np.maximum(a.max() - a.min(), b.max() - b.min()) + 1
#     r = max_num*np.arange(a.shape[0])[:, None]
#     p = np.searchsorted((a+r).ravel(), (b+r).ravel(),
#                         side='right').reshape(m, -1)
#     return p - n*(np.arange(m)[:, None])
# --------------------------- Interpolation (vectorized) ----------------------


def _search_indices_rowwise(x0: np.ndarray, x: np.ndarray,
                            side: str = "right") -> Tuple[np.ndarray,
                                                          np.ndarray]:
    """
    Returns (lo, hi) with shape (n_rows, n_queries).
    """
    assert x0.ndim == 2 and x.ndim == 2, "x0 and x must be 2-D"
    n_rows, n_cols = x0.shape
    n_rows_x, n_q = x.shape
    assert n_rows == n_rows_x, "x0 and x must have same number of rows"

    # Perform per-row searchsorted by broadcasting each row with its queries.
    # To keep memory in check, loop rows but only over indices (fast).
    hi = np.empty((n_rows, n_q), dtype=np.int64)
    for r in range(n_rows):
        hi[r, :] = np.searchsorted(x0[r, :], x[r, :], side=side)
    # clamp to valid range [0, n_cols-1]
    np.clip(hi, 0, n_cols - 1, out=hi)
    lo = hi - 1
    np.clip(lo, 0, n_cols - 1, out=lo)
    return lo, hi


def interp_coef(x0: np.ndarray, x: np.ndarray, kind: str = "linear",
                side: str = "right") -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray]:
    """
    Compute row-wise interpolation coefficients.
    Returns a tuple (lo, hi, w) so that y(x) = w*y0[lo] + (1-w)*y0[hi] for
    'linear', and for 'previous'/'next' we return w=1 or w=0 to pick
    y0[lo]/y0[hi] respectively.
    """
    lo, hi = _search_indices_rowwise(x0, x, side=side)

    if kind == "linear":
        # Distance within the range, shape (n_rows, n_q)
        # To get d_left and d_right, index x0 using row/col pairs.
        n_rows, n_q = x.shape
        # Gather with advanced indexing
        d_left = x - x0[np.arange(n_rows)[:, None], lo]
        d_right = x0[np.arange(n_rows)[:, None], hi] - x
        d_total = d_left + d_right
        # Avoid division by zero (when lo==hi).
        # Set weight to 0 in that case (choose hi).
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(d_total != 0.0, d_right / d_total, 0.0)
    elif kind == "previous":
        w = np.ones_like(lo, dtype=float)  # choose lo
    elif kind == "next":
        w = np.zeros_like(lo, dtype=float)  # choose hi
    else:
        raise ValueError(f"Unknown interpolation kind: {kind}")
    return lo, hi, w


def interp_2d(y0: np.ndarray, x0: np.ndarray, x: np.ndarray,
              kind: str = "linear", side: str = "right") -> np.ndarray:
    assert x0.ndim == 2 and x.ndim == 2 and y0.ndim == 2, "x0, x, y0 must be 2-D"
    n_rows, n_cols = x0.shape
    n_rows2, n_q = x.shape
    assert n_rows == n_rows2 and y0.shape == x0.shape, "Shapes must align"

    # If row starts decreasing, flip row (your heuristic)
    dec_mask = (x0[:, 1] < x0[:, 0])
    x0_inc = x0.copy()
    y0_inc = y0.copy()
    if np.any(dec_mask):
        x0_inc[dec_mask] = x0_inc[dec_mask][:, ::-1]
        y0_inc[dec_mask] = y0_inc[dec_mask][:, ::-1]

    # Row-wise searchsorted
    hi = np.empty((n_rows, n_q), dtype=np.int64)
    for r in range(n_rows):
        hi[r, :] = np.searchsorted(x0_inc[r, :], x[r, :], side=side)
    hi = np.clip(hi, 0, n_cols - 1)
    lo = np.clip(hi - 1, 0, n_cols - 1)

    return yint


def interp2d_wrap(x0: np.ndarray, x: np.ndarray, y0: np.ndarray,
                  side: str = "right", kind: str = "linear") -> np.ndarray:
    """
    Wrapper providing the old signature: y = interp2d_wrap(x0, x, y0,
                                                           side, kind)
    where x0 and y0 are the base arrays (2-D) and x is the query grid (2-D).
    """
    return interp_2d(y0=y0, x0=x0, x=x, kind=kind, side=side)


def interp_linear_bool(x0: np.ndarray,
                       x:  np.ndarray,
                       y0_bool: np.ndarray,
                       tie_break: str = 'nearest',   # 'nearest', 'nearest-up'
                       constant_extrap: bool = True  # extension at ends
                       ) -> np.ndarray:
    """
    Vectorized boolean interpolation via linear on {0,1} followed by threshold.

    Parameters
    ----------
    x0 : (N, M) float
        Base x grid per row, not necessarily sorted (we sort per row).
    x  : (N, Q) float
        Query x grid per row.
    y0_bool : (N, M) bool or {0,1} float
        Base y values (boolean steps).
    tie_break : {'nearest','nearest-up'}
        Midpoint decision: 'nearest' -> 0 at exact 0.5; 'nearest-up' -> 1 at
        exact 0.5.
    constant_extrap : bool
        If True, outside the data range use constant extension
        (first/last sample).
        If False, keep linear extrapolation from interp_2d
        (not recommended for steps).

    Returns
    -------
    y_bool : (N, Q) bool
        Interpolated booleans (nearest-step behavior).
    """

    # 1) Linear interpolation on floats
    y_lin = interp_2d(y0=y0_bool.astype(float), x0=x0, x=x,
                      kind='linear', side='right')

    # 2) Optionally override extrapolation to constant extension
    if constant_extrap:
        # We need the sorted x0 to construct masks correctly.
        # interp_2d already sorts per row,
        # but since we don't expose that here,
        # rebuild sorted x0 and y0 for masks:
        n_rows, n_cols = x0.shape
        row_index = np.arange(n_rows)[:, None]
        sort_idx = np.argsort(x0, axis=1)
        x0_sorted = x0[row_index, sort_idx]
        y0_sorted = y0_bool[row_index, sort_idx].astype(float)

        left_mask = x < x0_sorted[:, :1]   # (N, Q)
        right_mask = x > x0_sorted[:, -1:]  # (N, Q)

        # Constant extension: first / last sample
        y_lin[left_mask] = y0_sorted[:, :1]
        y_lin[right_mask] = y0_sorted[:, -1:]

    # 3) Threshold to boolean with desired tie-breaking
    if tie_break == 'nearest-up':
        y_bool = (y_lin >= 0.5)
    elif tie_break == 'nearest':
        y_bool = (y_lin > 0.5)
    else:
        raise ValueError("tie_break must be 'nearest' or 'nearest-up'")

    return y_bool.astype(bool)


def calcMPP_IscVocFFBPD(Isys, Vsys, Psys, bypassed_mod_arr,
                        run_bpact=True, run_annual=False):
    """
    Calculate MPP IV parameters. This method is vectorized.

    Parameters
    ----------
    Isys : np.ndarray
        2-D array of current curves.
    Vsys : np.ndarray
        2-D array of voltage curves.
    Psys : np.ndarray
        2-D array of power curves.
    bypassed_mod_arr : np.ndarray
        3-D or 5-D arr of bypass diode act curves for substring in mod of sys.
    run_bpact : bool, optional
        Flag to run bypass diode activation logic. The default is True.
    run_annual : bool, optional
        Flag to delete large bypass diode act arr in case of annual sim.
        The default is False.

    Returns
    -------
    Imp : np.ndarray
        1-D array of Imp.
    Vmp : np.ndarray
        1-D array of Vmp.
    Pmp : np.ndarray
        1-D array of Pmp.
    Isc : np.ndarray
        1-D array of Isc.
    Voc : np.ndarray
        1-D array of Voc.
    FF : np.ndarray
        1-D array of FF.
    BpDmp : np.ndarray
        Bypass diode activation at MPP for substr in each mod of the system.
    num_bpd_active : np.ndarray
        1-D array of number of bypass diodes active.

    """
    # Reverse direction of Psys
    rev_P = Psys[:, ::-1]
    mpp = rev_P.shape[1] - np.argmax(rev_P, axis=1) - 1
    check_max_idx = (mpp == rev_P.shape[1]-1)
    mpp[check_max_idx] = rev_P.shape[1] - 2
    mpp = np.reshape(mpp, (len(mpp), 1))
    mpp_lohi = np.concatenate([mpp-1, mpp, mpp+1], axis=1)
    mpp_row = np.reshape(np.arange(Psys.shape[0]), (Psys.shape[0], 1))
    P = Psys[mpp_row, mpp_lohi]
    V = Vsys[mpp_row, mpp_lohi]
    Curr = Isys[mpp_row, mpp_lohi]
    # calculate derivative dP/dV using central difference
    dP = np.diff(P, axis=1)  # size is (2, 1)
    dV = np.diff(V, axis=1)  # size is (2, 1)
    Pv = dP / dV  # size is (2, 1)
    # dP/dV is central difference at midpoints,
    Vmid = (V[:, 1:] + V[:, :-1]) / 2.0  # size is (2, 1)
    Imid = (Curr[:, 1:] + Curr[:, :-1]) / 2.0  # size is (2, 1)
    # interpolate to find Vmp
    Vmp = (-Pv[:, 0].flatten() * np.diff(Vmid, axis=1).flatten() /
           np.diff(Pv, axis=1).flatten() + Vmid[:, 0])
    Imp = (-Pv[:, 0].flatten() * np.diff(Imid, axis=1).flatten() /
           np.diff(Pv, axis=1).flatten() + Imid[:, 0])
    # calculate max power at Pv = 0
    Pmp = Imp * Vmp
    # calculate Voc, current must be increasing so flipup()
    Voc = np.zeros(Pmp.shape)
    Isc = np.zeros(Pmp.shape)
    for idx_time in range(Psys.shape[0]):
        # Only interpolate if Current data is non-zero
        if Vsys[idx_time, :].nonzero()[0].size != 0:
            Voc[idx_time] = np.interp(np.float64(0),
                                      np.flipud(Isys[idx_time, :]),
                                      np.flipud(Vsys[idx_time, :]))
            Isc[idx_time] = np.interp(np.float64(
                0), Vsys[idx_time, :], Isys[idx_time, :])  # calculate Isc
    FF = Pmp / Isc / Voc
    if run_bpact:
        # Use nearest interpolation to obtain bypass diode activation at MPP
        if len(bypassed_mod_arr.shape) == 3:
            BpD_Active = bypassed_mod_arr[mpp_row, :, mpp_lohi]
            BpDmp = np.zeros((Pmp.shape[0], BpD_Active.shape[2]), dtype=bool)
            for idx_row in range(BpD_Active.shape[0]):
                for idx_col in range(BpD_Active.shape[2]):
                    interpolator = interp1d(P[idx_row, :],
                                            BpD_Active[idx_row, :, idx_col],
                                            kind='previous',
                                            fill_value='extrapolate')
                    BpDmp[idx_row, idx_col] = interpolator(
                        Pmp[idx_row]).astype(bool)
            num_bpd_active = BpDmp.sum(axis=1)
        else:
            BpD_Active = bypassed_mod_arr[mpp_row, :, :, :, mpp_lohi]
            BpDmp = np.zeros((Pmp.shape[0], BpD_Active.shape[2],
                              BpD_Active.shape[3], BpD_Active.shape[4]),
                             dtype=bool)
            for idx_row in range(BpD_Active.shape[0]):
                for idx_str in range(BpD_Active.shape[2]):
                    for idx_mod in range(BpD_Active.shape[3]):
                        for idx_substr in range(BpD_Active.shape[4]):
                            interpolator = interp1d(P[idx_row, :],
                                                    BpD_Active[idx_row, :,
                                                               idx_str,
                                                               idx_mod,
                                                               idx_substr],
                                                    kind='previous',
                                                    fill_value='extrapolate')
                            BpDmp[idx_row, idx_str,
                                  idx_mod,
                                  idx_substr] = interpolator(
                                      Pmp[idx_row]).astype(bool)
            num_bpd_active = BpDmp.sum(axis=3).sum(axis=2).sum(axis=1)
    else:
        BpDmp = np.nan
        num_bpd_active = 0
    if run_annual:
        del bypassed_mod_arr

    return Imp, Vmp, Pmp, Isc, Voc, FF, BpDmp, num_bpd_active


def calcMPP_IscVocFF(Isys, Vsys, Psys):
    """
    Calculate MPP IV parameters. This method is vectorized.

    Parameters
    ----------
    Isys : np.ndarray
        2-D array of current curves.
    Vsys : np.ndarray
        2-D array of voltage curves.
    Psys : np.ndarray
        2-D array of power curves.
    bypassed_mod_arr : np.ndarray
        3-D or 5-D arr of bypass diode act curves for substring in mod of sys.
    run_bpact : bool, optional
        Flag to run bypass diode activation logic. The default is True.
    run_annual : bool, optional
        Flag to delete large bypass diode act arr in case of annual sim.
        The default is False.

    Returns
    -------
    Imp : np.ndarray
        1-D array of Imp.
    Vmp : np.ndarray
        1-D array of Vmp.
    Pmp : np.ndarray
        1-D array of Pmp.
    Isc : np.ndarray
        1-D array of Isc.
    Voc : np.ndarray
        1-D array of Voc.
    FF : np.ndarray
        1-D array of FF.
    BpDmp : np.ndarray
        Bypass diode activation at MPP for substr in each mod of the system.
    num_bpd_active : np.ndarray
        1-D array of number of bypass diodes active.

    """
    rev_P = Psys[:, ::-1]
    mpp = rev_P.shape[1] - np.argmax(rev_P, axis=1) - 1
    check_max_idx = (mpp == rev_P.shape[1]-1)
    mpp[check_max_idx] = rev_P.shape[1] - 2
    mpp = np.reshape(mpp, (len(mpp), 1))
    mpp_lohi = np.concatenate([mpp-1, mpp, mpp+1], axis=1)
    mpp_row = np.reshape(np.arange(Psys.shape[0]), (Psys.shape[0], 1))
    P = Psys[mpp_row, mpp_lohi]
    V = Vsys[mpp_row, mpp_lohi]
    Icurr = Isys[mpp_row, mpp_lohi]
    # calculate derivative dP/dV using central difference
    dP = np.diff(P, axis=1)  # size is (2, 1)
    dV = np.diff(V, axis=1)  # size is (2, 1)
    Pv = dP / dV  # size is (2, 1)
    # dP/dV is central difference at midpoints,
    Vmid = (V[:, 1:] + V[:, :-1]) / 2.0  # size is (2, 1)
    Imid = (Icurr[:, 1:] + Icurr[:, :-1]) / 2.0  # size is (2, 1)
    # interpolate to find Vmp
    Vmp = (-Pv[:, 0].flatten() * np.diff(Vmid, axis=1).flatten() /
           np.diff(Pv, axis=1).flatten() + Vmid[:, 0])
    Imp = (-Pv[:, 0].flatten() * np.diff(Imid, axis=1).flatten() /
           np.diff(Pv, axis=1).flatten() + Imid[:, 0])
    # calculate max power at Pv = 0
    Pmp = Imp * Vmp
    # calculate Voc, current must be increasing so flipup()
    Voc = np.zeros(Pmp.shape)
    Isc = np.zeros(Pmp.shape)
    for idx_time in range(Psys.shape[0]):
        # Only interpolate if Current data is non-zero
        if Vsys[idx_time, :].nonzero()[0].size != 0:
            Voc[idx_time] = np.interp(np.float64(0),
                                      np.flipud(Isys[idx_time, :]),
                                      np.flipud(Vsys[idx_time, :]))
            Isc[idx_time] = np.interp(np.float64(
                0), Vsys[idx_time, :], Isys[idx_time, :])  # calculate Isc
    FF = Pmp / Isc / Voc

    return Imp, Vmp, Pmp, Isc, Voc, FF


def round_to_dec(vector, val):
    """
    Round to nearest value or number. Example 2.03, 0.02 becomes 2.02.

    Parameters
    ----------
    vector : numpy.array
        Array of numbers.
    val : float
        Resolution.

    Returns
    -------
    numpy.array
        Rounded array.

    """
    return np.round(vector / val) * val


def save_pickle(filename, variable):
    """
    Save pickle file.

    Parameters
    ----------
    filename : str
        File path.
    variable : python variable
        Variable to save to pickle file.

    Returns
    -------
    None.

    """
    with open(filename, 'wb') as handle:
        pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    """
    Load data from a pickle file.

    Parameters
    ----------
    filename : str
        File path.

    Returns
    -------
    db : python variable
        Variable that is stored in the pickle file.

    """
    with open(filename, 'rb') as handle:
        db = pickle.load(handle)
    return db


def find_row_index(array_2d, array_1d):
    """
    Check if a D arr exists as row in 2D arr and returns index of the row.

    Args
    ----
        array_2d (numpy.ndarray): The 2D array to search within.
        array_1d (numpy.ndarray): The 1D array to search for.

    Returns
    -------
        int or None: The index of the row if found, otherwise None.
    """
    # Check if number of columns is the same.
    # This is to consider veriable string sizes within each diode subsection
    if array_2d.shape[1] < len(array_1d):
        zeros_column = np.zeros((array_2d.shape[0],
                                 len(array_1d) - array_2d.shape[1]))
        array_2d = np.hstack((array_2d, zeros_column))
    elif array_2d.shape[1] > len(array_1d):
        array_1d = np.pad(array_1d, (0, array_2d.shape[1] - len(array_1d)),
                          'constant')
    for col_idx in range(len(array_1d)):
        if col_idx == 0:
            mask = (array_2d[:, col_idx] == array_1d[col_idx])
        else:
            mask = mask & (array_2d[:, col_idx] == array_1d[col_idx])
    row_indices = np.where(mask)
    return row_indices[0][0] if row_indices[0].size > 0 else None


def prevstep_rows_bool_exact(x0_rows: np.ndarray,
                             y0_rows_bool: np.ndarray,
                             xq_all_same: np.ndarray) -> np.ndarray:
    R, N = x0_rows.shape
    Q = xq_all_same.size
    idx = np.empty((R, Q), dtype=np.int64)
    for r in range(R):
        idx[r] = np.searchsorted(x0_rows[r], xq_all_same, side='right')
    np.clip(idx, 0, N - 1, out=idx)
    idx -= 1
    np.clip(idx, 0, N - 1, out=idx)
    row_idx = np.arange(R)[:, None]
    picked = y0_rows_bool.astype(np.uint8)[row_idx, idx]
    return (picked != 0)


def prevstep_nd_bool_exact(x0_rows: np.ndarray,
                           y0_nd_bool: np.ndarray,
                           xq_all_same: np.ndarray) -> np.ndarray:
    R, N = x0_rows.shape
    Q = xq_all_same.size
    idx = np.empty((R, Q), dtype=np.int64)
    for r in range(R):
        idx[r] = np.searchsorted(x0_rows[r], xq_all_same, side='right')
    np.clip(idx, 0, N - 1, out=idx)
    idx -= 1
    np.clip(idx, 0, N - 1, out=idx)
    expand = (slice(None),) + (None,) * (y0_nd_bool.ndim - 2) + (slice(None),)
    idx_exp = idx[expand]
    gathered = np.take_along_axis(y0_nd_bool.astype(np.uint8), idx_exp,
                                  axis=-1)
    return (gathered != 0)


def build_row_key(row: np.ndarray) -> bytes:
    r = np.ascontiguousarray(row)
    void_dt = np.dtype((np.void, r.dtype.itemsize * r.size))
    return r.view(void_dt).tobytes()


def build_row_index_db(arr_2d: np.ndarray) -> Dict[bytes, int]:
    arr = np.ascontiguousarray(arr_2d)
    void_dt = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    keys = arr.view(void_dt).ravel()
    return {bytes(keys[i]): i for i in range(arr.shape[0])}


def find_row_index_fast(row_1d: np.ndarray,
                        index_db: Dict[bytes, int]) -> Optional[int]:
    return index_db.get(build_row_key(row_1d), None)


def _bytes_key(arr: np.ndarray) -> bytes:
    """
    Fast, stable bytes key for a 1-D contiguous array (dtype+shape+data).
    """
    a = np.ascontiguousarray(arr)
    hdr = f"{a.dtype.str}:{a.shape}".encode('ascii')
    return hdr + a.tobytes()


def _hash_rows_2d(arr_2d: np.ndarray) -> bytes:
    """
    Hash the full content of a 2-D array (dtype+shape+data).
    """
    a = np.ascontiguousarray(arr_2d)
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{a.dtype.str}:{a.shape}".encode('ascii'))
    h.update(a.tobytes())
    return h.digest()


def make_vrbd_cache_key(
    idxs: Tuple[int, ...],
    Ee_rows: np.ndarray,        # 1-D: Ee_mod[idxs]
    VRBD_rows: np.ndarray,      # 1-D: VRBD[idxs]
    Vcell_rows: np.ndarray,     # 2-D: Vcell[idxs]
    Icell_rows: np.ndarray      # 2-D: Icell[idxs]
) -> Tuple:
    """
    Strict cache key that captures everything that can change the
    np.interp(vrbd, vrow, irow):
    - which positions (idxs),
    - irradiance for those positions (Ee_rows),
    - the query points (VRBD_rows),
    - the actual IV arrays used (hashes of Vcell_rows, Icell_rows).
    """
    return (
        idxs,
        _bytes_key(np.asarray(Ee_rows).ravel()),
        _bytes_key(np.asarray(VRBD_rows).ravel()),
        _hash_rows_2d(np.asarray(Vcell_rows)),
        _hash_rows_2d(np.asarray(Icell_rows)),
    )


def row_void_view_1d(rows_2d: np.ndarray) -> np.ndarray:
    """
    Return a 1-D void-view of a 2-D array where each row is treated as one
    element.
    Ensures contiguity and computes the correct itemsize for the view.
    The output shape is (rows_2d.shape[0],).
    """
    rows_c = np.ascontiguousarray(rows_2d)
    void_dt = np.dtype((np.void, rows_c.dtype.itemsize * rows_c.shape[1]))
    return rows_c.view(void_dt).reshape(rows_c.shape[0])


def _step_previous_extrapolate(xq: Iterable,
                               x: Iterable,
                               y: Iterable) -> np.ndarray:
    """
    Previous-step interpolation with extrapolation.
    Equivalent to interp1d(x, y, kind='previous',
                           fill_value='extrapolate')(xq),
    but pure NumPy.

    Assumptions:
      - x is 1D and strictly/weakly monotonic (ascending or descending).
      - y is 1D, same length as x.
      - xq is 1D.

    Returns:
      - yq : 1D array same shape as xq
    """
    xq = np.asarray(xq)
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1 or xq.ndim != 1:
        raise ValueError("x, y, xq must be 1-D.")
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")

    # Work in ascending order
    if x[0] <= x[-1]:
        x_asc = x
        y_asc = y
    else:
        x_asc = x[::-1]
        y_asc = y[::-1]

    # 'previous' index: last j with x_asc[j] <= xq[k]
    idx = np.searchsorted(x_asc, xq, side='right') - 1
    np.clip(idx, 0, x_asc.size - 1, out=idx)

    # Gather
    return y_asc[idx]


def step_previous_bool(xq: Iterable,
                       x: Iterable,
                       y_bool: Iterable) -> np.ndarray:
    """
    Same as _step_previous_extrapolate but returns a boolean array.
    y_bool is a 1D boolean series sampled at x.
    """
    xq = np.asarray(xq)
    x = np.asarray(x)
    yb = np.asarray(y_bool, dtype=bool)

    # Index directly with bool dtype (no need to cast to int8)
    yq_bool = _step_previous_extrapolate(xq, x, yb)
    return yq_bool.astype(bool, copy=False)


def previous_indices(xq: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute the 'previous' indices once for a given x (curve) and xq (target).
    Returns idx such that x[idx] <= xq < x[idx+1] under 'previous' semantics
    with left extrapolation (clamped to [0, len(x)-1]).
    Works for ascending or descending x.
    """
    xq = np.asarray(xq)
    x = np.asarray(x)

    # Ascending orientation
    if x[0] <= x[-1]:
        x_asc = x
        asc = True
    else:
        x_asc = x[::-1]
        asc = False

    idx = np.searchsorted(x_asc, xq, side='right') - 1
    np.clip(idx, 0, x_asc.size - 1, out=idx)

    # If x was descending, idx refers to reversed x; but since we're going to
    # index y_rev = y[::-1] (like you already do), we can keep idx as-is.
    # We'll always index a y that matches the x orientation used to build idx.
    return idx


def delete_files_with_substrings(
    folder_path: str,
    substrings: Iterable[str],
    extension: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> List[str]:
    """
    Delete files in a folder whose filename contains ANY of the specified
    substrings AND ends with the given file extension.

    Parameters
    ----------
    folder_path : str
        Directory to scan (non-recursive).
    substrings : iterable of str
        List/tuple/set of strings. If ANY appears in the filename, the file is
        matched.
        Example: ['sfname_pre', 'stfname_pre']
    extension : str
        File extension to match (e.g. '.pickle' or 'pickle').
    dry_run : bool, optional (default=False)
        If True, do NOT delete files. Just report what would be deleted.
    verbose : bool, optional (default=True)
        If True, print actions taken.

    Returns
    -------
    deleted_files : list of str
        Absolute paths of files that were deleted
        (or would be deleted in dry-run).
    """

    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Folder does not exist: {folder_path}")

    # Normalize extension (allow '.pickle' or 'pickle')
    if not extension.startswith("."):
        extension = "." + extension

    substrings = list(substrings)
    if not substrings:
        raise ValueError(
            "substrings must contain at least one non-empty string")

    deleted_files = []

    for fname in os.listdir(folder_path):
        full_path = os.path.join(folder_path, fname)

        # Only operate on regular files
        if not os.path.isfile(full_path):
            continue

        # Match extension
        if not fname.endswith(extension):
            continue

        # Match ANY substring
        if not any(s in fname for s in substrings):
            continue

        deleted_files.append(full_path)

        if verbose:
            if dry_run:
                print(f"[DRY-RUN] Would delete: {full_path}")
            else:
                print(f"Deleting: {full_path}")

        if not dry_run:
            os.remove(full_path)

    if verbose:
        if dry_run:
            print(f"[DRY-RUN] {len(deleted_files)} file(s) would be deleted.")
        else:
            print(f"{len(deleted_files)} file(s) deleted.")

    return deleted_files
