# -*- coding: utf-8 -*-
"""Helper functions that are vectorized."""

import numpy as np
from functools import partial
from scipy.interpolate import interp1d
import pickle

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


def interp_coef(x0: np.ndarray, x: np.ndarray, side='right', kind='linear'):
    """
    Calculate interpolation coefficiencts for N-dimensional arrays.

    Parameters
    ----------
    x0 : numpy.ndarray
        Array to interpolate from.
    x : numpy.ndarray
        Array to interpolate to.
    side : str, optional
        Which side to interpolate from (input for numpy.searchsorted).
        The default is 'right'.
    kind : str, optional
        What type of interpolation.
        The options are 'linear', 'previous', and 'next'.
        The default is 'linear'.

    Returns
    -------
    lo_col : numpy.ndarray
        Left index.
    hi_col : numpy.ndarray
        Right index.
    hi_row : numpy.ndarray
        Rows.
    w : numpy.ndarray
        Weights for interpolation.

    """
    # https://towardsdatascience.com/linear-interpolation-in-python-a-single-line-of-code-25ab83b764f9
    # Modified to do sorting in 2d
    # find the indices into the original array
    # hi_col = np.minimum(x0.shape[1] - 1,
    #      searchsorted2d(x0, x)) # Doesn't work for large float numbers >1e20
    mapfunc = partial(np.searchsorted, side=side)
    hi_col = np.minimum(
        x0.shape[1] - 1, np.array(list(map(mapfunc, x0, x))))
    hi_row = np.array(list(range(hi_col.shape[0])))
    hi_row = np.repeat(hi_row, hi_col.shape[1])
    lo_col = np.maximum(0, hi_col - 1)

    # calculate the distance within the range
    if kind == 'linear':
        d_left = x - x0[hi_row.flatten(),
                        lo_col.flatten()].reshape(lo_col.shape)
        d_right = x0[hi_row.flatten(),
                     hi_col.flatten()].reshape(hi_col.shape) - x
        d_total = d_left + d_right
        # weights are the proportional distance
        w = d_right / d_total
    elif kind == 'previous':
        w = np.ones(lo_col.shape)
    elif kind == 'next':
        w = np.zeros(lo_col.shape)
    # correction if we're outside the range of the array
    # w[np.isinf(w)] = 0.0
    # return the information contained by the projection matrices
    return (lo_col, hi_col, hi_row, w)


def interp_2d(y0: np.ndarray, coef, x, x0):
    """
    Perform interpolation in N-dimensions with base x0.

    Parameters
    ----------
    y0 : np.ndarray
        Y Array to interpolate from.
    coef : tuple
        Output tuple from interp_coef.
    x : np.ndarray
        X Array to interpolate to.
    x0 : np.ndarray
        X Array to interpolate from.

    Returns
    -------
    yint : np.ndarray
        Interpolated Y array.

    """
    # https://towardsdatascience.com/linear-interpolation-in-python-a-single-line-of-code-25ab83b764f9
    term1 = coef[3] * y0[coef[2].flatten(), coef[0].flatten()
                         ].reshape(coef[0].shape)
    term2 = (1 - coef[3]) * y0[coef[2].flatten(),
                               coef[1].flatten()].reshape(coef[0].shape)
    yint = term1 + term2
    # Extrapolate
    # extrapolate left only
    x0_fr = x0[:, 0]
    left = np.logical_and(np.isnan(yint), x < x0_fr[:, None])
    idx_nan = np.where(left)
    xleft = x[left]
    yleft = y0[idx_nan[0], 0] + (xleft - x0[idx_nan[0], 0]) / \
        (x0[idx_nan[0], 1] - x0[idx_nan[0], 0]) * (
            y0[idx_nan[0], 1] - y0[idx_nan[0], 0])
    yint[idx_nan[0], idx_nan[1]] = yleft
    # extrapolate right only
    # extrapolate right
    x0_lr = x0[:, -1]
    right = np.logical_and(np.isnan(yint), x > x0_lr[:, None])
    idx_nan = np.where(right)
    xright = x[right]
    yright = y0[idx_nan[0], -1] + (xright - x0[idx_nan[0], -1]) / \
        (x0[idx_nan[0], -2] - x0[idx_nan[0], -1]) * (
            y0[idx_nan[0], -2] - y0[idx_nan[0], -1])
    yint[idx_nan[0], idx_nan[1]] = yright
    return yint


def interp2d_wrap(x0, x, y0, side='right', kind='linear'):
    """
    Wrap function for N-dimensional interpolation.

    Parameters
    ----------
    x0 : np.ndarray
        X Array to interpolate from.
    x : np.ndarray
        X Array to interpolate to.
    y0 : np.ndarray
        Y Array to interpolate from.
    side : str, optional
        Which side to interpolate from (input for numpy.searchsorted).
        The default is 'right'.
    kind : str, optional
        What type of interpolation.
        The options are 'linear', 'previous', and 'next'.
        The default is 'linear'.

    Returns
    -------
    yint : np.ndarray
        Interpolated Y array.

    """
    # Calculate weighting coefficients
    coef = interp_coef(x0, x, side=side, kind=kind)
    # Interpolate
    yint = interp_2d(y0, coef, x, x0)

    return yint


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
    comparison = array_2d == array_1d
    row_indices = np.where(np.all(comparison, axis=1))[0]
    return row_indices[0] if row_indices.size > 0 else None
