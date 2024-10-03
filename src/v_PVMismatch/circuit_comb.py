# -*- coding: utf-8 -*-
"""Circuit combination and bypass diode configuration functions that are vectorized."""

import numpy as np
from scipy.interpolate import interp1d
from .pvmismatch import pvconstants

# ------------------------------------------------------------------------------
# CIRCUIT COMBINATION FUNCTIONS-------------------------------------------------

DEFAULT_BYPASS = 0
MODULE_BYPASS = 1
CUSTOM_SUBSTR_BYPASS = 2


def calcSeries(Curr, V, meanIsc, Imax, Imod_pts, Imod_negpts, npts):
    """
    Calculate IV curve for cells and substrings in series.

    Given current and voltage in increasing order by voltage,
    the average short circuit current and the max current at the breakdown voltage.

    :param Curr: cell or substring currents [A]
    :param V: cell or substring voltages [V]
    :param meanIsc: average short circuit current [A]
    :param Imax: maximum current [A]
    :return: current [A] and voltage [V] of series

    """
    # make sure all inputs are numpy arrays, but don't make extra copies
    Curr = np.asarray(Curr)  # currents [A]
    V = np.asarray(V)  # voltages [V]
    meanIsc = np.asarray(meanIsc)  # mean Isc [A]
    Imax = np.asarray(Imax)  # max current [A]
    # create array of currents optimally spaced from mean Isc to  max VRBD
    Ireverse = (Imax - meanIsc) * Imod_pts + meanIsc
    Imin = np.minimum(Curr.min(), 0.)  # minimum cell current, at most zero
    # range of currents in forward bias from min current to mean Isc
    Iforward = (Imin - meanIsc) * Imod_negpts + meanIsc
    # create range for interpolation from forward to reverse bias
    Itot = np.concatenate((Iforward, Ireverse), axis=0).flatten()
    Vtot = np.zeros((2 * npts,))
    # add up all series cell voltages
    # t0 = time.time()
    for i, v in zip(Curr, V):
        # interp requires x, y to be sorted by x in increasing order
        Vtot += pvconstants.npinterpx(Itot, np.flipud(i), np.flipud(v))
    # print('Time elapsed to run loopy interp: ' + str(time.time() - t0) + ' s')
    # Itot = np.repeat(Itot[:,np.newaxis], Curr.shape[0], axis=1).T
    # t0 = time.time()
    # Vtot = interp2d_wrap(np.flip(Curr, axis = 1), Itot, np.flip(V, axis = 1))
    # Vtot = Vtot.sum(axis=1)
    # print('Time elapsed to run vectorized interp: ' + str(time.time() - t0) + ' s')
    return np.flipud(Itot), np.flipud(Vtot)


def calcSeries_with_bypass(Curr, V, meanIsc, Imax, Imod_pts, Imod_negpts, npts, substr_bypass, run_bpact=True):
    """
    Calculate IV curve for cells and substrings in series.

    Given current and voltage in increasing order by voltage,
    the average short circuit current and the max current at
    the breakdown voltage.

    :param Curr: cell or substring currents [A]
    :param V: cell or substring voltages [V]
    :param meanIsc: average short circuit current [A]
    :param Imax: maximum current [A]
    :return: current [A] and voltage [V] of series
    """
    # make sure all inputs are numpy arrays, but don't make extra copies
    Curr = np.asarray(Curr)  # currents [A]
    V = np.asarray(V)  # voltages [V]
    meanIsc = np.asarray(meanIsc)  # mean Isc [A]
    Imax = np.asarray(Imax)  # max current [A]

    # create array of currents optimally spaced from mean Isc to  max VRBD
    Ireverse = (Imax - meanIsc) * Imod_pts + meanIsc
    Imin = np.minimum(Curr.min(), 0.)  # minimum cell current, at most zero
    # range of currents in forward bias from min current to mean Isc
    Iforward = (Imin - meanIsc) * Imod_negpts + meanIsc
    # create range for interpolation from forward to reverse bias
    Itot = np.concatenate((Iforward, Ireverse), axis=0).flatten()
    Vtot = np.zeros((2 * npts,))

    # t0 = time.time()
    # add up all series cell voltages
    for i, v in zip(Curr, V):
        # interp requires x, y to be sorted by x in increasing order
        Vtot += pvconstants.npinterpx(Itot, np.flipud(i), np.flipud(v))

    # Logic to check the shape of substr_bypass and ensure interpolations is done correctly
    # Case 1 (Module): I, V, sb have same shape (n x npt)
    # Case 2 (String): I, & V and sb have different dimensions (n x npt and m x n x npt)
    if run_bpact:
        if len(Curr.shape) == len(substr_bypass.shape):
            bypassed_mod = []
            for i, v, bypassed in zip(Curr, V, substr_bypass):
                if ~bypassed.all():
                    bp_interp = bypassed[0]*np.ones(Itot.shape)
                else:
                    interpolator = interp1d(np.flipud(i), np.flipud(bypassed),
                                            kind='previous', fill_value='extrapolate')
                    bp_interp = interpolator(Itot)
                bp_interp = bp_interp.astype(bool)
                bypassed_mod.append(np.flipud(bp_interp))
            bypassed_mod = np.asarray(bypassed_mod)
        else:
            bypassed_mod = np.zeros((Curr.shape[0], substr_bypass.shape[1], 2 * npts))
            idx_mod = 0
            for i, v, bypassed_strs in zip(Curr, V, substr_bypass):
                for idx_substr in range(bypassed_strs.shape[0]):
                    bypassed = bypassed_strs[idx_substr, :]
                    if ~bypassed.all() or bypassed.all():
                        bp_interp = bypassed[0]*np.ones(Itot.shape)
                    else:
                        interpolator = interp1d(np.flipud(i), np.flipud(bypassed),
                                                kind='previous', fill_value='extrapolate')
                        bp_interp = interpolator(Itot)
                    bp_interp = bp_interp.astype(bool)
                    bypassed_mod[idx_mod, idx_substr, :] = np.flipud(bp_interp)
                idx_mod += 1
    else:
        bypassed_mod = np.nan
    return np.flipud(Itot), np.flipud(Vtot), bypassed_mod


def calcParallel(Curr, V, Vmax, Vmin, negpts, pts, npts):
    """
    Calculate IV curve for cells and substrings in parallel.

    :param Curr: currents [A]
    :type: Curr: list, :class:`numpy.ndarray`
    :param V: voltages [V]
    :type: V: list, :class:`numpy.ndarray`
    :param Vmax: max voltage limit, should be max Voc [V]
    :param Vmin: min voltage limit, could be zero or Vrbd [V]
    """
    Curr, V = np.asarray(Curr), np.asarray(V)
    Vmax = np.asarray(Vmax)
    Vmin = np.asarray(Vmin)
    Vreverse = Vmin * negpts
    Vforward = Vmax * pts
    Vtot = np.concatenate((Vreverse, Vforward), axis=0).flatten()
    Itot = np.zeros((2 * npts,))
    for i, v in zip(Curr, V):
        Itot += pvconstants.npinterpx(Vtot, v, i)
    return Itot, Vtot


def calcParallel_with_bypass(Curr, V, Vmax, Vmin, negpts, pts, npts, substr_bypass, run_bpact=True):
    """
    Calculate IV curve for cells and substrings in parallel.

    :param Curr: currents [A]
    :type: Curr: list, :class:`numpy.ndarray`
    :param V: voltages [V]
    :type: V: list, :class:`numpy.ndarray`
    :param Vmax: max voltage limit, should be max Voc [V]
    :param Vmin: min voltage limit, could be zero or Vrbd [V]
    """
    Curr, V = np.asarray(Curr), np.asarray(V)
    Vmax = np.asarray(Vmax)
    Vmin = np.asarray(Vmin)
    Vreverse = Vmin * negpts
    Vforward = Vmax * pts
    Vtot = np.concatenate((Vreverse, Vforward), axis=0).flatten()
    Itot = np.zeros((2 * npts,))
    for i, v in zip(Curr, V):
        Itot += pvconstants.npinterpx(Vtot, v, i)

    if run_bpact:
        if len(Curr.shape) == len(substr_bypass.shape):
            bypassed_mod = []
            for i, v, bypassed in zip(Curr, V, substr_bypass):
                if ~bypassed.all():
                    bp_interp = bypassed[0]*np.ones(Vtot.shape)
                else:
                    interpolator = interp1d(v, bypassed, kind='previous',
                                            fill_value='extrapolate')
                    bp_interp = interpolator(Vtot)
                bp_interp = bp_interp.astype(bool)
                bypassed_mod.append(bp_interp)
            bypassed_mod = np.asarray(bypassed_mod)
            # bypassed_mod = interp2d_wrap(V, Vtot1, substr_bypass.astype(float), kind='previous')
            # bypassed_mod = np.round(bypassed_mod).astype(bool)
        else:
            bypassed_mod = np.zeros((Curr.shape[0], substr_bypass.shape[1],
                                    substr_bypass.shape[2], 2 * npts), dtype=bool)
            idx_str = 0
            for i, v, bypassed_strs in zip(Curr, V, substr_bypass):
                idx_mod = 0
                for idx_mod in range(bypassed_strs.shape[0]):
                    for idx_substr in range(bypassed_strs.shape[1]):
                        bypassed = bypassed_strs[idx_mod, idx_substr, :]
                        if ~bypassed.all() or bypassed.all():
                            bp_interp = bypassed[0]*np.ones(Vtot.shape)
                        else:
                            interpolator = interp1d(v, bypassed, kind='previous',
                                                    fill_value='extrapolate')
                            bp_interp = interpolator(Vtot)
                        bp_interp = bp_interp.astype(bool)
                        bypassed_mod[idx_str, idx_mod, idx_substr, :] = bp_interp
                    idx_mod += 1
                idx_str += 1
    else:
        bypassed_mod = np.nan

    return Itot, Vtot, bypassed_mod


def combine_parallel_circuits(IVprev_cols, pvconst,
                              negpts, pts, Imod_pts, Imod_negpts, npts):
    """
    Combine crosstied circuits in a substring.

    :param IVprev_cols: lists of IV curves of crosstied and series circuits
    :return:
    """
    # combine crosstied circuits
    Irows, Vrows = [], []
    Isc_rows, Imax_rows = [], []
    for IVcols in zip(*IVprev_cols):
        Iparallel, Vparallel = zip(*IVcols)
        Iparallel = np.asarray(Iparallel)
        Vparallel = np.asarray(Vparallel)
        Irow, Vrow = calcParallel(
            Iparallel, Vparallel, Vparallel.max(),
            Vparallel.min(), negpts, pts, npts
        )
        Irows.append(Irow)
        Vrows.append(Vrow)
        Isc_rows.append(np.interp(np.float64(0), Vrow, Irow))
        Imax_rows.append(Irow.max())
    Irows, Vrows = np.asarray(Irows), np.asarray(Vrows)
    Isc_rows = np.asarray(Isc_rows)
    Imax_rows = np.asarray(Imax_rows)
    return calcSeries(
        Irows, Vrows, Isc_rows.mean(), Imax_rows.max(),
        Imod_pts, Imod_negpts, npts
    )


def parse_diode_config(Vbypass, cell_pos):
    """
    Parse diode configuration from the Vbypass argument.

    :param Vbypass: Vbypass config
    :type Vbypass: float|list|tuple
    :param cell_pos:
    :type cell_pos:
    :return: bypass config
    :rtype: str
    """
    try:
        # check if float or list/tuple
        num_bypass = len(Vbypass)
    except TypeError:
        # float passed - default case - Vbypass across every cell string
        return DEFAULT_BYPASS
    else:
        # if only one value is passed in the list- assume only one
        # bypass diode  across the PV module
        if len(Vbypass) == 1:
            return MODULE_BYPASS
        # if more than 1 values are passed, apply them across
        # the cell strings in ascending order
        elif len(cell_pos) == num_bypass:
            return CUSTOM_SUBSTR_BYPASS
        else:
            print("wrong number of bypass diode values passed : %d" %
                  (len(Vbypass)))
