# -*- coding: utf-8 -*-
"""
This module contains configuration constants for PVMismatch, such as number of
points in IV curve to calculate, flag to use parallel processing and parallel
processing parameters. This module also contains some utility functions like
:func:`~pvmismatch.pvmismatch_lib.pvconstants.npinterpx()` and
:func:`~pvmismatch.pvmismatch_lib.pvconstants.get_series_cells()` are defined
here too.
"""

# TODO: move this to pvmismatch_lib/__init__.py

import numpy as np
import scipy.constants

# Constants
NPTS = 101  # number of I-V points to calculate
MODSIZES = [24, 72, 96, 128]  # list of possible number of cells per module
NUMBERCELLS = MODSIZES[2]  # default is 96-cell module
NUMBERMODS = 10  # default number of modules
NUMBERSTRS = 10  # default number of strings

EPS = np.finfo(np.float64).eps


def npinterpx(x, xp, fp):
    """
    Numpy interpolation with linear extrapolation on both ends.
    - x, xp, fp: 1-D float64 arrays
    - xp must be increasing
    """
    # Interior interpolation (constant ends inside np.interp)
    y = np.interp(x, xp, fp)

    # Precompute slopes at left/right ends once
    sL = (fp[1] - fp[0]) / (xp[1] - xp[0] + EPS)
    sR = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2] + EPS)

    # Left extrapolation
    left_mask = (x < xp[0])
    if left_mask.any():
        xleft = x[left_mask]
        y[left_mask] = fp[0] + (xleft - xp[0]) * sL

    # Right extrapolation
    right_mask = (x > xp[-1])
    if right_mask.any():
        xright = x[right_mask]
        y[right_mask] = fp[-1] + (xright - xp[-1]) * sR

    return y


class PVconstants(object):
    """
    Class for configuration constants

    :param npts: number of points in IV curve
    :type npts: int
    """
    # hard constants
    k = scipy.constants.k  #: [J/K] Boltzmann constant
    q = scipy.constants.e  #: [Coloumbs] elementary charge
    E0 = 1000.  #: [W/m^2] irradiance of 1 sun
    T0 = 298.15  #: [K] reference temperature

    def __init__(self, npts=NPTS):
        self._npts = None
        self.pts = None
        """array of points with decreasing spacing from exactly zero to one"""
        self.negpts = None
        """array of points with increasing spacing from one to ``0.1/npts``"""
        self.Imod_pts = None
        """array of points with increasing spacing from exactly zero to one"""
        self.Imod_pts_sq = None
        """array of points from EPS to 1 with very close spacing near EPS"""
        self.Imod_negpts = None
        """array of points with decreasing spacing from one to  ``0.1/npts``"""
        self.Vmod_q4pts = None
        """array of points with decreasing spacing from ``0.1/npts`` to one"""
        # call property setter
        self.npts = npts  #: number of points in IV curves

    @property
    def npts(self):
        """number of points in IV curves"""
        return self._npts

    @npts.setter
    def npts(self, npts):
        # set number of points in IV curve(s)
        self._npts = npts  # number of points in IV curves
        # point spacing from 0 to 1, used for Vcell, Vmod, Vsys and Istring
        # decrease point spacing as voltage approaches Voc by using logspace
        pts = (11. - np.logspace(np.log10(11.), 0., self._npts)) / 10.
        pts[0] = 0.  # first point must be exactly zero
        self.pts = pts.reshape((self._npts, 1))
        negpts = (11. - np.logspace(np.log10(11. - 1. / float(self._npts)),
                                    0., self._npts)) / 10.
        negpts = negpts.reshape((self._npts, 1))
        self.Imod_negpts = 1 + 1. / float(self._npts) / 10. - negpts
        self.negpts = np.flipud(negpts)  # reverse the order
        # shift and concatenate pvconst.negpts and pvconst.pts
        # so that tight spacing is around MPP and RBD
        self.Imod_pts = 1 - np.flipud(self.pts)
        self.Imod_pts_sq = self.Imod_pts**2 + EPS
        self.Vmod_q4pts = np.flipud(self.Imod_negpts)

    def __str__(self):
        return '<PVconstants(npts=%d)>' % self.npts

    def __repr__(self):
        return str(self)

    def calcSeries(self, I, V, meanIsc, Imax):
        """
        Calculate IV curve for cells and substrings in series given current and
        voltage in increasing order by voltage, the average short circuit
        current and the max current at the breakdown voltage.

        :param I: cell or substring currents [A]
        :param V: cell or substring voltages [V]
        :param meanIsc: average short circuit current [A]
        :param Imax: maximum current [A]
        :return: current [A] and voltage [V] of series
        """
        # make sure all inputs are numpy arrays, but don't make extra copies
        I = np.asarray(I)  # currents [A]
        V = np.asarray(V)  # voltages [V]
        meanIsc = np.asarray(meanIsc)  # mean Isc [A]
        Imax = np.asarray(Imax)  # max current [A]
        # create array of currents optimally spaced from mean Isc to  max VRBD
        Ireverse = (Imax - meanIsc) * self.Imod_pts_sq + meanIsc
        # range of currents in forward bias from 0 to mean Isc
        Iforward = meanIsc * self.pts
        Imin = np.minimum(I.min(), 0.)  # minimum cell current, at most zero
        # range of negative currents in the 4th quadrant from min current to 0
        Iquad4 = Imin * self.Imod_negpts
        # create range for interpolation from forward to reverse bias
        Itot = np.concatenate((Iquad4, Iforward, Ireverse), axis=0).flatten()
        Vtot = np.zeros((3 * self.npts,))
        # add up all series cell voltages
        for i, v in zip(I, V):
            # interp requires x, y to be sorted by x in increasing order
            Vtot += npinterpx(Itot, np.flipud(i), np.flipud(v))
        return np.flipud(Itot), np.flipud(Vtot)

    def calcParallel(self, I, V, Vmax, Vmin, Voc=None):
        """
        Calculate IV curve for cells and substrings in parallel.

        :param I: currents [A]
        :type: I: list, :class:`numpy.ndarray`
        :param V: voltages [V]
        :type: V: list, :class:`numpy.ndarray`
        :param Vmax: max voltage limit, should be max Voc [V]
        :param Vmin: min voltage limit, could be zero or Vrbd [V]
        :param Voc: (``None``) open circuit voltage [V]
        """
        if Voc is None:
            Voc = Vmax
        I, V = np.asarray(I), np.asarray(V)
        Vmax = np.asarray(Vmax)
        Vmin = np.asarray(Vmin)
        Voc = np.asarray(Voc)
        Vff = Voc
        delta_Voc = Vmax - Voc
        if np.isclose(delta_Voc, 0):
            Vff = 0.8 * Voc
            delta_Voc = 0.2 * Voc
        elif delta_Voc < 0:
            Vff = Vmax
            delta_Voc = -delta_Voc
        Vquad4 = Vff + delta_Voc * self.Vmod_q4pts
        Vreverse = Vmin * self.negpts
        Vforward = Vff * self.pts
        Vtot = np.concatenate((Vreverse, Vforward, Vquad4), axis=0).flatten()
        Itot = np.zeros((3 * self.npts,))
        for i, v in zip(I, V):
            Itot += npinterpx(Vtot, v, i)
        return Itot, Vtot


def Vdiode(Icell, Vcell, Rs):
    """
    Calculate Vdiode from current, voltage and series resistance.

    :param Icell: cell current [A]
    :param Vcell: cell voltage [V]
    :param Rs: cell series resistance [:math:`\Omega`]
    :return: diode voltage [V]
    """
    return Vcell + Icell * Rs


def Idiode(Isat, Vdiode, Vt, n):
    """
    Calculate diode current using `Shockley diode model`_.

    :param Isat: diode saturation current [A]
    :param Vdiode: diode voltage [V]
    :param Vt: thermal voltage [V]
    :param n: diode ideality factor
    :return: diode current

    .. LaTeX math directives need double backslashes in comments

    .. note::
       Thermal voltage is defined as ...

       .. math::
           V_t = \\frac {k_B * T} {q_e}

    .. _Shockley diode model: https://en.wikipedia.org/wiki/Diode_modelling#Shockley_diode_model
    """
    return Isat * (np.exp(Vdiode / n / Vt) - 1.)


def Ishunt(Vdiode, Rsh):
    return Vdiode / Rsh


def Igen(Aph, Ee, Isc0):
    return Aph * Ee * Isc0


def get_series_cells(cell_pos_column, prev_col=None):
    """
    Get the sequence of series cells between parallel crossties.

    :param cell_pos_column: column in cell position pattern
    :param prev_col: previous column in cell position pattern
    :return: indices of series cells
    """
    series_cells = []  # empty list of indices of cells in series
    # if the previous column is specified, find the indices of cells in the
    # current column that correspond to cells between parallel crossties in the
    # previous column
    if prev_col:
        cell_pos_column = zip(prev_col, cell_pos_column)
    for cell in cell_pos_column:
        if prev_col:
            cell, next_col = cell
        else:
            next_col = None
        # noinspection PyTypeChecker
        if cell['crosstie'] == True:
            yield series_cells
            series_cells = []
        # if the next column is specified, return the cell indices that
        # correspond to the previous column since they must be the same
        if next_col:
            cell_idx = next_col['idx']
        else:
            # noinspection PyTypeChecker
            cell_idx = cell['idx']
        series_cells.append(cell_idx)
    yield series_cells
