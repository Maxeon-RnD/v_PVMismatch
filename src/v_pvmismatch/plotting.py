# -*- coding: utf-8 -*-
"""Plotting functions."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt


def plot_cell(cell_data, idx_cell):
    """
    Plot the cell IV and PV curves for a given cell index.

    Parameters
    ----------
    cell_data : dict
        Dictionary containing cell IV curves.
    idx_cell : int
        Index of cell IV curves to plot.

    Returns
    -------
    cell_plot : plotly.figure
        IV & PV curve.

    """
    VRBD = cell_data['VRBD'][idx_cell, 0] - 1
    Isc = cell_data['Isc'][idx_cell, 0]
    Voc = cell_data['Voc'][idx_cell, 0]
    Icell = cell_data['Icell'][idx_cell, :].copy()
    Vcell = cell_data['Vcell'][idx_cell, :].copy()
    Pcell = cell_data['Pcell'][idx_cell, :].copy()
    cell_plot = plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(Vcell, Icell)
    plt.title('Cell Reverse I-V Characteristics')
    plt.ylabel('Cell Current, I [A]')
    plt.xlim(VRBD - 1, 0)
    plt.ylim(0, Isc + 10)
    plt.grid()
    plt.subplot(2, 2, 2)
    plt.plot(Vcell, Icell)
    plt.title('Cell Forward I-V Characteristics')
    plt.ylabel('Cell Current, I [A]')
    plt.xlim(0, Voc)
    plt.ylim(0, Isc + 1)
    plt.grid()
    plt.subplot(2, 2, 3)
    plt.plot(Vcell, Pcell)
    plt.title('Cell Reverse P-V Characteristics')
    plt.xlabel('Cell Voltage, V [V]')
    plt.ylabel('Cell Power, P [W]')
    plt.xlim(VRBD - 1, 0)
    plt.ylim((Isc + 10) * (VRBD - 1), -1)
    plt.grid()
    plt.subplot(2, 2, 4)
    plt.plot(Vcell, Pcell)
    plt.title('Cell Forward P-V Characteristics')
    plt.xlabel('Cell Voltage, V [V]')
    plt.ylabel('Cell Power, P [W]')
    plt.xlim(0, Voc)
    plt.ylim(0, (Isc + 1) * Voc)
    plt.grid()
    plt.tight_layout()

    return cell_plot


def plot_module(mod_data, idx_mod, curve='IV'):
    """
    Plot the module IV and PV curves for a given module index.

    Parameters
    ----------
    mod_data : dict
        Dictionary containing module IV curves.
    idx_mod : int
        Index of module IV curves to plot.

    Returns
    -------
    fig : plotly.figure
        IV & PV curve.

    """
    num_str = mod_data['Bypassed_substr'].shape[1]
    Ipos = (mod_data['Vmod'][idx_mod, :] > 0)
    Imax = mod_data['Imod'][idx_mod, Ipos].max()
    Pmax = mod_data['Pmod'][idx_mod, Ipos].max()
    Vmax = mod_data['Vmod'][idx_mod, :].max()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if curve == 'IV':
        fig.add_trace(go.Scatter(name="I-V",
                                 x=mod_data['Vmod'][idx_mod, :].tolist(),
                                 y=mod_data['Imod'][idx_mod, :].tolist()),
                      secondary_y=False)
        for idx_str in range(num_str):
            fig.add_trace(go.Scatter(name=' '.join(["BP Active String",
                                                    str(idx_str+1)]),
                                     x=mod_data['Vmod'][idx_mod, :].tolist(),
                                     y=mod_data['Bypassed_substr'][idx_mod,
                                                                   idx_str,
                                                                   :].tolist()),
                          secondary_y=True)
        fig.update_yaxes(title_text='Current [A]', range=[0, Imax+1],
                         secondary_y=False)
        fig.update_yaxes(title_text='Bypass Diode Active',
                         secondary_y=True)
        fig.update_xaxes(title_text='Voltage [V]', range=[-10, Vmax+1])

    else:
        fig.add_trace(go.Scatter(name="P-V",
                                 x=mod_data['Vmod'][idx_mod, :].tolist(),
                                 y=mod_data['Pmod'][idx_mod, :].tolist()),
                      secondary_y=False)
        for idx_str in range(num_str):
            fig.add_trace(go.Scatter(name=' '.join(["BP Active String",
                                                    str(idx_str+1)]),
                                     x=mod_data['Vmod'][idx_mod, :].tolist(),
                                     y=mod_data['Bypassed_substr'][idx_mod,
                                                                   idx_str,
                                                                   :].tolist()),
                          secondary_y=True)
        fig.update_yaxes(title_text='Power [W]', range=[0, Pmax+1],
                         secondary_y=False)
        fig.update_yaxes(title_text='Bypass Diode Active',
                         secondary_y=True, row=2, col=1)
        fig.update_xaxes(title_text='Voltage [V]', range=[-10, Vmax+1])

    fig.update_layout(
        title_text=' '.join(['Module IV/ PV curves', str(idx_mod)]),
        autosize=False,
        width=800,
        height=800,
    )

    return fig


def plot_string(str_data, idx_str, idx_mod, curve='IV'):
    """
    Plot the string IV and PV curves for a given string & module index.

    Parameters
    ----------
    str_data : dict
        Dictionary containing string IV curves.
    idx_str : int
        Index of string IV curves to plot.
    idx_mod : int
        Index of module IV curves to plot.

    Returns
    -------
    fig : plotly.figure
        IV & PV curve.

    """
    num_str = str_data['Bypassed_substr'].shape[2]
    Ipos = (str_data['Vstring'][idx_str, :] > 0)
    Imax = str_data['Istring'][idx_str, Ipos].max()
    Pmax = str_data['Pstring'][idx_str, Ipos].max()
    Vmax = str_data['Vstring'][idx_str, :].max()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if curve == 'IV':
        fig.add_trace(go.Scatter(name="I-V",
                                 x=str_data['Vstring'][idx_str, :].tolist(),
                                 y=str_data['Istring'][idx_str, :].tolist()),
                      secondary_y=False)
        for idx_sstr in range(num_str):
            fig.add_trace(go.Scatter(name=' '.join(["BP Active String",
                                                    str(idx_sstr+1)]),
                                     x=str_data['Vstring'][idx_str,
                                                           :].tolist(),
                                     y=str_data['Bypassed_substr'][idx_str,
                                                                   idx_mod,
                                                                   idx_sstr,
                                                                   :].tolist()),
                          secondary_y=True)
        fig.update_yaxes(title_text='Current [A]', range=[0, Imax+1],
                         secondary_y=False)
        fig.update_yaxes(title_text='Bypass Diode Active',
                         secondary_y=True)
        fig.update_xaxes(title_text='Voltage [V]', range=[-10, Vmax+1])
    else:
        fig.add_trace(go.Scatter(name="P-V",
                                 x=str_data['Vstring'][idx_str, :],
                                 y=str_data['Pstring'][idx_str, :]),
                      secondary_y=False)
        for idx_sstr in range(num_str):
            fig.add_trace(go.Scatter(name=' '.join(["BP Active String",
                                                    str(idx_sstr+1)]),
                                     x=str_data['Vstring'][idx_str,
                                                           :].tolist(),
                                     y=str_data['Bypassed_substr'][idx_str,
                                                                   idx_mod,
                                                                   idx_sstr,
                                                                   :].tolist()),
                          secondary_y=True)
        fig.update_yaxes(title_text='Power [W]', range=[0, Pmax+1],
                         secondary_y=False)
        fig.update_yaxes(title_text='Bypass Diode Active',
                         secondary_y=True)
        fig.update_xaxes(title_text='Voltage [V]', range=[-10, Vmax+1])

    fig.update_layout(
        title_text=' '.join(['String IV/ PV curves', str(idx_str),
                             'Module index', str(idx_mod)]),
        autosize=False,
        width=800,
        height=800,
    )

    return fig


def plot_system(sys_data, idx_sys, idx_str, idx_mod, curve='IV'):
    """
    Plot the system IV and PV curves for a given system, string & module index.

    Parameters
    ----------
    sys_data : dict
        Dictionary containing system IV curves.
    idx_sys : int
        Index of string IV curves to plot.
    idx_str : int
        Index of string IV curves to plot.
    idx_mod : int
        Index of module IV curves to plot.

    Returns
    -------
    fig : plotly.figure
        IV & PV curve.

    """
    num_str = sys_data['Bypass_activation'].shape[3]
    Ipos = (sys_data['Vsys'][idx_sys, :] > 0)
    Imax = sys_data['Isys'][idx_sys, Ipos].max()
    Pmax = sys_data['Psys'][idx_sys, Ipos].max()
    Vmax = sys_data['Vsys'][idx_sys, :].max()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if curve == 'IV':
        fig.add_trace(go.Scatter(name="I-V",
                                 x=sys_data['Vsys'][idx_sys, :].tolist(),
                                 y=sys_data['Isys'][idx_sys, :].tolist()),
                      secondary_y=False)
        for idx_sstr in range(num_str):
            fig.add_trace(go.Scatter(name=' '.join(["BP Active String",
                                                    str(idx_sstr+1)]),
                                     x=sys_data['Vsys'][idx_sys, :].tolist(),
                                     y=sys_data['Bypass_activation'][idx_sys,
                                                                     idx_str,
                                                                     idx_mod,
                                                                     idx_sstr,
                                                                     :].tolist()),
                          secondary_y=True)
        fig.update_yaxes(title_text='Current [A]', range=[0, Imax+1],
                         secondary_y=False)
        fig.update_yaxes(title_text='Bypass Diode Active',
                         secondary_y=True)
        fig.update_xaxes(title_text='Voltage [V]', range=[-10, Vmax+1])

    else:
        fig.add_trace(go.Scatter(name="P-V",
                                 x=sys_data['Vsys'][idx_sys, :],
                                 y=sys_data['Psys'][idx_sys, :]),
                      secondary_y=False)
        for idx_sstr in range(num_str):
            fig.add_trace(go.Scatter(name=' '.join(["BP Active String",
                                                    str(idx_sstr+1)]),
                                     x=sys_data['Vsys'][idx_sys, :].tolist(),
                                     y=sys_data['Bypass_activation'][idx_sys,
                                                                     idx_str,
                                                                     idx_mod,
                                                                     idx_sstr,
                                                                     :].tolist()),
                          secondary_y=True)
        fig.update_yaxes(title_text='Power [W]', range=[0, Pmax+1],
                         secondary_y=False)
        fig.update_yaxes(title_text='Bypass Diode Active',
                         secondary_y=True)
        fig.update_xaxes(title_text='Voltage [V]', range=[-10, Vmax+1])

    fig.update_layout(
        title_text=' '.join(['System IV/ PV curves', str(idx_sys),
                             'String index', str(idx_str),
                             'Module index', str(idx_mod)]),
        autosize=False,
        width=800,
        height=800,
    )

    return fig


def plot_heatmap(cell_curr, idx_sim, dkey='cell_Imps',
                 show_vals=True, cmap='viridis', rounding=1):
    """
    Plot a heatmap of the outputs of the cell current estimation model.

    Parameters
    ----------
    cell_curr : dict
        Dictionary containing the outputs.
    idx_sim : int
        Simulation index.
    dkey : str, optional
        Which parameter to plot? This is the key of the dict.
        The default is 'cell_Imps'.
    show_vals : bool, optional
        Set to True if values to be displayed in the heatmap.
        The default is True.
    cmap : str, optional
        Heatmap cmap. The default is 'viridis'.
    rounding : int, optional
        Round values to specified significant digits. The default is 1.

    Returns
    -------
    fig : plotly.figure
        Heatmap.

    """
    cell_Imps = cell_curr[dkey].copy()
    sys_shp = cell_Imps[idx_sim, :, :, :, :].shape
    str_len = sys_shp[1]
    num_str = sys_shp[0]

    fig = make_subplots(rows=num_str, cols=str_len)
    for midx in range(str_len):
        for sidx in range(num_str):
            Ee = cell_Imps[idx_sim, sidx, midx, :, :]
            if show_vals:
                fig.add_trace(go.Heatmap(z=Ee.tolist(),
                                         text=Ee.round(rounding).tolist(),
                                         texttemplate="%{text}",
                                         coloraxis="coloraxis"),
                              row=sidx+1, col=midx+1)
            else:
                Ee = Ee.astype(int)
                fig.add_trace(go.Heatmap(z=Ee.tolist(),
                                         # text=Ee.round(1).tolist(),
                                         # texttemplate="%{text}",
                                         coloraxis="coloraxis"),
                              row=sidx+1, col=midx+1)
            fig.update_yaxes(autorange='reversed', row=sidx+1, col=midx+1)
    fig.update_layout(
        title_text='_'.join([dkey, str(idx_sim)]),
        autosize=False,
        width=1000,
        height=800,
        coloraxis={'colorscale': cmap}
    )

    return fig


def plot_heatmap_diode(cell_curr, idx_sim, dkey='diode_Imps',
                       show_vals=True, cmap='viridis', rounding=1):
    """
    Plot a heatmap of the outputs of the cell current estimation model.

    Parameters
    ----------
    cell_curr : dict
        Dictionary containing the outputs.
    idx_sim : int
        Simulation index.
    dkey : str, optional
        Which parameter to plot? This is the key of the dict.
        The default is 'diode_Imps'.
    show_vals : bool, optional
        Set to True if values to be displayed in the heatmap.
        The default is True.
    cmap : str, optional
        Heatmap cmap. The default is 'viridis'.
    rounding : int, optional
        Round values to specified significant digits. The default is 1.

    Returns
    -------
    fig : plotly.figure
        Heatmap.

    """
    cell_Imps = cell_curr[dkey].copy()
    sys_shp = cell_Imps[idx_sim, :, :, :].shape
    str_len = sys_shp[1]
    num_str = sys_shp[0]

    fig = make_subplots(rows=num_str, cols=str_len)
    for midx in range(str_len):
        for sidx in range(num_str):
            Ee = np.array([cell_Imps[idx_sim, sidx, midx, :].tolist()])
            if show_vals:
                fig.add_trace(go.Heatmap(z=Ee.tolist(),
                                         text=Ee.round(rounding).tolist(),
                                         texttemplate="%{text}",
                                         coloraxis="coloraxis"),
                              row=sidx+1, col=midx+1)
            else:
                Ee = Ee.astype(int)
                fig.add_trace(go.Heatmap(z=Ee.tolist(),
                                         # text=Ee.round(1).tolist(),
                                         # texttemplate="%{text}",
                                         coloraxis="coloraxis"),
                              row=sidx+1, col=midx+1)
            fig.update_yaxes(autorange='reversed', row=sidx+1, col=midx+1)
    fig.update_layout(
        title_text='_'.join([dkey, str(idx_sim)]),
        autosize=False,
        width=1000,
        height=800,
        coloraxis={'colorscale': cmap}
    )

    return fig
