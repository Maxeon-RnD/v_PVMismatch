# -*- coding: utf-8 -*-
"""Estimate the current, voltage, power, cells in reverse during mismatch.

The functions also calculate the diode currents.
The methodology is solving the circuit (similar to LTSpice).
"""

import numpy as np


def est_cell_current_DC(sim_data, str_data, mod_data, cell_index):
    cell_currs = {}
    # Bypass diode activation flags at MPP (same at short circuit).
    bpd_mpp = sim_data['Bypass_Active_MPP']
    # System level voltagee at MPP.
    Vmp = sim_data['Vmp']
    # All the system level IV curves.
    full_data = sim_data['full_data']
    # Initialize output arrays.
    np_shape = list(bpd_mpp.shape[:-1]) + list(cell_index.shape)
    cell_Imps = np.zeros(np_shape)
    cell_Iscs = cell_Imps.copy()
    cell_Vmps = cell_Imps.copy()
    cell_Vscs = cell_Imps.copy()
    cell_isRev_mp = np.zeros(np_shape, dtype=bool)
    cell_isRev_sc = cell_isRev_mp.copy()
    diode_Imps = np.zeros(bpd_mpp.shape)
    diode_Iscs = np.zeros(bpd_mpp.shape)
    # Run through each simulation and calculate the String currents.
    for idx_sim, Vsim in enumerate(Vmp):
        Istrings = full_data[idx_sim]['Istrings']
        Vstrings = full_data[idx_sim]['Vstrings']
        idxs_str = full_data[idx_sim]['Str_idxs']
        # Run through each string.
        for idx_str in range(Istrings.shape[0]):
            Istr = Istrings[idx_str, :]
            Vstr = Vstrings[idx_str, :]
            # The string current = current @ system Vmp or Vsc.
            I_str_mp = np.interp(Vsim, Vstr, Istr)
            I_str_sc = np.interp(0., Vstr, Istr)
            str_full_data = str_data['full_data'][idxs_str[idx_str]]
            # Run through each module.
            for idx_mod in range(str_full_data['Imods'].shape[0]):
                # Module currents, voltages
                # Get module data for specific string
                mod_idx = int(str_full_data['Mod_idxs'][idx_mod])
                # Get diode data for specific module
                mod_full_data = mod_data['full_data'][mod_idx]
                # Run through each bypass diode subsection.
                for idx_dio in range(mod_full_data['Vsubstr'].shape[0]):
                    # Extract bypass diode activation at MPP.
                    diode_act = bpd_mpp[idx_sim, idx_str, idx_mod, idx_dio]
                    if diode_act:
                        # Logic for when diode is active.
                        # Istr = Idiode + (Isubsection @ Vbypass)
                        # Step 1: Find last index where V = Vbypass.
                        Vd = mod_full_data['Vsubstr'][idx_dio, :]
                        Id = mod_full_data['Isubstr'][idx_dio, :]
                        last_index = np.max(np.argwhere(Vd == np.min(Vd)))
                        # Step 2: Remove data before last index.
                        Vd = Vd[last_index:]
                        Id = Id[last_index:]
                        # Step 3: Calculate Idiode and Isubsection @ Vbypass.
                        if I_str_mp > Id[0]:
                            # If Istr > Isubsection @ Vbypass.
                            # Isubsection = I @ Vbypass.
                            # Current through diode = Istr - Isubsection.
                            Idiode = I_str_mp - Id[0]
                            Issmp = Id[0]
                            diode_Imps[idx_sim, idx_str,
                                       idx_mod, idx_dio] = Idiode
                        else:
                            # If not, entire Istr is through subsection.
                            # Idiode = 0, already initialized.
                            Issmp = I_str_mp
                        # Repeat for short circuit string current.
                        if I_str_sc > Id[0]:
                            Idiode = I_str_sc - Id[0]
                            Isssc = Id[0]
                            diode_Iscs[idx_sim, idx_str,
                                       idx_mod, idx_dio] = Idiode
                        else:
                            Isssc = I_str_sc
                    else:
                        # If bypass diode doesn't activate, Isubsection = Istr.
                        # Idiode = 0, already initialized.
                        Vd = mod_full_data['Vsubstr_pre_bypass'][idx_dio, :]
                        Id = mod_full_data['Isubstr_pre_bypass'][idx_dio, :]
                        Issmp = I_str_mp
                        Isssc = I_str_sc
                    # Series crosstie currents, voltages
                    num_sct = [x for x in mod_full_data[idx_dio]
                               if isinstance(x, int)]
                    for idx_sct in num_sct:
                        Vsct = mod_full_data[idx_dio][idx_sct]['Vsubstr']
                        Isct = mod_full_data[idx_dio][idx_sct]['Isubstr']
                        if diode_act:
                            # Logic for when diode is active.
                            # Step 1: Find last index where V = Vbypass.
                            byp_v = np.min(Vd)
                            Vsct[Vsct < byp_v] = byp_v
                            last_index = np.max(
                                np.argwhere(Vsct == np.min(Vsct)))
                            # Step 2: Remove data before last index.
                            Vsct = Vsct[last_index:]
                            Isct = Isct[last_index:]
                        # The series crosstie voltage @ Isubsection.
                        V_sct_mp = np.interp(Issmp, np.flipud(Isct),
                                             np.flipud(Vsct))
                        V_sct_sc = np.interp(Isssc, np.flipud(Isct),
                                             np.flipud(Vsct))
                        # Parallel crosstie currents, voltages
                        num_pct = [x for x in mod_full_data[idx_dio]
                                   [idx_sct] if isinstance(x, int)]
                        for idx_pct in num_pct:
                            Vpct = mod_full_data[idx_dio][idx_sct][idx_pct]['Vsubstr']
                            Ipct = mod_full_data[idx_dio][idx_sct][idx_pct]['Isubstr']
                            if diode_act:
                                # Logic for when diode is active.
                                # Step 1: Find last index where V = Vbypass.
                                byp_v = np.min(Vd)
                                Vpct[Vpct < byp_v] = byp_v
                                last_index = np.max(
                                    np.argwhere(Vpct == np.min(Vpct)))
                                # Step 2: Remove data before last index.
                                Vpct = Vpct[last_index:]
                                Ipct = Ipct[last_index:]
                            # The cell current @ Vseriescrosstie.
                            Imp_pct = np.interp(V_sct_mp, Vpct, Ipct)
                            Isc_pct = np.interp(V_sct_sc, Vpct, Ipct)
                            # Cell currents, voltages
                            cell_idxs = mod_full_data[idx_dio][idx_sct][idx_pct]['cell_idxs']
                            cell_currents = mod_full_data[idx_dio][idx_sct][idx_pct]['cell_currents']
                            cell_voltages = mod_full_data[idx_dio][idx_sct][idx_pct]['cell_voltages']
                            for idx_row, idx_cell in enumerate(cell_idxs):
                                # The cell voltage @ Icell.
                                V_cell_mp = np.interp(Imp_pct,
                                                      np.flipud(
                                                          cell_currents[idx_row, :]),
                                                      np.flipud(cell_voltages[idx_row, :]))
                                V_cell_sc = round(np.interp(Isc_pct,
                                                            np.flipud(
                                                                cell_currents[idx_row, :]),
                                                            np.flipud(cell_voltages[idx_row, :])), 2)
                                row, col = np.where(cell_index == idx_cell)
                                cell_Imps[idx_sim, idx_str,
                                          idx_mod, row[0], col[0]] = Imp_pct
                                cell_Iscs[idx_sim, idx_str,
                                          idx_mod, row[0], col[0]] = Isc_pct
                                cell_Vmps[idx_sim, idx_str,
                                          idx_mod, row[0], col[0]] = V_cell_mp
                                cell_Vscs[idx_sim, idx_str,
                                          idx_mod, row[0], col[0]] = V_cell_sc
    # Populate is reverse matrices.
    isRev = cell_Vmps < 0.
    cell_isRev_mp[isRev] = True
    isRev = cell_Vscs < 0.
    cell_isRev_sc[isRev] = True
    # Calculate Power dissipation (or generation if not in reverse).
    cell_Pmps = cell_Imps * cell_Vmps
    cell_Pscs = cell_Iscs * cell_Vscs
    # Store results
    cell_currs['cell_Imps'] = cell_Imps.copy()
    cell_currs['cell_Iscs'] = cell_Iscs.copy()
    cell_currs['cell_Vmps'] = cell_Vmps.copy()
    cell_currs['cell_Vscs'] = cell_Vscs.copy()
    cell_currs['cell_Pmps'] = cell_Pmps.copy()
    cell_currs['cell_Pscs'] = cell_Pscs.copy()
    cell_currs['cell_isRev_mp'] = cell_isRev_mp.copy()
    cell_currs['cell_isRev_sc'] = cell_isRev_sc.copy()
    cell_currs['diode_Imps'] = diode_Imps.copy()
    cell_currs['diode_Iscs'] = diode_Iscs.copy()

    return cell_currs


def est_cell_current_AC(sim_data, cell_index):
    cell_currs = {}
    # Get system MPP data & bypass diode activation
    bpd_mpp = sim_data['Bypass_Active_MPP']
    # Initialize output arrays.
    np_shape = list(bpd_mpp.shape[:-2]) + list(cell_index.shape)
    cell_Imps = np.zeros(np_shape)
    cell_Iscs = cell_Imps.copy()
    cell_Vmps = cell_Imps.copy()
    cell_Vscs = cell_Imps.copy()
    cell_isRev_mp = np.zeros(np_shape, dtype=bool)
    cell_isRev_sc = cell_isRev_mp.copy()
    diode_shp = list(bpd_mpp.shape[:-2]) + [sim_data['full_data']
                                            [0][0][0]['full_data']['bypassed_mod'].shape[0]]
    diode_Imps = np.zeros(diode_shp)
    diode_Iscs = np.zeros(diode_shp)
    # All the system level IV curves.
    full_data = sim_data['full_data']
    # Run through each simulation.
    for idx_sim in range(np_shape[0]):
        sim_data = full_data[idx_sim]
        # Run through each string.
        for idx_str in range(np_shape[1]):
            str_data = sim_data[idx_str]
            # Run through each module.
            for idx_mod in range(np_shape[2]):
                mod_data = str_data[idx_mod]
                Imp = mod_data['Imp'][0]
                Isc = mod_data['Isc'][0]
                bpd_act_mod = mod_data['BPDMPP'][0, :]
                mod_full_data = mod_data['full_data']
                for idx_dio in range(mod_full_data['Vsubstr'].shape[0]):
                    # Extract bypass diode activation at MPP.
                    diode_act = bpd_act_mod[idx_dio]
                    if diode_act:
                        # Logic for when diode is active.
                        # Istr = Idiode + (Isubsection @ Vbypass)
                        # Step 1: Find last index where V = Vbypass.
                        Vd = mod_full_data['Vsubstr'][idx_dio, :]
                        Id = mod_full_data['Isubstr'][idx_dio, :]
                        last_index = np.max(np.argwhere(Vd == np.min(Vd)))
                        # Step 2: Remove data before last index.
                        Vd = Vd[last_index:]
                        Id = Id[last_index:]
                        # Step 3: Calculate Idiode and Isubsection @ Vbypass.
                        if Imp > Id[0]:
                            # If Istr > Isubsection @ Vbypass.
                            # Isubsection = I @ Vbypass.
                            # Current through diode = Istr - Isubsection.
                            Idiode = Imp - Id[0]
                            Issmp = Id[0]
                            diode_Imps[idx_sim, idx_str,
                                       idx_mod, idx_dio] = Idiode
                        else:
                            # If not, entire Istr is through subsection.
                            # Idiode = 0, already initialized.
                            Issmp = Imp
                        # Repeat for short circuit string current.
                        if Isc > Id[0]:
                            Idiode = Isc - Id[0]
                            Isssc = Id[0]
                            diode_Iscs[idx_sim, idx_str,
                                       idx_mod, idx_dio] = Idiode
                        else:
                            Isssc = Isc
                    else:
                        # If bypass diode doesn't activate, Isubsection = Istr.
                        # Idiode = 0, already initialized.
                        Vd = mod_full_data['Vsubstr_pre_bypass'][idx_dio, :]
                        Id = mod_full_data['Isubstr_pre_bypass'][idx_dio, :]
                        Issmp = Imp
                        Isssc = Isc
                    # Series crosstie currents, voltages
                    num_sct = [x for x in mod_full_data[idx_dio]
                               if isinstance(x, int)]
                    for idx_sct in num_sct:
                        Vsct = mod_full_data[idx_dio][idx_sct]['Vsubstr']
                        Isct = mod_full_data[idx_dio][idx_sct]['Isubstr']
                        if diode_act:
                            # Logic for when diode is active.
                            # Step 1: Find last index where V = Vbypass.
                            byp_v = np.min(Vd)
                            Vsct[Vsct < byp_v] = byp_v
                            last_index = np.max(
                                np.argwhere(Vsct == np.min(Vsct)))
                            # Step 2: Remove data before last index.
                            Vsct = Vsct[last_index:]
                            Isct = Isct[last_index:]
                        # The series crosstie voltage @ Isubsection.
                        V_sct_mp = np.interp(Issmp, np.flipud(Isct),
                                             np.flipud(Vsct))
                        V_sct_sc = np.interp(Isssc, np.flipud(Isct),
                                             np.flipud(Vsct))
                        # Parallel crosstie currents, voltages
                        num_pct = [x for x in mod_full_data[idx_dio]
                                   [idx_sct] if isinstance(x, int)]
                        for idx_pct in num_pct:
                            Vpct = mod_full_data[idx_dio][idx_sct][idx_pct]['Vsubstr']
                            Ipct = mod_full_data[idx_dio][idx_sct][idx_pct]['Isubstr']
                            if diode_act:
                                # Logic for when diode is active.
                                # Step 1: Find last index where V = Vbypass.
                                byp_v = np.min(Vd)
                                Vpct[Vpct < byp_v] = byp_v
                                last_index = np.max(
                                    np.argwhere(Vpct == np.min(Vpct)))
                                # Step 2: Remove data before last index.
                                Vpct = Vpct[last_index:]
                                Ipct = Ipct[last_index:]
                            # The cell current @ Vseriescrosstie.
                            Imp_pct = np.interp(V_sct_mp, Vpct, Ipct)
                            Isc_pct = np.interp(V_sct_sc, Vpct, Ipct)
                            # Cell currents, voltages
                            cell_idxs = mod_full_data[idx_dio][idx_sct][idx_pct]['cell_idxs']
                            cell_currents = mod_full_data[idx_dio][idx_sct][idx_pct]['cell_currents']
                            cell_voltages = mod_full_data[idx_dio][idx_sct][idx_pct]['cell_voltages']
                            for idx_row, idx_cell in enumerate(cell_idxs):
                                # The cell voltage @ Icell.
                                V_cell_mp = np.interp(Imp_pct,
                                                      np.flipud(
                                                          cell_currents[idx_row, :]),
                                                      np.flipud(cell_voltages[idx_row, :]))
                                V_cell_sc = round(np.interp(Isc_pct,
                                                            np.flipud(
                                                                cell_currents[idx_row, :]),
                                                            np.flipud(cell_voltages[idx_row, :])), 2)
                                row, col = np.where(cell_index == idx_cell)
                                cell_Imps[idx_sim, idx_str, idx_mod,
                                          row[0], col[0]] = Imp_pct
                                cell_Iscs[idx_sim, idx_str, idx_mod,
                                          row[0], col[0]] = Isc_pct
                                cell_Vmps[idx_sim, idx_str, idx_mod,
                                          row[0], col[0]] = V_cell_mp
                                cell_Vscs[idx_sim, idx_str, idx_mod,
                                          row[0], col[0]] = V_cell_sc
    # Populate is reverse matrices.
    isRev = cell_Vmps < 0.
    cell_isRev_mp[isRev] = True
    isRev = cell_Vscs < 0.
    cell_isRev_sc[isRev] = True
    # Calculate Power dissipation (or generation if not in reverse).
    cell_Pmps = cell_Imps * cell_Vmps
    cell_Pscs = cell_Iscs * cell_Vscs
    # Store results
    cell_currs['cell_Imps'] = cell_Imps.copy()
    cell_currs['cell_Iscs'] = cell_Iscs.copy()
    cell_currs['cell_Vmps'] = cell_Vmps.copy()
    cell_currs['cell_Vscs'] = cell_Vscs.copy()
    cell_currs['cell_Pmps'] = cell_Pmps.copy()
    cell_currs['cell_Pscs'] = cell_Pscs.copy()
    cell_currs['cell_isRev_mp'] = cell_isRev_mp.copy()
    cell_currs['cell_isRev_sc'] = cell_isRev_sc.copy()
    cell_currs['diode_Imps'] = diode_Imps.copy()
    cell_currs['diode_Iscs'] = diode_Iscs.copy()

    return cell_currs
