#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:16:20 2022

@author: jiedz

Quick calculation of the prepare & measure covariance matrix with
state monitoring at the transmitter.

For reference see my PhD thesis
"""
#%%
from quik.qip.nmodes_symbolic import Vacuum, CovarianceMatrix
import sympy
sympy.init_printing()
#%%
"""
Construct the PM covariance matrix with state monitoring at the transmitter, 
and verify that it still has normal form. This ensures that the same 
resulting state can still be purified by a two-mode squeezed vacuum state
as in the ideal transmitter model (see ideal-transmitter.py)
"""

V_q_0_A, V_p_0_A = sympy.symbols("V_{q_{0_A}}, V_{p_{0_A}}", real=True, nonnegative=True)
R_M_A, R_M, T_M_A_q, T_M_A_p = sympy.symbols("R_{MA}, R_M, T_{MA_q}, T_{MA_p}", real=True, nonnegative=True)
nu_M_A_q, nu_M_A_p, xi_M_A_q, xi_M_A_p = sympy.symbols("\\nu_{MA_q}, \\nu_{MA_p}, \\xi_{MA_q}, \\xi_{MA_p}", real=True, positive=True)
#1.
V_PM = Vacuum(7)
V_PM[12:, 12:] = CovarianceMatrix([[V_q_0_A, 0], [0, V_p_0_A]])
#Split at the tap beam splitter
V_PM = V_PM.pick_modes(1, 2, 3, 4, 5, 7, 6)
V_PM = V_PM.bs(6, 7, R_M_A)
#Split at the state monitor beam splitter
V_PM = V_PM.bs(5, 6, R_M)
#Define the two-mode squeeze vacuum states that models the additive Gaussian noise
#at the state monitor detectors
V_PM = V_PM.epr_state(1, 2, nu_M_A_q)
V_PM = V_PM.epr_state(3, 4, nu_M_A_p)
#Interfere each mode with the corresponding mode of the monitored signal,
#onto the beam splitter that model the corresponding detector loss
V_PM = V_PM.bs(2, 6, 1-T_M_A_q)
V_PM = V_PM.bs(4, 5, 1-T_M_A_p)
V_PM = V_PM.subs([((1-T_M_A_q)*nu_M_A_q, xi_M_A_q), ((1-T_M_A_p)*nu_M_A_p, xi_M_A_p)])
V_PM = V_PM.pick_modes(2, 4, 5, 6, 7)
V_PM = V_PM.pick_modes(1, 2, 5)
#Measure the monitored modes
V_PM = sympy.simplify(V_PM.homodyne_detection(1, "x").homodyne_detection(1, "p"))
#%%

