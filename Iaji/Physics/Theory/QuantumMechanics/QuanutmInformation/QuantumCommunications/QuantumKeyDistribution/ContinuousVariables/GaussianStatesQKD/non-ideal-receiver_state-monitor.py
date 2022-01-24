#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:16:20 2022

@author: jiedz

Quick calculation of the prepare & measure covariance matrix with
state monitoring at the receiver.

For reference see my PhD thesis
"""
#%%
from quik.qip.nmodes_symbolic import Vacuum, CovarianceMatrix
import sympy
sympy.init_printing()
#%%
"""
We start from the PM covariance matrix of mode B output from the channel, 
and propagate it towards the receiver homodyne detectors, including a state
monitor.
"""

V_q_B, V_p_B = sympy.symbols("V_{q_{B}}, V_{p_{B}}", real=True, nonnegative=True)
T_B, R_m, T_q, T_p = sympy.symbols("T_B, R_m, T_q, T_p", real=True, nonnegative=True)
zeta_q, zeta_p, nu_q, nu_p, xi_q, xi_p = sympy.symbols("\\zeta_q, \\zeta_p, \\nu_q, \\nu_p, \\xi_q, \\xi_p", real=True, nonnegative=True)
R_MB, R_B, T_MB_q, T_MB_p = sympy.symbols("R_{MB}, R_B, T_{MB_q}, T_{MB_p}", real=True, nonnegative=True)
nu_MB_q, nu_MB_p, xi_MB_q, xi_MB_p = sympy.symbols("\\nu^{(PM)}_{MB_q}, \\nu^{(PM)}_{MB_p}, \\xi_{MB_q}, \\xi_{MB_p}", real=True, positive=True)
#1.
V_PM = Vacuum(12)
V_PM[22:, 22:] = CovarianceMatrix([[V_q_B, 0], [0, V_p_B]])
#Propagation loss on mode B
V_PM = V_PM.opticalefficiency(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, T_B)
#Split at the state monitor tap beam splitter
V_PM = V_PM.pick_modes(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 11)
V_PM = V_PM.bs(11, 12, R_B)
#Split at the state monitor beam splitter
V_PM = V_PM.bs(10, 11, R_MB)
#Define the two-mode squeeze vacuum states that models the additive Gaussian noise
#at the state monitor detectors
V_PM = V_PM.epr_state(6, 7, nu_MB_q)
V_PM = V_PM.epr_state(8, 9, nu_MB_p)
#Interfere each mode with the corresponding mode of the monitored signal,
#onto the beam splitter that model the corresponding detector loss
V_PM = V_PM.bs(7, 11, 1-T_MB_q)
V_PM = V_PM.bs(9, 10, 1-T_MB_p)
V_PM = V_PM.subs([((1-T_MB_q)*nu_MB_q, xi_MB_q), ((1-T_MB_p)*nu_MB_p, xi_MB_p)])
V_PM = V_PM.pick_modes(1, 2, 3, 4, 5, 7, 9, 12) 
#Draw the correspondence between the measured quadrature variances at the 
#state monitor and the original quadrature variances
V_q_MB, V_p_MB = sympy.symbols("V_{q_{MB}}, V_{p_{MB}}", real=True, nonnegative=True)
eq1 = sympy.Eq(V_PM[10, 10], V_q_MB)
eq2 = sympy.Eq(V_PM[13, 13], V_p_MB)
solution_V_B_from_state_monitor  = sympy.simplify(sympy.solve((eq1, eq2), (V_q_B, V_p_B), dict=True)[0])
#Measure the monitored modes
V_PM = sympy.simplify(V_PM.homodyne_detection(6, "x").homodyne_detection(6, "p"))
#Define the two-mode squeeze vacuum states that models the additive Gaussian noise
#at the measurement system
V_PM = CovarianceMatrix(V_PM)
#Split at the measurement beam splitter
V_PM = V_PM.pick_modes(1, 2, 3, 4, 6, 5)
V_PM = V_PM.bs(5, 6, R_m)
V_PM = V_PM.epr_state(1, 2, nu_q)
V_PM = V_PM.epr_state(3, 4, nu_p)
#Interfere each mode with the corresponding mode of the measured signal,
#onto the beam splitter that model the corresponding detector loss
V_PM = V_PM.bs(2, 6, 1-T_q)
V_PM = V_PM.bs(4, 5, 1-T_p)
V_PM = V_PM.subs([((1-T_q)*nu_q, xi_q), ((1-T_p)*nu_p, xi_p)])
V_PM = V_PM.pick_modes(1, 2, 3, 4, 6, 5)
V_PM = V_PM.pick_modes(2, 4)
#Draw the correspondence between the measured quadrature variances at the 
#measurement HDs and the original quadrature variances
V_q_B1, V_p_B2 = sympy.symbols("V_{q_{B1}}, V_{p_{B2}}", real=True, nonnegative=True)
eq1 = sympy.Eq(V_PM[0, 0], V_q_B1)
eq2 = sympy.Eq(V_PM[3, 3], V_p_B2)
solution_V_B_from_measurement  = sympy.simplify(sympy.solve((eq1, eq2), (V_q_B, V_p_B), dict=True)[0])