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
C_q_SA, C_p_SA = sympy.symbols("C_{q_{SA}}, C_{p_{SA}}", real=True)
V_q_S, V_p_S = sympy.symbols("V_{q_{S}}, V_{p_{S}}", real=True, nonnegative=True)
R_MA, R_A, T_MA_q, T_MA_p = sympy.symbols("R_{MA}, R_A, T_{MA_q}, T_{MA_p}", real=True, nonnegative=True)
nu_MA_q, nu_MA_p, xi_MA_q, xi_MA_p = sympy.symbols("\\nu^{(PM)}_{MA_q}, \\nu^{(PM)}_{MA_p}, \\xi_{MA_q}, \\xi_{MA_p}", real=True, positive=True)
#1.
V_PM = Vacuum(7)
V_PM[12:, 12:] = CovarianceMatrix([[V_q_0_A, 0], [0, V_p_0_A]])
#Split at the state monitor tap beam splitter
V_PM = V_PM.pick_modes(1, 2, 3, 4, 5, 7, 6)
V_PM = V_PM.bs(6, 7, R_A)
#Split at the state monitor beam splitter
V_PM = V_PM.bs(5, 6, R_MA)
#Define the two-mode squeeze vacuum states that models the additive Gaussian noise
#at the state monitor detectors
V_PM = V_PM.epr_state(1, 2, nu_MA_q)
V_PM = V_PM.epr_state(3, 4, nu_MA_p)
#Interfere each mode with the corresponding mode of the monitored signal,
#onto the beam splitter that model the corresponding detector loss
V_PM = V_PM.bs(2, 6, 1-T_MA_q)
V_PM = V_PM.bs(4, 5, 1-T_MA_p)
V_PM = V_PM.subs([((1-T_MA_q)*nu_MA_q, xi_MA_q), ((1-T_MA_p)*nu_MA_p, xi_MA_p)])
V_PM = V_PM.pick_modes(2, 4, 5, 6, 7)
V_PM = V_PM.pick_modes(1, 2, 5)
#Draw the correspondence between the measured quadrature variances at the 
#state monitor and the original quadrature variance
V_q_MA, V_p_MA = sympy.symbols("V_{q_{MA}}, V_{p_{MA}}", real=True, nonnegative=True)
eq1 = sympy.Eq(V_PM[0, 0], V_q_MA)
eq2 = sympy.Eq(V_PM[3, 3], V_p_MA)
solution_V_0  = sympy.solve((eq1, eq2), (V_q_0_A, V_p_0_A), dict=True)[0]
#Measure the monitored modes
V_PM = sympy.simplify(V_PM.homodyne_detection(1, "x").homodyne_detection(1, "p"))
#%%
"""
Construct the EB covariance matrix
"""
r_1_EB_0, nu_2_EB_0 = sympy.symbols("r^{(EB)}_{1_0}, \\nu^{(EB)}_{2_0}", real=True, positive=True)
R_m = sympy.symbols("R_m", real=True, nonnegative=True)
#theta_EB, t_theta_EB = sympy.symbols("\\theta^{(EB)}, t^{(EB)}_\\theta", real=True)
#1.
#Build a purification of V_PM as a two-mode squeezed vacuum state, further squeezed
#along the p quadrature, to simulate asymmetry
V_EB = Vacuum(9).epr_state(8, 9, mu=nu_2_EB_0)
V_EB = V_EB.squeeze(0, 0, 0, 0, 0, 0, 0, 0, -r_1_EB_0)
#V_EB = V_EB.squeeze(0, 0, r_2_EB)
#V_EB = V_EB.rotate(0, 0, theta_EB)
V_EB = V_EB.pick_modes(1, 2, 3, 4, 5, 6, 8, 7, 9)
#Tap off mode A for state monitoring
V_EB = V_EB.pick_modes(1, 2, 3, 4, 5, 6, 9, 7, 8)
V_EB = V_EB.pick_modes(1, 2, 3, 4, 6, 5, 7, 8, 9)
V_EB = V_EB.bs(6, 7, R_A)
V_EB = V_EB.pick_modes(1, 2, 3, 4, 5, 7, 6, 8, 9)
V_EB = V_EB.bs(5, 6, R_MA)
#Split at the state monitor beam splitter
#Define the two-mode squeeze vacuum states that models the additive Gaussian noise
#at the state monitor detectors
V_EB = V_EB.epr_state(1, 2, nu_MA_q)
V_EB = V_EB.epr_state(3, 4, nu_MA_p)
#Interfere each mode with the corresponding mode of the monitored signal,
#onto the beam splitter that model the corresponding detector loss
V_EB = V_EB.bs(2, 6, 1-T_MA_q)
V_EB = V_EB.bs(4, 5, 1-T_MA_p)
V_EB = CovarianceMatrix(V_EB.subs([((1-T_MA_q)*nu_MA_q, xi_MA_q), ((1-T_MA_p)*nu_MA_p, xi_MA_p)]))
V_EB = V_EB.pick_modes(2, 4, 7, 8, 9)
#Measure at the state monitor
V_EB = V_EB.homodyne_detection(1, "x").homodyne_detection(1, "p")
V_EB_before_measurement_Aprime = V_EB[:-2, :-2]
#Split mode A' for measurement
V_EB = V_EB.bs(2, 3, R_m)
#Measure A'
V_EB_after_measurement_Aprime = sympy.simplify(V_EB.homodyne_detection(2, "x").homodyne_detection(2, "p"))
#%%
"""
Once it is verified that the PM state can be still purified by a two-mode
squeezed vacuum state further squeezed in one mode, one must impose the 
relation between V_q_A and V_p_A, C_q_SA, C_p_SA, V_q_S and V_p_S as found
in the thesis, or equivalently, in "ideal-transmitter.py"
"""
eq = sympy.Eq((V_PM[0, 0] - C_q_SA**2/V_q_S)*(V_PM[1, 1] - C_p_SA**2/V_p_S), 1)
#solution = sympy.solve(eq, R_M_A, dict=True)