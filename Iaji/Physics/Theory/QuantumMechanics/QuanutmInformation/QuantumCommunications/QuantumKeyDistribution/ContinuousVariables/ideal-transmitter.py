#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:16:20 2022

@author: jiedz

Quick calculation of the equivalence between prepare & measure (PM) 
and entanglement-based (EB) protocols for Gaussian CVQKD with ideal EB transmitter, 
modeling asymmetry in the quadratures.

For reference see my PhD thesis
"""
#%%
from quik.qip.nmodes_symbolic import Vacuum, CovarianceMatrix
import sympy
sympy.init_printing()
#%%
"""
1. Construct the covariance matrix of the mixed one-mode Gaussian state
   in the prepare & measure protocol, by using the solution found in the previous step.

2. Construct the covariance matrix of the pure two-mode 
   Gaussian state in the ideal entanglement-based protocol, by:
       - Creating a two-mode squeezed vacuum state
       - Applying single-mode squeezing of one mode
       - Rotating the squeezed mode
   
Establish the equivalence:
    1. The PM covariance matrix has to be equal to the submatrix of the EB
       mode propagating through towards the channel
       
    2. The quadrature variances of the PM mode conditioned on knowing the 
       corresponding symbol must be equal to the quadrature variance of the 
       mode measured by Alice in the EB protocol, conditioned on her measurement
"""

V_q_A, V_p_A, V_q_S, V_p_S, R_m = sympy.symbols("V_{q_A}, V_{p_A}, V_{q_S}, V_{p_S}, R_m", real=True, nonnegative=True)
V_q_A_given_S, V_p_A_given_S = sympy.symbols("V_{q_{A|S}}, V_{q_{A|S}}", real=True, nonnegative=True)
C_q_SA, C_p_SA = sympy.symbols("C_{q_{SA}}, C_{p_{SA}}", real=True)
r_1_EB, t_r_1_EB, r_2_EB, t_r_2_EB, nu_2_EB = sympy.symbols("r^{(EB)}_1, t^{(EB)}_{r_1}, r^{(EB)}_2, t^{(EB)}_{r_2}, \\nu^{(EB)}_2", real=True, positive=True)
theta_EB, t_theta_EB = sympy.symbols("\\theta^{(EB)}, t^{(EB)}_\\theta", real=True)
#1.
V_PM = CovarianceMatrix([[V_q_A, 0], [0, V_p_A]])
print("Expression of the PM covariance matrix")
print(V_PM)
print(sympy.latex(V_PM))
#1.
#Build a purification of V_PM as a two-mode squeezed vacuum state, further squeezed
#along the p quadrature, to simulate asymmetry
V_EB = Vacuum(3).epr_state(2, 3, mu=nu_2_EB)
V_EB = V_EB.squeeze(0, 0, -r_1_EB)
#V_EB = V_EB.squeeze(0, 0, r_2_EB)
#%%
#V_EB = V_EB.rotate(0, 0, theta_EB)
#%%
print("Expression of the EB covariance matrix before measurement on mode A'")
print(V_EB[2:, 2:])
print(sympy.latex(V_EB[2:, 2:]))
#Establish condition 1
eq1 = sympy.simplify(sympy.Eq(V_PM[0, 0], V_EB[4, 4]))
eq2 = sympy.simplify(sympy.Eq(V_PM[1, 1], V_EB[5, 5]))
#Perform the measurement on mode A' in the EB version
V_EB = CovarianceMatrix(V_EB.pick_modes(2, 1, 3)).bs(1, 2, R_m)
V_EB = V_EB.homodyne_detection(1, "x").homodyne_detection(1, "p")
V_EB = sympy.simplify(V_EB)
print("Expression of the EB covariance matrix after measurement on mode A'")
print(V_EB)
print(sympy.latex(V_EB))
#Establish the condition 2

eq3 = sympy.simplify(sympy.Eq(V_PM[0, 0] - C_q_SA**2/V_q_S, V_EB[0, 0]))
eq4 = sympy.simplify(sympy.Eq(V_PM[1, 1] - C_p_SA**2/V_p_S, V_EB[1, 1]))
#Solve the system of equations
system = [eq1, eq2, eq3, eq4]

solution = sympy.solve(system, (nu_2_EB, r_1_EB, R_m, V_q_A), dict=True)

