#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 11:40:30 2022

@author: jiedz

Quick calculation of the equivalence between prepare & measure (PM) 
and entanglement-based (EB) protocols for Gaussian CVQKD with ideal and
non-ideal EB transmitter, 
using a four-mode Bloch-Messiah reduction. 

For reference, see my PhD thesis.
"""
#%%
from quik.qip.nmodes_symbolic import Vacuum, CovarianceMatrix
import sympy
sympy.init_printing()
#%%
"""
Establish the equivalence between PM
and the EB covariance matrices by requiring that:
    
    1. the quadrature variances of mode A are the same in both protocols
    
    2. the variance of each quadrature of mode A conditioned on knowning the
      corresponding symbol in the PM protocol, is the same as the variance of
      each quadrature of mode A conditioned on measuring the same quadrature
      on the second mode in the EB protocol
"""

V_q_Aprime, V_p_Aprime, C_q_AprimeA, C_p_AprimeA, V_q_A, V_p_A,\
V_q_S, V_p_S, C_q_SA, C_p_SA, R_m = sympy.symbols(names="V_{q_{A'}}, V_{p_{A'}}, C_{q_{A'A}}, C_{p_{A'A}}, V_{q_A}, V_{p_A}, V_{q_S}, V_{p_S}, C_{q_{SA}}, C_{p_{SA}}, R_m", \
                                             real=True, nonnegative=True)
V_AprimeA = Vacuum(3)
V_AprimeA[2:, 2:] = sympy.Matrix([[V_q_Aprime, 0, C_q_AprimeA, 0], [0, V_p_Aprime, 0, C_p_AprimeA], [C_q_AprimeA, 0, V_q_A, 0], [0, C_p_AprimeA, 0, V_p_A]])
V_AprimeA_before_splitting = V_AprimeA[2:, 2:]
#Apply the measurement
V_AprimeA = CovarianceMatrix(sympy.simplify(V_AprimeA.bs(1, 2, R=R_m)))
V_A_given_Aprime = V_AprimeA.homodyne_detection(1, "p").homodyne_detection(1, "x")
#Apply condition 2.
eq_q = sympy.Eq(V_A_given_Aprime[0, 0], V_q_A - C_q_SA**2/V_q_S) 
eq_p = sympy.Eq(V_A_given_Aprime[1, 1], V_p_A - C_p_SA**2/V_p_S) 

system = (eq_q, eq_p)
solution = sympy.solve(system, (C_q_AprimeA, C_p_AprimeA), dict=True)[1]
#Substitute the solution
#V_AprimeA = V_AprimeA.subs([(C_q_AprimeA, solution[C_q_AprimeA]), (C_p_AprimeA, solution[C_p_AprimeA])])
#V_A_given_Aprime = V_A_given_Aprime.subs([(C_q_AprimeA, solution[C_q_AprimeA]), (C_p_AprimeA, solution[C_p_AprimeA])])
#V_AprimeA_before_splitting = V_AprimeA_before_splitting.subs([(C_q_AprimeA, solution[C_q_AprimeA]), (C_p_AprimeA, solution[C_p_AprimeA])])
#Since no condition is imposed on V_q_Aprime or V_p_Aprime, set them equal to
#V_q_A and V_p_A
#V_AprimeA = V_AprimeA.subs([(V_q_Aprime, V_q_A), (V_p_Aprime, V_p_A)])
#V_A_given_Aprime = V_A_given_Aprime.subs([(V_q_Aprime, V_q_A), (V_p_Aprime, V_p_A)])
#V_AprimeA_before_splitting = V_AprimeA_before_splitting.subs([(V_q_Aprime, V_q_A), (V_p_Aprime, V_p_A)])
#%%
"""
Ideal transmitter
---------------------------------------------------
    1. Construct the covariance matrix of the mixed one-mode Gaussian state
       in the prepare & measure protocol, by using the solution found in the previous step.

    2. Construct the covariance matrix of the pure two-mode 
       Gaussian state in the ideal entanglement-based protocol, by:
           - Creating a two-mode squeezed vacuum state
           - Applying single-mode squeezing of one mode
    
Compare the two matrices to find the relation between the prepare & measure
and the entanglement-based parameters
"""
"""
V_s, zeta_1_PM, mu, zeta_1_EB, r_1_EB, zeta_2_EB, nu, n_q, n_p = sympy.symbols("V_s, \zeta^{(PM)}_1, mu, \zeta^{(EB)}_1, r^{(EB)}_1, \zeta^{(EB)}_2, nu, \\bar{n}_q, \\bar{n}_p")
#1.
V_AprimeA_ideal_PM = V_AprimeA_before_splitting.subs([(V_q_A, 1/V_s+2*n_q), (V_p_A, V_s+2*n_p), (C_q_SA, 2*n_q),\
                                  (C_p_SA, 2*n_p), (V_q_S, 2*n_q), (V_p_S, 2*n_p)])
#2.
V_AprimeA_ideal_EB = Vacuum(2)
V_AprimeA_ideal_EB = CovarianceMatrix(V_AprimeA_ideal_EB.epr_state(1, 2, nu))
V_AprimeA_ideal_EB = V_AprimeA_ideal_EB.squeeze(0, -r_1_EB)
"""
#%%
"""
Non-ideal transmitter
--------------------------------------------------
    1 Construct the four-mode decomposition of the EB two-mode Gaussian state
      according to the Bloch-Messiah decomposition

    2.equate the covariance matrix
      to the mixed two-mode EB covariance matrix defined in the first step,
      after partial tracing of the two extra modes.
"""
T_1, T_2, nu_C, nu_D = sympy.symbols(names="T_1, T_2, nu_C, nu_D", real=True, nonnegative=True)
r_1, r_2 = sympy.symbols(names="r_1, r_2", real=True)
V_AprimeACD = Vacuum(4)
V_AprimeACD =  V_AprimeACD.epr_state(1, 2, mu=nu_C)
V_AprimeACD =  CovarianceMatrix(V_AprimeACD.epr_state(3, 4, mu=nu_D))
V_AprimeACD = V_AprimeACD.pick_modes(3, 1, 2, 4)
V_AprimeACD = CovarianceMatrix(sympy.simplify(V_AprimeACD.bs(1, 2, R = 1-T_1)))
V_AprimeACD = V_AprimeACD.squeeze(r_1, r_2, 0, 0)
V_AprimeACD = V_AprimeACD.pick_modes(2, 1, 3, 4)
V_AprimeACD = CovarianceMatrix(sympy.simplify(V_AprimeACD.bs(1, 2, R = 1-T_2)))
#Partial trace modes C and D
V_AprimeACD_partial_trace_CD = sympy.simplify(V_AprimeACD.pick_modes(1, 2))
#Substitute particular values for the transmittances
#V_AprimeACD_partial_trace_CD = V_AprimeACD_partial_trace_CD.subs([(T_1, 1)])
#V_AprimeACD_partial_trace_CD = sympy.simplify(V_AprimeACD_partial_trace_CD.subs([(T_2, 0.5)]))
#Set up system of equations
eq1 = sympy.Eq(V_AprimeACD_partial_trace_CD[0, 0], V_q_Aprime)
eq2 = sympy.Eq(V_AprimeACD_partial_trace_CD[1, 1], V_p_Aprime)
eq3 = sympy.Eq(V_AprimeACD_partial_trace_CD[2, 2], V_q_A)
eq4 = sympy.Eq(V_AprimeACD_partial_trace_CD[3, 3], V_p_A)
eq5 = sympy.Eq(V_AprimeACD_partial_trace_CD[0, 2], C_q_AprimeA)
eq6 = sympy.Eq(V_AprimeACD_partial_trace_CD[1, 3], C_p_AprimeA)
system = (eq1, eq2, eq3, eq4, eq5, eq6)
#Solve the system
solution_Bloch_Messiah = sympy.solve(system, (nu_C, nu_D, r_1, r_2, T_1, T_2), dict=True)
"""
You only need to keep the real solutions
"""
#solution_Bloch_Messiah = solution_Bloch_Messiah[11]

        