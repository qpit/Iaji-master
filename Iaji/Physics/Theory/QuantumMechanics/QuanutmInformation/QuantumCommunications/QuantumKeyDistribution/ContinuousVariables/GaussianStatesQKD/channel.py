#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 15:20:55 2022

@author: jiedz

Quick symbolic calculation general one-mode Gaussian channels.

For a reference, see 

https://arxiv.org/pdf/1110.3234.pdf or 
https://arxiv.org/abs/quant-ph/0607051v2
"""
#%%
from quik.qip.nmodes_symbolic import Vacuum, CovarianceMatrix
import sympy
sympy.init_printing()
#%%
def G(channel_class, tau_c=None, n_c=None):
    """
    Return the canonical form of the Gaussian channel of class 'channel_class'
    """
    if tau_c is None:
        tau_c = sympy.symbols("\\tau_\mathcal{C}", real=True) #generalized transmissivity
    if n_c is None:
        n_c = sympy.symbols("\\bar{n}_\mathcal{C}", real=True, nonnegative=True)
    T_c = None
    N_c = None
    Z = sympy.eye(2)
    Z[1, 1] = -1
    I = sympy.eye(2)
    if channel_class == "A1":
        T_c = sympy.zeros((2, 2))
        N_c = (2*n_c+1)*I
    elif channel_class == "A2":        
        T_c = (I+Z)/2
        N_c = (2*n_c+1)*I
    elif channel_class == "B1":
        T_c = I
        N_c = (I-Z)/2
    elif channel_class == "B2":
        T_c = I
        N_c = n_c*I
    elif channel_class == "C(loss)":
        T_c = sympy.sqrt(tau_c)*I
        N_c = (1-tau_c)*(2*n_c+1)*I
    elif channel_class == "C(amp)":
        T_c = sympy.sqrt(tau_c)*I
        N_c = (tau_c-1)*(2*n_c+1)*I
    elif channel_class == "D":
        T_c = sympy.sqrt(-tau_c)*I
        N_c = (1-tau_c)*(2*n_c+1)*I   
    else:
        raise ValueError("Not a valid channel class")
    return T_c, N_c

def apply(V, T_c, N_c):
    return T_c * V * T_c.T + N_c
#%%
"""
Derive the expression of a one-mode Gaussian channel corresponding to the 
asymmetric thermal-lossy channel, and check if such a channel exists.
"""
V_q_Aprime, V_p_Aprime, C_q_AprimeA, C_p_AprimeA, V_q_A, V_p_A = sympy.symbols(names="V_{q_{A'}}, V_{p_{A'}}, C_{q_{A'A}}, C_{p_{A'A}}, V_{q_A}, V_{p_A}", \
                                             real=True, nonnegative=True)
eta_q, eta_p = sympy.symbols("\\eta_q, \\eta_p", real=True, positive=True)
n_q, n_p = sympy.symbols("\\bar{n}_q, \\bar{n}_p", real=True, positive = True)
V_AprimeA = Vacuum(2)
V_AprimeA[:, :] = sympy.Matrix([[V_q_Aprime, 0, C_q_AprimeA, 0], [0, V_p_Aprime, 0, C_p_AprimeA], [C_q_AprimeA, 0, V_q_A, 0], [0, C_p_AprimeA, 0, V_p_A]])
V_AprimeB = CovarianceMatrix(V_AprimeA)
V_AprimeB[2, 2] = V_AprimeB[2, 2]*eta_q+(1-eta_q)*(2*n_q+1)
V_AprimeB[3, 3] = V_AprimeB[3, 3]*eta_p+(1-eta_p)*(2*n_p+1)
V_AprimeB[0, 2] = V_AprimeB[2, 0] = V_AprimeB[0, 2]*sympy.sqrt(eta_q)
V_AprimeB[1, 3] = V_AprimeB[3, 1] = V_AprimeB[1, 3]*sympy.sqrt(eta_p)

t_c_q, t_c_p = sympy.symbols("t_{\mathcal{C}_q}, t_{\mathcal{C}_p}", real=True, positive=True)
n_c_q, n_c_p = sympy.symbols("\\bar{n}_{\mathcal{C}_q}, \\bar{n}_{\mathcal{C}_p}", real=True, positive=True)
T_c = sympy.diag(1, 1, t_c_q, t_c_p)
N_c = sympy.diag(0, 0, n_c_q, n_c_p)
Omega = sympy.zeros(4, 4)
Omega[0, 1] = Omega[2, 3] = 1
Omega[1, 0] = Omega[3, 2] = -1
eq = sympy.Eq(V_AprimeB, T_c*V_AprimeA*T_c.T + N_c)
solution = sympy.solve(eq, (t_c_q, t_c_p, n_c_q, n_c_p), dict=True)[0]

physicality_matrix = V_AprimeB + sympy.I*Omega
#physicality_eigenvalues = physicality_matrix.eigenvals()
#%%
"""
Perform channel parameter estimation
"""
V_q_B, V_p_B, C_q_AprimeB, C_p_AprimeB = sympy.symbols(names="V_{q_B}, V_{p_B}, C_{q_{A'B}}, C_{p_{A'B}}", \
                                             real=True, nonnegative=True)
eq1 = sympy.Eq(V_AprimeB[2, 2], V_q_B)
eq2 = sympy.Eq(V_AprimeB[3, 3], V_p_B)
eq3 = sympy.Eq(V_AprimeB[0, 2], C_q_AprimeB)
eq4 = sympy.Eq(V_AprimeB[1, 3], C_p_AprimeB)
system = (eq1, eq2, eq3, eq4)
solution_channel_parameter_estimation = sympy.solve(system, (eta_q, eta_p, n_q, n_p), dict=True)