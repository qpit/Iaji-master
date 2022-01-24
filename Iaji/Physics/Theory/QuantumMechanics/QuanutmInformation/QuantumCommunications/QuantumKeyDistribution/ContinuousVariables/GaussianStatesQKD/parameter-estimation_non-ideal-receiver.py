#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:18:30 2022

@author: jiedz

Quick estimation of the transmitter and receiver parameters
"""
#%%
from quik.qip.nmodes_symbolic import Vacuum, CovarianceMatrix
import numpy, sympy
sympy.init_printing()
#%%
V_q_B1, V_p_B2 = sympy.symbols("V_{q_{B1}}, V_{p_{B2}}", real=True, nonnegative=True)
C_q_SB1, C_p_SB2 = sympy.symbols("C_{q_{SB1}}, C_{p_{SB2}}", real=True, nonnegative=False)
R_B, T_B, T_q, T_p = sympy.symbols("R_B, T_B, T_q, T_p", real=True, nonnegative=True)
xi_q, xi_p =  sympy.symbols("xi_q, xi_p", real=True, positive=True)
#Build system of equations
