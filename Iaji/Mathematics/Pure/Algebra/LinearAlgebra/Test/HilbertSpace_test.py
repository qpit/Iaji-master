#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 15:44:16 2022

@author: jiedz

This script tests the HilberSpace module
"""
#%%
import sympy
sympy.init_printing()
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.HilbertSpace import HilbertSpace
#%%
H1 = HilbertSpace(dimension=2, scalars=sympy.Complexes, name="H_{1}")
print(H1)
H2 = HilbertSpace(dimension=2, name="H_{2}")
print(H2)
#Compose the two Hilbert spaces
H = H1 * H2
print(H)