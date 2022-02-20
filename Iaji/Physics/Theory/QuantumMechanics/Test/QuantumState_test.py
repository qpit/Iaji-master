#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 20:55:23 2022

@author: jiedz
This script tests the module QuantumState.py
"""

from Iaji.Physics.Theory.QuantumMechanics.QuantumState import QuantumState
import sympy, numpy
sympy.init_printing()

state = QuantumState()
print(state)
