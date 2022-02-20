#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 23:52:52 2022

@author: jiedz
This script tests the QuantumStateFock module
"""
# In[]
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.QuantumStateFock import QuantumStateFock
import sympy, numpy
sympy.init_printing()
# In[]
from matplotlib import pyplot
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.Utilities import *
# In[]
state = QuantumStateFock(truncated_dimension=5).symbolic.NumberState(1)
print(state)
#Plot the Wigner function of the quantum state
p, q = [numpy.linspace(-5, 5, 200) for j in range(2)]
Q, P = numpy.meshgrid(q, p)
W = state.wigner_function.expression_lambda(Q, P)
plotWignerFunction(Q, P, W, alpha=.7)