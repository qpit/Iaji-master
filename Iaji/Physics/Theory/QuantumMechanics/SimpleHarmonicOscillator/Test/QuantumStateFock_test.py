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
# In[symbolic state]
state_sym = QuantumStateFock(truncated_dimension=5).numeric.NumberState(3)
print(state_sym)
#Plot the Wigner function of the quantum state
p, q = [numpy.linspace(-5, 5, 200) for j in range(2)]
state_sym.PlotWignerFunction(q, p, alpha=0.5, plot_name="symbolic - Wigner function")
#Plot the density operator
state_sym.PlotDensityOperator(plot_name="symbolic - density operator")
# In[numeric state]
state_num = QuantumStateFock(truncated_dimension=5).numeric.NumberState(3)
print(state_num)
#Plot the Wigner function of the quantum state
p, q = [numpy.linspace(-5, 5, 200) for j in range(2)]
state_num.PlotWignerFunction(q, p, alpha=0.5, plot_name="numeric - Wigner function")
#Plot the density operator
state_num.PlotDensityOperator(plot_name="numeric - density operator")