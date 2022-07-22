#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 23:52:52 2022

@author: jiedz
This script tests the QuantumStateFock module
"""
# In[]
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.QuantumStateFock import *
import sympy, numpy
sympy.init_printing()
# In[]
state = QuantumStateFockNumeric(truncated_dimension=10, name="original").NumberState(9)
p, q = [numpy.linspace(-5, 5, 200) for j in range(2)]
state.PlotWignerFunction(q, p, plot_name="original")
state.PlotNumberDistribution(plot_name="original")
state.PlotDensityOperator(plot_name="original")
#Expand
state = state.Resize(15).NumberState(14)
state.name = "resized"
state.PlotWignerFunction(q, p, plot_name="resized")
state.PlotNumberDistribution(plot_name="resized")
state.PlotDensityOperator(plot_name="resized")