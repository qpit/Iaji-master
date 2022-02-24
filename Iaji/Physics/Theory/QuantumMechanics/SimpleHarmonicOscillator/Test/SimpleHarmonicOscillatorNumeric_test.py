#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 23:52:52 2022

@author: jiedz
This script tests the SimpleHarmonicOscillator module
"""
# In[imports]
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.SimpleHarmonicOscillator import SimpleHarmonicOscillator
from Iaji.Mathematics.Parameter import ParameterNumeric
import sympy, numpy
sympy.init_printing()
# In[more imports]
from matplotlib import pyplot
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.Utilities import *
# In[]
p, q = [numpy.linspace(-5, 5, 200) for j in range(2)]
# In[basic harmonic oscillator]
system = SimpleHarmonicOscillator(truncated_dimension=10).numeric.NumberState(0)
#Plot the Wigner function of the quantum state
system.state.PlotWignerFunction(q, p, plot_name="basic - Wigner function")
#Plot the density operator
system.state.PlotDensityOperator(plot_name="basic - density operator")
# In[displaced]
#alpha = ParameterNumeric(name="\\alpha")
alpha_0 = 1+1j
system_displaced = system.Vacuum().Displace(alpha_0)
system_displaced.state.PlotWignerFunction(q, p, plot_name="displaced - Wigner function")
#Plot the density operator
system_displaced.state.PlotDensityOperator(plot_name="displaced - density operator")
# In[squeezed]
s = ParameterNumeric(name="s")
zeta_0 = numpy.log(2)/2
system_squeezed = system\
                  .Vacuum().Squeeze(zeta_0)
#Plot the Wigner function of the quantum state
system_squeezed.state.PlotWignerFunction(q, p, plot_name="squeezed - Wigner function")
#Plot the density operator
system_squeezed.state.PlotDensityOperator(plot_name="squeezed - density operator")
# In[rotated]
t = ParameterNumeric(name="t")
theta_0 = numpy.pi/6
system_rotated = system_squeezed\
                .Rotate(theta_0)
#Plot the Wigner function of the quantum state
system_rotated.state.PlotWignerFunction(q, p,  plot_name="rotate - Wigner function")
#Plot the density operator
system_rotated.state.PlotDensityOperator(plot_name="rotate - density operator")