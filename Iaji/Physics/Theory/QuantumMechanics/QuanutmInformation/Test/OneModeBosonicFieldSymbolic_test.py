#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 23:52:52 2022

@author: jiedz
This script tests the OneModeBosonicFieldSymbolic module
"""
# In[imports]
from Iaji.Physics.Theory.QuantumMechanics.OneModeBosonicField.OneModeBosonicField import OneModeBosonicField
from Iaji.Mathematics.Parameter import ParameterSymbolic
import sympy, numpy
sympy.init_printing()
# In[more imports]
from matplotlib import pyplot
from Iaji.Physics.Theory.QuantumMechanics.OneModeBosonicField.Utilities import *
# In[]
p, q = [numpy.linspace(-5, 5, 200) for j in range(2)]
# In[basic harmonic oscillator]
mode = OneModeBosonicField(truncated_dimension=5).symbolic.NumberState(0)
#Plot the Wigner function of the quantum state
mode.state.PlotWignerFunction(q, p, plot_name="basic - Wigner function")
#Plot the density operator
mode.state.PlotDensityOperator(plot_name="basic - density operator")
# In[displaced]
#alpha = ParameterSymbolic(name="\\alpha")
alpha_0 = 0.2+0.1j
mode_displaced = mode.Vacuum().Displace(alpha_0)
mode_displaced.state.PlotWignerFunction(q, p, plot_name="displaced - Wigner function")
#Plot the density operator
mode_displaced.state.PlotDensityOperator(plot_name="displaced - density operator")
# In[squeezed]
s = ParameterSymbolic(name="s")
zeta_0 = numpy.log(2)/2
mode_squeezed = mode\
                  .Vacuum().Squeeze(zeta_0)
#Plot the Wigner function of the quantum state
mode_squeezed.state.PlotWignerFunction(q, p, plot_name="squeezed - Wigner function")
#Plot the density operator
mode_squeezed.state.PlotDensityOperator(plot_name="squeezed - density operator")
# In[rotated]
t = ParameterSymbolic(name="t")
theta_0 = numpy.pi/6
mode_rotated = mode_squeezed\
                .Rotate(theta_0)
#Plot the Wigner function of the quantum state
mode_rotated.state.PlotWignerFunction(q, p,  plot_name="rotate - Wigner function")
#Plot the density operator
mode_rotated.state.PlotDensityOperator(plot_name="rotate - density operator")