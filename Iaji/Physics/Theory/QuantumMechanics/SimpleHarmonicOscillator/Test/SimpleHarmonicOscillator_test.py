#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 23:52:52 2022

@author: jiedz
This script tests the SimpleHarmonicOscillator module
"""
# In[]
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.SimpleHarmonicOscillator import SimpleHarmonicOscillatorSymbolic
import sympy, numpy
sympy.init_printing()
# In[]
from matplotlib import pyplot
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.Utilities import *
# In[]
system = SimpleHarmonicOscillatorSymbolic(truncated_dimension=4).NumberState(1)
#Plot the Wigner function of the quantum state
p, q = [numpy.linspace(-5, 5, 200) for j in range(2)]
Q, P = numpy.meshgrid(q, p)
W = system.state.wigner_function.expression_lambda(Q, P)
plotWignerFunction(Q, P, W, alpha=.7)
# In[]
#Displacement
system_displaced = system.Displace(0.2-1j*0.2)
#Plot the Wigner function of the quantum state
p, q = [numpy.linspace(-5, 5, 200) for j in range(2)]
Q, P = numpy.meshgrid(q, p)
W = system_displaced.state.wigner_function.expression_lambda(Q, P)
plotWignerFunction(Q, P, W, alpha=.7, plot_name="displaced")
# In[]
#Squeezing
zeta = numpy.log(2)/2*numpy.exp(1j*numpy.pi/4)
system_squeezed = SimpleHarmonicOscillatorSymbolic(truncated_dimension=8).NumberState(1)\
                  .Squeeze(zeta)
#Plot the Wigner function of the quantum state
p, q = [numpy.linspace(-5, 5, 200) for j in range(2)]
Q, P = numpy.meshgrid(q, p)
W = system_squeezed.state.wigner_function.expression_lambda(Q, P)
plotWignerFunction(Q, P, W, alpha=.7, plot_name="squeezed")
# In[]
#Rotation
zeta = -numpy.log(2)/2
theta = numpy.pi/4
system_rotated = SimpleHarmonicOscillatorSymbolic(truncated_dimension=8)\
    .Squeeze(zeta).Rotate(theta)
#Plot the Wigner function of the quantum state
p, q = [numpy.linspace(-5, 5, 200) for j in range(2)]
Q, P = numpy.meshgrid(q, p)
W = system_rotated.state.wigner_function.expression_lambda(Q, P)
plotWignerFunction(Q, P, W, alpha=.7, plot_name="squeezed and rotated")