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
from Iaji.Utilities.statistics import *
import sympy, numpy
sympy.init_printing()
# In[more imports]
from matplotlib import pyplot
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.Utilities import *
# In[]
p, q = [numpy.linspace(-5, 5, 200) for j in range(2)]
# In[basic harmonic oscillator]
system = SimpleHarmonicOscillator(truncated_dimension=20).numeric.NumberState(1)
#Plot the Wigner function of the quantum state
system.state.PlotWignerFunction(q, p, plot_name="basic - Wigner function")
#Plot the density operator
system.state.PlotDensityOperator(plot_name="basic - density operator")
# In[displaced]
#alpha = ParameterNumeric(name="\\alpha")
alpha_0 = 1.5+1j
system_displaced = system.Vacuum().Displace(alpha_0)
system_displaced.state.PlotWignerFunction(q, p, plot_name="displaced - Wigner function")
#Plot the density operator
system_displaced.state.PlotDensityOperator(plot_name="displaced - density operator")
# In[squeezed]
s = ParameterNumeric(name="s")
zeta_0 = numpy.log(4)/2
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
# In[projective measurement of the number operator]
samples = system.ProjectiveMeasurement(measurable="n", ntimes=int(1e4))[0]

figure_measure_n = pyplot.figure(figsize=(11, 8))
figure_measure_n.subplots_adjust(wspace=0.5)
axis_samples = figure_measure_n.add_subplot(1, 2, 1)
axis_samples.plot(samples)
axis_samples.set_xlabel("sample number", font=axis_font)
axis_samples.set_ylabel("quantum number", font=axis_font)

axis_pdf = figure_measure_n.add_subplot(1, 2, 2)
axis_pdf.set_xlabel("quantum number", font=axis_font)
axis_pdf.set_ylabel("probability", font=axis_font)
#Plot
x_range = numpy.array([0, system.hilbert_space.dimension-1])
PDF_values, PDF = PDF_histogram(samples, x_range=x_range, n_bins=system.hilbert_space.dimension)
axis_pdf.bar(PDF_values, PDF, color="tab:blue", alpha=0.5)
# In[projective measurement of a generalized quadrature]
figure_measure_quadrature = pyplot.figure(figsize=(11, 8))

samples = system_displaced.ProjectiveMeasurement(measurable="x", ntimes=int(1e6), theta=numpy.pi/2)[0]

figure_measure_quadrature.subplots_adjust(wspace=0.5)
axis_samples = figure_measure_quadrature.add_subplot(1, 2, 1)
axis_samples.plot(samples)
axis_samples.set_xlabel("sample number", font=axis_font)
axis_samples.set_ylabel("quadrature value (SNU)", font=axis_font)

axis_pdf = figure_measure_quadrature.add_subplot(1, 2, 2)
axis_pdf.set_xlabel("quadrature value (SNU)", font=axis_font)
axis_pdf.set_ylabel("probability density", font=axis_font)
#Plot
x_range = numpy.array([numpy.min(samples), numpy.max(samples)])
PDF_values, PDF = PDF_histogram(samples, x_range=x_range, n_bins=int(len(samples)*0.01))
axis_pdf.scatter(PDF_values, PDF, color="tab:blue", alpha=0.5)
