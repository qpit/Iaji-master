#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 23:52:52 2022

@author: jiedz
This script tests the OneModeBosonicFieldNumeric module
"""
# In[imports]
from Iaji.Physics.Theory.QuantumMechanics.QuanutmInformation.OneModeBosonicField \
    import OneModeBosonicField
from Iaji.Mathematics.Parameter import ParameterNumeric
from Iaji.Utilities.statistics import *
import sympy, numpy
sympy.init_printing()
# In[more imports]
from matplotlib import pyplot, font_manager
# In[Plot settings]
pyplot.close('all')
#General plot settings
default_marker = ""
default_figure_size = (11, 8)
default_fontsize = 50
title_fontsize = default_fontsize
title_font = font_manager.FontProperties(family='Times New Roman',
                                   weight='bold',
                                   style='normal', size=title_fontsize)
axis_font = font_manager.FontProperties(family='Times New Roman',
                                   weight='normal',
                                   style='normal', size=title_fontsize*0.7)
legend_font = font_manager.FontProperties(family='Times New Roman',
                                   weight='normal',
                                   style='normal', size=int(numpy.floor(0.9*title_fontsize)))
ticks_fontsize = axis_font.get_size()*0.8
# In[]
p, q = [numpy.linspace(-5, 5, 200) for j in range(2)]
# In[basic mode]
mode = OneModeBosonicField(truncated_dimension=5).numeric.NumberState(1)
#Plot the Wigner function of the quantum state
mode.state.PlotWignerFunction(q, p, plot_name="basic - Wigner function")
#Plot the density operator
mode.state.PlotDensityOperator(plot_name="basic - density operator")
# In[displaced]
#alpha = ParameterNumeric(name="\\alpha")
alpha_0 = 1.5+1j
mode_displaced = mode.Vacuum().Displace(alpha_0)
mode_displaced.state.PlotWignerFunction(q, p, plot_name="displaced - Wigner function")
#Plot the density operator
mode_displaced.state.PlotDensityOperator(plot_name="displaced - density operator")
# In[squeezed]
zeta_0 = numpy.log(10**(2/10))/2
mode_squeezed = mode\
                  .Vacuum().Squeeze(zeta_0)
#Plot the Wigner function of the quantum state
mode_squeezed.state.PlotWignerFunction(q, p, plot_name="squeezed - Wigner function")
#Plot the density operator
mode_squeezed.state.PlotDensityOperator(plot_name="squeezed - density operator")
# In[rotated]
t = ParameterNumeric(name="t")
theta_0 = numpy.pi/6
mode_rotated = mode_squeezed\
                .Rotate(theta_0)
#Plot the Wigner function of the quantum state
mode_rotated.state.PlotWignerFunction(q, p,  plot_name="rotate - Wigner function")
#Plot the density operator
mode_rotated.state.PlotDensityOperator(plot_name="rotate - density operator")
# In[projective measurement of the number operator]
samples = mode.ProjectiveMeasurement(measurable="n", ntimes=int(1e4))[0]

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
x_range = numpy.array([0, mode.hilbert_space.dimension-1])
PDF_values, PDF = PDF_histogram(samples, x_range=x_range, n_bins=mode.hilbert_space.dimension)
axis_pdf.bar(PDF_values, PDF, color="tab:blue", alpha=0.5)
# In[projective measurement of a generalized quadrature]
figure_measure_quadrature = pyplot.figure(figsize=(11, 8))

samples = mode_squeezed.ProjectiveMeasurement(measurable="x", ntimes=int(1e6), theta=numpy.pi/2)[0]

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
