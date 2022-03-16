#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 18:38:23 2022

@author: jiedz
This script tests the NModeBosonicFieldNumeric module
"""
# In[imports]
from Iaji.Physics.Theory.QuantumMechanics.QuanutmInformation.NModeBosonicField \
    import NModeBosonicFieldNumeric
from Iaji.Physics.Theory.QuantumMechanics.QuanutmInformation.OneModeBosonicField \
    import OneModeBosonicFieldNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix import \
    MatrixNumeric
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
# In[Basic system]
system = NModeBosonicFieldNumeric.Vacuum(N=2, truncated_dimensions=[15, 15], name="A", hbar=1)
#%%
system_BS = system.Squeeze("A_{0}", 0.6).BeamSplitter(system.mode_names, R=0.5)\
    .SelectModes("A_{0}")
system_BS.state.PlotWignerFunction(q, p, plot_name="BS")
#%%
system_loss = system.Squeeze("A_{0}", 0.6).Loss(["A_{0}"], [0.5]).SelectModes("A_{0}")
system_loss.state.PlotWignerFunction(q, p, plot_name="loss")
# In[Two-mde squeezed vacuum state] 
#Two-mode squeezed vacuum from single-mode squeezing and beam splitting
zeta = numpy.log(10**(5/10))/2
#zeta = 0.001
system_TMSV = system.Squeeze("A_{0}", zeta).Squeeze("A_{1}", -zeta).BeamSplitter(modes=system.mode_names, R=0.5)
# In[Simulate the conditional generation of a single-boson state from two-mode squeezed vacuum] 
#Perform boson number measurements and find a configuration of the field where
#a single boson has been measured
outcomes, _, _, post_measurement_fields = system_TMSV\
    .ProjectiveMeasurement(mode="A_{0}", measurable="n", ntimes=int(1e4), return_all_fields=True)
system_single_boson = post_measurement_fields[numpy.where(outcomes==1)[0][0]]
system_single_boson.SelectModes("A_{0}").state.PlotWignerFunction(q, p, plot_name="single-boson - measured mode")
system_single_boson.SelectModes("A_{1}").state.PlotWignerFunction(q, p, plot_name="single-boson - other mode")
# In[Simulate the conditional generation of single-mode displaced squeezed states form two-mode squeezed vacuum]
#Perform a quadrature measurement
outcomes, _, _, system_displaced_squeezed = system_TMSV\
    .ProjectiveMeasurement(mode="A_{0}", measurable="x", ntimes=1, theta=numpy.pi/2)
system_displaced_squeezed.SelectModes("A_{0}").state.PlotWignerFunction(q, p, plot_name="single-mode squeezed and displaced - measured mode")
system_displaced_squeezed.SelectModes("A_{1}").state.PlotWignerFunction(q, p, plot_name="single-mode squeezed and displaced - other mode")
# In[Simulate on/off detection from two-mode squeezed vacuum]
outcomes, _, _, post_measurement_fields = system_TMSV\
    .POVM(mode="A_{0}", ntimes=int(1e4), return_all_fields=True, povm_type="on/off detection")
system_on_off = post_measurement_fields[numpy.where(outcomes==1)[0][0]]
system_on_off.SelectModes("A_{0}").state.PlotWignerFunction(q, p, plot_name="single-boson - on/off -measured mode")
system_on_off.SelectModes("A_{1}").state.PlotWignerFunction(q, p, plot_name="single-boson - on/off - other mode")
# In[Simulate ON detection only, from two-mode squeezed vacuum]
#Prepare the POVM element corresponding to ON outcome
mode_index = 0
mode = system_TMSV.modes_list[mode_index]
modes_before = system_TMSV.modes_list[0:mode_index]
modes_after = system_TMSV.modes_list[mode_index+1:]
d = mode.hilbert_space.dimension
e0 = mode.hilbert_space.CanonicalBasisVector(0).numeric
ON_operator = MatrixNumeric.TensorProduct([*[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_before], \
                                         MatrixNumeric.Eye(d) - e0 @ e0.T(), \
                                          *[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_after]])
#Simulate the ON outcome according to the generalized Born rule
post_measurement_field = system_TMSV._GeneralizedBornRule(ON_operator)[0]
post_measurement_field.SelectModes("A_{0}").state.PlotNumberDistribution(plot_name="ON detection - measured mode")
post_measurement_field.SelectModes("A_{1}").state.PlotNumberDistribution(plot_name="ON detection - other mode")