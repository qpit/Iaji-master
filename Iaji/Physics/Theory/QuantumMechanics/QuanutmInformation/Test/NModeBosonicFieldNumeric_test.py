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
system = NModeBosonicFieldNumeric.Vacuum(N=2, truncated_dimensions=[10, 10], name="A", hbar=1)
#system = NModeBosonicFieldNumeric.NumberStates(N=3, truncated_dimensions=[2, 5, 2], number_state_orders=[1, 3, 0], name="A", hbar=1)
#mode1 = OneModeBosonicFieldNumeric(truncated_dimension=2, name="A").NumberState(1).Displace(0.1)
#mode2 = OneModeBosonicFieldNumeric(truncated_dimension=5, name="B").Squeeze(numpy.log(2)/2)
#mode3 = OneModeBosonicFieldNumeric(truncated_dimension=3, name="C").NumberState(2)
#system = NModeBosonicFieldNumeric(modes_list=[mode1, mode2, mode3], name="S")
system_squeezed = system.Squeeze([0.3, 0])
#system_squeezed = system.TwoModeSqueeze(modes=system.mode_names, zeta=0.15)
#%%
system_BS = system_squeezed.BeamSplitter(system_squeezed.mode_names, R=0.7)\
    .PartialTrace("A_{1}")
system_BS.state.PlotWignerFunction(q, p, plot_name="BS")
#%%
system_loss = system_squeezed.Loss(["A_{0}"], [0.7]).PartialTrace("A_{1}")
system_loss.state.PlotWignerFunction(q, p, plot_name="loss")
# In[Simulate the conditional generation of a single-boson state from two-mode squeezed vacuum]
#Two-mode squeezed vacuum from single-mode squeezing and beam splitting
zeta = 0.1
system_TMSV = system.Squeeze([zeta, zeta]).BeamSplitter(modes=system.mode_names, R=0.5)
#Perform boson number measurements and find a configuration of the field where
#a single boson has been measured
outcomes, _, _, post_measurement_fields = system_TMSV\
    .ProjectiveMeasurement(mode="A_{0}", measurable="n", ntimes=int(1e4), return_all_fields=True)
system_single_boson = post_measurement_fields[numpy.where(outcomes==1)[0][0]]
system_single_boson.PartialTrace("A_{1}").state.PlotWignerFunction(q, p, plot_name="single-boson - measured mode")
system_single_boson.PartialTrace("A_{0}").state.PlotWignerFunction(q, p, plot_name="single-boson - other mode")
# In[Simulate the conditional generation of single-mode displaced squeezed states form two-mode squeezed vacuum]
zeta = 0.5
system_TMSV = system.Squeeze([zeta, zeta]).BeamSplitter(modes=system.mode_names, R=0.5)
#Perform a quadrature measurement
outcomes, _, _, system_displaced_squeezed = system_TMSV\
    .ProjectiveMeasurement(mode="A_{0}", measurable="x", ntimes=1, theta=-numpy.pi/4)
system_displaced_squeezed.PartialTrace("A_{1}").state.PlotWignerFunction(q, p, plot_name="single-mode squeezed and displaced - measured mode")
system_displaced_squeezed.PartialTrace("A_{0}").state.PlotWignerFunction(q, p, plot_name="single-mode squeezed and displaced - other mode")