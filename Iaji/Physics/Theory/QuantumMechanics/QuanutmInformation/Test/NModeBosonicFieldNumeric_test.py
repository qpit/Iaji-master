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
system_squeezed = system.Squeeze([0.1, 0.1]).BeamSplitter(["A_{0}", "A_{1}"], 0.5)
#system_squeezed = system.TwoModeSqueeze(modes=system.mode_names, zeta=0.15)
