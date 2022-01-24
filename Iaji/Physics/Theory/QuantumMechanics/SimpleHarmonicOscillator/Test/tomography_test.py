#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 00:34:51 2022

@author: jiedz
This script tests the 'QuantumStateTomography' module by performing maximum likelihood
tomographic state reconstruction from numerically simulated quadrature measurements.
"""


import numpy
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.QuantumStateTomography import QuadratureTomographer as Tomographer
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator import Utilities as qfutils
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import font_manager

#%%
plt.close('all')
#General plot settings
default_marker = ""
default_figure_size = (16, 14)
default_fontsize = 60
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

#Funciton to plot Wigner function
c1 = [(0.,'#ffffff'), (1/3.,'#FEFEFE'), (1,'#CC0000')]
c2 = [(0.,'#ffffff'), (0.1,'#0C50B7'), (0.2,'#2765C2'), (0.3,'#5889D3'), \
      (0.4,'#A2BEE8'), (0.49,'#FFFFFF'), (0.51,'#FFFFFF'), (0.6,'#E8A2A2'), \
      (0.7,'#D35858'), (0.8,'#C22727'), (0.9,'#B70C0C'), (1.,'#B20000')]
c3 = [(0.,'#ffffff'), (0.1,'#0C50B7'), (0.2,'#2765C2'), (0.3,'#5889D3'), \
      (0.4,'#A2BEE8'), (0.49,'#F9F9F9'), (0.51,'#F9F9F9'), (0.6,'#E8A2A2'), \
      (0.7,'#D35858'), (0.8,'#C22727'), (0.9,'#B70C0C'), (1.,'#B20000')]
cmwig1 = matplotlib.colors.LinearSegmentedColormap.from_list('cmwig1',c2)
cmwig2 = matplotlib.colors.LinearSegmentedColormap.from_list('cmwig2',c3)

#%%
def PDF_histogram(x, x_range, n_bins):
    """
    This function computes the histogram-based probability density function of the inumpyut
    samples x, over the range of values described by 'x_range', divided into 'n_bins' bins.
    """
    PDF_histogram, PDF_bin_edges = numpy.histogram(x, n_bins, x_range, density = True)
    half_bin = (PDF_bin_edges[1]-PDF_bin_edges[2])/2
    PDF_values = PDF_bin_edges+half_bin #center the bin values
    PDF_values = PDF_values[1:] #discard the first bin value
    return PDF_values, PDF_histogram

#%%
#Set up quantum state tomography
n_max = 15# Fock space truncation
mode_function_type = 'double exponential filter'
mode_function_parameters = (2*numpy.pi*16e6, 2*numpy.pi*20e6, -171e-9)
convergence_rule = 'fidelity of state' #'fidelity of iteration operator'#
tomographer = Tomographer(n_max=n_max, mode_function_type=mode_function_type, mode_function_parameters=mode_function_parameters, \
                          convergence_rule=convergence_rule)

#%%
#Generate artificial data to test the tomography code
#Try with a coherent state
phases = numpy.pi/180*numpy.linspace(0, 150, 6)
alpha = 1/numpy.sqrt(2)*(0+0j)
squeezing_variance = 1/4
squeezing_angle = -numpy.pi/2
n_samples = int(1e6)
vacuum_noise = numpy.random.normal(loc=0, scale=numpy.sqrt(1/2), size=n_samples)
quadratures = {}
for phase in phases:
    quadrature_mean = numpy.sqrt(2)*(numpy.real(alpha)*numpy.cos(phase) + numpy.imag(alpha)*numpy.sin(phase))
    quadrature_variance = 1/2*(squeezing_variance*numpy.cos(phase-squeezing_angle/2)**2+1/squeezing_variance*numpy.sin(phase-squeezing_angle/2)**2)
    quadratures[phase] = numpy.random.normal(loc=quadrature_mean, scale=numpy.sqrt(quadrature_variance), size=n_samples)
Ts = 1/25e6
#%%
#Subtract the electronic noise variance from the vacuum quadrature variance
vacuum_var = numpy.var(vacuum_noise)# - numpy.var(electronic_noise)
#Normalize quadrature measurements by vacuum quadrature standard deviation
n_samples = len(quadratures[phases[0]])
quadratures_array = numpy.zeros((n_samples, len(phases)))
for j in range(len(phases)):
    phase = phases[j]
    quadratures_array[:, j] = quadratures[phase]

tomographer.setQuadratureData(quadratures=quadratures_array, vacuum=vacuum_noise, phases=phases, dt=Ts, apply_mode_function=False)
#%%
tomographer.reconstruct(quadratures_range_increase=1, convergence_rule="fidelity of state", convergence_parameter=1e-7)
#%%
q, p = [numpy.linspace(-5, 5, 200) for j in range(2)]
Q, P, wigner_function = qfutils.WignerFunctionFromDensityMatrix(rho=tomographer.rho, q=q, p=p)

#%%
#Check marginal probability densities
n_bins = tomographer.n_bins
histograms = {}
plt.figure()
for phase in phases:
    histograms[phase] = PDF_histogram(tomographer.quadratures[phase], [numpy.min(tomographer.quadratures[phase]), numpy.max(tomographer.quadratures[phase])], n_bins=n_bins)
    plt.plot(histograms[phase][0], histograms[phase][1],label="%.1f"%(phase*180/numpy.pi), marker='.')
histogram_vacuum = PDF_histogram(tomographer.vacuum, [numpy.min(tomographer.vacuum), numpy.max(tomographer.vacuum)], n_bins=n_bins)
plt.plot(histogram_vacuum[0], histogram_vacuum[1],label='vacuum quadrature', marker='.')
plt.grid()
plt.legend()
#%%
#Plot the iteration operator
figure_R = plt.figure(figsize=default_figure_size)
axis_R = figure_R.add_subplot(111)
N, M = numpy.meshgrid(numpy.arange(n_max+1)/2, numpy.arange(n_max+1)/2)
R_max = numpy.max(numpy.abs(tomographer.R))
axis_R.imshow(numpy.abs(tomographer.R), cmap=cmwig1, alpha=0.6, norm=matplotlib.colors.Normalize(vmin=-R_max, vmax=R_max))    
axis_R.set_xlabel("n", font=axis_font)
axis_R.set_ylabel("m", font=axis_font)
plt.pause(.05)
axis_R.set_xticks(numpy.arange(0, n_max+1))
axis_R.set_yticks(numpy.arange(0, n_max+1))
axis_R.set_xticklabels(numpy.arange(0, n_max+1), fontsize=ticks_fontsize)
axis_R.set_yticklabels(numpy.arange(0, n_max+1), fontsize=ticks_fontsize)
axis_R.grid(True, color="grey", alpha=0.2)
#Plot the density matrix
rho_max = numpy.max(numpy.abs(tomographer.rho))
figure_rho = plt.figure(figsize=default_figure_size)
axis_rho = figure_rho.add_subplot(111)
axis_rho.imshow(numpy.abs(tomographer.rho), cmap=cmwig1, alpha=0.6, norm=matplotlib.colors.Normalize(vmin=-rho_max, vmax=rho_max))    
axis_rho.set_xlabel("n", font=axis_font)
axis_rho.set_ylabel("m", font=axis_font) 
plt.pause(.05)
axis_rho.set_xticks(numpy.arange(0, n_max+1))
axis_rho.set_yticks(numpy.arange(0, n_max+1))
axis_rho.set_xticklabels(numpy.arange(0, n_max+1), fontsize=ticks_fontsize)
axis_rho.set_yticklabels(numpy.arange(0, n_max+1), fontsize=ticks_fontsize)
axis_rho.grid(True, color="grey", alpha=0.2)
#axis_rho.grid(True)
#%%
figures_wigner = qfutils.plotWignerFunction(Q, P, wigner_function, alpha=0.5)
axis_wigner_2D = figures_wigner["2D"].axes[0]
axis_wigner_3D = figures_wigner["3D"].axes[0]
#Set the ticks and legends
plt.pause(.05)
axis_wigner_2D.set_xticklabels(axis_wigner_2D.get_xticklabels(), fontsize=ticks_fontsize)
axis_wigner_2D.set_yticklabels(axis_wigner_2D.get_yticklabels(), fontsize=ticks_fontsize)
#axis_PSD.legend(prop=legend_font)
plt.pause(.05)
axis_wigner_3D.set_xticks([])
axis_wigner_3D.set_yticks([])
axis_wigner_3D.set_zticklabels(axis_wigner_3D.get_zticklabels(), fontsize=ticks_fontsize)
#axis_wigner_3D.set_yticklabels(axis_wigner_3D.get_yticklabels(), fontsize=ticks_fontsize)
#%%
figure_convergence_metrics = plt.figure(figsize=(11, 8))
n_iterations = len(tomographer.log_likelihood)
axis = figure_convergence_metrics.add_subplot(111)
for rule in ["fidelity of state", "fidelity of iteration operator"]:
    axis.scatter(numpy.arange(n_iterations), tomographer.convergence_metris[rule], label=rule, s=100)
axis.set_xlabel("iteration index", font=axis_font)
plt.pause(.05)
axis.set_xticklabels(axis.get_xticklabels(), fontsize=ticks_fontsize)
axis.set_yticklabels(axis.get_yticklabels(), fontsize=ticks_fontsize)
axis.grid(True)
#axis.legend(loc="center", prop=axis_font)