#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:10:25 2022

@author: jiedz
This script test sampling of a random variable with known numerical probability density
function.
"""
# In[imports]
import numpy, sympy
from sympy import stats
from sympy import factorial
sympy.init_printing()
# In[GUI imports]
from matplotlib import pyplot, font_manager
#%%
#General plot settings
default_marker = ""
default_figure_size = (14.5, 10)
default_fontsize = 40
title_fontsize = default_fontsize
title_font = font_manager.FontProperties(family='Times New Roman',
                                   weight='bold',
                                   style='normal', size=title_fontsize)
axis_font = font_manager.FontProperties(family='Times New Roman',
                                   weight='normal',
                                   style='normal', size=title_fontsize*0.8)
legend_font = font_manager.FontProperties(family='Times New Roman',
                                   weight='normal',
                                   style='normal', size=int(numpy.floor(0.9*title_fontsize)))
ticks_fontsize = axis_font.get_size()*0.8
# In[define PDFs]
k = sympy.symbols("k")
lamda = sympy.sympify("lamda")
pdf_poisson = lamda**k*sympy.exp(-lamda)/factorial(k)
poisson = stats.DiscreteRV(k, pdf_poisson, set=sympy.Naturals0)
 #%%
def PDF_histogram(x, x_range, n_bins):
    """
    This function computes the histogram-based probability density function of the input
    samples x, over the range of values described by 'x_range', divided into 'n_bins' bins.
    """
    PDF_histogram, PDF_bin_edges = numpy.histogram(x, n_bins, x_range, density = True)
    half_bin = (PDF_bin_edges[1]-PDF_bin_edges[2])/2
    PDF_values = PDF_bin_edges+half_bin #center the bin values
    PDF_values = PDF_values[1:] #discard the first bin value
    return PDF_values, PDF_histogram

# In[discrete random sampling]
l = 50
n_samples = 1000
samples = stats.sample(poisson, size=(n_samples,))
#samples = stats.sample(sympy.stats.Poisson("poisson", lamda=l), numsamples=n_samples)
figure_discrete = pyplot.figure(figsize=(11, 8))
figure_discrete.subplots_adjust(wspace=0.5)
axis_samples = figure_discrete.add_subplot(1, 2, 1)
axis_samples.plot(samples)
axis_samples.set_xlabel("sample number", font=axis_font)
axis_samples.set_ylabel("sample value", font=axis_font)

axis_pdf = figure_discrete.add_subplot(1, 2, 2)
axis_pdf.set_xlabel("sample value", font=axis_font)
axis_pdf.set_ylabel("probability", font=axis_font)
#Plot
axis_pdf.hist(samples, color="tab:blue", alpha=0.7, ec="black")
#pyplot.pause(.05)
#axis_pdf.set_xticklabels(axis_pdf.get_xticklabels(), fontsize=ticks_fontsize)
#axis_pdf.set_yticklabels(axis_pdf.get_yticklabels(), fontsize=ticks_fontsize)
# In[continuous random sampling]
n = 0
theta = 0
rv_continuous = QuadratureNumberState()
n_samples = 10000
samples = rv_continuous.rvs(n, theta, size=n_samples)
title = "Generalized quadrature distribution for the %d-th number state"\
        " along the angle %.1f degrees"%(n, theta)
figure_continuous = pyplot.figure(num=title, figsize=(11, 8))
figure_continuous.subplots_adjust(wspace=0.5)
axis_samples = figure_continuous.add_subplot(1, 2, 1)
axis_samples.plot(samples)
axis_samples.set_xlabel("sample number", font=axis_font)
axis_samples.set_ylabel("sample value", font=axis_font)

axis_pdf = figure_continuous.add_subplot(1, 2, 2)
axis_pdf.set_xlabel("sample value", font=axis_font)
axis_pdf.set_ylabel("probability density", font=axis_font)
#Plot
values, pdf =PDF_histogram(samples, [numpy.min(samples), numpy.max(samples)],\
                            n_bins=20)
axis_pdf.plot(values, pdf, color="tab:blue", alpha=0.7, marker="o", \
                              linestyle="None")