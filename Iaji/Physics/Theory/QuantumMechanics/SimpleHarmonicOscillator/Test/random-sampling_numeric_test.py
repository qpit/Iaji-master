#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:10:25 2022

@author: jiedz
This script test sampling of a random variable with known numerical probability density
function.
"""
# In[imports]
import numpy, scipy
from scipy.stats import rv_continuous, rv_discrete
from scipy.special import factorial, hermite
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
class Poisson(rv_discrete):
    def _pmf(self, k, l):
        return l**k * numpy.exp(-l)/factorial(k)

class Gaussian(rv_continuous):
    def _pdf(self, x, mu, sigma):
       return 1/numpy.sqrt(2*numpy.pi*sigma**2)*numpy.exp(-(x-mu)**2/(2*sigma**2))
n = 1
theta = 0  
class QuadratureNumberState(rv_continuous):
    def _pdf(self, x):
        return numpy.math.pow(numpy.abs(1/numpy.sqrt(2**n*factorial(n)*numpy.sqrt(numpy.pi))\
                *numpy.exp(-x**2/2)*numpy.exp(-1j*n*theta)*hermite(n)(x)), 2)
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
rv_discrete = Poisson()
n_samples = 1000
samples = rv_discrete.rvs(l=l, size=n_samples)

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
n = 1
theta = 0
continuous = QuadratureNumberState()
n_samples = 1000
samples = continuous.rvs(size=n_samples)
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
                            n_bins=50)
axis_pdf.plot(values, pdf, color="tab:blue", alpha=0.7, marker="o", \
                              linestyle="None")