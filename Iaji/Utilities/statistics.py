#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:05:13 2022

@author: jiedz
"""
# In[imports]
import numpy
# In[]
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

