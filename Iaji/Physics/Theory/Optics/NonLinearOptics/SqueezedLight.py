#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:18:26 2020
@author: jiedz
"""
#%%
#imports
from matplotlib import pyplot as plt
#from scipy import signal
import numpy as np
import lmfit as lm
from scipy.optimize import curve_fit
#%%
        #General plot settings
default_marker = ""
default_figure_size = (16*0.9, 9*0.9)
default_fontsize = 16
title_fontsize = default_fontsize
axis_fontsize = title_fontsize
legend_fontsize = int(np.floor(0.8*title_fontsize))
#%%
def quadraturePSD(frequency, bandwidth, gain_factor, efficiency, \
                  phase_noise_std=0, quadrature_type="squeezed", plot=False):
   """
   This function calculates the power spectral density (PSD) of the squeezed and antisqueezed
   quadratures of a pure squeezed state of the electromagnetic radiation field, 
   generated through second harmonic generation inside a double-resonant optical parametric oscillator (OPO).
   The power spectral density is calculated using the formula given in https://doi.org/10.1364/OE.27.037877
   The calculation takes into account a Gaussian noise added to the phase of one beam.
   INPUTS: 
   - quadrature_type: can be 'squeezed' or 'antisqueezed' - string
   - frequency: frequency vector where the PSD is evaluated [Hz] - array-like of float
   - efficiency: total power transmission efficiency from the output of the OPO (including escape efficiency) to the evaluation point - float (in [0,1])
   - bandwidth: full-width-half-maximum of the OPO frequency response [Hz] - float 
   - phase_noise_std: standard deviation of the phase noise added to one beam [rad] - float (>=0) 
   - gain_factor: square-root of the ratio between the pump optical power and the threshold pump optical power [adimensional] - float (>0) 
    
   OUTPUTS:
   - power spectral density of the selected quadrature, evaluated at frequencies contained in 'frequency' [(a.u.)^2/Hz]- array of float
   """
   #-----------------------------------------------
   #Calculate the power spectral density
   frequency = np.array(frequency)
   sign = -1*(quadrature_type=='squeezed')+1*(quadrature_type=='antisqueezed')
   #print("\n\n \sign*2 = "+str(sign*2))
   b = 2/bandwidth
   PSD = 1+efficiency*4*gain_factor*\
         (sign*(1+np.exp(-2*phase_noise_std**2))/2/((1-sign*gain_factor)**2+(b*frequency)**2)\
          -sign*(1-np.exp(-2*phase_noise_std**2))/2/((1+sign*gain_factor)**2+(b*frequency)**2))
   #Plot the PSD, if required
   if plot:
       figure_squeezed_light_quadrature_PSD = plt.figure(num="Squeezed light quadrature power spectral density", figsize=default_figure_size)
       axis = figure_squeezed_light_quadrature_PSD.add_subplot(111)
       axis.set_title("Power spectral density of the "+quadrature_type+" quadrature", fontsize=title_fontsize)
       axis.set_xlabel("frequency (MHz)", fontsize=axis_fontsize)
       axis.set_ylabel("power spectral density (dB/Hz)", fontsize=axis_fontsize)
       axis.grid(True)
       axis.plot(frequency*1e-6, 10*np.log10(PSD), marker=default_marker)
       axis.legend(loc="upper right", fontsize=legend_fontsize)
   return PSD
#%%
def quadraturePSD2(frequency, bandwidth, gain_factor, efficiency, \
                   phase_noise_std):

    """
    This function computes simultaneously the squeezed and antisqueezed quadratures of a single-mode squeezed light field,
    using the function 'quadraturePSD'.
    
    INPUTS
    ---------------
    - frequency: frequency vector where the PSD is evaluated [Hz] - array-like of float
   - efficiency: total power transmission efficiency from the output of the OPO (including escape efficiency) to the evaluation point - float (in [0,1])
   - bandwidth: full-width-half-maximum of the OPO frequency response [Hz] - float 
   - phase_noise_std: standard deviation of the phase noise added to one beam [rad] - float (>=0) 
   - gain_factor: square-root of the ratio between the pump optical power and the threshold pump optical power [adimensional] - float (>0) 
   
    OUTPUTS
    --------------
    a matrix containing the power spectral densities of the squeezed and antisqueezed quadratures as two column vectors
    """
    PSDs = np.zeros((len(frequency), 2))
    PSDs[:, 0] = quadraturePSD(frequency, bandwidth, gain_factor, efficiency, phase_noise_std=phase_noise_std, quadrature_type='squeezed')
    PSDs[:, 1] = quadraturePSD(frequency, bandwidth, gain_factor, efficiency, phase_noise_std=phase_noise_std, quadrature_type='antisqueezed')
    return PSDs
#%%
def quadraturePSD2_1D(frequency, bandwidth, gain_factor, efficiency, \
                   phase_noise_std):

    """
    This function computes simultaneously the squeezed and antisqueezed quadratures of a single-mode squeezed light field,
    using the function 'quadraturePSD' and concatenates the output arrays into a 1-D array.
    
    INPUTS
    ---------------
    - frequency: frequency vector where the PSD is evaluated [Hz] - array-like of float
   - efficiency: total power transmission efficiency from the output of the OPO (including escape efficiency) to the evaluation point - float (in [0,1])
   - bandwidth: full-width-half-maximum of the OPO frequency response [Hz] - float 
   - phase_noise_std: standard deviation of the phase noise added to one beam [rad] - float (>=0) 
   - gain_factor: square-root of the ratio between the pump optical power and the threshold pump optical power [adimensional] - float (>0) 
   
    OUTPUTS
    --------------
    a matrix containing the power spectral densities of the squeezed and antisqueezed quadratures as two column vectors
    """
    quadratures = quadraturePSD2(frequency, bandwidth, gain_factor, efficiency, phase_noise_std)
    return np.concatenate((10*np.log10(quadratures[:, 0]), 10*np.log10(quadratures[:, 1])))
#%%
def fitQuadraturePSD2(frequency, quadrature_PSD_squeezed, quadrature_PSD_antisqueezed, initial_guess, bounds=None):
    #Put the input data arrays into a format suited for 1-D optimization
    samples = np.empty((0,))
    samples = np.concatenate((10*np.log10(quadrature_PSD_squeezed), 10*np.log10(quadrature_PSD_antisqueezed)))
    fit_result = curve_fit(f=quadraturePSD2_1D, xdata=frequency, ydata=samples, p0=initial_guess, bounds=bounds)
    return fit_result 