#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:35:46 2020

@author: jiedz

This module contains a set of utility functions, to analyze the spectrum of an optical cavity.
It currently contains:
    - modeMatching(): calculates the mode-matching efficiency of the fundamental mode of an optical cavity, given the cavity power spectrum;

"""
#%%
#Imports
import sys
modules_path = "/home/jiedz/Jiedz/University/PhD-@-DTU/Software/Python"
sys.path.append(modules_path)
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import font_manager
import numpy 
import scipy
from Iaji.Physics.Experiment.Optics.Cavity.peakdetect import peakdetect
matplotlib.rcParams['agg.path.chunksize'] = 10000
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
#%%
def modeMatching(cavity_spectrum, x = None, scanning_signal = None, beam_name = "", color="tab:blue"):
    """
    This function calculates the mode-matching efficiency of the fundamental mode of an optical cavity, given its power spectrum.
    It does so, by applying the following procedure:
        - Let the user select a free spectral range;
        - Smoothen the selected power spectrum;
        - Automatically detect peaks;
        - Calculate the mode-matching efficiency, as the fundamental mode peak height over the 
          sum of the heights of all the peaks.
    INPUTS: 
        - cavity_spectrum: the transmission power spectrum of the cavity [a.u.]: array-like of float
        - x: independent variable (e.g., time, frequency or wavelength) [a.u.]: array-like of float
        - scanning_signal: signal used for scanning the cavity length, to yield the power spectrum [a.u.]: array-like of float
        - beam_name: name given to the analyzed cavity mode - string
    
    OUTPUTS:
        - modematching: estimated mode-matching efficiency [adimensional] - float (in [0, 1])
        - figure_cavity_spectrum: figure displaying the cavity spectrum and the detected peaks
    """
    #%%
    #Prepare the raw data
    n_samples = len(cavity_spectrum)
    #Make the cavity spectrum positive (remove offset)
    #cavity_spectrum -= numpy.min(cavity_spectrum)
    #Rescale the raw data
    cavity_spectrum /= (numpy.max(cavity_spectrum)-numpy.min(cavity_spectrum))
    #Prepare a dummy normalized x vector, if None is provided
    x_is_none = False
    if x is None:
        x = numpy.linspace(start=0, stop=1, num=n_samples)
        x_is_none = True
    #Remove offset from the cavity spectrum
    cavity_spectrum -= numpy.mean(cavity_spectrum[numpy.where(numpy.isclose(cavity_spectrum, numpy.min(cavity_spectrum), rtol=1e-2))])
    #%%
    #Plot the raw data
    figure_cavity_spectrum = plt.figure(num="Cavity spectrum", figsize = default_figure_size)
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    
    figure_title = "Cavity spectrum"
    if beam_name!="":
        figure_title += " of the "+beam_name
        
    user_instructions = 'Select the start and end points of a free spectral range (only one main peak)'
    
    axis=plt.subplot(2, 1, 1)
    axis#.set_title(figure_title+"\n\n"+user_instructions, font=title_font)
    axis.set_xlabel("x (a.u.)", font=axis_font)
    axis.set_ylabel("power spectrum (a.u.)", font=axis_font)
    plt.plot(x, cavity_spectrum, label="cavity spectrum", linewidth=4, color=color, alpha = 0.7) 
    if not(scanning_signal is None):
        plt.plot(x, scanning_signal, label="scanning signal", color="black", alpha = 0.5, linewidth=4)
    plt.pause(.05)
    axis.set_xticklabels(axis.get_xticklabels(), fontsize=ticks_fontsize*0.8, family="Times New Roman")
    axis.set_yticklabels(axis.get_yticklabels(), fontsize=ticks_fontsize*0.8, family="Times New Roman")
    axis.grid()
    if x_is_none:
        axis.set_xticks([])
        axis.set_xlabel("")
    #%%
    #Design a smoothing filter and apply it to the cavity spectrum trace
    #Savitsky-Golay filter
    print("\nApplying a smoothening filter to the cavity spectrum trace.")
    cavity_spectrum_smooth = scipy.signal.savgol_filter(cavity_spectrum, window_length=25, polyorder=5, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
    #Plot the smoothened data
    plt.plot(x, cavity_spectrum_smooth, label="cavity spectrum - smoothened", linewidth=4, color=color, alpha = 0.5, linestyle="-")
    axis#.legend(loc="upper right", font=legend_font)
    #%%
    #Ask the user to select a free spectral range
    plt.show()
    selected_points = plt.ginput(2)
    axis#.set_title(figure_title, font=title_font)
    free_spectral_range = numpy.sort([numpy.array(selected_points[0])[0], numpy.array(selected_points[1])[0]])
    #Define a zoomed-in version of the original data, over the selected FSR
    cavity_spectrum_FSR = cavity_spectrum[numpy.where((x>=free_spectral_range[0]) & (x<=free_spectral_range[1]))]
    cavity_spectrum_smooth_FSR = cavity_spectrum_smooth[numpy.where((x>=free_spectral_range[0]) & (x<=free_spectral_range[1]))]
    if not(scanning_signal is None):
        scanning_signal_FSR = scanning_signal[numpy.where((x>=free_spectral_range[0]) & (x<=free_spectral_range[1]))]
    x_FSR = x[numpy.where((x>=free_spectral_range[0]) & (x<=free_spectral_range[1]))]
    
    #%%
    #Plot the selected cavity spectrum free spectral range
    axis=plt.subplot(2, 1, 2)
    axis#.set_title(figure_title+" in a free spectral range", font=title_font)
    axis.set_xlabel("x (a.u.)", font=axis_font)
    axis.set_ylabel("power spectrum (a.u.)", font=axis_font)
    
    plt.plot(x_FSR, cavity_spectrum_FSR, label="cavity spectrum", linewidth=4, color=color, alpha = 0.7)
    if not(scanning_signal is None):
        plt.plot(x_FSR, scanning_signal_FSR, label="scanning signal", linewidth=4, color="black", alpha = 0.7)
    plt.plot(x_FSR, cavity_spectrum_smooth_FSR, label="cavity spectrum - smoothened",marker=default_marker)
    plt.pause(.05)
    axis.set_xticklabels(axis.get_xticklabels(), fontsize=ticks_fontsize*0.8, family="Times New Roman")
    axis.set_yticklabels(axis.get_yticklabels(), fontsize=ticks_fontsize*0.8, family="Times New Roman")
    if x_is_none:
        axis.set_xticks([])
        axis.set_xlabel("")
    axis.grid()
    #%%
    #IMPORTANT: subtract the mean value of the baseline noise, i.e., the minimum value of
    #           the smoothened cavity spectrum (over the considered FSR)
    #cavity_spectrum_smooth_FSR = cavity_spectrum_FSR
    noise_floor_mean =  numpy.min(cavity_spectrum_smooth_FSR)
    cavity_spectrum_smooth_FSR -= noise_floor_mean
    #Now find the peaks whithin a free spectral range
    print("\nFinding the peaks corresponding to the cavity frequency modes")
    #Find the peaks using a custom made function, by Sixten Bergman
    peaks = peakdetect(y_axis = cavity_spectrum_smooth_FSR, x_axis=x_FSR, lookahead=250)[0]
    peaks = numpy.transpose(peaks)
    peak_positions = peaks[0]
    peak_heights = peaks[1]
    #Add the baseline noise for plotting purposes
    cavity_spectrum_smooth_FSR += noise_floor_mean
    #%%
    #Plot the peak points and calculate the modematching
    plt.scatter(peak_positions, peak_heights+numpy.min(cavity_spectrum_smooth_FSR), label="detected cavity frequency modes", marker="x")
    #plt.scatter(x_FSR[peak_search_range, highest_peak_height/2*numpy.ones((2, )), label="secondary peaks search range", marker="o")
    axis#.legend(loc="upper right", font=legend_font)
    #Calculate mode matching
    modematching = numpy.max(peak_heights)/numpy.sum(peak_heights)
    
    axis#.set_title(figure_title+" in a free spectral range"
     #         +"\nFundamental mode mode-matching: "+str(numpy.round(modematching*100, 2))+"%",\
     #         font=title_font)
        
    return modematching, figure_cavity_spectrum
