#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:33:17 2020

@author: jiedz
This module contains functions related to optical homodyne detection.
It currently comprises:
    - vacuumNoiseLinearity() - vacuum noise linearity characterization of a homodyne detector;
"""

#%%
#Imports
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import font_manager
import numpy as np
import scipy
from scipy import signal
import uncertainties
from uncertainties import unumpy
matplotlib.rcParams['agg.path.chunksize'] = 10000
#%%
#Clear variables or plots
plt.close('all')
#%%
#General plot settings
default_marker = ""
default_figure_size = (16, 9)
default_fontsize = 16
title_fontsize = default_fontsize
title_font = font_manager.FontProperties(family='Times New Roman',
                                   weight='bold',
                                   style='normal', size=title_fontsize)
axis_font = font_manager.FontProperties(family='Times New Roman',
                                   weight='normal',
                                   style='normal', size=title_fontsize)
legend_font = font_manager.FontProperties(family='Times New Roman',
                                   weight='normal',
                                   style='normal', size=int(np.floor(0.9*title_fontsize)))
#%%
#Define a function that computes the sample variance of a vector of samples drawn from
#i.i.d. Gaussian random variables, with an underlying Gaussian distribution. as well as the standard deviation of the sample variance
def sampleVariance(x, name=None):
    """
    This function computes the sample variance of a Gaussian random variables, as well as the theoretical standard deviation
    of the estimator, given one vectors of i.i.d. samples whose elements are realizations of the original Gaussian random variables.
    The standard deviation of the estimator is drawn from the theory of estimating the covariance matrix of a Gaussian random vector using the sample covariance matrix.
    
    INPUTS
    -----------------
        x : array-like of float
            array of input samples.
        
        name : string
            name associated to this sample variance
        
    OUTPUTS
    ------------------
        sample_variance : uncertainties.ufloat
            distribution containing both the nominal value and the standard deviation of the sample variance
    """
    x = np.array(x)
    N = len(x) #total number of samples
    sample_variance = 1/N * (x-np.mean(x)).T @ (x-np.mean(x)) #sample variance
    sample_variance_std = (2/(N-1))**0.5 * sample_variance #theoretical standard deviation of the sample variance
 
    return uncertainties.ufloat(nominal_value=sample_variance, std_dev=sample_variance_std, tag=name)

#%%
#Define a function that converts a list of uncertainties.ufloat objects to pairs of numpy arrays, containing (nominal_values, std_dev)
def ufloatsToNumpy(x):
    """
    This function converts a list of uncertainties.ufloat objects to pairs of numpy arrays, containing (nominal_values, std_dev)

    INPUTS
    ----------
    x : array-like of uncertainties.ufloat
        array of uncertainty.ufloat objects

    OUTPUTS
    -------
    nominal_values : array-like of float
        array containing the nominal values associated to the elements of 'x'
    
    std_dev : array-like of float
        array containing the standard deviations associated to the elements of 'x'
    """
    
    nominal_values = []
    std_devs = []
    for j in range(len(x)):
        nominal_values.append(x[j].nominal_value)
        std_devs.append(x[j].std_dev)  
    return np.array(nominal_values), np.array(std_devs)
#%%
def affine(x, a, b):
         """
         This function calculates the affine transformation x |--> b+a*x
         INPUTS:
             - x: independent variable [a.u.] - array-like of float
             - a: angular coefficient [a.u.] - float
             - b: intercept [a.u.] - float
         """
         return (b+a*x)
#%%
def linear(x, a):
         """
         This function calculates the linear transformation x |--> a*x
         INPUTS:
             - x: independent variable [a.u.] - array-like of float
             - a: angular coefficient [a.u.] - float
         """
         return a*x
#%%
def PDF_histogram(x, x_range, n_bins):
    """
    This function computes the histogram-based probability density function of the input
    samples x, over the range of values described by 'x_range', divided into 'n_bins' bins.
    """
    PDF_histogram, PDF_bin_edges = np.histogram(x, n_bins, x_range, density = True)
    half_bin = (PDF_bin_edges[1]-PDF_bin_edges[2])/2
    PDF_values = PDF_bin_edges+half_bin #center the bin values
    PDF_values = PDF_values[1:] #discard the first bin value
    return PDF_values, PDF_histogram
#%%

# =============================================================================
# def vacuumNoiseLinearity(frequency, optical_powers, power_spectral_densities, optical_powers_std=None, measurement_range = np.array([2, 5])*1e6, plot=True):
#     """
#     This function performs a vacuum noise linearity characterization of a homodyne detector,
#     by taking power spectral densities of detected vacuum quadratures from the homodyne detector, 
#     with increasing input local oscillator optical powers, and the corresponding local oscillator powers, 
#     to perform a linear fit of the statistical power over the local oscillator powers. If a homodyne detector
#     is "shot-noise-limited", then there is a linear dependence of the quadrature statistical power (over a measurement bandwidth)
#     on the input local oscillator optical powers. The power spectral densities are assumed to be measured with an electronic
#     spectrum analyzer, in units of dBm/Hz.
#     
#     INPUTS:
#         - frequency: frequency vector where the PSD is evaluated [Hz] - array-like of float
#         - optical_powers: vector of the input local oscillator optical powers [W] - array-like of float
#         - power_spectral_densities: matrix of the vacuum quadrature power spectral densities, detected with increasing
#           input local oscillator optical powers [dBm/Hz] - 2D array-like of float
#         - measurement_range: frequency limits within which the statistical power is computed [Hz] - 2-element 1D array-like of float
#         - plot: if True, plotting is activated - boolean
#     
#     OUTPUTS:
#         - statistical_powers: vector of the statistical powers, calculated from the power spectral densities over measurement range [W] - array-like of float
#         - linear_fit: dictionary structured as follows:
#             - ["coefficients"]: the optimal linear fit coefficients {a, b : f=b+a*x} [a=adimensional][b=W] - array-like of float
#             - ["covariance"]: the covariance matrix of the fitted paramters - 2d array-like of float
#         - figures: dictionary structured as follows:
#             - ["power spectral densities"]: power spectral densities
#             - ["vacuum noise linearity"]: figure containing the linear fit between statistical powers and local oscillator optical powers
#     """
#     #Convert input to numpy arryas
#     frequency = np.array(frequency)
#     optical_powers = np.array(optical_powers)
#     power_spectral_densities = np.array(power_spectral_densities)
#     measurement_range = np.array(measurement_range)
#     #Determine the shape if the power specral densities matrix
#     PSD_shape = np.shape(power_spectral_densities) 
#     n_measurements = np.min(PSD_shape)-1 #number of power spectral densities taken
#     #Correct the shape of the input power spectral densities, to arrange it in columns
#     #---------------------------------------------------------------------
#     if np.argmax(PSD_shape) == 1: #if the power_spectral densities were input as subsequent rows
#         power_spectral_densities = np.transpose(power_spectral_densities) #turn them into columns
#     #--------------------------------------------------------------------
#     #Determine the measurement bandwidth and the resolution bandwidth
#     band = np.where(np.logical_and(frequency>=measurement_range[0], frequency<=measurement_range[1]))
#     df = frequency[1]-frequency[0]
#     #Sort the arrays in increasing statistical and optical power order
#     #--------------------------------------------------------------
#     #Sort the optical powers in increasing order
#     optical_powers = np.sort(optical_powers)
#     #Sort the bounds of the measurement frequency range
#     measurement_range = np.sort(measurement_range)
#     #Sort the power spectral densities in increasing statistical power order
#     statistical_powers = np.zeros((n_measurements+1, ))
#     for j in range(n_measurements+1):
#         PSD = 10**((power_spectral_densities[:, j]-30)/10) #[W/Hz]
#         statistical_powers[j] = np.mean(band*PSD[band])
#     PSD_order = np.argsort(statistical_powers)
#     print("\norder: "+str(PSD_order))
#     PSD_sorted = np.zeros(np.shape(power_spectral_densities))
#     for j in range(n_measurements+1):
#         PSD_sorted[:, PSD_order[j]] = power_spectral_densities[:, j]
#     power_spectral_densities = PSD_sorted
#     #--------------------------------------------------------------
#     #%%
#     #Check errors in the input
#     if (frequency is None) or len(frequency)==0:
#         print("\nERROR in function Iyad_homodyneDetection.vacuumNoiseLinearity(): 'frequency' must be a non-empty array.")
#         return
#     if (optical_powers is None) or len(optical_powers)==0:
#         print("\nERROR in function Iyad_homodyneDetection.vacuumNoiseLinearity(): 'optical_powers' must be a non-empty array.")
#         return 
#     if (power_spectral_densities is None) or len(power_spectral_densities)==0:
#         print("\nERROR in function Iyad_homodyneDetection.vacuumNoiseLinearity(): 'power_spectral_densities' must be a non-empty array.")
#         return  
#     if len(optical_powers) != n_measurements:
#         print("\nERROR in function Iyad_homodyneDetection.vacuumNoiseLinearity(): there are "+str(len(optical_powers))\
#               +" optical power measurements, but "+str(n_measurements)+" power vacuum spectral densities were found.")
#         return
#     if len(measurement_range) != 2:
#         print("\nERROR in function Iyad_homodyneDetection.vacuumNoiseLinearity(): 'measurement_range' must be 2-element 1D array.")
#         return  
#     if optical_powers_std is None:
#         optical_powers_std = np.zeros((n_measurements, ))
#     #%%
#     #Define the statistical powers vector
#     statistical_powers = np.zeros((n_measurements+1, ))
#     statistical_powers_std = np.zeros((n_measurements+1, ))
#     #%%
#     #Create the figures for plotting
#     figure_PSD = None
#     #Fill the figures, if required
#     if plot:
#         electronic_noise = power_spectral_densities[:, 0]
#         PSD_lowest = power_spectral_densities[:, 1]
#         PSD_lowest = 10**(PSD_lowest/10)-10**(electronic_noise/10)
#         #Power spectral densities
#         figure_PSD = plt.figure(num="Vacuum quadrature power spectral densities", figsize=default_figure_size)
#         figure_PSD.subplots_adjust(wspace=0.2, hspace=0.8)
#         #Raw power spectral densities
#         axis_raw = figure_PSD.add_subplot(2, 1, 1)
#         axis_raw.set_title("Raw quadrature measurements", fontsize=title_fontsize)
#         axis_raw.set_xlabel("frequency (MHz)", fontsize=axis_fontsize)
#         axis_raw.set_ylabel("power spectral density (dBm/Hz)", fontsize=axis_fontsize)
#         axis_raw.grid(True)
#         #Noise subtracted power spectral densities
#         axis = figure_PSD.add_subplot(2, 1, 2)
#         axis.set_title("Electronic noise-subctracted and normalized quadrature measurements", fontsize=title_fontsize)
#         axis.set_xlabel("frequency (MHz)", fontsize=axis_fontsize)
#         axis.set_ylabel("power spectral density (dB/Hz)", fontsize=axis_fontsize)
#         axis.grid(True) 
#     for j in range(n_measurements+1):
#         #Plot the raw power spectral densities
#         if plot:
#             if j == 0:
#                 label = "electronic noise"
#             else:
#                 label = "vacuum quadrature - local oscillator power: "+str(np.round(optical_powers[j-1]*1e6, 1))+" $\mu$W"
#               
#             axis_raw.plot(frequency*1e-6, power_spectral_densities[:, j], label=label, marker=default_marker)
#         #Noise-subtracted power spectral densities
#         if j>0:
#             PSD = power_spectral_densities[:, j]
#             #Subtract the electronic noise
#             PSD = 10**((PSD-30)/10)-10**((electronic_noise-30)/10) #[W/Hz]
#             #Determine the statistical power associated to the current power spectral density
#             statistical_powers[j-1] =  np.mean(df*PSD[band]) #[W]
#             statistical_powers_std[j-1] = np.std(df*PSD[band]) #[W]
#             #Plot the noise-subtracted power spectral densities
#             if plot:
#                 label = ("vacuum quadrature - local oscillator power: "+str(np.round(optical_powers[j-1]*1e6, 1))+" $\mu$W")
#                 axis.plot(frequency*1e-6, 10*np.log10(PSD/PSD_lowest), label=label, marker=default_marker)    
#     if plot:    
#         axis_raw.legend(loc="upper right", fontsize=legend_fontsize)
#         axis.legend(loc="upper right", fontsize=legend_fontsize)
#     #%%
#     #Perform the linear fit between the vacuum quadrature statistical power and the local oscillator optical power
#     #Define the affine transformation of a 1D vector
#     def affine(x, a, b):
#         """
#         This function calculates the affine transformation x |--> b+a*x
#         INPUTS:
#             - x: independent variable [a.u.] - array-like of float
#             - a: angular coefficient [a.u.] - float
#             - b: intercept [a.u.] - float
#         """
#         return (b+a*x)
#     #Exclude invalid results, where the statistical power is negative
#     print("\nstatistical_powers: "+str(statistical_powers))
#     valid_indices = np.where(statistical_powers > 0)
#     statistical_powers = statistical_powers[valid_indices] 
#     statistical_powers_std = statistical_powers_std[valid_indices] 
#     optical_powers = optical_powers[valid_indices]
#     optical_powers_std = optical_powers_std[valid_indices]
#     #Perform linear fitHH
#     linear_fit_coefficients, linear_fit_covariance = scipy.optimize.curve_fit(affine, optical_powers, statistical_powers) 
#     statistical_powers_fitted = linear_fit_coefficients[1]+linear_fit_coefficients[0]*optical_powers
#     #Build figure and plot
#     
#     figure_vacuum_linearity = None
#     if plot:
#         figure_vacuum_linearity = plt.figure(num="Vacuum linearity fit", figsize=default_figure_size)
#         #Raw power spectral densities
#         axis_vacuum_linearity = figure_vacuum_linearity.add_subplot(1, 1, 1)
#         axis_vacuum_linearity .set_title("Vacuum noise linearity", fontsize=title_fontsize)
#         axis_vacuum_linearity .set_xlabel("relative optical power ($\emptyset$)", fontsize=axis_fontsize)
#         axis_vacuum_linearity .set_ylabel("relative statistical power ($\emptyset$)", fontsize=axis_fontsize)
#         axis_vacuum_linearity.set_yscale("log")  
#         axis_vacuum_linearity.set_xscale("log")
#         #Plot both axes in log scale, normalizing by the lowest power
#         optical_powers_plot = optical_powers/optical_powers[0]
#         optical_powers_std_plot = optical_powers_std/optical_powers[0]
#         statistical_powers_plot = statistical_powers/statistical_powers[0]  
#         statistical_powers_std_plot = statistical_powers_std/statistical_powers[0]
#         statistical_powers_fitted_plot = statistical_powers_fitted/statistical_powers_fitted[0]
#         #Plot
#         axis_vacuum_linearity.errorbar(optical_powers_plot, statistical_powers_plot,\
#                                        xerr=optical_powers_std_plot, yerr=statistical_powers_std_plot,\
#                                        label="raw", marker="o")
#         axis_vacuum_linearity.plot(optical_powers_plot, statistical_powers_fitted_plot, label="linear fit", marker=default_marker)
#         axis_vacuum_linearity.legend(loc="upper right", fontsize=legend_fontsize) 
#         axis_vacuum_linearity .grid(True) 
#     #%%
#     #Build dictionaries with data to be returned
#     linear_fit = dict(zip(["coefficients", "covariance"], [linear_fit_coefficients, linear_fit_covariance]))
#     #linear_fit = None
#     figures = dict(zip(["power spectral densities", "vacuum noise linearity"], [figure_PSD, figure_vacuum_linearity]))
#     #Return
#     return statistical_powers, linear_fit, figures
# =============================================================================


def vacuumNoiseLinearity(electronic_noise, vacuum_quadratures, Fs, LO_optical_powers, measurement_sideband, plot=True):
    """
    This script takes samples of the electronic noise and vacuum quadrature measurements over time
    and does the following:
        
        - Compute the power spectral density of the measured vacuum after, subtracting the electronic noise
        - Compute the statistical power of the measured vacuum (after electronic noise subtraction) over the electronic noise
        - Fit a linear curve into the statistical powers as function of local oscillator optical powers, to check that the measurements
          represent the measurement of the vacuum quadrature 
        
    
    INPUTS
    ----------
    electronic_noise : array-like of float 
        The electronic noise 
    vacuum_quadratures : 2-D array like of float
        The measured vacuum quadrature
    Fs : float (>0)
        The acquisition sampling rate [Hz]
    measurement_sideband : 1-D array like of float of length 2 
        The boundary of the measurement sideband [Hz]
    plot : boolean
        If true, plots will be shown

    OUTPUTS
    -------
    PSDs: 2-D array like of float
        The power spectral densities of the measurements, with keys equal to the measured LO optical powers
    statistical_powers: 2D array-like of float
        The statistical powers of the measurements. The electronic noise statistical power is subtracted from the vacuum quadrature measurements.
        The first column contains the estimated statistical powers, the second column contains the standard deviation of the estimates
    PDFs: 2-D array like of float
        The histogram-based probability density functions of the measurements, with keys equal to the measured LO optical powers
    """
    #Compute the power spectral densities
    n_samples = len(electronic_noise)
    LO_optical_powers = np.asarray(LO_optical_powers)
    n_LO_powers = len(LO_optical_powers)
    #Subtract the mean value
    electronic_noise -= np.mean(electronic_noise)
    for j in range(n_LO_powers):
        vacuum_quadratures[:, j] -= np.mean(vacuum_quadratures[:, j])
    #Compute the power spectra densities using the Welch periodogram
    n_samples_per_segment = int(n_samples/2000)
    n_samples_overlap = 20
    frequency, electronic_noise_PSD = signal.welch(electronic_noise, fs=Fs, nperseg=n_samples_per_segment, noverlap=n_samples_overlap)
    #Compute the vacuum quadrature PSD
    PSDs = np.zeros((len(frequency), n_LO_powers+1))
    for j in range(n_LO_powers):
        _, PSDs[:, j+1] = signal.welch(vacuum_quadratures[:, j], fs=Fs, nperseg=n_samples_per_segment, noverlap=n_samples_overlap)
        PSDs[:, j+1] -= electronic_noise_PSD #subtract the electronic noise
    PSDs[:, 0] = electronic_noise_PSD
    PSDs /= 50
    #Compute the statistical powers
    statistical_powers = np.zeros((n_LO_powers+1, 2))
    b_filter = signal.firwin(1001, cutoff=measurement_sideband, fs=Fs, pass_zero=False) #band-pass filter coefficients [a.u.]
    #Electronic noise
    statistical_power_en = sampleVariance(signal.filtfilt(b_filter, 1, electronic_noise))
    statistical_powers[0, 0] = statistical_power_en.nominal_value
    statistical_powers[0, 1] = statistical_power_en.std_dev
    #Vacuum quadratures
    for j in range(n_LO_powers):
        statistical_power = sampleVariance(signal.filtfilt(b_filter, 1, vacuum_quadratures[:, j]))
        statistical_power -= statistical_power_en
        statistical_powers[j+1, 0] = statistical_power.nominal_value
        statistical_powers[j+1, 1] = statistical_power.std_dev
    #Exclude invalid results, where the statistical power is negative
    print("\nstatistical_powers: "+str(statistical_powers))
    #Perform linear fit
    linear_fit_coefficients, linear_fit_covariance = scipy.optimize.curve_fit(linear, LO_optical_powers, statistical_powers[1:, 0]) 
    statistical_powers_fitted = linear_fit_coefficients[0]*LO_optical_powers
    linear_fit = dict(zip(["coefficients", "covariance"], [linear_fit_coefficients, linear_fit_covariance]))
    #Plot results if requested
    if plot:
        #Plot the power spectral densities
        figure_PSD = plt.figure(num="Vacuum quadrature power spectral densities", figsize=default_figure_size)        
        figure_PSD.subplots_adjust(wspace=0.2, hspace=0.8)
        #Raw power spectral densities
        axis_raw = figure_PSD.add_subplot(1, 1, 1)
        axis_raw.set_title("Raw quadrature measurements", font=title_font)
        axis_raw.set_xlabel("frequency (MHz)", font=axis_font)
        axis_raw.set_ylabel("power spectral density (dBm/Hz)", font=axis_font)
        axis_raw.grid(True)
        for j in range(n_LO_powers+1):
            if j == 0:
                label = "electronic noise"
            else:
                label = "vacuum quadrature - local oscillator power: "+str(np.round(LO_optical_powers[j-1]*1e3, 1))+" mW"
                  
            axis_raw.plot(frequency*1e-6, 10*np.log10(PSDs[:, j]/1e-3), label=label, marker='o', linestyle='None')
        axis_raw.legend(prop=legend_font)
        #Plot the statistical powers VS LO optical power
        figure_vacuum_linearity = plt.figure(num="Vacuum linearity fit", figsize=default_figure_size)
        #Raw power spectral densities
        axis_vacuum_linearity = figure_vacuum_linearity.add_subplot(1, 1, 1)
        axis_vacuum_linearity.set_title("Vacuum noise linearity", font=title_font)
        axis_vacuum_linearity.set_xlabel("relative optical power (dB)", font=axis_font)
        axis_vacuum_linearity.set_ylabel("relative statistical power (dB)", font=axis_font)
        #Plot both axes in log scale, normalizing by the lowest power
        LO_optical_powers_plot = 10*np.log10(LO_optical_powers/LO_optical_powers[0])
       # optical_powers_std_plot = LO_optical_powers_std/LO_optical_powers[0]
        statistical_powers_plot = 10*np.log10(statistical_powers[1:, 0]/statistical_powers[1, 0])
        statistical_powers_std_plot = 10/(np.log10(np.exp(1))*statistical_powers[1:, 0]/statistical_powers[1, 0])*statistical_powers[1:, 1] #error propagation formula for statistical powers expressed in dB
        statistical_powers_fitted_plot = 10*np.log10(statistical_powers_fitted/statistical_powers_fitted[0])
        #Plot
        axis_vacuum_linearity.errorbar(LO_optical_powers_plot, statistical_powers_plot,\
                                       yerr=statistical_powers_std_plot,\
                                       label="experiment", marker="o", linestyle='None', linewidth=3)
        axis_vacuum_linearity.plot(LO_optical_powers_plot, statistical_powers_fitted_plot, label="linear fit", marker=default_marker)
        axis_vacuum_linearity.legend(loc="upper right", prop=legend_font) 
        axis_vacuum_linearity.grid(True) 
        
    return PSDs, statistical_powers, linear_fit

        