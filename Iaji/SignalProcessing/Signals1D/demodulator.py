#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:52:42 2020

@author: jiedz
"""

#%%
#imports
import numpy as np
from scipy import signal, optimize
from .correlator import correlator
#%%
class demodulator:
    """
    This class defines a demodulator object, useful for demodulating functions of 
    one independent variable (e.g., time), and for demodulating pairs of functions.
    The input functions are assumed to be 'discrete-time', where the 
    independent variable varies by constant increments, called 'sampling period'.
    
    A simple demodulation system is composed of two transformations:
        1) Downmixing: the input signal is multiplied by a sinusoidal signal, having a constant frequency called 'downmixing frequency', and a constant phase called 'downmixing phase';
        2) Low-pass filtering: the output of the downmixing system is input to a low-pass fiter, with specified cutoff frequency.
    """
    
    def __init__(self, sampling_frequency, downmixing_frequency=None, downmixing_phase=0, cutoff_frequency=None):
            """
            INPUTS
            -----------------
            sampling_period: float (>0)
                default sampling frequency at which the demodulator works [a.u.].
                
            downmixing_frequency: float (>0)
                default Fourier downmixing frequency [same units as 'sampling_frequency']. It must not exceed half the sampling frequency, by the Nyquist theorem.
            
            downmixing_phase: float
                default downmixing phase [rad].
                
            cutoff_frequency: float (>0)
                default Fourier cutoff frequency [same units as 'sampling_frequency']. It must not exceeed half the sampling frequency, by the Nyquist theorem.
                   
                
            OUTPUS
            -----------------
            None
            """     
            #Check errors in the input
            if sampling_frequency<0:
                raise ValueError("'sampling_frequency' must be a positive real number.")
            if downmixing_frequency<0:
                raise ValueError("'downmixing_frequency' must be a positive real number.")
            if cutoff_frequency<0:
                raise ValueError("'cutoff_frequency' must be a positive real number.")
            if downmixing_frequency>sampling_frequency/2:
                raise ValueError("'downmixing_frequency' must not exceed half of 'sampling_frequency'.")
            if cutoff_frequency>sampling_frequency/2:
                raise ValueError("'cutoff_frequency' must not exceed half of 'sampling_frequency'.")
            #Initialize internal variables
            self.Fs = sampling_frequency 
            self.downmixing_frequency = downmixing_frequency
            self.downmixing_phase = downmixing_phase
            self.cutoff_frequency = cutoff_frequency
            self.b_low_pass = None #coefficients of the demodulation low-pass FIR filter
            if cutoff_frequency:
                #Design the demodulation low-pass FIR filter, with the specified sampling frequency and  cutoff frequency
                self.b_low_pass = signal.firwin(1001, self.cutoff_frequency, nyq=0.5*self.Fs, window='hamming', scale=False) #demodulation low-pass filter coefficients [a.u.]
        

    def demodulate(self, f):
            """
            This function demodulates a signal.
            
            INPUTS
            ----------
            f : array-like of float
                signal to be demodulated [a.u.].
        
            OUTPUS
            -------
            f_demodulated: array-like of float
                demodulated signal [same units as 's']
            """
            n_samples = len(f) #number of samples 
            downmixing_signal = np.cos(2*np.pi*self.downmixing_frequency*np.array(range(n_samples))/self.Fs+self.downmixing_phase) #[a.u.]
            #Downmix and low-pass filter
            return signal.filtfilt(self.b_low_pass, 1, f*downmixing_signal) #[a.u.]
        
    def demodulatePiecewise(self, f, n_piece, downmixing_frequencies=None, downmixing_phases=None, n_offset=0, n_exclude=0):
            """
            This function demodulates a signal in a piecewise fashion, 
            using the function demodulate(). The number of samples of each piece is 'n_piece', while the start index for the recorrelation procedure is 'offset'.
            Depending on the nature of the signals, the user may choose to discard 'n_exclude' samples before and after the boundary between contiguous pieces.
            
            INPUTS
            ----------
            f : array-like of float
                signal to be demodulated [a.u.].
            n_piece : int
                number of samples of each piece of the input signals to be demodulated.
            downmixing_frequencies: array-like of float (elements>0)
                the downmixing frequencies at which the corresponding pieces of the signals must be demodulated
            downmixing_phases: array-like of float (elements>0)
                the downmixing phases which the corresponding pieces of the signals must be demodulated with          
            n_offset : int
                number of samples after which the demodulation starts.
            n_exclude : int
                number of samples, before and after the separation points of adjacent pieces, to be discarded before demodulating
                
        
            OUTPUS
            -------
            f_demodulated: array-like of float
                demodulated signal [same units as 's']
            """
             
            n_samples = len(f)
            n_pieces = int(np.ceil((n_samples-n_offset)/n_piece)) #total number of pieces of signals to be demodulated
        
            if downmixing_frequencies is None:
                downmixing_frequencies = self.downmixing_frequency*np.ones((n_pieces,))
            if downmixing_phases is None:
                downmixing_phases = self.downmixing_phase*np.ones((n_pieces,))
            
            downmixing_frequencies = np.atleast_1d(downmixing_frequencies)
            downmixing_phases = np.atleast_1d(downmixing_phases)
            #Check errors in the input
            #--------------------
            if len(downmixing_frequencies) != n_pieces:
                raise ValueError("'downmixing_frequencies' must have length equal to 'n_piece'")
            if len(downmixing_phases) != n_pieces:
                raise ValueError("'downmixing_phases' must have length equal to 'n_piece'")    
            #--------------------
            f_demodulated = np.zeros((n_samples,))
            #Save the default values of the downmixing frequency and phase
            downmixing_frequency_default = self.downmixing_frequency
            downmixing_phase_default = self.downmixing_phase
            
            for j in range(n_pieces):
                start_index = n_offset + (j*n_piece) + n_exclude #start index of the current piece
                if j<n_pieces-1:
                    stop_index = n_offset + ((j+1)*n_piece-1) - n_exclude #stop index of the current piece
                else:
                    stop_index = n_samples-1   
                #Demodulate
                self.setDownmixingFrequency(downmixing_frequencies[j])
                self.setDownmixingPhase(downmixing_phases[j])
                f_demodulated[start_index:stop_index+1] = self.demodulate(f[start_index:stop_index+1])    
           
            #Reset the downmixing frequency and phase to the default values
            self.setDownmixingFrequency(downmixing_frequency_default)
            self.setDownmixingPhase(downmixing_phase_default)           
            return f_demodulated #[a.u.]


    def demodulate2(self, relative_downmixing_phase, f_1, f_2):
        """
        This function demodulates two signals at the same downmixing frequency, with a relative phase between the respective downmixing signals.
        
        INPUTS
        ----------
        relative_downmixing_phase : float
            phase difference between the downmixing sinusoidal signals for 'f_1' and 'f_2'[rad].
        downmixing_frequency : float
            frequency of the downmixing sinusoidal signals [Hz].
        f_1 : array-like of float
            time signal to be demodulated [a.u.].
        f_2 : array-like of float
            time signal to be demodulated [a.u.].
    
        OUTPUS
        -------
        f_1_demodulated: array-like of float
            f_1 demodulated.
        f_2_demodulated: array-like of float
            f_2 demodulated
        """
        f_1_demodulated = self.demodulate(f=f_1)
        self.setDownmixingPhase(self.downmixing_phase+relative_downmixing_phase) #increment the downmixing phase for the second signal
        f_2_demodulated = self.demodulate(f=f_2)
        self.setDownmixingPhase(self.downmixing_phase-relative_downmixing_phase) #reset the downmixing phase to the default one
        return f_1_demodulated, f_2_demodulated
    

    def demodulateDownsampleAndCorrelate(self, relative_downmixing_phase, f_1, f_2):
        """
        This function performs the following on two signals 'f_1' and 'f_2':
            1) Demodulate them with a relative phase between the respetive downmixing sinusoidal signals;
            2) Re-correlate the two signals, i.e., align the signals using the normalized correlation function;
            3) Downsample the re-correlated signals by an integer factor, corresponding to the low-pass filter cutoff frequency;
            4) Compute the covariance of f_2, given f_1. The higher it is, the more correlated the two signals.
        
        INPUTS
        ----------
        relative_downmixing_phase : float
            phase difference between the downmixing sinusoidal signals for 'f_1' and 'f_2'[rad].
        f_1 : array-like of float
           signal to be demodulated [a.u.].
        f_2 : array-like of float
            signal to be demodulated [a.u.].
    
        OUTPUS
        -------
        covariance: float
            the covariance between 'f_1' and 'f_2' after demodulation and downsampling. The higher it is, the higher the correlation between 
        """
        #1) Demodulate
        f_1_demodulated, f_2_demodulated = self.demodulate2(relative_downmixing_phase=relative_downmixing_phase, f_1=f_1, f_2=f_2)
        #Initialize a correlator object
        correlator_obj = correlator(f_1=f_1_demodulated, f_2=f_2_demodulated, sampling_period = 1/self.Fs)
        #2) Re-correlate the demodulated signals
        correlator_obj.recorrelateSignals(delete_correlation_function=True)
        #Get the re-correlated time signals
        f_1_recorrelated, f_2_recorrelated, _ = correlator_obj.getRecorrelatedSignals()
        #3) Downsample
        downsampling_factor = int(self.Fs/(2*self.cutoff_frequency)) 
        f_1_recorrelated = f_1_recorrelated[::downsampling_factor]
        f_2_recorrelated = f_2_recorrelated[::downsampling_factor]
        #Make sure all the time signals have the same length
        n_samples = correlator_obj.n_samples_recorrelated
        #4) Compute the conditional variance
        covariance = np.abs(1/n_samples * np.sum(f_1_recorrelated*f_2_recorrelated)) #covariance between the recorrelated signals
        return np.abs(covariance)
    
    
    def demodulateAndCorrelate(self, relative_downmixing_phase, f_1, f_2):
        """
        This function performs the following on the signals 'f_1' and 'f_2':
            1) Demodulate them with a relative phase between the respetive downmixing sinusoidal signals;
            2) Compute the modulus of the normalized correlation coefficient between the two time signals;
        
        INPUTS
        ----------
        relative_downmixing_phase : float
            phase difference between the downmixing sinusoidal signals for 'signal_1' and 'signal_2'[rad].
        f_1 : array-like of float
            input signal.
        f_2 : array-like of float
            input signal.
    
        OUTPUTS
        -------
        correlation_coefficient: float
            1-complementary of the normalized correlation coefficient between the demodulated signals.
        """
        #Demodulate
        f_1_demodulated, f_2_demodulated = self.demodulate2(relative_downmixing_phase, f_1, f_2)
        #Initialize a correlator object
        correlator_obj = correlator(signal_1=f_1_demodulated, signal_2=f_2_demodulated, sampling_period=1/self.Fs)
        #Compute the normalized correlation coefficient
        correlator_obj.computeCorrelationCoefficient(delete_correlation_function=True)
        #Get the normalized correlation coefficient
        correlation_coefficient = correlator_obj.getCorrelationCoefficient()
        return abs(correlation_coefficient)
    
    def demodulateAndDecorrelate(self, relative_downmixing_phase, f_1, f_2):
        """
        This function performs the following on the signals 'f_1' and 'f_2':
            1) Demodulate them with a relative phase between the respetive downmixing sinusoidal signals;
            2) Compute the 1-complement of the normalized correlation coefficient between the two time signals;
        
        INPUTS
        ----------
        relative_downmixing_phase : float
            phase difference between the downmixing sinusoidal signals for 'signal_1' and 'signal_2'[rad].
        f_1 : array-like of float
            input signal.
        f_2 : array-like of float
            input signal.
    
        OUTPUTS
        -------
        decorrelation_coefficient: float
            1-complementary of the normalized correlation coefficient between the demodulated signals.
        """
        
        return 1-self.demodulateAndCorrelate(relative_downmixing_phase, f_1, f_2)
    
    def demodulateDownsampleAndDecorrelate(self, relative_downmixing_phase, f_1, f_2):
        """
        This function performs the following on the signals 'f_1' and 'f_2':
            1) Demodulate them with a relative phase between the respetive downmixing sinusoidal signals;
            2) Re-correlate the two signals, i.e., align the signals using the normalized correlation function;
            3) Downsample the re-correlated signals by an integer factor, corresponding to the low-pass filter cutoff frequency;
            4) Compute the conditional variance of f_2, given f_1. The lower it is, the more correlated the two signals.
        
        INPUTS
        ----------
        relative_downmixing_phase : float
            phase difference between the downmixing sinusoidal signals for 'f_1' and 'f_2'[rad].
        downmixing_frequency : float
            frequency of the downmixing sinusoidal signals [Hz].
        b_low_pass : array-like of float
            demodulation low-pass filter coefficients [a.u.].
        downsampling_factor : float or integer
            downsampling factor [adimensional]. It will be truncated to the closest lower integer.
        f_1 : array-like of float
            input signal 
        f_2 : array-like of float
            input signal 
    
        OUTPUTS
        -------
        conditional_variance: float
            the residual variance of 'f_2', given the knowlegde on 'f_1' after downsampling. The lower it is, the higher the correlation between 
        """ 
    
        #1) Demodulate
        f_1_demodulated, f_2_demodulated = self.demodulate2(relative_downmixing_phase=relative_downmixing_phase, f_1=f_1, f_2=f_2)
        #Initialize a correlator object
        correlator_obj = correlator(signal_1=f_1_demodulated, signal_2=f_2_demodulated, sampling_period = 1/self.Fs)
        #2) Re-correlate the demodulated signals
        correlator_obj.recorrelateSignals(delete_correlation_function=True)
        #Get the re-correlated time signals
        f_1_recorrelated, f_2_recorrelated, _ = correlator_obj.getRecorrelatedSignals()
        #3) Downsample
        downsampling_factor = int(self.Fs/(2*self.cutoff_frequency)) 
        f_1_recorrelated = f_1_recorrelated[::downsampling_factor]
        f_2_recorrelated = f_2_recorrelated[::downsampling_factor]
        #Make sure all the time signals have the same length
        n_samples = len(f_1_recorrelated)
        #4) Compute the conditional variance
        covariance = 1/n_samples * np.sum(f_1_recorrelated*f_2_recorrelated) #covariance between the recorrelated signals
        return -abs(covariance)
    
    
    def optimalRelativeDownmixingPhase(self, f_1, f_2):
        """
        This function computes the optimum relative demodulation phase between two signals, so as to maximize the correlation between them.
        
        INPUTS
        ----------
    
        f_1 : array-like of float
            input signal 
        f_2 : array-like of float
            input signal 
    
        OUTPUS
        -------
        optimal_downmixing_phase : float
            phase difference between the downmixing sinusoidal signals for 'f_1' and 'f_2' [rad].
        """   
        optimal_downmixing_phase = optimize.fminbound(self.demodulateAndDecorrelate, x1=0, x2=np.pi, args=(f_1, f_2))
        return optimal_downmixing_phase
    
    
    
    def optimalRelativeDownmixingPhasePiecewise(self, f_1, f_2, n_piece, n_offset=0, n_exclude=0):
        """
        This function computes the optimum relative demodulation phase between two signals, so as to maximize the correlation between them, in a piecewise fashion, 
        using the function optimalRelativeDownmixingPhase(). The number of samples of each piece is 'n_piece', while the start index for the recorrelation procedure is 'n_offset'.
        Depending on the nature of the signals, the user may choose to discard 'n_exclude' samples before and after the boundary between contiguous pieces.
        
        INPUTS
        ----------
    
        f_1 : array-like of float
            input signal 
        f_2 : array-like of float
            input signal 
        n_piece : int
            number of samples of each piece of the input signals which the relative downmixing phase is optimized for
        n_offset : int
            number of samples after which the optimization procedure starts
        n_exclude : int
            number of samples, before and after the separation points of adjacent pieces, to be discarded before performing the optimization
    
        OUTPUS
        -------
        optimal_downmixing_phases : array-like of float
            phase differences between corresponding pieces of the downmixing sinusoidal signals for 'f_1' and 'f_2' [rad].
        """ 
        if len(f_1) != len(f_2):
            raise ValueError("The two input signals must have equal length.")
            
        n_samples = len(f_1)
        n_pieces = int(np.ceil((n_samples-n_offset)/n_piece)) #total number of pieces of signals for which the optimization procedure will be performed
        optimal_downmixing_phases = np.empty((n_pieces,))
        for j in range(n_pieces):
            start_index = n_offset + (j*n_piece) + n_exclude #start index of the current piece
            if j<n_pieces-1:
                stop_index = n_offset + ((j+1)*n_piece-1) - n_exclude #stop index of the current piece
            else:
                stop_index = n_samples-1
            optimal_downmixing_phases[j]= self.optimalRelativeDownmixingPhase(f_1[start_index:stop_index+1], f_2[start_index:stop_index+1])
        return optimal_downmixing_phases 
    

    def setDownmixingFrequency(self, downmixing_frequency):
        """
        This function sets as desired downmixing signal to the demodulator.

        INPUTS
        ----------
        downmixing_frequency : float (>0)
            downmixing frequency to be set [same units as 'self.Fs'].

        OUTPUTS
        -------
        None.

        """
        self.downmixing_frequency = downmixing_frequency
    
    def setDownmixingPhase(self, downmixing_phase):
        """
        This function sets as desired downmixing signal to the demodulator.

        INPUTS
        ----------
        downmixing_phase : float
            downmixing phase to be set [rad].

        OUTPUTS
        -------
        None.

        """
        self.downmixing_phase = downmixing_phase
        
    def setLowPassFilterCoefficients(self, b_low_pass):
        """
        This function sets the desired low-pass FIR filter coefficients to the demodulator, 
        and re-estimates the cutoff frequency of the filter, by computing the half-width-half maximum
        of the modulus of the corresponding frequency response. The filter is assumed to be defined on the 
        same sampling frequency as the internal sampling frequency of the demodulator ('self.Fs').

        INPUTS
        ----------
        b_low_pass : array-like of float
            low-pass filter coefficients to be set to be set [rad].

        OUTPUTS
        -------
        None.

        """
        self.b_low_pass = b_low_pass
        #Compute the frequency response associated to the input low-pass filter coefficients
        frequency, H = signal.freqz(b_low_pass, worN=2000, fs=self.Fs) #frequency response of the demodulation filter
        #Estimate the cutoff frequency
        H = abs(H)
        self.cutoff_frequency = np.max(frequency[np.where(H>=1/2*np.max(H))]) #full-width-half-maximum of the filter frequency response
        
