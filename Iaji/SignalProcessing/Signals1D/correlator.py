#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:04:29 2020

@author: jiedz


"""

#%%
#imports
import numpy
from scipy import signal
#%%
class correlator:
    """
    This class defines the 'correlator' object. A correlator is an object that can host two functions of one real independent variable ('signal_1' and 'signal_2'), 
    and manages the correlation and delay between the two. The two functions are assumed to be discrete in the independent variable,
    i.e., the independent variable varies by constant increments, called 'sampling period'.
    """
    
    def __init__(self, signal_1, signal_2, sampling_period):
        """
        INPUTS:
            
            - signal_1: input function 1 [a.u.] - array-like of float;
            - signal_2: input function 2 [a.u.] - array-like of float;
            - sampling_period: sampling period [a.u.] - float (>0)
        """
        #Check for errors in the input
        #----------------------------------------------------------
        #Empty or None signals
        if len(signal_1)==0:
            print("\nERROR in function correlator.__init__(): 'signal_1' cannot be empty or None.")
        if len(signal_2)==0:
            print("\nERROR in function correlator.__init__(): 'signal_2' cannot be empty or None.") 
        #Non-positive sampling period
        if sampling_period <= 0: 
            print("\nERROR in function correlator.__init__(): 'sampling_period' must be a positive real number.") 
        #----------------------------------------------------------
        #Make the input vectors numpy arrays
        self.signal_1 = numpy.atleast_1d(signal_1) #[a.u.]
        self.signal_2 = numpy.atleast_1d(signal_2) #[a.u.]
        self.Xs = sampling_period #sampling period [same units as 'x']
        self.n_samples = numpy.max([len(signal_1), len(signal_2)]) #maximum number of samples
        #Add samples to the shortest signal, through zero-pad
        #---------------------------------------------------------
        #Signal 1
        Delta_n = numpy.max([0, self.n_samples-len(self.signal_1)])
        self.signal_1 = numpy.concatenate((self.signal_1, numpy.zeros((Delta_n, ))))
        #Signal 2
        Delta_n = numpy.max([0, self.n_samples-len(self.signal_2)])
        self.signal_2 = numpy.concatenate((self.signal_2, numpy.zeros((Delta_n, )))) 
        #---------------------------------------------------------
        #Define variables related to correlation
        self.correlation_function = None; #normalized correlation function  [adimensional]
        self.lags = None; #lags [same units as 'x']
        self.correlation_coefficient = None; #normalized correlation coefficient [adimensional]
        self.delay = None; #delay between the input signals [same units as 'x']
        self.coherence_width = None #half-width-half-maximum of the correlation function [same units as 'x']
        self.x_max = (self.n_samples-1)*self.Xs #maximum value for the independent variable [same units as Xs]
        
        #Define the re-correlated versions of the input signals
        #When self.correlate() is called, these variables will become the versions of the input signals
        #that are aligned in x.
        self.signal_1_recorrelated = None
        self.signal_2_recorrelated = None
        self.n_samples_recorrelated = None
      
    def correlationFunction(self):
        self.computeCorrelationFunction()
        return self.getCorrelationFunction()
    
    def covarianceFunction(self):
        self.computeCorrelationFunction()
        correlation, lags = self.getCorrelationFunction()
        return numpy.std(self.signal_1)*numpy.std(self.signal_2)*self.n_samples*correlation, \
            lags
        
    def computeCorrelationFunction(self):
        """
        This function computes the normalized correlation function between the two signals, as well as the 
        lags, used as the independent variable for the correlation function.
        """  
        #Compute and subtract the mean value of the input signals
        signal_1_mean = numpy.mean(self.signal_1)
        signal_2_mean = numpy.mean(self.signal_2)
        self.signal_1 -= signal_1_mean
        self.signal_2 -= signal_2_mean
        self.correlation_function = signal.correlate(self.signal_1, self.signal_2)/(self.n_samples*numpy.std(self.signal_1)*numpy.std(self.signal_2)) #normalized correlation function [adimensional]
        self.lags = numpy.flipud(numpy.linspace(start=-self.x_max+self.Xs/2, stop=self.x_max-self.Xs/2, num=len(self.correlation_function))) #correlation delays [same units as 'x']
        #Add the mean values back
        self.signal_1 += signal_1_mean
        self.signal_2 += signal_2_mean
        
    def getCorrelationFunction(self):
        """
        This function returns the normalized correlation function and the associated lags.
        
        OUTPUTS:
            
            - correlation_function: normalized correlation function [adimensional] - array-like of float
            - lags: delays, used as independent variable for the correlation function [same units as 'x'] - array-like of float
        """
        return self.correlation_function, self.lags
    
    def correlationCoefficient(self, delete_correlation_function=False):
        self.computeCorrelationCoefficient(delete_correlation_function)
        return self.getCorrelationCoefficient()
    
    def computeCorrelationCoefficient(self, delete_correlation_function=False):
        """
        This function computes the normalized correlation coefficient. If the correlation function has not been computed yet, 
        it will compute it.
        
        INPUTS:
            
            - delete_correlation_function: if set to True, a newly computed correlation function is deleted - boolean
        """
        #If the correlation coefficient has not been computed yet, compute it
        could_delete = False
        if self.correlation_function is None: 
            self.computeCorrelationFunction()
            could_delete = True
        index = numpy.argmax(numpy.abs(self.correlation_function))
        self.correlation_coefficient = self.correlation_function[index] #correlation coefficient [adimensional]
        #If the user wanXs to delete the correlation function, and if it was newly computed, then delete it
        if delete_correlation_function and could_delete:
            self.correlation_function, self.lags = [None for j in range(2)]
            
        
    def getCorrelationCoefficient(self):
        """
        This function returns the normalized correlation coefficient
        
        OUTPUTS:
            
            - correlation_coefficient: normalized correlation coefficient [adimensional] - float (>=0)
        """
        return self.correlation_coefficient   
    
    def computeCoherenceWidth(self, delete_correlation_function=False):
        """
        This function computes coherence width, defined as the half-width-half-maximum of the modulus of the correlation function. 
        If the correlation function has not been computed yet, it will compute it.
        
        INPUTS:
            
            - delete_correlation_function: if set to True, a newly computed correlation function is deleted - boolean
        """
        #If the correlation coefficient has not been computed yet, compute it
        could_delete = False
        if self.correlation_function is None: 
            self.computeCorrelationFunction()
            could_delete = True
        self.coherence_width = -numpy.min(self.lags[numpy.abs(self.correlation_function)>=numpy.max(abs(self.correlation_function))/2])
        #If the user wanXs to delete the correlation function, and if it was newly computed, then delete it
        if delete_correlation_function and could_delete:
            self.correlation_function, self.lags = [None for j in range(2)] 
            
    def getCoherenceWidth(self):
        """
        This function returns the coherence width. 
        
        OUTPUTS:           
            - coherence_width: coherence width, defined as the half-width-half-maximum of the modulus of the correlation function [same units as 'x'] - float (>=0)
        """
        return self.coherence_width 
    
    
    def computeDelay(self, delete_correlation_function=False):
        """
        This function computes the delay between the two signals, by extremizing the correlation function 
        
        INPUTS:
            
            - delete_correlation_function: if set to True, a newly computed correlation function is deleted - boolean
        """
        #If the correlation coefficient has not been computed yet, compute it
        could_delete = False
        if self.correlation_function is None: 
            self.computeCorrelationFunction()
            could_delete = True
        self.delay =self.Xs*(numpy.argmax(numpy.abs(self.correlation_function))-self.n_samples+1)
        #If the user wanXs to delete the correlation function, and if it was newly computed, then delete it
        if delete_correlation_function and could_delete:
            self.correlation_function, self.lags = [None for j in range(2)] 
            
    def getDelay(self):
        """
        This function returns the delay between the input signals. 
        
        OUTPUTS:           
            - delay: delay between the input signals [same units as 'x'] - float (>=0)
        """
        return self.delay
    
    def recorrelate(self, delete_correlation_function=False):
        self.recorrelateSignals(delete_correlation_function)
        return self.getRecorrelatedSignals()
    
    def recorrelateSignals(self, delete_correlation_function=False):
        """
        This function overlaps the input signals in x.
        
        INPUTS:
            
            - delete_correlation_function: if set to True, a newly computed correlation function is deleted - boolean
        """
        #If the correlation coefficient and delay has not been computed yet, compute it
        if (self.delay is None) or (self.correlation_coefficient is None): 
            #The delay and correlation coefficient are newly computed
            self.computeDelay(delete_correlation_function)
            self.computeCorrelationCoefficient(delete_correlation_function)  
        #Compute the delay in samples
        delay_samples = abs(int(numpy.ceil(self.delay/self.Xs)))
        #sign = numpy.sign(self.correlation_coefficient)
        self.n_samples_recorrelated = self.n_samples-delay_samples #number of correlated samples
        #Select the portions of the signals that are overlapped, according to the delay         
        if self.delay > 0 :       
            self.signal_1_recorrelated = self.signal_1[delay_samples:]
            self.signal_2_recorrelated = self.signal_2[0:self.n_samples_recorrelated]
        else:
            self.signal_2_recorrelated = self.signal_2[delay_samples:]
            self.signal_1_recorrelated = self.signal_1[0:self.n_samples_recorrelated]
            
       
            
    def getRecorrelatedSignals(self):
        """
        This function returns the delay between the input signals. 
        INPUTS
        -------------------
        
        None
        
        OUTPUTS
        ---------------------
        
            - signal_1_recorrelated: re-correlated signal 1 [a.u.] - array-like of float
            - signal_2_recorrelated: re-correlated signal 2 [a.u.] - array-like of float
            - x_recorrelated: x vecotr for the re-correlated signals [same units as 'x'] - array-like of float
        """
        x_recorrelated = numpy.linspace(start=0, stop=(self.n_samples_recorrelated-1)*self.Xs, num=self.n_samples_recorrelated)
        return self.signal_1_recorrelated, self.signal_2_recorrelated, x_recorrelated
        
    
    def recorrelateSignalsPiecewise(self, n_piece, n_offset=0, n_exclude=0):
        """
        This function overlaps the input signals along the independent variable in a piecewise fashion, i.e., 
        it overlaps corresponding pieces of the input signals, using the function recorrelateSignals().
        The number of samples of each piece is 'n_piece', while the start index for the recorrelation procedure is 'offset'.
        Depending on the nature of the signals, the user may choose to discard 'n_exclude' samples before and after the boundary between contiguous pieces.
        
        INPUTS
        ---------------
        
        n_piece : int
            number of samples of each piece of signal to be recorrelated.
        
        n_offset: int
            number of samples after which the signals need to be recorrelated.
            
        n_exclude : int
            number of samples, before and after the separation points of adjacent pieces, to be discarded before overlapping the signals.
        
        OUTPUTS
        --------------
        
        None
        """
        
        #Create empty signals
        signal_1_recorrelated, signal_2_recorrelated, x_recorrelated = [numpy.empty((0,)) for s in range(3)]

        n_pieces = int(numpy.ceil((self.n_samples-n_offset)/n_piece)) 
        #Construct the temporary signals containing the parXs to be recorrelated, discarding 2*n_exclude samples from the boundaries between pieces
        start_signal = n_offset #starting index for the parXs of the signals to be recorrelated
        for j in range(n_pieces):
            start_index = start_signal + (j*n_piece) + n_exclude #start index of the current piece
            if j<n_pieces-1:
                stop_index = start_signal + ((j+1)*n_piece-1) - n_exclude #stop index of the current piece
            else:
                stop_index = self.n_samples-1
            #Define a new correlator to overlap the two signal pieces in x
            corr = correlator(signal_1=self.signal_1[start_index:stop_index+1], signal_2=self.signal_2[start_index:stop_index+1], sampling_period=self.Xs)
            #Overlap the two pieces in x
            corr.recorrelateSignals(delete_correlation_function=True)
            #Append the new re-correlated pieces
            signal_1_temp, signal_2_temp, _ = corr.getRecorrelatedSignals()
            signal_1_recorrelated = numpy.concatenate((signal_1_recorrelated, signal_1_temp))
            signal_2_recorrelated = numpy.concatenate((signal_2_recorrelated, signal_2_temp))
            
        self.n_samples_recorrelated = len(signal_1_recorrelated)
        self.signal_1_recorrelated = signal_1_recorrelated
        self.signal_2_recorrelated = signal_2_recorrelated
            
        
    
                                            
        
        
        
        
        
        
        
        
            
            
            

        

