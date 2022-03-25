#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:44:00 2021

@author: jiedz

This module defines objects and functions to perform tomographic state
reaconstruction of quantum states
"""

#%%
#imports
import numpy 
import scipy as sp
import scipy.special as sps
from matplotlib import pyplot as plt
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator import Utilities as qfutils
#%%
def homodyneFockPOVM(n_max, x, theta):
    """
    This function computes, the homodyne detection POVM in the Fock basis, up
    to order "N" (i.e, in a reduced-dimension Hilbert space). 
    
    For a quadrature value x, the (n, m) element of the POVM matrix is
    <X|m>*<X|n>. The expression of <X|k> depends on the associated LO phase
    "theta"
    
    n_max: positive integer
    x: scalar
    theta: scalar
    """
    #For each element of x, a POVM matrix is computed
    if n_max < 2:
        raise InvalidDimensionError('The Hilbert space dimension must be at least 2')
    POVM = numpy.zeros((n_max+1, ), dtype=complex)
    for n in numpy.arange(n_max):
        if n==0:
            POVM[n] = 1/(numpy.sqrt(numpy.sqrt(numpy.pi)))*numpy.exp(-0.5*x**2)
        elif n==1:
            POVM[n] = x*numpy.sqrt(2)*numpy.exp(-1j*theta) * POVM[0]
        else:
            POVM[n] = numpy.exp(-1j*theta)/numpy.sqrt(n)*(numpy.sqrt(2)*x*POVM[n-1] - numpy.exp(-1j*theta)*numpy.sqrt(n-1)*POVM[n-2])
    return numpy.tensordot(POVM, numpy.conj(POVM), axes=0)

#%%
def homodyneFockPOVM_2(n_max, x, theta):
    if n_max < 2:
        raise InvalidDimensionError('The Hilbert space dimension must be at least 2')
    POVM = numpy.zeros((n_max+1, ), dtype=complex)
    for n in numpy.arange(n_max+1):
        POVM[n] = numpy.exp(-1j*n*theta)/numpy.sqrt(numpy.sqrt(numpy.pi)*2**n*sp.special.gamma(n+1)) * numpy.exp(-0.5*x**2) * sp.special.hermite(n)(x)
    return numpy.tensordot(POVM, numpy.conj(POVM), axes=0)
#%%
CONVERGENCE_RULES = ['fidelity of state', 'fidelity of iteration operator', 'log-likelihood', 'trace distance(iteration operator, identity)']
RECONSTRUCTION_METHODS = ['maximum likelihood']
class QuadratureTomographer:
    """
    This class defines a quantum state tomographer, an object that performs 
    tomographic state reconstruction of one-mode quantum states of harmonic oscillators.
    
    It is based on the measurement of generalized quadratures of the harmonic oscillator, 
    previously normalized to vacuum quadrature noise, with vacuu quadrature variance equal to 1/2.
    
    Currently, it only uses the maximum likelihood method for state reconstruction.
    """
    def __init__(self, n_max, mode_function_type='doulbe exponential', temporal_mode_function=None, convergence_rule='fidelity of state'):
        self.temporal_mode_function = temporal_mode_function
        self.phases = None #quadrature phases associate tot he input quadrature data
        self.n_phases = None #number of quadrature phases
        self.quadratures = {} #the (non-filtered) quadrature traces corresponding to the quadrature angles, normalized by vacuum quadrature noise
        #self.n_quadrature_observations = {} #the number of quadrature observations computed during the maximum likelihood algorithm
        #self.quadrature_wavefunctions = {} #the wavefunctions of the input quadratures, in Fock basis, computed up to order n_max
        #self.projection_operators = {} #the average projection operators of the generalized quadratures, in Fock basis.
        self.n_max = n_max #maximum order of representation of the density operator in Fock space
        self.rho = 1/(self.n_max+1) * numpy.eye(N=self.n_max+1, dtype=complex) #estimated density operator
        self.rho_last = numpy.zeros((self.n_max+1, self.n_max+1), dtype=complex)#estimated density operator at the previous iteration of the reconstruction method
        self.log_likelihood = [] #log-likelihood of the maximum likelihood estimation
        self.convergence_parameter = 1e-9 #parameter deciding the convergence rule of the maximum likelihood estimation algorithm
        self.convergence_rule = convergence_rule
        return
    
    def setQuadratureData(self, quadratures, vacuum, phases, dt, apply_mode_function=False):
        """
        This function sets the quadrature data and associated quadrature angles.
        
        INPUTS
        -------------
            quadratures : 2-D array-like of float 
                Matrix of quadrature data, each column representing a sequence of quadrature measurements.
                Quadrature values are assumed to be normalized to vacuum noise standard deviation.
                The variance of vacuum quadrature is assumed to be 1/2.
            phases : array-like of float
                Quadrature phases associated to the input quadrature sequences [rad].
            dt : float (>0)
                Time separation between adjacent quadrature values in a quadrature sequence [s]. 
                If the quadratures are sampled digitally, dt is the acqusition sampling period.        
        OUTPUTS
        ------------
            None
        """
        self.dt = dt #time separation between adjacent raw quadrature samples [s]
        self.n_samples = len(quadratures[:, 0]) #number of samples per raw quadratures
        self.phases = phases
        self.n_phases = len(phases)
        if apply_mode_function:
            self.vacuum = self.temporal_mode_function.apply(x=vacuum.copy(), dt=self.dt)
        else:
            self.vacuum = vacuum.copy()
        if numpy.shape(quadratures)[1] != self.n_phases:
            raise InvalidQuadratureDataError('The number of quadrature sequences does not match the number of quadrature phases.')  
        for j in range(self.n_phases):
            if apply_mode_function:
                #Store after applying the mode function
                self.quadratures[self.phases[j]] = self.mode_function.apply(x=quadratures[:, j].copy(), dt=self.dt)/(numpy.var(self.vacuum)*2)**0.5
            else: 
                self.quadratures[self.phases[j]] = quadratures[:, j].copy()/(numpy.var(self.vacuum)*2)**0.5
        self.n_samples_filtered = len(self.quadratures[phases[0]]) #number of samples per filtered quadrature measurement
        self.dt_filtered = self.dt*float(int(self.n_samples/len(self.quadratures[self.phases[0]]))) #time separation between adjacent filtered quadrature samples [s]
        self.vacuum /= (numpy.var(self.vacuum)*2)**0.5 
        
    def reconstruct(self, n_bins=None, n_max=None, quadratures_range_increase=2, convergence_rule=None, convergence_parameter=None, method='maximum likelihood'):
        """
        This function performs tomographic state reconstruction.
        
        INPUTS
        ------------
            n_bins : integer (>0)
                Number of bins in which quadrature values are divided
            n_max : integer (>0)
                Dimension of the truncated Hilbert space
            quadratures_range_factor : float
                If x_max (x_min) is the maximum (minimum) observed quadrature value, the quadrature edges
                considered by the reconstruction algorithm is 
                     - x_min*(1-quadratures_range_increase/2)
                     - x_max*(1+quadrature_range_increase/2)
            convergence_rule : string
                Name of the criterion used to check that the state reconstruction method has converged.
            convergence_parameter : float (>0)
                Convergence threshold
            method : string
                State reconstruction method
        OUTPUS
        -----------
        None
        """
        
        if n_max is None:
            n_max = self.n_max
        if convergence_rule is None:
            convergence_rule = self.convergence_rule
        else:
            self.convergence_rule = convergence_rule
        if convergence_parameter is None:
            convergence_parameter = self.convergence_parameter
        else:
            self.convergence_parameter = convergence_parameter
        if method not in RECONSTRUCTION_METHODS:
            raise InvalidMethodError('The specified reconstruction method is invalid. Valid reconstruction methods are:\n'+str(RECONSTRUCTION_METHODS))
        #Precompute the maximum and minimum measured quadrature value
        x_max = [numpy.max(self.quadratures[phase]) for phase in self.phases]
        x_min = [numpy.min(self.quadratures[phase]) for phase in self.phases]
        x_max = numpy.max(x_max)*(1+quadratures_range_increase/2)
        x_min = numpy.min(x_min)*(1-quadratures_range_increase/2)
        if n_bins is None:
            #Try automatic binning
            #I want to resolve at least n_vacuum times the variance of the vacuum quadrature
            #So I'll set the number of bins to...
            n_vacuum = 10
            n_bins = int(numpy.ceil(numpy.abs(x_max-x_min)/(1/numpy.sqrt(2) / n_vacuum)))
        self.n_bins = n_bins
        #Bin the quadrature values accordingly and compute the central values of each bin
        x_bin_edges = numpy.linspace(x_min, x_max, self.n_bins+1)
        #Initialize
        #---------------------------------------------------------------------
        self.rho = (1/(self.n_max+1)) * numpy.eye(self.n_max+1, dtype=complex) #estimated density operator
        self.log_likelihood = []
        self.convergence_metris = {"fidelity of state": [], "fidelity of iteration operator": [], \
                                   "trace distance(iteration operator, identity)": [], \
                                    "likelihood": []}
        #---------------------------------------------------------------------
        #For each angle:
            #- Precompute the projectors on the generalized quadratures, averaged over each bin
            #- Precompute the number of quadrature observations per each bin
        #-------------------------------------------------------------------
        n_x = 1
        self.projection_operators = numpy.zeros((self.n_max+1, self.n_max+1, self.n_bins, self.n_phases), dtype=complex)
        self.n_observations = numpy.zeros((self.n_bins, self.n_phases))
        for p in range(self.n_phases):
            phase = self.phases[p]
            x = self.quadratures[phase]
            for j in range(self.n_bins): #for each bin
                projection_operator = numpy.zeros((self.n_max+1, self.n_max+1), dtype=complex)
                #Compute the average projection operator and the number of quadrature observations               
                x_bin = x[numpy.where(numpy.logical_and(x>x_bin_edges[j], x<x_bin_edges[j+1]))]
                x_bin_continuous = numpy.linspace(x_bin_edges[j], x_bin_edges[j+1], n_x)
                self.n_observations[j, p] = len(x_bin)
                #Compute the average projection operator
                #print('Computing wavefunctions')
                for k in range(n_x):
                    projection_operator += homodyneFockPOVM(n_max=n_max, x=x_bin_continuous[k], theta=phase)
                self.projection_operators[:, :, j, p] = projection_operator *(x_bin_edges[j+1]-x_bin_edges[j])
        #-----------------------------------------------
        #Run the maximum likelihood algorithm
        #-----------------------------------------------
        has_converged = [False]
        while not has_converged[0]:
            self.rho_last = self.rho 
            self.R = numpy.zeros((self.n_max+1, self.n_max+1), dtype=complex)
            self.measurement_probabilities = numpy.zeros((self.n_bins, self.n_phases))
            log_likelihood = 0
            for p in range(self.n_phases):
                for j in range(self.n_bins): #for each bin
                    self.measurement_probabilities[j, p] = numpy.trace(self.projection_operators[:, :, j, p] @ self.rho)
            self.measurement_probabilities /= self.n_phases
            self.R = numpy.tensordot(self.n_observations/self.measurement_probabilities, self.projection_operators, axes=([0, 1], [2, 3]))              
            self.R /= numpy.trace(self.R)
            log_likelihood = numpy.sum(self.n_observations*numpy.log(self.measurement_probabilities), (0, 1))
            self.R = (self.R+numpy.transpose(numpy.conj(self.R)))/2
           # self.R /= numpy.trace(self.R)
            self.rho = self.R @ (self.rho @ self.R)
            #self.rho = (self.rho+numpy.transpose(numpy.conj(self.rho)))/2
            self.rho /= numpy.trace(self.rho)
            #Check that it is a density matrix
            is_density_matrix = qfutils.isDensityMatrix(self.rho)
            if not is_density_matrix[0]:
                raise NotADensityMatrixError(str(is_density_matrix[1]))
            self.log_likelihood.append(log_likelihood)
            has_converged = self.hasConverged(convergence_rule=convergence_rule, convergence_parameter=convergence_parameter)
            for rule in ["fidelity of state", "fidelity of iteration operator", "trace distance(iteration operator, identity)",\
                         "likelihood"]:
                self.convergence_metris[rule].append(has_converged[1][rule])
            print(self.convergence_rule+': '+str(has_converged))
            
        #----------------------------------------------

    def hasConverged(self, convergence_rule=None, convergence_parameter=None):
        """
        This function verifies the convergence of the quantum state tomography method
        
        INPUTS
        ---------------
            convergence_rule : string
                Name of the criterion used to check that the state reconstruction method has converged.
            convergence_parameter : float (>0)
                Convergence threshold       
        OUTPUTS
        --------------
            True if the method has converged, False otherwise.
            
        """
        if convergence_rule is None:
            convergence_rule = self.convergence_rule         
        if convergence_parameter is None:
            convergence_parameter = self.convergence_parameter
        convergence_metrics = {"fidelity of state": None, "fidelity of iteration operator": None, \
                                   "trace distance(iteration operator, identity)": None, \
                                    "likelihood": None}
        has_converged = False
        convergence_metrics["fidelity of state"] = float(qfutils.quantumStateFidelity(self.rho, self.rho_last))
        convergence_metrics["fidelity of iteration operator"] = numpy.abs(qfutils.quantumStateFidelity(self.R, 1/(self.n_max+1)*numpy.eye(self.n_max+1, dtype=complex)))
        convergence_metrics["trace distance(iteration operator, identity)"] = float(qfutils.traceDistance(self.R, 1/(self.n_max+1)*numpy.eye(self.n_max+1, dtype=complex)))
        convergence_metrics["likelihood"] = numpy.exp(self.log_likelihood[-1])
        if convergence_rule == 'fidelity of state':
            has_converged = convergence_metrics["fidelity of state"] > 1-convergence_parameter
        elif convergence_rule == 'fidelity of iteration operator':
            has_converged = convergence_metrics["fidelity of iteration operator"] > 1-convergence_parameter
        elif convergence_rule == 'trace distance(iteration operator, identity)':
            has_converged = convergence_metrics["trace distance(iteration operator, identity)"] < convergence_parameter
        elif convergence_rule == 'likelihood':
            has_converged = convergence_metrics["likelihood"] > 1-convergence_parameter
        else:
            raise InvalidConvergenceRuleError('The valid convergence rules are:\n'+str(CONVERGENCE_RULES))
        return has_converged, convergence_metrics
        
#%%     
class InvalidQuadratureDataError(Exception):
    def __init__(self, error_message):
        print(error_message)
        
class InvalidMethodError(Exception):
    def __init__(self, error_message):
        print(error_message)
        
class InvalidConvergenceRuleError(Exception):
    def __init__(self, error_message):
        print(error_message)
        
class InvalidDimensionError(Exception):
    def __init__(self, error_message):
        print(error_message)
        

class NotADensityMatrixError(Exception):
    def __init__(self, error_message):
        print(error_message)

               
