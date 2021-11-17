#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:51:59 2021

@author: jiedz
This module defines objects and functions useful to handle bosonic quantum fields modeling and analysis.
"""
#%%
import numpy as np
import scipy as sp
import inspect
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import font_manager
#%%
#General plot settings
default_marker = ""
default_figure_size = (9, 9)
default_fontsize = 20
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
MODE_FUNCTION_TYPES = ['double exponential', 'double exponential filter', 'delta filter']
MODE_FUNCTIONS = [( lambda gamma, t0, t: 
                        (gamma)**0.5*np.exp(-gamma*abs(t-t0)) ), \
                  ( lambda gamma, kappa, t0, t:
                        1/(gamma**2/kappa-kappa/(1+kappa/gamma)+kappa**2/gamma)**0.5*(kappa*np.exp(-gamma*abs(t-t0)) - 
                            gamma*np.exp(-kappa*abs(t-t0))) ),   \
                  ( lambda kappa, t0, t:
                        (2*kappa)**0.5*np.exp(-kappa*abs(t-t0)) * (np.sign(t0-t)+1)/2)]
MODE_FUNCTIONS_DICT = dict(zip(MODE_FUNCTION_TYPES, MODE_FUNCTIONS))
#Define the characteristic duration of a temporal mode function, as the 
#time delaty at which the temporal mode function has decreased to CHARACTERISTIC_DURATION_FACTOR
CHARACTERISTIC_DURATION_FACTOR = 1/10 
class TemporalModeFunction():
    """
    This object represents a temporal mode function, which is necessary to consider
    when dealing with quantum fields, where temporal modes are correlated in time
    according to a specific process.
    
    For example, different temporal modes of the 
    electromagnetic field output from a type-0 optical parametric oscillator (OPO) are
    correlated in time according to a temporal mode function, that is defined
    by the OPO frequency response. Such temporal mode function typically has the
    shape of a doulbe-sided decaying exponential.
    
    Applying a temporal mode function is necessary if one wants to correctly
    reconstruct the state output from an OPO (or other systems) via homodyne-detection
    tomographic state reconstruction.
    """
    
    
    
    def __init__(self, function_type, parameters=None):
        if function_type not in MODE_FUNCTION_TYPES:
            raise InvalidFunctionTypeError('The specified mode function is not defined. Valid mode function types are:\n'+str(MODE_FUNCTION_TYPES))
        self.type, self.function, self.parameters, self.duration = [None for j in range(4)] 
        self.type, self.function = (function_type, MODE_FUNCTIONS_DICT[function_type])
        if parameters is not None:
            self.setParameters(parameters)
        return
    
    def setParameters(self, *parameters):
        """
        Sets the function type and parameters.
        
        INPUTS
        -------------
            parameters : tuple of float
                The parameters of the specified function
                
        OUTPUTS
        ------------
        None
        
        """
        #Set the parameters
        if len(np.shape(parameters)) > 1:
            parameters = parameters[0]
        n_parameters = len(str(inspect.signature(self.function)).split(','))-1 #exclude the variable 't' from the parameters
        if n_parameters != len(parameters):
            number_of_parameters_difference = len(parameters) - n_parameters
            excess = ('in excess')*(number_of_parameters_difference > 0) + ('missing')*(number_of_parameters_difference < 0)  
            error_message = "The number of input parameters does not match the mode function.\n "+\
                            str(abs(number_of_parameters_difference))+' parameters '+excess                         
            raise InvalidParametersError(error_message)
            
        self.parameters = parameters
        #Compute the characteristic duration of the temporal mode function (derived analytically)
        #it is assumed that the exponential damping factor 'gamma' is the first specified parameter
        #Calculate the characteristic duration of the temporal mode function
        initial_period = 1e-4
        duration = None
        period_increase = 1
        while duration is None:
            t = np.linspace(-initial_period/2*period_increase/2, initial_period/2*period_increase/2, 1e6)
            coefficients = self.function(*parameters, t)
            t_duration = t[np.where(np.abs(coefficients) >= np.max(np.abs(coefficients))*CHARACTERISTIC_DURATION_FACTOR)]
            duration = np.abs(t_duration[-1] - t_duration[0])
            if duration == 0:
                duration = None
            period_increase *= 2
        self.characteristic_time = duration
        self.delay = t[np.argmax(np.abs(coefficients))]
        
    def apply(self, x, dt, idle_fraction):
        """
        Apply the mode function to the input data representing quadrature values
        in time domain. 
        
        1) The temporal mode function coefficients are calculated from its parameters
        2) The input data are partitioned into segments of duration equal
        to the characteristic duration of the temporal mode function, increased by 'idle_time'
        3) Each segment is multiplied by the temporal mode function coefficients and integrated (summed)
        
        INPUTS
        --------------
            x : array-like of float
                the input quadrature values, correctly normalized according to the field theory.
            dt : float (>0)
                the time separation between adjacent samples. If 'input_data' is a discrete-time 
                sequence, then dt is the sampling period [s].
            idle_fraction : float (>=0)
                Fraction of the temporal mode functions characteristic time to wait, between one segment
                integration and the next. It's a safety margin to avoid correlation between subsequent
                temporal modes.
        OUTPUTS
        -------------
        the 'filtered' data and the 
        """
        n_samples = len(x)
        duration = 4*self.characteristic_time*(1+idle_fraction)
        n_samples_segment = int(np.floor(duration/dt)) #number of samples of the temporal mode function
        n_segments = int(np.ceil(n_samples/n_samples_segment))
        #Construct the temporal mode function
        t = self.delay + np.linspace(-duration/2, duration/2-dt, n_samples_segment) #time axis centered around 0
        self.coefficients = self.function(*self.parameters, t)
        
        x_filtered = np.zeros((n_segments,))
        plt.plot(t, self.coefficients)
        for j in range(n_segments):
            end = int(np.min([n_samples, (j+1)*n_samples_segment]))
            if end - j*n_samples_segment < n_samples_segment:
                pass
            else:
                x_segment = x[j*n_samples_segment:end]
                #Apply the mode function to the current segment
                x_filtered[j] = x_segment.T @ self.coefficients
        
        return x_filtered[:-1], dt*(n_segments-1) 
        
#%%
def traceDistance(M_1, M_2):
    """
    Source: https://en.wikipedia.org/wiki/Trace_distance
    
    INPUTS
    ------------
        M_1 : 2-D array-like of complex
            Matrix 1
        M_2 : 2-D array-like of complex
            Matrix 2
    
    OUTPUTS
    -----------
    The trace distance
    """   
    X = M_1 - M_2
    return 0.5*np.trace(sp.linalg.sqrtm(np.transpose(np.conj(X)) @ X))


def quantumStateFidelity(rho_1, rho_2):
        """
        This function computes the fidelity of two quantum states, defined by
        the density matrices 'rho_1' and 'rho_2'.
        
        Source: https://en.wikipedia.org/wiki/Fidelity_of_quantum_states
        
        INPUTS
        ------------
            rho_1 : 2-D array-like of complex
                Density matrix 1
            rho_2 : 2-D array-like of complex
                Density matrix 2
        
        OUTPUTS
        -----------
        The fidelity (float in [0, 1])
        """                
        X = sp.linalg.sqrtm(rho_1) @ sp.linalg.sqrtm(rho_2)
        return (2*traceDistance(X, 0))**2
#%%
def WignerFunctionFromDensityMatrix(rho, q, p):
    """
    This function computes the Wigner function associated to a density operator, 
    in the basis of number states.
    
    INPUTS
    ---------------
        rho : 2-D array-like of complex
            The density operator in the number states basis
        q, p : 1-D array-like of float
            The q and p quadrature values, on which the Wigner function is computed
    OUTPUTS
    --------------
        Q, P : 2-D array-like of float
            grids of quadrature values
        W : 2-D array-like of float
            the Wigner function evaluated on q and p
    """
    
    P, Q = np.meshgrid(np.atleast_1d(p), np.atleast_1d(q))
    W = np.zeros((len(p), len(q)), dtype=complex)
    N = np.shape(rho)[0] - 1 #dimension of the (truncated) Hilbert space of rho
    W_nm = np.zeros((len(p), len(q), N+1, N+1), dtype=complex) #expansion coefficients of the Wigner function
    X = 2*(Q**2 + P**2)
    #Compute the lower triangle of the matrix W_nm
    for n in np.arange(N+1):
        for m in np.arange(n+1):
            k=m
            alpha = float(n-m)
            W_nm[:, :, n, m] = 1/np.pi * np.exp(-0.5*X) * (-1)**m \
                               *(Q-1j*P)**(n-m) * np.sqrt( 2**(n-m)*sp.special.gamma(m+1)/sp.special.gamma(n+1) )\
                               * sp.special.assoc_laguerre(x=X, n=k, k=alpha)     
    #Compute the upper triangle without the diagonal
    for m in np.arange(N+1):
        for n in np.arange(m):
            W_nm[:, :, n, m] = np.conj(W_nm[:, :, m, n])
    
    #Compute the Wigner function
    for n in np.arange(N+1):
        for m in np.arange(N+1):
            W += W_nm[:, :, n, m] * rho[n, m]
    #Normalize in L1
    dq = q[1] - q[0]
    dp = p[1] - p[0]
    W /= np.sum(np.sum(W) * dq*dp)
    return Q, P, W

#Wigner function fo the covariance matrix of a Gaussian state
def WignerFuntion_GaussianState(q, p, mean_vector, covariance_matrix):
    """
    INPUTS
    ----------
    q : 1-D array-like of float
        q quadrature values, normalized to vacuum noise units.
    p : 1-D array-like of float
        p quadrature values, normalized to vacuum noise units.
    mean_vector : 1-D array-like of float, length 2N
        Mean vector containing the mean value of the quadratures (q_1, p_1, ..., q_N, p_N)
    covariance_matrix : 2-D array-like of float, size 2N x 2N
        Covariance matrix of the quadratures

    OUTPUTS
    -------
    Wigner function, evaluated at the points specified by q and p
    
    """
    P, Q = np.meshgrid(p, q)
    W = np.zeros((len(p), len(q)))
    for j in range(len(p)):
        p_0 = p[j]
        for k in range(len(q)):
            q_0 = q[k]
            x_0 = np.array([q_0, p_0])
            W[j, k] = np.exp(-0.5*(x_0-mean_vector).T @ np.linalg.inv(covariance_matrix) @ (x_0-mean_vector))
    W /= 2*np.pi*np.linalg.det(covariance_matrix)
    return Q, P, W

#Wigner function of the coherent states
def WignerFunction_CoherentState(q, p, alpha):
    """
    

    INPUTS
    ----------
    q : 1-D array-like of float
        q quadrature values, normalized to vacuum noise units.
    p : 1-D array-like of float
        p quadrature values, normalized to vacuum noise units.
    alpha : complex
        complex amplitude of the coherent state

    OUTPUTS
    -------
    q and p meshgrid values : 2-D arrays-like of float
    
    wigner function : 2-D array-like of float
    """
    mean_vector = 2*np.array([np.real(alpha), np.imag(alpha)])
    covariance_matrix = np.eye(2)
    return WignerFuntion_GaussianState(q, p, mean_vector, covariance_matrix)

#Wigner function of the coherent states
def WignerFunction_SqueezedCoherentState(q, p, alpha, zeta):
    """
    

    INPUTS
    ----------
    q : 1-D array-like of float
        q quadrature values, normalized to vacuum noise units.
    p : 1-D array-like of float
        p quadrature values, normalized to vacuum noise units.
    alpha : complex
        complex amplitude of the squeezed coherent state
    zeta : complex
        complex squeezing parameter. The modulus is the squeezing strength and the 
        argument is the squeezing direction

    OUTPUTS
    -------
    q and p meshgrid values : 2-D arrays-like of float
    
    wigner function : 2-D array-like of float
    """
    mean_vector = 2*np.array([alpha+np.conj(alpha), 1j*(np.conj(alpha)-alpha)])
    covariance_matrix = np.eye(2)
    r = np.abs(zeta)
    theta = np.angle(zeta)
    covariance_matrix[0, 0] = np.exp(r)*np.sin(theta)**2 + np.exp(-r)*np.cos(theta)**2
    covariance_matrix[1, 1] = np.exp(r)*np.cos(theta)**2 + np.exp(-r)*np.sin(theta)**2
    covariance_matrix[0, 1] = covariance_matrix[1, 0] = np.sin(2*theta)*np.sinh(r)   
    return WignerFuntion_GaussianState(q, p, mean_vector, covariance_matrix)

#Wigner function of the thermal state
def WignerFunction_ThermalCoherentState(q, p, alpha, mean_photon_number):
    """ 
    INPUTS
    ----------
    q : 1-D array-like of float
        q quadrature values, normalized to vacuum noise units.
    p : 1-D array-like of float
        p quadrature values, normalized to vacuum noise units.
    mean_photon_number : float (>=0)
        mean photon number of the thermal state

    OUTPUTS
    -------
    q and p meshgrid values : 2-D arrays-like of float
    
    wigner function : 2-D array-like of float
    """
    mean_vector = 2*np.array([np.real(alpha), np.imag(alpha)])
    covariance_matrix = np.eye(2)
    covariance_matrix[0, 0] += mean_photon_number/2  
    covariance_matrix[1, 1] = covariance_matrix[0, 0]
    return WignerFuntion_GaussianState(q, p, mean_vector, covariance_matrix)

#Wigner function of the number states
def WignerFunction_NumberState(q, p, n):
    """ 
    INPUTS
    ----------
    q : 1-D array-like of float
        q quadrature values, normalized to vacuum noise units.
    p : 1-D array-like of float
        p quadrature values, normalized to vacuum noise units.
    n : natural
        number of photons

    OUTPUTS
    -------
    q and p meshgrid values : 2-D arrays-like of float
    
    wigner function : 2-D array-like of float
    """
    P, Q = np.meshgrid(p, q)
    W = np.zeros((len(p), len(q)))
    for j in range(len(p)):
        p_0 = p[j]
        for k in range(len(q)):
            q_0 = q[k]
            W[j, k] = (-1)**n/np.pi*np.exp(-(q_0**2+p_0**2))*sp.special.assoc_laguerre(x=2*(q_0**2+p_0**2), n=n)
    return P, Q, W

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

def plotWignerFunction(Q, P, W, colormap=cmwig1, alpha=0.7, plot_name='untitled', plot_contour_on_3D=True):
    W_max = np.max(np.abs(W))
    #Define the x and y axes lines as a 2D function
    xy_2D = np.zeros((len(P), len(Q)))
    xy_2D[np.where(np.logical_or(P==0, Q==0))] = 1
    #3D plot
    figure_3D = plt.figure(num="Wigner function - "+plot_name+" - 3D", figsize=default_figure_size)
    axis_3D = figure_3D.add_subplot(111,  projection='3d')
    #3D Wigner function
    axis_3D.plot_surface(Q, P, W, cmap=colormap, alpha=alpha, norm=matplotlib.colors.Normalize(vmin=-W_max, vmax=W_max))
    #Plot the contour of the xy projection
    axis_3D.contour(Q, P, W, zdir='z', offset=np.min(W)-0.1*W_max, cmap=colormap, norm=matplotlib.colors.Normalize(vmin=-W_max, vmax=W_max))
    #Plot the Q axis
    axis_3D.plot([np.min(Q), np.max(Q)], [0, 0], zs=np.min(W)-0.1*W_max, color='grey', alpha=alpha)
    #Plot the P axis
    axis_3D.plot([0, 0], [np.min(P), np.max(P)],zs=np.min(W)-0.1*W_max, color='grey', alpha=alpha)
    axis_3D.grid(False)
    axis_3D.set_zticks([])
    axis_3D.set_xlabel('q', font=axis_font)
    axis_3D.set_ylabel('p', font=axis_font)
    axis_3D.set_zlabel('W(q, p)', font=axis_font)
    #2D plot
    #Set the color of the plot to white, when the Wigner function is close to 0
    W[np.where(np.isclose(W, 0, rtol=1e-4))] = np.nan
    colormap.set_bad('w')
    figure_2D = plt.figure(num="Wigner function - "+plot_name+" - 2D", figsize=default_figure_size)
    axis_2D = figure_2D.add_subplot(111)
    axis_2D.set_aspect('equal')
    #2D plot
    axis_2D.contourf(Q, P, W, alpha=alpha, cmap=colormap, norm=matplotlib.colors.Normalize(vmin=-W_max, vmax=W_max))
    #Plot the Q axis
    axis_2D.plot([np.min(Q), np.max(Q)], [0, 0], color='grey', alpha=alpha)
    #Plot the P axis
    axis_2D.plot([0, 0], [np.min(P), np.max(P)], color='grey', alpha=alpha)
    axis_2D.grid(False)
    axis_2D.set_xlabel('q', font=axis_font)
    axis_2D.set_ylabel('p', font=axis_font)
    figures = {'2D': figure_2D, '3D': figure_3D}
    return figures
#%%
def isDensityMatrix(rho):
    """
    This function checks whether the input matrix is a density matrix.
    rho is a density matrix if and only if:
        
        - rho is Hermitian
        - rho is positive-semidefinite
        - rho has trace 1
        - rho^2 has trace <= 1
    """
    
    check = False
    check_details = {}
    check_details['is Hermitian'] = isHermitian(rho)
    check_details['is positive-semidefinite'] = isPositiveSemidefinite(rho)
    check_details['has trace 1'] = isTraceNormalized(rho)
    check_details['square has trace <= 1'] = isTraceConvex(rho)
    check = check_details['is Hermitian'] and check_details['is positive-semidefinite']\
            and check_details['has trace 1'] and check_details['square has trace <= 1']
    return check, check_details

def isPositiveSemidefinite(M, tolerance=1e-10):
    """
    This function checks whether the input Hermitian matrix is positive-semidefinite.
    M is positive-semidefinite if an only if all its eigenvalues are nonnegative
    
    INPUTS
    -------------
        M : 2-D array-like of complex
            The input matrix.
        
    OUTPUTS
    -----------
    True if M is positive-semidefinite
    """
    if not isHermitian(M):
        M = (M+np.transpose(np.conj(M)))/2 #take the Hermitian part of M
    eigenvalues, _ = np.linalg.eig(M)
    return np.all(np.real(np.array(eigenvalues)) >= 0-tolerance)

def isHermitian(M):
    """
    This function checks whether the input matrix is Hermitian.
    A matrix is hermitian if an only if all of its eigenvalues are real
    
    INPUTS
    -------------
        M : 2-D array-like of complex
            The input matrix.
        
    OUTPUTS
    -----------
    True if M is is Hermitian
    """
    
    eigenvalues, _ = np.linalg.eig(M)
    eigenvalues_imaginary = np.array([np.imag(e) for e in eigenvalues]) #imaginary part of all eigenvalues
    return np.all(np.isclose(eigenvalues_imaginary, 0)) #check that all eigenvalues are approximately real

def isTraceNormalized(M):
    """
    This function checks whether the input matrix has trace equal to 1.
    
    INPUTS
    -------------
        M : 2-D array-like of complex
            The input matrix.
        
    OUTPUTS
    -----------
    True if M has trace equal to 1.
    """    
    
    return np.isclose(np.real(np.trace(M)), 1)
    
def isTraceConvex(M):
    """
    This function checks whether the input matrix has trace(M^2) <= 1
    
    INPUTS
    -------------
        M : 2-D array-like of complex
            The input matrix.
        
    OUTPUTS
    -----------
    True if M has trace(M^2) <= 1
    """    
    
    return np.real(np.trace(M @ M)) <= 1
    
    
    
    
#%%
#Define exceptions
class InvalidParametersError(Exception):
    def __init__(self, error_message):
        print(error_message)
        
class InvalidFunctionTypeError(Exception):
    def __init__(self, error_message):
        print(error_message)
        
