#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:51:59 2021

@author: jiedz
This module defines objects and functions useful to handle bosonic quantum fields modeling and analysis.
"""
#%%
import numpy 
import scipy as sp
import inspect
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib import cm
#%%
plt.close('all')
#General plot settings
default_marker = ""
default_figure_size = (11, 8)
default_fontsize = 50
title_fontsize = default_fontsize
title_font = font_manager.FontProperties(family='Times New Roman',
                                   weight='bold',
                                   style='normal', size=title_fontsize)
axis_font = font_manager.FontProperties(family='Times New Roman',
                                   weight='normal',
                                   style='normal', size=title_fontsize*0.7)
legend_font = font_manager.FontProperties(family='Times New Roman',
                                   weight='normal',
                                   style='normal', size=int(numpy.floor(0.9*title_fontsize)))
ticks_fontsize = axis_font.get_size()*0.8
#%%
MODE_FUNCTION_TYPES = ['exponential', 'double exponential']
#%%
def exponential(t, *params):
    if len(params) != 2:
        raise ValueError("The exponential mode function must have 2 parameters, but %d were given"%len(params))
    gamma = params[0]
    t0 = params[1]
    y = numpy.exp(-gamma*numpy.abs(t-t0))
    return y/numpy.sqrt(y.T@y)


def double_exponential(t, *params):
    if len(params) != 3:
        raise ValueError("The double_exponential mode function must have 3 parameters, but %d were given"%len(params))
    gamma = params[0]
    kappa = params[1]
    t0 = params[2]
    print(t0)
    y = 1/gamma*numpy.exp(-gamma*numpy.abs(t-t0)) - 1/kappa*numpy.exp(-kappa*numpy.abs(t-t0))
    return y/numpy.sqrt(y.T@y)
#%%
MODE_FUNCTIONS_DICT = dict(zip(MODE_FUNCTION_TYPES, [exponential, double_exponential]))
#%%
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
        if len(numpy.shape(parameters)) > 1:
            parameters = parameters[0]
        self.parameters = parameters
        #Compute the characteristic duration of the temporal mode function (derived analytically)
        #it is assumed that the exponential damping factor 'gamma' is the first specified parameter
        #Calculate the characteristic duration of the temporal mode function
        initial_period = 1e-4
        duration = 0
        period_increase = 1
        while duration == 0:
            t = numpy.linspace(-initial_period/2*period_increase/2, initial_period/2*period_increase/2, int(1e6))
            coefficients = self.function(*parameters, t)
            t_duration = t[numpy.where(numpy.abs(coefficients) >= numpy.max(numpy.abs(coefficients))*CHARACTERISTIC_DURATION_FACTOR)]
            duration = numpy.abs(t_duration[-1] - t_duration[0])
            period_increase *= 2
        self.characteristic_time = duration
        self.delay = t[numpy.argmax(numpy.abs(coefficients))]
        
    def apply(self, x, dt):
        """
        Apply the mode function to the input data representing quadrature values
        in time domain. 
        """
        if len(x.shape) == 1:
            n_segments = 1
            n_per_segment = x.shape[0]
        else:
            n_segments = x.shape[0]
            n_per_segment = x.shape[1]           
        #Repeat the mode function for all the segments
        t = dt*numpy.arange(n_per_segment)
        f = numpy.repeat(a=numpy.matrix(self.function(t, *self.parameters)), repeats=n_segments,\
                         axis=0)
        f = numpy.asarray(f)
        x_filtered = numpy.sum(x*f, axis=1)
        return x_filtered
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
    return 0.5*numpy.trace(sp.linalg.sqrtm(numpy.transpose(numpy.conj(X)) @ X))


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
    
    P, Q = numpy.meshgrid(numpy.atleast_1d(p), numpy.atleast_1d(q))
    W = numpy.zeros((len(p), len(q)), dtype=complex)
    N = numpy.shape(rho)[0] - 1 #dimension of the (truncated) Hilbert space of rho
    W_nm = numpy.zeros((len(p), len(q), N+1, N+1), dtype=complex) #expansion coefficients of the Wigner function
    X = 2*(Q**2 + P**2)
    #Compute the lower triangle of the matrix W_nm
    for n in numpy.arange(N+1):
        for m in numpy.arange(n+1):
            k=m
            alpha = float(n-m)
            W_nm[:, :, n, m] = 1/numpy.pi * numpy.exp(-0.5*X) * (-1)**m \
                               *(Q+1j*P)**(n-m) * numpy.sqrt( 2**(n-m)*sp.special.gamma(m+1)/sp.special.gamma(n+1) )\
                               * sp.special.assoc_laguerre(x=X, n=k, k=alpha)     
    #Compute the upper triangle without the diagonal
    for m in numpy.arange(N+1):
        for n in numpy.arange(m):
            W_nm[:, :, n, m] = numpy.conj(W_nm[:, :, m, n])
    
    #Compute the Wigner function
    for n in numpy.arange(N+1):
        for m in numpy.arange(N+1):
            W += W_nm[:, :, n, m] * rho[n, m]
    #Normalize in L1
    dq = q[1] - q[0]
    dp = p[1] - p[0]
    W /= numpy.sum(numpy.sum(W) * dq*dp)
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
    P, Q = numpy.meshgrid(p, q)
    W = numpy.zeros((len(p), len(q)))
    for j in range(len(p)):
        p_0 = p[j]
        for k in range(len(q)):
            q_0 = q[k]
            x_0 = numpy.array([q_0, p_0])
            W[j, k] = numpy.exp(-0.5*(x_0-mean_vector).T @ numpy.linalg.inv(covariance_matrix) @ (x_0-mean_vector))
    W /= 2*numpy.pi*numpy.linalg.det(covariance_matrix)
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
    mean_vector = 2*numpy.array([numpy.real(alpha), numpy.imag(alpha)])
    covariance_matrix = numpy.eye(2)
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
    mean_vector = 2*numpy.array([alpha+numpy.conj(alpha), 1j*(numpy.conj(alpha)-alpha)])
    covariance_matrix = numpy.eye(2)
    r = numpy.abs(zeta)
    theta = numpy.angle(zeta)
    covariance_matrix[0, 0] = numpy.exp(r)*numpy.sin(theta)**2 + numpy.exp(-r)*numpy.cos(theta)**2
    covariance_matrix[1, 1] = numpy.exp(r)*numpy.cos(theta)**2 + numpy.exp(-r)*numpy.sin(theta)**2
    covariance_matrix[0, 1] = covariance_matrix[1, 0] = numpy.sin(2*theta)*numpy.sinh(r)   
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
    mean_vector = 2*numpy.array([numpy.real(alpha), numpy.imag(alpha)])
    covariance_matrix = numpy.eye(2)
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
    P, Q = numpy.meshgrid(p, q)
    W = numpy.zeros((len(p), len(q)))
    for j in range(len(p)):
        p_0 = p[j]
        for k in range(len(q)):
            q_0 = q[k]
            W[j, k] = (-1)**n/numpy.pi*numpy.exp(-(q_0**2+p_0**2))*sp.special.assoc_laguerre(x=2*(q_0**2+p_0**2), n=n)
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

#%%
#Funciton to plot Wigner function
def plotWignerFunction(Q, P, W, alpha, colormap=cmwig1, plot_name='untitled', plot_contour_on_3D=True):
    W *= numpy.pi
    W_max = numpy.max(numpy.abs(W))
    #Define the x and y axes lines as a 2D function
    xy_2D = numpy.zeros((len(P), len(Q)))
    xy_2D[numpy.where(numpy.logical_or(P==0, Q==0))] = 1
    #3D plot
    figure_3D = plt.figure(num="Wigner function - "+plot_name+" - 3D", figsize=default_figure_size)
    axis_3D = figure_3D.add_subplot(111,  projection='3d')
    #3D Wigner function
    W = W.astype(float)
    Q = Q.astype(float)
    P = P.astype(float)
    axis_3D.plot_surface(Q, P, W, alpha=alpha, cmap=colormap, norm=matplotlib.colors.Normalize(vmin=-W_max, vmax=W_max))
    plt.pause(0.5)
    #Plot the contour of the xy projection
    axis_3D.contour(Q, P, W, zdir='z', offset=numpy.min(W)-0.1*W_max, cmap=colormap, norm=matplotlib.colors.Normalize(vmin=-W_max, vmax=W_max))
    #Plot the Q axis
    axis_3D.plot([numpy.min(Q), numpy.max(Q)], [0, 0], zs=numpy.min(W)-0.1*W_max, color='grey', alpha=alpha)
    #Plot the P axis
    #axis_3D.plot([0, 0], [numpy.min(P), numpy.max(P)],zs=numpy.min(W)-0.1*W_max, color='grey', alpha=alpha)
    axis_3D.grid(False)
#    axis_3D.set_zlim([numpy.min(W)-0.1*W_max, W_max])
    axis_3D.set_xlabel('q', font=axis_font)
    axis_3D.set_ylabel('p', font=axis_font)
    #axis_3D.set_zlabel('W(q, p)', font=axis_font)
    #2D plot
    #Set the color of the plot to white, when the Wigner function is close to 0
   # W[numpy.where(numpy.isclose(W, 0, rtol=1e-3))] = numpy.nan
    #colormap.set_bad('w')
    figure_2D = plt.figure(num="Wigner function - "+plot_name+" - 2D", figsize=default_figure_size)
    axis_2D = figure_2D.add_subplot(111)
    axis_2D.set_aspect('equal')
    #2D plot
    axis_2D.contourf(Q, P, W, alpha=alpha, cmap=colormap, norm=matplotlib.colors.Normalize(vmin=-W_max, vmax=W_max))
    #Plot the Q axis
    axis_2D.plot([numpy.min(Q), numpy.max(Q)], [0, 0], color='grey', alpha=alpha)
    #Plot the P axis
    axis_2D.plot([0, 0], [numpy.min(P), numpy.max(P)], color='grey', alpha=alpha)
    axis_2D.grid(False)
    axis_2D.set_xlabel('q', font=axis_font)
    axis_2D.set_ylabel('p', font=axis_font)
    figures = {'2D': figure_2D, '3D': figure_3D}
    return figures
#%%
def isDensityMatrix(M, tolerance=1/100):
    """
    This function checks whether the matrix is a density matrix.
    M is a density matrix if and only if:

        - M is Hermitian
        - M is positive-semidefinite
        - M has trace 1
    """
    check_details = {}
    eigenvalues = numpy.linalg.eigvals(M)
    max_eigenval = numpy.max(numpy.abs(eigenvalues))
    check_details['is Hermitian'] = not numpy.any([not numpy.isclose(numpy.imag(e), 0) for e in eigenvalues])
    check_details['is positive-semidefinite'] = numpy.all([numpy.real(e)>=-tolerance*max_eigenval for e in eigenvalues])
    check_details['has trace 1'] = numpy.isclose(numpy.trace(M), 1)
    check = check_details['is Hermitian'] and check_details['is positive-semidefinite'] \
            and check_details['has trace 1']
    return check, check_details
# ----------------------------------------------------------
# ----------------------------------------------------------
def isHermitian(M, tolerance=1/100):
    """
    This function checks whether the input matrix is Hermitian.
    A matrix is hermitian if an only if all of its eigenvalues are real
    """
    if not M.isSymmetric(tolerance=tolerance):
        raise TypeError(
            "Cannot test Hermitianity because the value of matrix " + M.name + " is not symmetric.\n" + M.__str__())
    else:
        min_eigenvalue = numpy.min(numpy.abs(M.eigenvalues))  # modulus of the minimum eigenvalue of the input matrix
        tolerance = tolerance * min_eigenvalue
        return numpy.all(numpy.isclose(numpy.imag(M.eigenvalues), 0, atol=tolerance))  # check that all eigenvalues are approximately real
#%%
class InvalidFunctionTypeError(TypeError):
    def __init__(self, error_message):
        super().__init__(error_message)
        
class InvalidParametersError(ValueError):
    def __init__(self, error_message):
        super().__init__(error_message)