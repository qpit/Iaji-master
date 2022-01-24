#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 11:06:05 2022

@author: jiedz
This module contains utilities for Gaussian CVQKD
"""
#%%
import numpy
from quik.qip.nmodes import covariancematrix, vacuum
#%%
def secret_key_fraction_lossy_channel_bound(channel_total_efficiency, L):
    """
    Computes Pirandola's bound on the secret key fraction of CVQKD, for a 
    quantum channel partitioned in segments with equal efficiencies.
        channel_total_efficiency : iterable of float (elements in [0, 1])
            the transmission efficiencies of the channel segments.
    """
    return -numpy.log2(1-(channel_total_efficiency)**(1/L)) 
#%%
def mutual_information(variances, conditional_variances):
    """
    INPUTS
    -----------
        variances : array-like of float (shape in {(1, 2), (2, 1)}
            The variances V_x_{B} described in the last cell.
        conditional_variances : array-like of float (shape in {(1, 2), (2, 1)}
        quadrature_switching_ratio : float (in [0, 1])
            equal to the probability of modulating the q quadrature. Only valid 
            for unidimensional protocols
    OUTPUTS
    -----------
    The Shannon's mutual information
    """
    variances = numpy.array(variances)
    conditional_variances = numpy.array(conditional_variances)
    return 0.5*numpy.sum(numpy.log2(variances/conditional_variances))
#%%
def covariance_matrix_EB_ideal_transmitter(variances_A, correlations_A):
    """
    Returns the covariance matrix of the state prepared by Alice in the entanglement-based
    Gaussian CVQKD protocol with ideal transmitter.
    
    INPUTS
    -------------
        variances : array-like of float (shape in {(1, 2), (2, 1)}
            The variances V_x_{A} described in the last cell.
        correlations : array-like of float (shape in {(1, 2), (2, 1)}
            The correlations c_{x_{SA}} described in the last cell
    OUTPUTS
    -------------
        The covariance matrix of modes A' and A in the entanglement-based protocol.
        
        The reflectivity of the beam splitter used by Alice for her measurement
        
        Updated V_q_A, calculated from the other input parameters
        
        Updated V_q_B, calculated from the other input parameters
    """
    V_q_A = variances_A[0]
    V_p_A = variances_A[1]
    c_q_SA = correlations_A[0]
    c_p_SA = correlations_A[1]
    r_1_EB, nu_2_EB, R_m = [None for j in range(3)]
    if c_q_SA == 0:
        if V_p_A == 0: #V_p_A is not provided
            raise ValueError("In this case you have to provide V_p_A")
        #Apply the first system of equations
        r_1_EB = numpy.log((V_p_A*c_q_SA**2 - c_p_SA**2*c_q_SA**2 + 1)/(V_p_A*(V_p_A - c_p_SA**2)))/4
        nu_2_EB = V_p_A*numpy.sqrt((V_p_A*c_q_SA**2 - c_p_SA**2*c_q_SA**2 + 1)/(V_p_A*(V_p_A - c_p_SA**2)))
        R_m = c_q_SA**2*(V_p_A - c_p_SA**2)/(V_p_A*c_q_SA**2 - c_p_SA**2*c_q_SA**2 + c_p_SA**2*numpy.sqrt((V_p_A*c_q_SA**2 - c_p_SA**2*c_q_SA**2 + 1)/(V_p_A*(V_p_A - c_p_SA**2))))      
        V_q_A = (V_p_A*c_q_SA**2 - c_p_SA**2*c_q_SA**2 + 1)/(V_p_A - c_p_SA**2)
    elif c_p_SA == 0:
        if V_q_A == 0: #V_q_A is not provided
            raise ValueError("In this case you have to provide V_q_A")
        #Apply the second system of equations
        r_1_EB = numpy.log(V_q_A**(1/4)*((V_q_A - c_q_SA**2)/(V_q_A*c_p_SA**2 - c_p_SA**2*c_q_SA**2 + 1))**(1/4))
        nu_2_EB = numpy.sqrt(V_q_A)/numpy.sqrt((V_q_A - c_q_SA**2)/(V_q_A*c_p_SA**2 - c_p_SA**2*c_q_SA**2 + 1))
        R_m = numpy.sqrt(V_q_A)*c_q_SA**2*numpy.sqrt((V_q_A - c_q_SA**2)/(V_q_A*c_p_SA**2 - c_p_SA**2*c_q_SA**2 + 1))*(V_q_A*c_p_SA**2 - c_p_SA**2*c_q_SA**2 + 1)/(-V_q_A**2 + V_q_A*c_q_SA**2 + (V_q_A*c_p_SA**2 - c_p_SA**2*c_q_SA**2 + 1)*(numpy.sqrt(V_q_A)*c_q_SA**2*numpy.sqrt((V_q_A - c_q_SA**2)/(V_q_A*c_p_SA**2 - c_p_SA**2*c_q_SA**2 + 1)) + V_q_A**2 - V_q_A*c_q_SA**2))
        if V_p_A == 0: #V_p_A is not provided
            V_p_A = (V_q_A*c_p_SA**2 - c_p_SA**2*c_q_SA**2 + 1)/(V_q_A - c_q_SA**2)
    else:
        if V_q_A == 0 or V_p_A == 0: #V_q_A or V_p_A is not provided
            raise ValueError("In this case you have to provide V_q_A and V_p_A")
        #Apply whichever system of equations (e.g., the first)
        r_1_EB = numpy.log((V_p_A*c_q_SA**2 - c_p_SA**2*c_q_SA**2 + 1)/(V_p_A*(V_p_A - c_p_SA**2)))/4
        nu_2_EB = V_p_A*numpy.sqrt((V_p_A*c_q_SA**2 - c_p_SA**2*c_q_SA**2 + 1)/(V_p_A*(V_p_A - c_p_SA**2)))
        R_m = c_q_SA**2*(V_p_A - c_p_SA**2)/(V_p_A*c_q_SA**2 - c_p_SA**2*c_q_SA**2 + c_p_SA**2*numpy.sqrt((V_p_A*c_q_SA**2 - c_p_SA**2*c_q_SA**2 + 1)/(V_p_A*(V_p_A - c_p_SA**2))))
        if V_q_A == 0: #V_q_A is not provided
            V_q_A = (V_p_A*c_q_SA**2 - c_p_SA**2*c_q_SA**2 + 1)/(V_p_A - c_p_SA**2)
    #Construct the covariance matrix
    V = numpy.matrix(numpy.zeros((4, 4)))
    V[0, 0] = V[1, 1] = nu_2_EB
    V[2, 2] = nu_2_EB*numpy.exp(2*r_1_EB)
    V[3, 3] = nu_2_EB*numpy.exp(-2*r_1_EB)
    V[0, 2] = V[2, 0] = numpy.sqrt(nu_2_EB**2-1)*numpy.exp(r_1_EB)
    V[1, 3] = V[3, 1] = -numpy.sqrt(nu_2_EB**2-1)*numpy.exp(-r_1_EB)
    return V, R_m, V_q_A, V_p_A
#%%
def covariance_matrix_EB_ideal_transmitter_after_channel(variances_A, correlations_A, channel_efficiencies=[1, 1], channel_thermal_numbers=[0, 0]):
    """
    INPUTS
    -----------
        variances : array-like of float (shape in {(1, 2), (2, 1)}
            The variances V_x_{A} described before.
        correlations : array-like of float (shape in {(1, 2), (2, 1)}
            The correlations c_{x_{SA}} described before
        channel_efficiencies : array-like of float (shape in {(1, 2), (2, 1)}
            Transmission efficiencies of the channel, for respectively the q and p quadrature
        channel_thermal_numbers : array-like of float (shape in {(1, 2), (2, 1)}
            Thermal numbers of the channel, for respectively the q and p quadrature
    
    OUTPUTS
    -----------
        The covariance matrix of modes A' and A in the entanglement-based protocol.
        
        The reflectivity of the beam splitter used by Alice for her measurement
        
        Updated V_q_A, calculated from the other input parameters
        
        Updated V_q_B, calculated from the other input parameters   
    """
    V_AprimeA, R_m, V_q_A, V_p_A = covariance_matrix_EB_ideal_transmitter(variances_A, correlations_A)
    V_AprimeB = numpy.matrix(V_AprimeA)
    eta_q, eta_p = [channel_efficiencies[j] for j in range(2)]
    n_q, n_p = [channel_thermal_numbers[j] for j in range(2)]
    #Apply channel loss
    V_AprimeB[0, 2] *= numpy.sqrt(eta_q)
    V_AprimeB[2, 0] = V_AprimeB[0, 2]
    V_AprimeB[1, 3] *= numpy.sqrt(eta_p)
    V_AprimeB[3, 1] = V_AprimeB[1, 3]
    #Apply channel additive Gaussian noise
    V_AprimeB[2, 2] = eta_q*V_AprimeB[2, 2] + (1-eta_q)*(2*n_q+1)
    V_AprimeB[3, 3] = eta_p*V_AprimeB[3, 3] + (1-eta_p)*(2*n_p+1)
    return V_AprimeB, R_m, V_q_A, V_p_A
#%%
def symplecticOmega(n_modes):
    """
    This function computes the basic symplectic matrix associated to an N-modes bosonic field in a Gaussian state.
    
    INPUTS
    --------------
    
    n_modes : int (>0)
        number of modes of the field.
    OUTPUTS
    -------------
    
    Omega: 2D array-like of float
        the basic symplectic matrix, of dimension (2*'n_modes')X(2*'n_modes') 
    """
    #Construct the generator of the symplectic Omega
    omega = [[0, 1], [-1, 0]]
    Omega = numpy.eye(2*n_modes)
    #Construct the larger symplectic matrix
    for j in range(n_modes):
        Omega[2*j:2*j+1+1, 2*j:2*j+1+1] = omega
    return Omega
#%%
def symplecticEigenvalues(V, Omega=None):
    """
    This function computes the symplectic eigenvalues of the specified covariance matrix.
    Given the covariance matrix of a Gaussian quantum state, V, the simplectic eigenvalues of V are the set
    of the distinct eigenvalues of the matrix
                                |i Omega V|
    where Omega is the symplectic matrix.
    
    INPUTS
    -----------------
    V: 2D array-like of float
        the covariance matrix of the Gaussian quantum state of which the symplectic values are to be calculated.
        The covariance matrix is assumed to be squared, with even dimension.
    
    OUTPUTS
    -----------------
    symplectic_eigenvalues: array-like of float
        symplectic eigenvalues of the covariance matrix
    """
    N = int(V.shape[0]/2) #number of modes of the bosonic field.
    if Omega is None:
        Omega = symplecticOmega(N)    
    V = numpy.matrix(V)   
    #Construct the symplectic matrix corresponding the input covariance matrix  
    N = int(V.shape[0]/2) #number of modes of the bosonic field.
    C_symplectic = 1j*Omega @ V
    #Compute the symplectic eigenvalues of the input covariance matrix
    symplectic_eigenvalues = numpy.real(numpy.linalg.eigvals(C_symplectic))
    symplectic_eigenvalues = symplectic_eigenvalues[numpy.where(symplectic_eigenvalues>0)]
    for j in range(N):
        s = symplectic_eigenvalues[j]
        if numpy.isclose(s, 1):
            symplectic_eigenvalues[j] = 1
    if numpy.any(symplectic_eigenvalues < 1):   
        print("unphysical covariance matrix")
    return numpy.array(symplectic_eigenvalues)
#%%
def VonNeumannEntropy(V, Omega=None, print_warnings=False):
    """
    This function computes the von Neumann entropy of the Gaussian quantum state
    with 'V' as the associated covariance matrix.
    
    INPUTS
    ----------
    V : 2D array-like of float
       input covariance matrix, assumed to be squared, with even dimension.
    print_warnings: boolean
        If 'True', the function will print eventual warnings regarding the calculations. 
    OUTPUTS
    -------
    S: float
        von Neumann entropy of the Gaussian state with input covariance matrix.
    """
    ni = symplecticEigenvalues(V, Omega) #symplectic eigenvalues of the covariance matrixvacuum]:
    g = ((ni+1)/2)*numpy.log2((ni+1)/2) - ((ni-1)/2)*numpy.log2((ni-1)/2) #composite function of the symplectic eigenvalues
    g = [g_0 for g_0 in g if str(g_0) != "nan"] #if the symplectic eigenvalues are equal to 1, the numerical evaluation contains a term 0*numpy.inf
    S = numpy.sum(g) #von Neumann entropy
    if print_warnings and S<0:
        print("Warning in vonNeumannEntropy(): the calculated entropy is negative.")
    return S
#%%
def secret_key_fraction_Gaussian_CVQKD(variances_A, correlations_A, \
                                       channel_efficiencies=[1, 1], channel_thermal_numbers=[0, 0],\
                                        beta=1, Omega=None, ideal_transmitter=True, switching_probability=numpy.nan):
    """
    Computes the secret key fraction of Gaussian CVQKD against collective attacks in the asymptotic limit.
    
    INPUTS
    -------------
        variances : array-like of float (shape in {(1, 2), (2, 1)}
            The variances V_x_{A} described before.
        correlations : array-like of float (shape in {(1, 2), (2, 1)}
            The correlations c_{x_{SA}} described before
        channel_efficiencies : array-like of float (shape in {(1, 2), (2, 1)}
            Transmission efficiencies of the channel, for respectively the q and p quadrature
        channel_thermal_numbers : array-like of float (shape in {(1, 2), (2, 1)}
            Thermal numbers of the channel, for respectively the q and p quadrature  
        beta : float (in [0, 1])
            Assumed error correction efficiency
        Omega : array-like of float (shape (2N, 2N))
            Symplectic Omega.
        ideal_transmitter : bool
            If True, the key rate is computed considering an ideal transmitter, otherwise
            with a non-ideal transmitter
        switching_probability : float (in [0, 1])
            probability that the q quadrature is modulated and measured. Only valid in 
            two-dimensional protocols.
    """
    if numpy.any(correlations_A == 0) and not ideal_transmitter:
        raise TypeError("A non-ideal transmitter is not supported if the protocol is unidimensional.")
    #Calculate the covariance matrix before the channel
    V_AprimeA, R_m = covariance_matrix_EB_ideal_transmitter(variances_A, correlations_A)[0:2]
    V = covariance_matrix_EB_ideal_transmitter_after_channel(variances_A, correlations_A,\
                                                        channel_efficiencies, channel_thermal_numbers)[0]
    N = int(V.shape[0]/2)
    V = covariancematrix(data=V, N=N)
    if numpy.isnan(switching_probability) or numpy.any(correlations_A == 0):
        #Non-switching protocol
        V_given_A = vacuum(N+2)
        V_given_A[2:2+2*N, 2:2+2*N] = V
        #Propagate mode A' until Alice's homodyne detectors
        V_given_A = V_given_A.bs(1, 2, R=R_m)
        #Propagate mode B until Bob's homodyne detectors
        V_given_A = V_given_A.bs(3, 4, R=R_m)
        variances_B = numpy.array([V_given_A[4, 4], V_given_A[7, 7]])
        #Measure at Alice
        #print(V_given_A)
        V_given_A = V_given_A.homodyne_detection(1, "p").homodyne_detection(1, "x")
        conditional_variances_B = numpy.array([V_given_A[3, 3], V_given_A[0, 0]])
        #Calculate the Shannon's mutual information
        I_AB = mutual_information(variances_B, conditional_variances_B)
        #Perform measurement at Bob's
        V_given_B = vacuum(N+1)
        V_given_B[0:2*N, 0:2*N] = V
        V_given_B = V_given_B.bs(2, 3, R=R_m)
        V_given_B = V_given_B.homodyne_detection(2, "x").homodyne_detection(2, "p")
        #Compute the Von Neumann entropies
        S_V = VonNeumannEntropy(V, Omega)
        S_V_given_B = VonNeumannEntropy(V_given_B, Omega)
        Holevo_information = S_V - S_V_given_B
    else:
        I_AB, Holevo_information = (0, 0)
        for j in range(2):
            switching_factor = switching_probability*(j==0) + (1-switching_probability)*(j==1)
            R_m = 1*(j==0) + 0*(j==1)
            V_given_A = vacuum(N+2)
            V_given_A[2:2+2*N, 2:2+2*N] = V
            #Propagate mode A' until Alice's homodyne detectors
            V_given_A = V_given_A.bs(1, 2, R=R_m)
            #Propagate mode B until Bob's homodyne detectors
            V_given_A = V_given_A.bs(3, 4, R=R_m)
            variances_B = numpy.array([V_given_A[4, 4], V_given_A[7, 7]])
            #Measure at Alice
            #print(V_given_A)
            V_given_A = V_given_A.homodyne_detection(1, "p").homodyne_detection(1, "x")
            conditional_variances_B = numpy.array([V_given_A[3, 3], V_given_A[0, 0]])
            #Calculate the Shannon's mutual information
            I_AB += switching_factor*mutual_information(variances_B, conditional_variances_B)
            #Perform measurement at Bob's
            V_given_B = vacuum(N+1)
            V_given_B[0:2*N, 0:2*N] = V
            V_given_B = V_given_B.bs(2, 3, R=R_m)
            V_given_B = V_given_B.homodyne_detection(2, "x").homodyne_detection(2, "p")
            #Compute the Von Neumann entropies
            S_V = VonNeumannEntropy(V, Omega)
            S_V_given_B = VonNeumannEntropy(V_given_B, Omega)
            Holevo_information += switching_factor*(S_V - S_V_given_B)
    kappa = beta*I_AB - Holevo_information
    return kappa, I_AB, Holevo_information, V
