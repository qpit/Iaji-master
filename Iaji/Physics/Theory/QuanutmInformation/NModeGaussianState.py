#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:28:10 2021

@author: jiedz
This file contains methods that are useful in quantum information, such as
checking the physicality of quantum states, computing symplectic eigenvalues of
covariance matrices, and calculating the Von Neumann entropy of a Gaussian state.
"""
import numpy as np
import numpy.linalg
from uncertainties import ufloat
from uncertainties import umath
import sympy
sympy.init_printing()
from scipy.linalg import sqrtm
from quik.qip.nmodes_symbolic import CovarianceMatrix
from quik.qip.nmodes import covariancematrix
from quik.qip.nmodes import vacuum
#%%
def symplecticOmega(n_modes, form='symbolic'):
    """
    This function computes the basic symplectic matrix associated to an N-modes bosonic field in a Gaussian state.
    
    INPUTS
    --------------
    
    n_modes : int (>0)
        number of modes of the field.
    form : string
        'numeric' or 'symbolic'
    
    OUTPUTS
    -------------
    
    Omega: 2D array-like of float
        the basic symplectic matrix, of dimension (2*'n_modes')X(2*'n_modes') 
    """
    #Construct the generator of the symplectic Omega
    omega = [[0, 1], [-1, 0]]
    Omega = np.eye(2*n_modes)
    #Construct the larger symplectic matrix
    for j in range(n_modes):
        Omega[2*j:2*j+1+1, 2*j:2*j+1+1] = omega
    return Omega
#%%
def symplecticEigenvalues(covariance_matrix):
    """
    This function computes the symplectic eigenvalues of the specified covariance matrix.
    Given the covariance matrix of a Gaussian quantum state, V, the simplectic eigenvalues of V are the set
    of the distinct modulii of the eigenvalues of the matrix
                                i Omega V
    where Omega is the symplectic matrix.
    
    INPUTS
    -----------------
    covariance_matrix: 2D array-like of float
        the covariance matrix of the Gaussian quantum state of which the symplectic values are to be calculated.
        The covariance matrix is assumed to be squared, with even dimension.
    
    OUTPUTS
    -----------------
    symplectic_eigenvalues: array-like of float
        symplectic eigenvalues of the covariance matrix
    """
    N = int(covariance_matrix.shape[0]/2) #number of modes of the bosonic field.
    Omega = symplecticOmega(N)    
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)   
    #Construct the symplectic matrix corresponding the input covariance matrix  
    N = int(covariance_matrix.shape[0]/2) #number of modes of the bosonic field.
    #Construct the basic symplectic matrix
    Omega = np.asarray(symplecticOmega(N, form="numeric"), dtype=float)
    C_symplectic = 1j*Omega @ covariance_matrix
    #Compute the symplectic eigenvalues of the input covariance matrix
    symplectic_eigenvalues = np.abs(np.linalg.eig(C_symplectic)[0])
    symplectic_eigenvalues = symplectic_eigenvalues[::3]
    for j in range(N):
        s = symplectic_eigenvalues[j]
        if np.isclose(s, 1):
            symplectic_eigenvalues[j] = 1
    if np.any(symplectic_eigenvalues < 1):   
        print("unphysical covariance matrix")
    return np.array(symplectic_eigenvalues)
#%%
def VonNeumannEntropy(covariance_matrix, print_warnings=False):
    """
    This function computes the von Neumann entropy of the Gaussian quantum state
    with 'covariance_matrix' as the associated covariance matrix.
    
    INPUTS
    ----------
    covariance_matrix : 2D array-like of float
       input covariance matrix, assumed to be squared, with even dimension.
    print_warnings: boolean
        If 'True', the function will print eventual warnings regarding the calculations. 
    OUTPUTS
    -------
    S: float
        von Neumann entropy of the Gaussian state with input covariance matrix.
    """
    ni = symplecticEigenvalues(covariance_matrix) #symplectic eigenvalues of the covariance matrixvacuum]:   
    g = ((ni+1)/2)*np.log2((ni+1)/2) - ((ni-1)/2)*np.log2((ni-1)/2) #composite function of the symplectic eigenvalues
    g = [g_0 for g_0 in g if str(g_0) != "nan"]
    S = np.sum(g) #von Neumann entropy
    if print_warnings and S<0:
        print("Warning in vonNeumannEntropy(): the calculated entropy is negative.")
    return S
#%%
def mutualInformation(variance_1, variance_2, covariance):
    """
    This function computes the Shannon's mutual information of two
    dependent Gaussian random variables, with respective variances 'variance_1' and 'variance_2' and
    covariance 'covariance'
    
    INPUTS
    ---------------------
    variance_1: float (>0)
        variance of the first variable [a.u.].
        
    variance_2: float(>0)
        variance of the second variable [a.u.].
        
    covariance: float
        covariance between the two variables.
    print_warnings: boolean
        If 'True', the function will print eventual warnings regarding the calculations.     
    
    OUTPUTS
    -------------------
    mutual_information: float (>0)
        Shannon's mutual information of the two independent variables.
    """
    return -1/2*np.log2(1-covariance**2/(variance_1*variance_2))
#%%
#Validity check functions
def isPositiveDefinite(matrix):
    """
    This function checks whether the input matrix is positive definite, i.e., whether all its eigenvalues are greater than zero.

    INPUTS
    ----------
    matrix : 2D array-like of floats
        input square matrix.

    OUTPUTS
    -------
    is_positive_definite: boolean
        it is true if and only if the input matrix is positive definite
    """
    
    return np.all(np.linalg.eigvals(matrix) > 0)

def isPositiveSemiDefinite(matrix):
    """
    This function checks whether the input matrix is positive semidefinite, i.e., whether all its eigenvalues are not lower than zero.

    INPUTS
    ----------
    matrix : 2D array-like of floats
        input square matrix.

    OUTPUTS
    -------
    is_positive_definite: boolean
        it is true if and only if the input matrix is positive semidefinite
    """
    try:
        return np.all(np.linalg.eigvals(matrix) >= 0) 
    except:
        raise TypeError
    
def isCovarianceMatrix(matrix):
    """
    This function checks whether the input square matrix is a covariance matrix, i.e., it satisfies the following:
        - It is positive semidefinite (i.e., all is eigenvalue are not lower than zero);
        - It is symmetric (i.e., it is equal to its transpose);
        - It has an inverse (i.e., the determinant is not close to zero);
        
    INPUTS
    ----------
    matrix : array-like of float
        input square matrix

    OUTPUTS
    -------
    is_covariance_matrix : boolean
        it is true if an only if the input matrix is a covariance matrix

    """
    try:
        matrix = np.matrix(matrix)
        min_eigenvalue = np.min(np.abs(np.linalg.eigvals(matrix))) #modulus of the minimum eigenvalue of the input matrix
        tolerance = 1/100*min_eigenvalue
        return isPositiveSemiDefinite(matrix) and np.allclose(matrix, matrix.T, atol = tolerance) and not np.isclose(np.linalg.det(matrix), 0, atol=tolerance)
    except:
        raise TypeError
        

#%%
#Define exceptions
class UnphysicalError(Exception):
    """
    Exception raised when a covariance matrix is unphysical
    """
    pass

class NotACovarianceMatrixError(Exception):
    """
    Exeption raised when a matrix is not a covariance matrix
    """
    pass
