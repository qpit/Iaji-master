
import numpy as np
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt
#import sympy
#from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.DensityMatrix import DensityMatrix
#from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.QuantumStateFock import *
#sympy.init_printing()

'''
#TODO
- Include scissor as a function of input state and Tb
- Include success rate
- Include transmission losses
'''

# Functions that will be used in the code
def factorial(n):
    '''
    Returns the factorial of n
    :param n: int
        It should be non negative
    :return: n!: int
    '''
    fact = 1
    if n > 0:
        for i in range(1, n+1):
            fact = fact*i
    if n < 0:
        print('ERROR: n should be a positive integer')
    return fact

def density_matrix(x):
    '''
    Returns the density matrix of a state written in the Fock basis
    :param x: 1D array
        Fock number distribution of the state
    :return: rho: 2D array
    '''
    x_conj = np.conjugate(x)
    n_max = len(x)
    rho = np.zeros((n_max, n_max))
    for i in range(n_max):
        for j in range(n_max):
            rho[i][j] = x_conj[j]*x[i]
    return rho

def trace(M):
    '''
    Computes the trace of a matrix
    :param M: 2D array
        Matrix
    :return: trace: float
    '''
    if len(M.shape) != 2:
        print("Input is not a matrix.")
        return
    if M.shape[0] != M.shape[1]:
        print("Matrix is not square.")
        return
    tr = 0
    for i in range(M.shape[0]):
        tr += M[i][i]
    return tr

def beam_splitter(transmission, input_state):
    '''
    Beam splitter operation for density matrices
    :param transmission: float
        Transmission of the beam splitter
    :param input_state: 2D array
        Matrix describing the input state, elements i,j correspond to coefficients of |\\psi\\rangle = \\sum_{i,j} c_{i,j} |i\\rangle|j\\rangle
    :return:
    '''
    dimension = len(input_state)
    bs = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            k = 0
            n = 0
            sum = 0
            while k <= n:
                while n <= i + j:
                    sum += factorial(i)*factorial(j)/(factorial(k)*factorial(i-k)*factorial(n-k)*factorial(j-n+k))
            bs[i][j] =

#-------------------------------------------------------------------------
# Setting the initial parameters
n_max = 10

# Defining the states
def coherent_state_fock(alpha):
    return np.array([np.exp(-abs(alpha)**2/2)*alpha**i/np.sqrt(factorial(i)) for i in range(n_max)])
def coherent_state_rho(alpha):
     return density_matrix(coherent_state_fock(alpha))

null_fill = [0]*(n_max-2)
def scissor_output_fock(alpha):
    return (1/np.sqrt(coherent_state_fock(alpha)[0]**2 + coherent_state_fock(alpha)[1]**2))*np.array([coherent_state_fock(alpha)[0], coherent_state_fock(alpha)[1], *null_fill])
def scissor_output_rho(alpha):
    return density_matrix(scissor_output_fock(alpha))

def fidelity(alpha):
    return np.real((trace(linalg.sqrtm(linalg.sqrtm(coherent_state_rho(alpha))*scissor_output_rho(alpha)*linalg.sqrtm(coherent_state_rho(alpha)))))**2)

# Plot the fidelity
alphas = np.linspace(0,1,1000)
f = [fidelity(alpha) for alpha in alphas]

plt.figure()
plt.plot(alphas, f, color = 'm')
plt.title("Fidelity for different input state sizes")
plt.xlabel("\\alpha")
plt.ylabel("\\mathcal(F)")
plt.show()