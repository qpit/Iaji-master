#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 23:50:58 2022

@author: jiedz

This module describes the quantum state of a simple harmonic oscillator in the 
number states (Fock) basis.
"""
# In[]
from Iaji.Mathematics.Parameter import ParameterSymbolic, ParameterNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.DensityMatrix import DensityMatrixSymbolic, \
     DensityMatrixNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.CovarianceMatrix import CovarianceMatrixSymbolic, \
     CovarianceMatrixNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.HilbertSpace import HilbertSpace
from Iaji.Physics.Theory.QuantumMechanics.QuantumState import QuantumStateSymbolic, QuantumStateNumeric, QuantumState
import sympy, numpy, scipy
from sympy import assoc_laguerre
# In[]
class QuantumStateFock(QuantumState):
    """
    This class describes the quantum state of a simple harmonic oscillator
    in the number states (Fock) basis. 
    TODONOTE: infinite-dimensional Hilbert spaces are not handeled yet, so the
    dimension of the Hilbert space is truncated up to a finite integer.
    """    
    def __init__(self, truncated_dimension, name="A"):
        super().__init__(name=name)
        self._numeric = QuantumStateFockNumeric(truncated_dimension=truncated_dimension, name=name)
        self._symbolic = QuantumStateFockSymbolic(truncated_dimension=truncated_dimension, name=name)
# In[]
class QuantumStateFockSymbolic(QuantumStateSymbolic):
    """
    This class describes the symbolic quantum state of a simple harmonic oscillator
    in the number states (Fock) basis. 
    TODONOTE: infinite-dimensional Hilbert spaces are not handeled yet, so the
    dimension of the Hilbert space is truncated up to a finite integer.
    """
    #------------------------------------------------------------
    def __init__(self, truncated_dimension, name="A"):
        super().__init__(name=name)
        self._hilbert_space = \
            HilbertSpace(dimension=truncated_dimension, name="H_{%s}"%self.name)
        self.Vacuum()
    # ---------------------------------------------------------- 
    def Vacuum(self):
         """
         Sets the quantum state to the vacuum
         Returns
         -------
         self
    
         """
         e0 = self.hilbert_space.canonical_basis[0].symbolic
         rho = e0 @ e0.T()
         self._density_operator = DensityMatrixSymbolic(name="\\hat{\\rho}_{%s}"%self.name) 
         self.density_operator.expression = rho.expression
         self.WignerFunction()
         return self
    # ---------------------------------------------------------- 
    def NumberState(self, n):
          """
          Sets the quantum state to the n-th number state
          Returns
          -------
          self
     
          """
          assert n == int(n)
          en= self.hilbert_space.canonical_basis[n].symbolic
          rho = en @ en.T()
          self._density_operator = DensityMatrixSymbolic(name="\hat{\\rho}_{%s}"%self.name) 
          self.density_operator.expression = rho.expression
          self.WignerFunction()
          return self
    #----------------------------------------------------------
    def WignerFunction(self):
        """
        Calculates the Wigner function in the number states basis from the
        density matrix
        """
        q, p = sympy.symbols("q_{%s}, p_{%s}"%(self.name, self.name), real=True)
        self._wigner_function = ParameterSymbolic(name="W_{%s}"%self.name, real=True)
        self._wigner_function.expression = 0
        N = self.hilbert_space.dimension - 1 
        W_nm = sympy.zeros(N+1, N+1) #expansion coefficients of the Wigner function
        x = 2*(q**2 + p**2)
        #Compute the lower triangle of the matrix W_nm
        for n in numpy.arange(N+1):
            for m in numpy.arange(n+1):
                k=m
                alpha = float(n-m)
                W_nm[n, m] = 1/sympy.pi * sympy.exp(-0.5*x) * (-1)**m \
                                   *(q-sympy.I*p)**(n-m) * sympy.sqrt( 2**(n-m)*sympy.factorial(m)/sympy.factorial(n) )\
                                   * assoc_laguerre(k, alpha, x)     
        #Compute the upper triangle without the diagonal
        for m in numpy.arange(N+1):
            for n in numpy.arange(m):
                W_nm[n, m] = sympy.conjugate(W_nm[m, n])
        
        #Compute the Wigner function
        for n in numpy.arange(N+1):
            for m in numpy.arange(N+1):
                self.wigner_function.expression += W_nm[n, m] * self.density_operator.expression[n, m]
        #Normalize the Wigner function
        self.wigner_function.expression /= \
            sympy.integrate(self.wigner_function.expression, (q, -sympy.oo, sympy.oo), (p, -sympy.oo, sympy.oo))
        return self.wigner_function
# In[]
class QuantumStateFockNumeric(QuantumStateNumeric):
    """
    This class describes the numeric quantum state of a simple harmonic oscillator
    in the number states (Fock) basis. 
    TODONOTE: infinite-dimensional Hilbert spaces are not handeled yet, so the
    dimension of the Hilbert space is truncated up to a finite integer.
    """
    #------------------------------------------------------------
    def __init__(self, truncated_dimension, name="A"):
        super().__init__(name=name)
        self._hilbert_space = \
            HilbertSpace(dimension=truncated_dimension, name="H_{%s}"%self.name)
        self.Vacuum()
    # ---------------------------------------------------------- 
    def Vacuum(self):
         """
         Sets the quantum state to the vacuum
         Returns
         -------
         self
    
         """
         e0 = self.hilbert_space.canonical_basis[0].numeric
         self._density_operator = e0 @ e0.T()
         #self.WignerFunction()
         return self
    # ---------------------------------------------------------- 
    def NumberState(self, n):
          """
          Sets the quantum state to the n-th number state
          Returns
          -------
          self
     
          """
          assert n == int(n)
          en = self.hilbert_space.canonical_basis[n].numeric
          self._density_operator = en @ en.T()
          self.WignerFunction()
          return self
    #----------------------------------------------------------
    def WignerFunction(self, q, p):
        """
        Calculates the Wigner function in the number states basis from the
        density matrix
        """
        P, Q = numpy.meshgrid(numpy.atleast_1d(p), numpy.atleast_1d(q))
        W = numpy.zeros((len(p), len(q)), dtype=complex)
        N = self.hilbert_space.dimension - 1 #dimension of the (truncated) Hilbert space of rho
        W_nm = numpy.zeros((len(p), len(q), N+1, N+1), dtype=complex) #expansion coefficients of the Wigner function
        X = 2*(Q**2 + P**2)
        #Compute the lower triangle of the matrix W_nm
        for n in numpy.arange(N+1):
            for m in numpy.arange(n+1):
                k=m
                alpha = float(n-m)
                W_nm[:, :, n, m] = 1/numpy.pi * numpy.exp(-0.5*X) * (-1)**m \
                                   *(Q-1j*P)**(n-m) * numpy.sqrt( 2**(n-m)*scipy.special.gamma(m+1)/scipy.special.gamma(n+1) )\
                                   * scipy.special.assoc_laguerre(x=X, n=k, k=alpha)     
        #Compute the upper triangle without the diagonal
        for m in numpy.arange(N+1):
            for n in numpy.arange(m):
                W_nm[:, :, n, m] = numpy.conj(W_nm[:, :, m, n])
        
        #Compute the Wigner function
        for n in numpy.arange(N+1):
            for m in numpy.arange(N+1):
                W += W_nm[:, :, n, m] * self.density_operator.value[n, m]
        #Normalize in L1
        dq = q[1] - q[0]
        dp = p[1] - p[0]
        W /= numpy.sum(numpy.sum(W) * dq*dp)
        self.wigner_function.value = W
        return Q, P, W
