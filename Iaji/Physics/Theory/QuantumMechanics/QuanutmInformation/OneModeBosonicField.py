#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 09:58:05 2022

@author: jiedz
This module defines a one-mode bosonic quantum field, and some of 
its transformations.
"""
# In[Generic imports]
import numpy, sympy
# In[Iaji imports]
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.SimpleHarmonicOscillator\
    import SimpleHarmonicOscillatorNumeric, SimpleHarmonicOscillatorSymbolic,\
        SimpleHarmonicOscillator
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix import \
    MatrixSymbolic, MatrixNumeric, Matrix
from Iaji.Mathematics.Parameter import \
    ParameterSymbolic, ParameterNumeric, Parameter
# In[One-mode bosonic field]
class OneModeBosonicField(SimpleHarmonicOscillator): #TODO
    """
    This class describes a one-mode bosonic quantum field. 
    It consists of a simple harmonic oscillator.
    It may be equipped with additional properties or functions, such as
    a composition operation, used to compose bosonic quantum fields.
    """
    #----------------------------------------------------------
    def __init__(self, truncated_dimension, name="A", hbar=1):
        self.name = name
        self._symbolic = OneModeBosonicFieldSymbolic(truncated_dimension=truncated_dimension, name=name, hbar=hbar)
        self._numeric = OneModeBosonicFieldNumeric(truncated_dimension=truncated_dimension, name=name, hbar=hbar)
    #----------------------------------------------------------
    @property 
    def name(self):
        return self._name
    @name.setter
    def name(self, name):
        self._name = name   
    @name.deleter
    def name(self):
        del self._name
    # ---------------------------------------------------------- 
    @property
    def symbolic(self):
        return self._symbolic

    @symbolic.deleter
    def symbolic(self):
        del self._symbolic
    # ----------------------------------------------------------
    @property
    def numeric(self):
        return self._numeric

    @numeric.deleter
    def numeric(self):
        del self._numeric
    # ----------------------------------------------------------
# In[Symbolic one-mode bosonic field]
class OneModeBosonicFieldSymbolic(SimpleHarmonicOscillatorSymbolic):
    """
    This class describes a symbolic one-mode bosonic quantum field
    """
    #----------------------------------------------------------
    def __init__(self, truncated_dimension, name="A", hbar=1):
        super().__init__(truncated_dimension, name, hbar)
    #----------------------------------------------------------
# In[Numeric one-mode bosonic field]
class OneModeBosonicFieldNumeric(SimpleHarmonicOscillatorNumeric):
    """
    This class describes a numeric one-mode bosonic quantum field
    """
    #----------------------------------------------------------
    def __init__(self, truncated_dimension, name="A", hbar=1):
        super().__init__(truncated_dimension, name, hbar)