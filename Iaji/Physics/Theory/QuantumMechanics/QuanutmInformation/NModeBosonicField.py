#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 09:58:05 2022

@author: jiedz
This module defines a N-mode bosonic quantum field, and some of 
its transformations.
"""
# In[Generic imports]
import numpy, sympy
# In[Iaji imports]
from Iaji.Physics.Theory.QuantumMechanics.QuanutmInformation.OneModeBosonicField \
    import OneModeBosonicFieldSymbolic, OneModeBosonicFieldNumeric, OneModeBosonicField
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix import \
    MatrixSymbolic, MatrixNumeric, Matrix
from Iaji.Mathematics.Parameter import \
    ParameterSymbolic, ParameterNumeric, Parameter
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.HilbertSpace import \
    HilbertSpace
from copy import deepcopy as copy
# In[]
MEASURABLES = ["n", "x"]
POVM_TYPES = ["on/off detection"]
# In[N-mode bosonic field]
class NModeBosonicField: #TODO
    """
    This class describes a N-mode bosonic quantum field. 
    It consists of a simple harmonic oscillator.
    It may be equipped with additional properties or functions, such as
    a composition operation, used to compose bosonic quantum fields.
    """
    #----------------------------------------------------------
    def __init__(self, truncated_dimension, name="A", modes_list=None, hbar=1):
        if modes_list is not None:
            assert numpy.all([truncated_dimension == mode.numeric.hilbert_space.dimension \
                              for mode in modes_list]), \
            "The input truncated dimension does not match the dimension of the truncated hilbert space of all input modes"
            assert numpy.all([hbar == mode.numeric.hbar.value \
                              for mode in modes_list]), \
            "The input value of hbar does not match the value of hbar of all input modes"    
        self.name = name
        self._symbolic = NModeBosonicFieldSymbolic(truncated_dimension=truncated_dimension, name=name, modes_list=None, hbar=hbar)
        self._numeric = NModeBosonicFieldNumeric(truncated_dimension=truncated_dimension, name=name, modes_list=None, hbar=hbar)
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
# In[Symbolic N-mode bosonic field]
class NModeBosonicFieldSymbolic: #TODO
    """
    This class describes a symbolic N-mode bosonic quantum field
    """
    #----------------------------------------------------------
    def __init__(self, truncated_dimension, name="A", modes_list=None, hbar=1):
        if modes_list is not None:
            assert numpy.all([truncated_dimension == mode.hilbert_space.dimension \
                              for mode in modes_list]), \
            "The input truncated dimension does not match the dimension of the truncated hilbert space of all input modes"
            assert numpy.all([hbar == float(mode.hbar.expression) \
                              for mode in modes_list]), \
            "The input value of hbar does not match the value of hbar of all input modes" 
            names = [mode.name for mode in modes_list]
            self._modes = dict(zip(names, modes_list))
        else:
            self._modes = {"A_{1}": OneModeBosonicFieldNumeric(truncated_dimension, name="A_{1}", hbar=hbar).Vacuum()}
    #----------------------------------------------------------
# In[Numeric N-mode bosonic field]
class NModeBosonicFieldNumeric:
    """
    This class describes a numeric N-mode bosonic quantum field
    """
    #----------------------------------------------------------
    def __init__(self, modes_list, name="A", hbar=1):
        """
        Parameters
        ----------
        modes_list : iterable of Iaji OneModeBosonicFieldNumeric
            The list of modes that make up the N-mode field.
        name : str
            Name given to the N-mode field
        hbar : type in {float, Iaji ParameterNumeric}
            the Planck's constant that defines the normalization of the theory.
        Returns
        -------
        None.

        """
        self._symbol = sympy.symbols(names=name)
        self._name = name
        self._hbar = ParameterNumeric(name="\\hbar")
        self.hbar.value = hbar
        self._mode_names = [mode._name for mode in modes_list]
        self._modes = dict(zip(self._mode_names, modes_list))
        self._modes_list = list(self.modes.values()) 
        #Number of modes
        self._N = len(list(self._mode_names)) 
        #Composite Hilbert space
        self._hilbert_space = HilbertSpace.TensorProduct([self.modes[name].hilbert_space for name in self.mode_names])
        #Composite quantum state
        self._state = self.modes[self.mode_names[0]].state
        for name in self.mode_names[1:]:
            self._state = self.state.Otimes(self.modes[name].state)
    #----------------------------------------------------------
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, name):
        self._name = name
        self._symbol = sympy.symbols(names=name)
    @name.deleter
    def name(self):
        del self._name
    #---------------------------------------------------------
    @property 
    def name(self):
        return self._name
    @name.setter
    def name(self, name):
        self._name = name 
        self._symbol = sympy.symbols(names=name, real=self.symbol.is_real, nonnegative=self.symbol.is_nonnegative)
    @name.deleter
    def name(self):
        del self._name
    # ---------------------------------------------------------- 
    @property
    def symbol(self):
        return self._symbol

    @symbol.setter
    def symbol(self, symbol):
        self._symbol = symbol

    @symbol.deleter
    def symbol(self):
        del self._symbol
    #----------------------------------------------------------
    @property 
    def hbar(self):
        return self._hbar   
    @hbar.deleter
    def hbar(self):
        del self._hbar
    # ---------------------------------------------------------- 
    @property
    def modes(self):
        return self._modes
    @modes.deleter
    def modes(self):
        del self._modes
    #---------------------------------------------------------
    @property
    def mode_names(self):
        return self._mode_names
    @mode_names.setter
    def mode_names(self, mode_names):
        assert len(mode_names) == len(self.mode_names), \
            "The field has %d modes, but %d new mode names were specified"\
                %(self.N, len(mode_names))
        self._mode_names = mode_names
        #Set the new name to each mode
        for j in range(self.N):
            self.modes_list[j]._name = mode_names[j]
        #Redefine the dictionary self.modes
        self._modes = dict(zip(self.mode_names, self.modes_list))
    @mode_names.deleter
    def mode_names(self):
        del self._mode_names
    #----------------------------------------------------------
    @property
    def modes_list(self):
        return self._modes_list
    @modes_list.deleter
    def modes_list(self):
        del self._modes_list
    #---------------------------------------------------------
    @property
    def N(self):
        self._N = len(self.modes)
        return self._N
    @N.deleter
    def N(self):
        del self._N
    #----------------------------------------------------------
    @property
    def hilbert_space(self):
        return self._hilbert_space
    @hilbert_space.deleter
    def hilbert_space(self):
        del self._hilbert_space
    #----------------------------------------------------------
    @property
    def state(self):
        return self._state
    @state.deleter
    def state(self):
        del self._state
    #----------------------------------------------------------
    @classmethod
    def Vacuum(cls, N, truncated_dimensions, name="A", mode_names=None, hbar=1):
        if mode_names is not None:
            assert len(mode_names) == N, \
                "list of names of length %d does not match the number of modes N = %d" \
                    %(len(mode_names), N)   
        else:
            mode_names = ["%s_{%d}"%(name, j) for j in range(N)]
        truncated_dimensions = numpy.atleast_1d(truncated_dimensions)
        if truncated_dimensions.size == 1:
            modes_list = [OneModeBosonicFieldNumeric(truncated_dimensions[0], name=mode_names[j], hbar=hbar).Vacuum() \
                      for j in range(N)]
        else:
            assert truncated_dimensions.size == len(mode_names), \
                "The array of Hilbert space dimensions has length %d, different from the number %d of mode names"\
                    %(truncated_dimensions.size, len(mode_names))
            modes_list = [OneModeBosonicFieldNumeric(truncated_dimensions[j], name=mode_names[j], hbar=hbar).Vacuum() \
                      for j in range(N)]
        return NModeBosonicFieldNumeric(name=name, modes_list=modes_list, hbar=1)
    #----------------------------------------------------------
    @classmethod
    def NumberStates(cls, N, truncated_dimensions, number_state_orders, mode_names=None, name="A", hbar=1):
        assert len(truncated_dimensions) == len(number_state_orders), \
            "%d Hilbert space dimensions were specified, while %d number state orders were specified"\
                %(len(truncated_dimensions), len(number_state_orders))
        if mode_names is not None:
            assert len(mode_names) == N, \
                "list of names of length %d does not match the number of modes N = %d" \
                    %(len(mode_names), N)   
        else:
            mode_names = ["%s_{%d}"%(name, j) for j in range(N)]
        truncated_dimensions = numpy.atleast_1d(truncated_dimensions)
        if truncated_dimensions.size == 1:
            modes_list = [OneModeBosonicFieldNumeric(truncated_dimensions[0], name=mode_names[j], hbar=hbar).Vacuum() \
                      for j in range(N)]
        else:
            assert truncated_dimensions.size == len(mode_names), \
                "The array of Hilbert space dimensions has length %d, different from the number %d of mode names"\
                    %(truncated_dimensions.size, mode_names)
            modes_list = [OneModeBosonicFieldNumeric(truncated_dimensions[j], name=mode_names[j], hbar=hbar).NumberState(number_state_orders[j]) \
                      for j in range(N)]
        return NModeBosonicFieldNumeric(name=name, modes_list=modes_list, hbar=1)
    #----------------------------------------------------------
    def PartialTrace(self, traced_mode_name):
        """
        Traces out the mode with name 'traced_mode'
        """
        field = copy(self)
        assert traced_mode_name in field.mode_names, \
            "No mode named %s in the field"%(traced_mode_name)
        traced_index = numpy.where(numpy.array(field.mode_names) == traced_mode_name)[0][0]
        traced_mode = field._modes_list[traced_index]
        modes_before = field._modes_list[0:traced_index]
        modes_after = field._modes_list[traced_index+1:]
        modes_list = modes_before + modes_after
        name = "Tr_{%s}\\left(%s\\right)"%(traced_mode_name, field._name)
        x = NModeBosonicFieldNumeric(modes_list, name, field._hbar.value)
        x.state.wigner_function.value = None
        x.state.covariance_matrix.value = None
        #Compute the partial trace of the density operator
        rho = copy(field).state._density_operator
        rho_new = MatrixNumeric.Zeros((x.hilbert_space.dimension, x.hilbert_space.dimension))
        for j in range(traced_mode.hilbert_space.dimension):
            ej = traced_mode.hilbert_space.CanonicalBasisVector(j)
            M_j = MatrixNumeric.TensorProduct(\
                  [*[MatrixNumeric.Eye(mode.hilbert_space.dimension) for mode in modes_before], \
                   ej.numeric.T(), \
                    *[MatrixNumeric.Eye(mode.hilbert_space.dimension) for mode in modes_after]])
            rho_new += M_j @ rho @ M_j.T()
        x.state._density_operator = rho_new
        x.state.name = "Tr_{%s}\\left(%s\\right)"%(traced_mode.name, field._state.name)
        x.state.density_operator.name = "Tr_{%s}\\left(%s\\right)"%(traced_mode.name, field._state.density_operator.name)
        x.state._density_operator = x.state.density_operator.Hermitian()
        return x
    #----------------------------------------------------------
    def SelectModes(self, modes):
        """
        Traces out all the modes that have not been selected
        
        INPUTS
        ----------------
            mode : type in {str, int}
                name or index of the selected modes
        """
        modes = numpy.atleast_1d(modes)
        if "str" in str(type(modes[0])):
            assert numpy.all([modes[j] in self.mode_names for j in range(len(modes))]), \
                "Not all the specified modes %s are contained in the field"\
                    %(modes)
        field = self
        traced_modes_names = list(set(self.mode_names)-set(modes))
        for mode_name in traced_modes_names:
            field = field.PartialTrace(mode_name)
        return field
    #----------------------------------------------------------
    def Displace(self, mode, alpha):
        """
        Performs single-mode displacement operations on the selected mode       
        
        INPUTS
        ------------------
            mode : type in {str, int}
                name or index of the mode to be displaced
            alpha: type in {Iaji ParameterNumeric, complex}
                Displacement to be applied on the selected mode
        """
        if "str" in str(type(mode)):
            assert mode in self.mode_names, \
                "No such mode with name %s in the field"%mode
            #Transform names in indices
            mode_index = numpy.where(numpy.array(self.mode_names) == mode)[0][0]
        #Compute the multimode displacement operator#
        field = copy(self)
        mode = field.modes_list[mode_index]
        modes_before = field.modes_list[0:mode_index]
        modes_after = field.modes_list[mode_index+1:]
        #Construct the displacement operator
        D = MatrixNumeric.TensorProduct([*[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_before], \
                                         mode._DisplacementOperator(alpha), \
                                        *[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_after]])
        #Apply the displacement operator
        rho_name = field.state.density_operator.name
        field.state._density_operator = D @ field.state.density_operator @ D.Dagger()
        field.state.density_operator.name = "%s\\left(%s\\right)"%(D.name, rho_name)
        field.state._density_operator = field.state.density_operator.Hermitian()
        return field
    #----------------------------------------------------------
    def Squeeze(self, mode, zeta):
        """
        Performs single-mode squeezing on the selected mode       
        INPUTS
        ------------------
            mode : type in {str, int}
                name or index of the mode to be displaced
            zeta : type in {Iaji ParameterNumeric, complex}
                Squeezing parameter to be applied on the selected mode
        """
        #OLD
        '''
        """
        Performs single-mode squeezing operations on the individual modes.
        
        INPUTS
        ------------------
            zetas: iterable of Iaji ParameterNumeric or complex
                Squeezings to be performed on each mode. If zetas
                is a dictionary, the corresponding keys must be equal to the mode names.
                If it is an array-like object, then the Squeezings will follow 
                the same order as self.modes_list
        """
        if type(zetas) == dict:
            assert set(list(zetas.keys())) == set(self.mode_names), \
                "names associated to the input zetas do not match the mode names"
            #reorder the zetas and return a list of zetas
            zetas = [zetas[mode.name] for mode in self.modes_list]
        #Compute the multimode Squeezing operator
        field = copy(self)
        S = MatrixNumeric.TensorProduct([field.modes_list[j]._SqueezingOperator(zetas[j]) \
                                           for j in range(field.N)])
        #Apply the Squeezing operator
        field.state._density_operator = S @ field.state.density_operator @ S.Dagger()
        field.state.density_operator.name = "%s\\left(%s\\right)"%(S.name, self.state.density_operator.name)
        return field
        '''
        #New
        if "str" in str(type(mode)):
            assert mode in self.mode_names, \
                "No such mode with name %s in the field"%mode
            #Transform names in indices
            mode_index = numpy.where(numpy.array(self.mode_names) == mode)[0][0]
        #Compute the multimode displacement operator#
        field = copy(self)
        mode = field.modes_list[mode_index]
        modes_before = field.modes_list[0:mode_index]
        modes_after = field.modes_list[mode_index+1:]
        #Construct the displacement operator
        S = MatrixNumeric.TensorProduct([*[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_before], \
                                         mode._SqueezingOperator(zeta), \
                                        *[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_after]])
        #Apply the displacement operator
        rho_name = field.state.density_operator.name
        field.state._density_operator = S @ field.state.density_operator @ S.Dagger()
        field.state.density_operator.name = "%s\\left(%s\\right)"%(S.name, rho_name)
        field.state._density_operator = field.state.density_operator.Hermitian()
        return field
    #----------------------------------------------------------
    def Rotate(self, mode, theta):
        """
        OLD
        Performs single-mode rotation operations on the individual modes.
        
        INPUTS
        ------------------
            thetas: iterable of Iaji ParameterNumeric or float
                Rotations to be performed on each mode. If thetas
                is a dictionary, the corresponding keys must be equal to the mode names.
                If it is an array-like object, then the Rotations will follow 
                the same order as self.modes_list
        """
        """
        if type(thetas) == dict:
            assert set(list(thetas.keys())) == set(self.mode_names), \
                "names associated to the input thetas do not match the mode names"
            #reorder the thetas and return a list of thetas
            thetas = [thetas[mode.name] for mode in self.modes_list]
        #Compute the multimode Rotation operator
        field = copy(self)
        R = MatrixNumeric.TensorProduct([field.modes_list[j]._RotationOperator(thetas[j]) \
                                           for j in range(field.N)])
        #Apply the Rotation operator        
        field.state._density_operator = R @ field.state.density_operator @ R.Dagger()
        field.state.density_operator.name = "%s\\left(%s\\right)"%(R.name, self.state.density_operator.name)
        return field
        """
        
        """
        Performs single-mode rotation on the selected mode       
        INPUTS
        ------------------
            mode : type in {str, int}
                name or index of the mode to be displaced
            theta: type in {Iaji ParameterNumeric, complex}
                Rotation angle to be applied on the selected mode
        """
        if "str" in str(type(mode)):
            assert mode in self.mode_names, \
                "No such mode with name %s in the field"%mode
            #Transform names in indices
            mode_index = numpy.where(numpy.array(self.mode_names) == mode)[0][0]
        #Compute the multimode displacement operator#
        field = copy(self)
        mode = field.modes_list[mode_index]
        modes_before = field.modes_list[0:mode_index]
        modes_after = field.modes_list[mode_index+1:]
        #Construct the displacement operator
        R = MatrixNumeric.TensorProduct([*[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_before], \
                                         mode._RotationOperator(theta), \
                                        *[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_after]])
        #Apply the displacement operator
        rho_name = field.state.density_operator.name
        field.state._density_operator = R @ field.state.density_operator @ R.Dagger()
        field.state.density_operator.name = "%s\\left(%s\\right)"%(R.name, rho_name)
        field.state._density_operator = field.state.density_operator.Hermitian()
        return field        
    #----------------------------------------------------------
    def TwoModeSqueeze(self, modes, zeta):
        """
        Performs two-mode squeezing on the input modes with two-mode squeezing
        parameter zeta
        
        INPUTS
        ----------------
            mode : type in {str, int}
                name or index of the selected mode
            zeta : type in {Iaji ParameterNumeric, complex}
                two-mode squeezing parameter
        """
        modes = numpy.atleast_1d(modes)
        assert modes.size == 2, \
            "Two modes must be specified"
        if "str" in str(type(modes[0])):
            #Transform names in indices
            mode_indices = [numpy.where(numpy.array(self.mode_names) == modes[j])[0][0] \
                            for j in range(modes.size)]
        mode_indices.sort()
        field = copy(self)
        modes = [field.modes_list[j] for j in mode_indices]
        modes_before = field.modes_list[0:mode_indices[0]]
        modes_between = field.modes_list[mode_indices[0]+1:mode_indices[1]]
        modes_after = field.modes_list[mode_indices[1]+1:]
        #Compute the two-mode squeezing operator
        #Exponent
        exponent1 = MatrixNumeric.TensorProduct([*[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_before], \
                                                modes[0].a.Dagger()**2, \
                                                *[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_between], \
                                                modes[1].a.Dagger()**2, \
                                                *[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_after]])
        exponent2 = MatrixNumeric.TensorProduct([*[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_before], \
                                                modes[0].a**2, \
                                                *[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_between], \
                                                modes[1].a**2, \
                                                *[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_after]])
        exponent1 *= zeta
        try:
            exponent2 *= zeta.Conjugate()
            name = "\\hat{\\mathcal{S}}_{%s%s}\\left(%s\\right)\\left(%s\\right)"\
                %(modes[0].name, modes[1].name, zeta.valuename, field.state.density_operator.name)
        except:
            exponent2 *= numpy.conjugate(zeta)
            name = "\\hat{\\mathcal{S}}_{%s%s}\\left(%s\\right)\\left(%s\\right)"\
                %(modes[0].name, modes[1].name, zeta, field.state.density_operator.name)
        S = (exponent1-exponent2).Exp()
        field.state._density_operator = S @ field.state.density_operator @ S.Dagger()
        #field.state._density_operator /= field.state._density_operator.Trace()
        field.state.density_operator.name = name
        field.state._density_operator = field.state.density_operator.Hermitian()
        return field
     #----------------------------------------------------------   
    def BeamSplitter(self, modes, R):
        """
        Applies a two-port beam splitter to the selected modes, with power
        reflectivity R. 
        
        INPUTS
        ---------------
            mode : type in {str, int}
                name or index of the selected modes
        """
        modes = numpy.atleast_1d(modes)
        assert modes.size == 2, \
            "Two modes must be specified"
        if "str" in str(type(modes[0])):
            assert numpy.all([modes[j] in self.mode_names for j in range(len(modes))]), \
                "Not all the specified modes %s are contained in the field"\
                    %(modes)
            #Transform names in indices
            mode_indices = [numpy.where(numpy.array(self.mode_names) == modes[j])[0][0] \
                            for j in range(modes.size)]
        mode_indices.sort()
        field = copy(self)
        modes = [field.modes_list[j] for j in mode_indices]
        modes_before = field.modes_list[0:mode_indices[0]]
        modes_between = field.modes_list[mode_indices[0]+1:mode_indices[1]]
        modes_after = field.modes_list[mode_indices[1]+1:]
        #Compute the two-mode squeezing operator
        try:
            theta = (R.Sqrt()).Arcsin()
        except:
           theta = numpy.arcsin(numpy.sqrt(R))
        #Exponent
        exponent1 = MatrixNumeric.TensorProduct([*[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_before], \
                                                modes[0].a.Dagger(), \
                                                *[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_between], \
                                                modes[1].a, \
                                                *[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_after]])
        exponent2 = MatrixNumeric.TensorProduct([*[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_before], \
                                                modes[0].a, \
                                                *[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_between], \
                                                modes[1].a.Dagger(), \
                                                *[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_after]])
        try:
             name = "\\hat{\\mathcal{B}}_{%s%s}\\left(R=%s\\right)\\left(%s\\right)"\
                 %(modes[0].name, modes[1].name, R.name, field.state.density_operator.name)
        except:
             name = "\\hat{\\mathcal{B}}_{%s%s}\\left(R=%s\\right)\\left(%s\\right)"\
                 %(modes[0].name, modes[1].name, R, field.state.density_operator.name)
        B = ((exponent1-exponent2)*theta).Exp()
        field.state._density_operator = B @ field.state.density_operator @ B.Dagger()
        #field.state._density_operator /= field.state._density_operator.Trace()
        field.state.density_operator.name = name
        field.state._density_operator = field.state.density_operator.Hermitian()
        return field      
    #----------------------------------------------------------
    def Loss(self, modes, etas):
        """
        Applies bosonic loss to the selected field modes
        
        INPUTS
        ----------------
            mode : type in {str, int}
                name or index of the selected modes
            etas : type in {Iaji ParameterNumeric, float}
                transmission efficiencies
        """
        field = copy(self)
        modes = numpy.atleast_1d(modes)
        etas = numpy.atleast_1d(etas)
        if "str" in str(type(modes[0])):
            assert numpy.all([modes[j] in self.mode_names for j in range(len(modes))]), \
                "Not all the specified modes %s are contained in the field"\
                    %(modes)
            #Transform names in indices
            mode_indices = [numpy.where(numpy.array(self.mode_names) == modes[j])[0][0] \
                            for j in range(modes.size)]
        else:
            mode_indices = modes
        modes = [field.modes_list[j] for j in mode_indices]
        vacuum_field = NModeBosonicFieldNumeric.Vacuum(N=len(modes), \
                                                       truncated_dimensions=[m.hilbert_space.dimension for m in modes], \
                                                           name="Vacuum")
        field = field.Otimes(vacuum_field)
        for j in range(len(mode_indices)):
            field = field.BeamSplitter(modes=[modes[j].name, vacuum_field.mode_names[j]], R=1-etas[j])\
                .PartialTrace(vacuum_field.mode_names[j])
        #Make the name of the new field
        name = "Loss_{"
        for mode in modes:
            name += "%s"%mode.name
        name += "}\\left("
        def value(eta):
            try:
                return eta.value
            except AttributeError:
                return eta
        for eta in etas:
            name += "%.2f"%value(eta)
        name += "\\right)\\left(%s\\right)"%self.state.density_operator.name
        field.state.density_operator.name = name
        return field
    #----------------------------------------------------------
    def _GeneralizedBornRule(self, measurement_operator):
        field = copy(self)
        p = (field.state.density_operator @ measurement_operator).Trace() #outcome probability density
        field.state._density_operator = measurement_operator @ field.state.density_operator \
                                  @ measurement_operator.Dagger()
        field.state._density_operator /= p
        return field, p
    #----------------------------------------------------------
    def ProjectiveMeasurement(self, mode, measurable, ntimes=1, return_all_fields=False, **kwargs):
        """
        Peforms a projective measurement of a measurable quantity associated
        to a linear operator on the selected mode, following the generalized Born rule.
        It repeats the measurement 'ntimes' times (assuming 'ntimes' identical
                                                   copies of the system exist)
        
        INPUTS
        --------------
            mode : type in {str, int}
                name or index of the mode to be measured
            measurable : str
                name of the measurable quantity
            ntimes : int
                number of times the measurement is performed
            return_all_states : bool
                If True, all the post-measurement modes are returned in an array.
                Otherwise, only the post-measurement field associated to the last
                measurement outcomes is returned.
                Strictly speaking, the quantum state is what is being transformed by
                a measurement and not the field, although the field is uniquely associated
                to one quantum state after a given transformation. For convenience, 
                the whole field is returned for easy manipulation in later code.
                
        """
        assert measurable in MEASURABLES,\
        "%s is not supported as a measurable quantity. \n It should be one of these: %s"\
            %(measurable, MEASURABLES)
        if "str" in str(type(mode)):
            #Transform names in indices
            mode_index = numpy.where(numpy.array(self.mode_names) == mode)[0][0]
        field = copy(self)
        mode = field.modes_list[mode_index]
        modes_before = field.modes_list[0:mode_index]
        modes_after = field.modes_list[mode_index+1:]
        def _generalized_born_rule(projector):
            field0 = copy(self)
            p = (field.state.density_operator @ projector).Trace() #outcome probability density
            field0.state._density_operator = projector @ field0.state.density_operator \
                                      @ projector
            field0.state._density_operator /= p
            return field0, p
        #-------------------
        if measurable == "n":
            def Projector(n):
                en = mode.hilbert_space.CanonicalBasisVector(n).numeric
                projector = MatrixNumeric.TensorProduct([*[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_before], \
                                                         en @ en.T(), \
                                                          *[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_after]])
                return projector
            #Calculate the probabilities of outcomes
            values = numpy.arange(mode.hilbert_space.dimension)
            p = numpy.zeros((len(values), ))           
            for j in numpy.arange(len(values)):
                n = values[j]
                projector = Projector(n) 
                p[j] = _generalized_born_rule(projector)[1].value
            p = numpy.abs(p)
            p /= numpy.sum(p)
            #Sample according to the calculated probabilities
            outcomes = numpy.random.choice(values, size=(ntimes,), p=p)
            if return_all_fields:
                #Apply the generalized born rule to the all measurements
                projector = []
                post_measurement_field = []
                for j in range(ntimes):
                    projector.append(Projector(outcomes[j]))
                    post_measurement_field.append(_generalized_born_rule(projector[j])[0])
            else:
                #Apply the generalized born rule to the last measurement
                projector = Projector(outcomes[-1])
                post_measurement_field = _generalized_born_rule(projector)[0]
        #-------------------
        elif measurable == "x":
            def Projector(x, theta):
                proj = MatrixNumeric.Zeros((N+1, 1))              
                for n in numpy.arange(N):
                    if n==0:
                        proj.value[n] = 1/(numpy.sqrt(numpy.sqrt(numpy.pi)))*numpy.exp(-0.5*x**2)
                    elif n==1:
                        proj.value[n] = x*numpy.sqrt(2)*numpy.exp(1j*theta) * proj.value[0]
                    else:
                        proj.value[n] = numpy.exp(1j*theta)/numpy.sqrt(n)*(numpy.sqrt(2)*x*proj.value[n-1] - numpy.exp(1j*theta)*numpy.sqrt(n-1)*proj.value[n-2]) 
                proj = proj @ proj.Dagger() 
                proj = MatrixNumeric.TensorProduct([*[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_before], \
                                                         proj, \
                                                          *[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_after]])
                proj /= proj.Trace()
                return proj   
            theta = kwargs["theta"]    
            N = mode.hilbert_space.dimension - 1
            #Consider a range of values that spans a few standard deviations beyond
            #the mean value of the number operator, which defines the energy
            #of the harmonic oscillator
            q_theta = mode.q*numpy.cos(theta) + mode.p*numpy.sin(theta)
            max_x = (mode.state.Mean(q_theta).value + mode.state.Std(q_theta).value*5)
            n_values = 300
            x_values = numpy.linspace(-max_x, max_x, n_values)
            p = numpy.zeros((n_values,))
            for j in range(n_values):
                x = x_values[j]
                projector = Projector(x, theta)
                p[j] = _generalized_born_rule(projector)[1].value
            p = numpy.abs(p)
            p /= numpy.sum(p) 
            #from matplotlib import pyplot
            #pyplot.figure()
            #pyplot.plot(x_values, p)
            #Sample according to the calculated probabilities
            outcomes = numpy.random.choice(x_values, size=(ntimes,), p=p)
            if return_all_fields:
                #Apply the generalized born rule to the all measurements
                projector = []
                post_measurement_field = []
                for j in range(ntimes):
                    projector.append(Projector(outcomes[j], theta))
                    post_measurement_field.append(_generalized_born_rule(projector[j])[0])
            else:
                #Apply the generalized born rule to the last measurement
                projector = Projector(outcomes[-1], theta)
                post_measurement_field = _generalized_born_rule(projector)[0]
            values = x_values
        return outcomes, values, p, post_measurement_field
    #----------------------------------------------------------
    def POVM(self, mode, measurement_operators=None, ntimes=1, return_all_fields=False, **kwargs):
        """
        Peforms a positive operator-valued measurement on the selected mode,
        following the generalized Born rule.
        It repeats the measurement 'ntimes' times (assuming 'ntimes' identical
                                                   copies of the system exist)
        
        INPUTS
        --------------
            mode : type in {str, int}
                name or index of the mode to be measured
            measurement_operators : iterable of Iaji Matrix
                operators forming the POVM. They must add up to the identity matrix.
            povm_type : str
                in-built type of the POVM to be performed.
                If measurement_operators is also specified, the function will
                use the latter to perform the POVM
            ntimes : int
                number of times the measurement is performed
            return_all_states : bool
                If True, all the post-measurement modes are returned in an array.
                Otherwise, only the post-measurement field associated to the last
                measurement outcomes is returned.
                Strictly speaking, the quantum state is what is being transformed by
                a measurement and not the field, although the field is uniquely associated
                to one quantum state after a given transformation. For convenience, 
                the whole field is returned for easy manipulation in later code.
                
        """
        measurement_operators = list(numpy.atleast_1d(measurement_operators))
        if "str" in str(type(mode)):
            #Transform names in indices
            mode_index = numpy.where(numpy.array(self.mode_names) == mode)[0][0]
        field = copy(self)
        mode_name = copy(mode)
        mode = field.modes_list[mode_index]
        modes_before = field.modes_list[0:mode_index]
        modes_after = field.modes_list[mode_index+1:]
        def _generalized_born_rule(measurement_operator):
            field0 = copy(self)
            p = (field.state.density_operator @ measurement_operator).Trace() #outcome probability density
            field0.state._density_operator = measurement_operator @ field0.state.density_operator \
                                      @ measurement_operator.Dagger()
            field0.state._density_operator /= p
            return field0, p
        #-------------------
        if len(measurement_operators) > 0 and measurement_operators[0] != None:
            #Fetch the values of the POVM
            if "values" not in list(kwargs.keys()):
                values = numpy.arange(len(measurement_operators))
            else:
                values = kwargs["values"]
            p = numpy.zeros((len(values), ))           
            for j in numpy.arange(len(values)):
                p[j] = _generalized_born_rule(measurement_operators[j])[1].value
            p = numpy.abs(p)
            p /= numpy.sum(p)
            #Sample according to the calculated probabilities
            outcomes = numpy.random.choice(values, size=(ntimes,), p=p)
            if return_all_fields:
                #Apply the generalized born rule to the all measurements
                post_measurement_field = []
                for j in range(ntimes):
                    value = outcomes[j]
                    index = numpy.where(numpy.isclose(values, value))[0][0]
                    post_measurement_field.append(_generalized_born_rule(measurement_operators[index])[0])
            else:
                #Apply the generalized born rule to the last measurement
                value = outcomes[-1]
                index = numpy.where(numpy.isclose(values, value))[0][0]
                post_measurement_field = _generalized_born_rule(measurement_operators[index])[0]
            return outcomes, values, p, post_measurement_field
        else:
            assert "povm_type" in list(kwargs.keys()), \
                "Measurement operators are not specified. You need to specify a standard povm type in %s" \
                    %POVM_TYPES
            povm_type = kwargs["povm_type"]
            assert povm_type in POVM_TYPES, \
                "Invalid POVM type. Accepted types are %s" \
                    %POVM_TYPES
            if povm_type == "on/off detection":
                d = mode.hilbert_space.dimension
                measurement_operators = [0, 0]
                e0 = mode.hilbert_space.CanonicalBasisVector(0).numeric
                measurement_operators[0] = MatrixNumeric.TensorProduct([*[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_before], \
                                                         e0 @ e0.T(), \
                                                          *[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_after]])
                measurement_operators[1] = MatrixNumeric.TensorProduct([*[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_before], \
                                                         MatrixNumeric.Eye(d) - e0 @ e0.T(), \
                                                          *[MatrixNumeric.Eye(m.hilbert_space.dimension) for m in modes_after]])
                return self.POVM(mode_name, measurement_operators, ntimes, return_all_fields, values=[0, 1])
    #----------------------------------------------------------
    def Otimes(self, other):
        """
        Composition of two N-mode bosonic fields
        """
        field_self = copy(self)
        field_other = copy(other)
        assert field_self.hbar.value == field_other.hbar.value, \
            "The two fields %s and %s must have the same value of hbar"\
                %(field_self.name, field_other.name)
        field = NModeBosonicFieldNumeric(modes_list = field_self.modes_list+field_other.modes_list, \
                                         name="%s%s"%(field_self.name, field_other.name), \
                                             hbar=field_self.hbar.value)
        field._state = field_self.state.Otimes(field_other.state)
        field._hilbert_space = field.state.hilbert_space
        return field

        