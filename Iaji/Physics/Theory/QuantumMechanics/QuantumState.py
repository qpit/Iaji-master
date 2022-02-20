"""
This module describes a mixed quantum state or a quantum harmonic oscillator
"""
# In[]
from Iaji.Mathematics.Parameter import ParameterSymbolic, ParameterNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.DensityMatrix import DensityMatrixSymbolic, \
     DensityMatrixNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.CovarianceMatrix import CovarianceMatrixSymbolic, \
     CovarianceMatrixNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.HilbertSpace import HilbertSpace
import sympy, numpy
from sympy import assoc_laguerre
# In[]
print_separator = "-----------------------------------------------"
# In[]
class QuantumState:
    """
    This class describes a quantum state.
    It consists of:
        - an associated Hilbert space on the complex numbers
        - a density operator
        - a Wigner function
    """
    #------------------------------------------------------------
    def __init__(self, name="\\rho"):
        self.name = name
        self._numeric = QuantumStateNumeric(name=name)
        self._symbolic = QuantumStateSymbolic(name=name)
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
    def __str__(self):
        s = "QUANTUM STATE: \n" + "name: " + self.name.__str__() + "\n" + print_separator+"\n"\
            + self.symbolic.__str__() + "\n" + print_separator+"\n"\
            + self.numeric.__str__() + "\n" + print_separator
        return s
    # ---------------------------------------------------------- 
    #Tensor product
    def otimes(self, other):
        x = QuantumState(name=self.name.__str__() + " \\otimes " + other.name.__str__())
        x._symbolic = self.symbolic.otimes(other.symbolic)
        x._numeric = self.numeric.otimes(other.numeric)
        return x
    # ----------------------------------------------------------
# In[]
class QuantumStateSymbolic:
    """
    This class describes a symbolic quantum state.
    It consists of:
        - an associated Hilbert space on the complex numbers
        - a density operator
        - a Wigner function
    """
    #------------------------------------------------------------
    def __init__(self, name="\\rho"):
        self.name = name
        self._hilbert_space = None
        self._density_operator = DensityMatrixSymbolic(name=self.name)
        self._wigner_function = ParameterSymbolic(name="W", real=True)
        self._covariance_matrix = CovarianceMatrixSymbolic(name="V")
    #------------------------------------------------------------
    @property
    def hilbert_space(self):
        return self._hilbert_space
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
    def density_operator(self):
        return self._density_operator
    @density_operator.deleter
    def density_operator(self):
        del self._density_operator
    # ---------------------------------------------------------- 
    @property 
    def wigner_function(self):
        return self._wigner_function
    @wigner_function.deleter
    def wigner_function(self):
        del self._wigner_function
    # ---------------------------------------------------------- 
    @property 
    def covariance_matrix(self):
        return self._covariance_matrix
    @covariance_matrix.deleter
    def covariance_matrix(self):
        del self._covariance_matrix
    # ---------------------------------------------------------- 
    def __str__(self):
        s = "SYMBOLIC QUANTUM STATE: \n" + "name: " + self.name.__str__() + "\n" + print_separator+"\n"\
            + "Hilbert space: \n" + self.hilbert_space.__str__()+ "\n" + print_separator+"\n"\
            +"density operator: \n" + self.density_operator.__str__() + "\n" + print_separator+"\n"\
            +"Wigner function: \n" + self.wigner_function.__str__() + "\n" + print_separator + "\n"\
            + "covariance matrix: \n" + self.covariance_matrix.__str__() + "\n" + print_separator
        return s
    # ----------------------------------------------------------
    # Tensor product
    def otimes(self, other):
        x = QuantumStateSymbolic(name=self.name.__str__() + " \\otimes " + other.name.__str__())
        x._hilbert_space = self.hilbert_space.otimes(other.hilbert_space)
        x._density_operator = self.density_operator.otimes(other.density_operator)
        x._wigner_function = self.wigner_function * other.wigner_function
        #x._covariance_matrix = self.covariance_matrix.oplus(other.covariance_matrix)
        return x
# In[]
class QuantumStateNumeric:
    """
    This class describes a numeric quantum state.
    It consists of:
        - an associated Hilbert space on the complex numbers
        - a density operator
        - a Wigner function
    """
    #------------------------------------------------------------
    def __init__(self, name="A"):
        self.name = name
        self._hilbert_space = None
        self._density_operator = DensityMatrixNumeric(name="\\rho_{%s}"%self.name)
        self._wigner_function = ParameterNumeric(name="W_{%s}"%self.name)
        self._covariance_matrix = CovarianceMatrixNumeric(name="V_{%s}"%self.name)
    #------------------------------------------------------------
    @property
    def hilbert_space(self):
        return self._hilbert_space
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
    def density_operator(self):
        return self._density_operator
    @density_operator.deleter
    def density_operator(self):
        del self._density_operator
    # ---------------------------------------------------------- 
    @property 
    def wigner_function(self):
        return self._wigner_function
    @wigner_function.deleter
    def wigner_function(self):
        del self._wigner_function
    # ---------------------------------------------------------- 
    @property 
    def covariance_matrix(self):
        return self._covariance_matrix
    @covariance_matrix.deleter
    def covariance_matrix(self):
        del self._covariance_matrix
    # ---------------------------------------------------------- 
    def __str__(self):
        s = "NUMERIC QUANTUM STATE: \n" + "name: " + self.name.__str__() + "\n" + print_separator+"\n"\
            + "Hilbert space: \n" + self.hilbert_space.__str__()+ "\n" + print_separator+"\n"\
            +"density operator: \n" + self.density_operator.__str__() + "\n" + print_separator+"\n"\
            +"Wigner function: \n" + self.wigner_function.__str__() + "\n" + print_separator + "\n" \
            + "covariance matrix: \n" + self.covariance_matrix.__str__() + "\n" + print_separator
        return s
    # ----------------------------------------------------------
    # Tensor product
    def otimes(self, other):
        name = "\\left(%s\\otimes%s\\right)"%(self.name, other.name)
        x = QuantumStateSymbolic(name=name)
        x._hilbert_space = self.hilbert_space.otimes(other.hilbert_space)
        x._density_operator = self.density_operator.otimes(other.density_operator)
        x._wigner_function = self.wigner_function * other.wigner_function
        #x._covariance_matrix = self.covariance_matrix.oplus(other.covariance_matrix)
        return x