"""
This module defines a simple harmonic oscillator.
"""
# In[]
import Iaji
from Iaji.Mathematics.Parameter import ParameterSymbolic, ParameterNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix import MatrixSymbolic, MatrixNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.DensityMatrix import DensityMatrixSymbolic, \
     DensityMatrixNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.CovarianceMatrix import CovarianceMatrixSymbolic, \
     CovarianceMatrixNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.HilbertSpace import HilbertSpace
from Iaji.Physics.Theory.QuantumMechanics.QuantumState import QuantumStateSymbolic, QuantumStateNumeric
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.QuantumStateFock import QuantumStateFockSymbolic, \
    QuantumStateFockNumeric
import sympy, numpy
from sympy import assoc_laguerre
# In[]
ACCEPTED_HBAR_VALUES = [1] #TODO: adapt the calculation of the Wigner function from the 
                           #density operator in the Fock basis to admit values of hbar 
                           #different from 1. The calculation is performed in the module QuantumStateFock
"""
For hbar=1, the vacuum quadrature variance is equal to 1/2 and the Heisenberg
inequality reads Var(q)Var(p) >= 1/4

Fro hbar=2, the vacuum quadrature variance is equal to 1 and the Heisenberg
inequality reads Var(q)Var(p) >= 1
"""
# In[]
class SimpleHarmonicOscillator: #TODO
    """
    This class describes a simple harmonic oscillator.
    It consists of:
        - a Hilbert space
        - a quantum state
        - an annihilation (creation) operator
        - a Hamiltonian operator
        - quadrature operators
    Quantum states and operators are represented in the number states basis, 
    and the Hilbert space is truncated to a finite dimension.
    The Planck length is set to an arbitrary value.
    The system is described in the Schrodinger picture, where the quantum state
    of the system evolves according to unitary transformations while the 
    Hamiltonian operator remains unchanged.
    """
# In[symbolic simple harmonic oscillator]
class SimpleHarmonicOscillatorSymbolic: 
    """
    This class describes a symbolic simple harmonic oscillator.
    """
    #----------------------------------------------------------
    def __init__(self, truncated_dimension, name="A", hbar=1):
        assert hbar in ACCEPTED_HBAR_VALUES
        self.name = name
        self._hilbert_space = HilbertSpace(dimension=truncated_dimension, name="H_{%s}"%self.name)
        self._hbar = ParameterSymbolic(name="\\hbar", real=True, nonnegative=True)
        self.hbar.expression = hbar
        self._state = QuantumStateFockSymbolic(truncated_dimension=truncated_dimension, name=name) 
        #Define the most relevant operators
        #Annihilation operator
        self._a = MatrixSymbolic(name="\\hat{a}_{%s}"%self.name)
        self.a.expression = sympy.zeros(self.hilbert_space.dimension, \
                                      self.hilbert_space.dimension)
        for j in range(self.hilbert_space.dimension-1):
            self.a.expression[j, j+1] = sympy.sqrt(sympy.sympify(float(j+1)))
        #Number operator
        self._n = self.a.Dagger()@self.a
        self.n.name = "\\hat{n}_{%s}"%self.name
        self.n.symbol = sympy.symbols(names=self.n.name)
        #Hamiltonian operator
        self._H = self.hbar*(self.n + 1/2)
        self.H.name = "\\hat{\\mathcal{H}}_{%s}"%self.name
        self.H.symbol = sympy.symbols(names=self.H.name)
        #Canonical variables
        self._q = self.hbar**(1/2)/(sympy.sqrt(2)*sympy.I)*(self.a + self.a.Dagger())
        self.q.name = "\\hat{q}_{%s}"%self.name
        self.q.symbol = sympy.symbols(names=self.q.name)
        
        self._p = self.hbar**(1/2)/(sympy.sqrt(2)*sympy.I)*(self.a - self.a.Dagger())
        self.p.name = "\\hat{p}_{%s}"%self.name
        self.p.symbol = sympy.symbols(names=self.p.name)
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
    def hbar(self):
        return self._hbar   
    @hbar.deleter
    def hbar(self):
        del self._hbar
    # ---------------------------------------------------------- 
    @property
    def hilbert_space(self):
        return self._hilbert_space
    #----------------------------------------------------------
    @property
    def state(self):
        return self._state
    #----------------------------------------------------------
    @property
    def a(self):
        return self._a
    #----------------------------------------------------------
    @property
    def n(self):
        return self._n
    #----------------------------------------------------------
    @property
    def H(self):
        return self._H
    #----------------------------------------------------------
    @property
    def q(self):
        return self._q
    #----------------------------------------------------------
    @property
    def p(self):
        return self._p
    # ---------------------------------------------------------- 
    def _AnnihilationOperator(self):
        """
        Calculates the annihilation operator of the system 
        in the Fock basis
        """
        self.a.expression = sympy.zeros(self.hilbert_space.dimension, \
                                      self.hilbert_space.dimension)
        for j in range(self.hilbert_space.dimension-1):
            self.a.expression[j, j+1] = sympy.sqrt(sympy.sympify(float(j+1)))
    #----------------------------------------------------------
    def Vacuum(self):
        """
        Sets the system to the vacuum state
        """
        x = SimpleHarmonicOscillatorSymbolic(self.hilbert_space.dimension, self.name, float(self.hbar.expression))
        x.state.Vacuum()
        return x
    #----------------------------------------------------------
    def NumberState(self, n):
        """
        Sets the system to the n-th number state
        """
        x = SimpleHarmonicOscillatorSymbolic(self.hilbert_space.dimension, self.name, float(self.hbar.expression))
        x.state.NumberState(n)
        return x
    #----------------------------------------------------------
    def Displace(self, alpha):
        """
        Displaces the system, by applying the unitary displacement operator
        """
        x = SimpleHarmonicOscillatorSymbolic(self.hilbert_space.dimension, self.name, float(self.hbar.expression))
        x._state = self._state
        try:
            #assume alpha is of type ParameterSymbolic
            D = (x.a.Dagger()*alpha-x.a*alpha.Conjugate()).ExpTruncated(10)
        except:
            D = (x.a.Dagger()*alpha-x.a*sympy.conjugate(alpha)).ExpTruncated(10)     
        print(D.name)
        x.state._density_operator = D @ x.state._density_operator @ D.Dagger()
        x.state.WignerFunction()
        return x
    #----------------------------------------------------------
    def Squeeze(self, zeta):
        """
        Squeezes the system, by applying the unitary squeezing operator
        """
        x = SimpleHarmonicOscillatorSymbolic(self.hilbert_space.dimension, self.name, float(self.hbar.expression))
        x._state = self._state
        try:
            #assume zeta is of type ParameterSymbolic
            S = ((x.a.Dagger()**2*zeta-x.a**2*zeta.Conjugate())*sympy.sympify(1/2)).ExpTruncated(10)
        except:
            S = ((x.a.Dagger()**2*zeta-x.a**2*sympy.conjugate(zeta))*sympy.sympify(1/2)).ExpTruncated(10)
        x.state._density_operator = S @ x.state._density_operator @ S.Dagger()
        x.state.WignerFunction()
        return x
    #----------------------------------------------------------
    def Rotate(self, theta):
        """
        Rotates the system, by applying the unitary rotation operator
        """
        x = SimpleHarmonicOscillatorSymbolic(self.hilbert_space.dimension, self.name, float(self.hbar.expression))
        x._state = self._state
        try:
            #assume theta is of type ParameterSymbolic
            R = (x.n*sympy.sympify(1j)*theta).ExpTruncated(10)
        except:
            R = (x.n*sympy.sympify(1j*float(theta))).ExpTruncated(10)
        x.state._density_operator = R @ x.state._density_operator @ R.Dagger()
        x.state.WignerFunction()
        return x
    #----------------------------------------------------------
    def Unitary(self, H):
        """
        Evolves the system according to the arbitrary interaction Hamiltonian defined by
        the operator H. The system is assumed to be stationary.
        """
        assert type(H) is Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix.MatrixSymbolic, \
            "H must be a symbolic matrix, instead it is %s"%(type(H))
        assert H.isHermitian(), \
            "The interaction Hamiltonian operator must be Hermitian"
        assert H.shape == self.H.shape, \
            "The interaction Hamiltonian has incompatible shape %s"%H.shape.__str__()
        x = SimpleHarmonicOscillatorSymbolic(self.hilbert_space.dimension, self.name, float(self.hbar.expression))
        x._state = self._state
        U = (H*x.hbar*sympy.sympify(-sympy.I)).ExpTruncated(10)
        x.state._density_operator = U @ x.state._density_operator @ U.Dagger()
        x.state.WignerFunction()
    
# In[symbolic simple harmonic oscillator]
class SimpleHarmonicOscillatorNumeric: 
    """
    This class describes a Numeric simple harmonic oscillator.
    """
    #----------------------------------------------------------
    def __init__(self, truncated_dimension, name="A", hbar=1):
        assert hbar in ACCEPTED_HBAR_VALUES
        self.name = name
        self._hilbert_space = HilbertSpace(dimension=truncated_dimension, name="H_{%s}"%self.name)
        self._hbar = ParameterNumeric(name="\\hbar")
        self.hbar.value = hbar
        self._state = QuantumStateFockNumeric(truncated_dimension=truncated_dimension, name=name) 
        #Define the most relevant operators
        #Annihilation operator
        self._a = MatrixNumeric(name="\\hat{a}_{%s}"%self.name)
        self.a.value = sympy.zeros(self.hilbert_space.dimension, \
                                      self.hilbert_space.dimension)
        for j in range(self.hilbert_space.dimension-1):
            self.a.value[j, j+1] = numpy.sqrt(j+1)
        #Number operator
        self._n = self.a.Dagger()@self.a
        self.n.name = "\\hat{n}_{%s}"%self.name
        self.n.symbol = sympy.symbols(names=self.n.name)
        #Hamiltonian operator
        self._H = self.hbar*(self.n + 1/2)
        self.H.name = "\\hat{\\mathcal{H}}_{%s}"%self.name
        self.H.symbol = sympy.symbols(names=self.H.name)
        #Canonical variables
        self._q = self.hbar**(1/2)/(numpy.sqrt(2)*1j)*(self.a + self.a.Dagger())
        self.q.name = "\\hat{q}_{%s}"%self.name
        self.q.symbol = sympy.symbols(names=self.q.name)
        
        self._p = self.hbar**(1/2)/(numpy.sqrt(2)*1j)*(self.a - self.a.Dagger())
        self.p.name = "\\hat{p}_{%s}"%self.name
        self.p.symbol = sympy.symbols(names=self.p.name)
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
    def hbar(self):
        return self._hbar   
    @hbar.deleter
    def hbar(self):
        del self._hbar
    # ---------------------------------------------------------- 
    @property
    def hilbert_space(self):
        return self._hilbert_space
    #----------------------------------------------------------
    @property
    def state(self):
        return self._state
    #----------------------------------------------------------
    @property
    def a(self):
        return self._a
    #----------------------------------------------------------
    @property
    def n(self):
        return self._n
    #----------------------------------------------------------
    @property
    def H(self):
        return self._H
    #----------------------------------------------------------
    @property
    def q(self):
        return self._q
    #----------------------------------------------------------
    @property
    def p(self):
        return self._p
    # ---------------------------------------------------------- 
    def _AnnihilationOperator(self):
        """
        Calculates the annihilation operator of the system 
        in the Fock basis
        """
        self.a.value = sympy.zeros(self.hilbert_space.dimension, \
                                      self.hilbert_space.dimension)
        for j in range(self.hilbert_space.dimension-1):
            self.a.value[j, j+1] = numpy.sqrt(j+1)
    #----------------------------------------------------------
    def Vacuum(self):
        """
        Sets the system to the vacuum state
        """
        x = SimpleHarmonicOscillatorNumeric(self.hilbert_space.dimension, self.name, float(self.hbar.value))
        x.state.Vacuum()
        return x
    #----------------------------------------------------------
    def NumberState(self, n):
        """
        Sets the system to the n-th number state
        """
        x = SimpleHarmonicOscillatorNumeric(self.hilbert_space.dimension, self.name, float(self.hbar.value))
        x.state.NumberState(n)
        return x
    #----------------------------------------------------------
    def Displace(self, alpha):
        """
        Displaces the system, by applying the unitary displacement operator
        """
        x = SimpleHarmonicOscillatorNumeric(self.hilbert_space.dimension, self.name, float(self.hbar.value))
        x._state = self._state
        try:
            #assume alpha is of type ParameterNumeric
            D = (x.a.Dagger()*alpha-x.a*alpha.Conjugate()).Exp()
        except:
            D = (x.a.Dagger()*alpha-x.a*numpy.conjugate(alpha)).Exp()
        
        x.state._density_operator = D @ x.state._density_operator @ D.Dagger()
        return x
    #----------------------------------------------------------
    def Squeeze(self, zeta):
        """
        Squeezes the system, by applying the unitary squeezing operator
        """
        x = SimpleHarmonicOscillatorNumeric(self.hilbert_space.dimension, self.name, float(self.hbar.value))
        x._state = self._state
        try:
            #assume zeta is of type ParameterNumeric
            S = ((x.a.Dagger()**2*zeta-x.a**2*zeta.Conjugate())*0.5).Exp()
        except:
            S = ((x.a.Dagger()**2*zeta-x.a**2*numpy.conjugate(zeta))*0.5).Exp()
        x.state._density_operator = S @ x.state._density_operator @ S.Dagger()
        return x
    #----------------------------------------------------------
    def Rotate(self, theta):
        """
        Rotates the system, by applying the unitary rotation operator
        """
        x = SimpleHarmonicOscillatorNumeric(self.hilbert_space.dimension, self.name, float(self.hbar.value))
        x._state = self._state
        R = (x.n*1j*theta).Exp()
        x.state._density_operator = R @ x.state._density_operator @ R.Dagger()
        return x
    #----------------------------------------------------------
    def Unitary(self, H):
        """
        Evolves the system according to the arbitrary interaction Hamiltonian defined by
        the operator H. The system is assumed to be stationary.
        """
        assert type(H) is Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix.MatrixNumeric, \
            "H must be a Numeric matrix, instead it is %s"%(type(H))
        assert H.isHermitian(), \
            "The interaction Hamiltonian operator must be Hermitian"
        assert H.shape == self.H.shape, \
            "The interaction Hamiltonian has incompatible shape %s"%H.shape.__str__()
        x = SimpleHarmonicOscillatorNumeric(self.hilbert_space.dimension, self.name, float(self.hbar.value))
        x._state = self._state
        U = (H*x.hbar*(-1j)).Exp()
        x.state._density_operator = U @ x.state._density_operator @ U.Dagger()