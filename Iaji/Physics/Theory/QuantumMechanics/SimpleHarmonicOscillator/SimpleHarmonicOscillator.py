"""
This module defines a simple harmonic oscillator.
"""
# In[]
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
# In[]
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
        self._n = self.a.dagger()@self.a
        #self.n.name = "\\hat{n}_{%s}"%self.name
        self.n.symbol = sympy.symbols(names=self.n.name)
        #Hamiltonian operator
        self._H = self.hbar*(self.n + 1/2)
        #self.H.name = "\\hat{\\mathcal{H}}_{%s}"%self.name
        self.H.symbol = sympy.symbols(names=self.H.name)
        #Canonical variables
        self._q = self.hbar**(1/2)/(sympy.sqrt(2)*sympy.I)*(self.a + self.a.dagger())
        #self.q.name = "\\hat{q}_{%s}"%self.name
        self.q.symbol = sympy.symbols(names=self.q.name)
        
        self._p = self.hbar**(1/2)/(sympy.sqrt(2)*sympy.I)*(self.a - self.a.dagger())
        #self.p.name = "\\hat{p}_{%s}"%self.name
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
    #----------------------------------------------------------
    def _AnnihilationOperator(self):
        """
        Calculates the annihilation operator of the system 
        in the Fock basis
        """
        self.a.expression = sympy.zeros(self.hilbert_space.dimension, \
                                      self.hilbert_space.dimension)
        for j in range(self.hilbert_space.dimension-1):
            self.a.expression[j, j+1] = j
    #----------------------------------------------------------
    def Vacuum(self):
        """
        Sets the system to the vacuum state
        """
        self.state.Vacuum()
        return self
    #----------------------------------------------------------
    def NumberState(self, n):
        """
        Sets the system to the n-th number state
        """
        self.state.NumberState(n)
        return self
    #----------------------------------------------------------
    def Displace(self, alpha):
        """
        Displaces the system, by applying the unitary displacement operator
        """
        #assert type(alpha) in [int, float, complex]
        D = (self.a.dagger()*alpha-self.a*numpy.conjugate(alpha)).Exp()
        self.state._density_operator = D @ self.state.density_operator @ D.dagger()
        self.state.WignerFunction()
        return self
    #----------------------------------------------------------
    def Squeeze(self, zeta):
        """
        Squeezes the system, by applying the unitary squeezing operator
        """
        #assert type(zeta) in [int, float, complex]
        S = ((self.a.dagger()**2*zeta-self.a**2*(numpy.conjugate(zeta)))*sympy.sympify(1/2)).Exp()
        self.state._density_operator = S @ self.state.density_operator @ S.dagger()
        self.state.WignerFunction()
        return self
    #----------------------------------------------------------
    def Rotate(self, theta):
        """
        Rotates the system, by applying the unitary rotation operator
        """
        #assert type(theta) in [int, float]
        R = (self.n*sympy.sympify(-1j*float(theta))).Exp()
        self.state._density_operator = R @ self.state.density_operator @ R.dagger()
        self.state.WignerFunction()
        return self
    #----------------------------------------------------------
    
# In[]
class SimpleHarmonicOscillatorNumeric: #TODO
    """
    This class describes a numeric simple harmonic oscillator.
    """
