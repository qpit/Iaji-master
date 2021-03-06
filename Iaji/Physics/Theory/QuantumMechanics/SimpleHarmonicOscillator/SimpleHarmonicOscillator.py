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
from copy import deepcopy as copy
# In[]
ACCEPTED_HBAR_VALUES = [1] #TODO: adapt the calculation of the Wigner function from the 
                           #density operator in the Fock basis to admit values of hbar 
                           #different from 1. The calculation is performed in the module QuantumStateFock
MEASURABLES = ["n", "a", "x"]
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
    def __init__(self, truncated_dimension, name="A", hbar=1):
        self.name = name
        self._symbolic = SimpleHarmonicOscillatorSymbolic(truncated_dimension=truncated_dimension, name=name, hbar=hbar)
        self._numeric = SimpleHarmonicOscillatorNumeric(truncated_dimension=truncated_dimension, name=name, hbar=hbar)
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
# In[symbolic simple harmonic oscillator]
class SimpleHarmonicOscillatorSymbolic: 
    """
    This class describes a symbolic simple harmonic oscillator.
    TODO: implement a symbolic version of a projective measurement.
    So far, I have only found a useful way of doing so in SimpleHarmonicOscillatorNumeric
    class.
    """
    #----------------------------------------------------------
    def __init__(self, truncated_dimension, name="A", hbar=1):
        assert hbar in ACCEPTED_HBAR_VALUES
        self.symbol = sympy.symbols(names=name)
        self._name = name    
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
        self._q = self.hbar**(1/2)/(sympy.sqrt(2))*(self.a + self.a.Dagger())
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
    def _GeneralizeQuadratureProjector(self, x, theta):
        """
        Computes, the homodyne detection POVM in the Fock basis, up
        to order "N" (i.e, in a reduced-dimension Hilbert space). 
        
        For a quadrature value x, the (n, m) element of the POVM matrix is
        <X|m>*<X|n>. The expression of <X|k> depends on the associated LO phase
        "theta"
        
        n_max: positive integer
        x: scalar
        theta: scalar
        """
        #--------------------------------------------
        def expression(x):
            try:
                return x.expression
            except:
                return x
        #--------------------------------------------
        #For each element of x, a POVM matrix is computed
        N = self.hilbert_space.dimension - 1
        try:
            if "pi" in theta.__str__():
                theta_sym = expression(theta)/sympy.pi
            else:
                theta_sym = expression(theta)/numpy.pi
        except:
            theta_sym = expression(theta)/numpy.pi
        name = "\\hat{\\Pi}_{q_{{%s}_{%.3f\\pi}}}(%s)"%(self.name, theta_sym, expression(x))
        projector = MatrixSymbolic(name=name)
        projector.expression = sympy.zeros(N+1, 1)
        for n in numpy.arange(N):
            if n==0:
                projector.expression[n] = 1/(sympy.sqrt(sympy.sqrt(sympy.pi)))*sympy.exp(-0.5*expression(x)**2)
            elif n==1:
                projector.expression[n] = expression(x)*sympy.sqrt(2)*sympy.exp(1j*expression(theta)) * projector.expression[0]
            else:
                projector.expression[n] = sympy.exp(1j*expression(theta))/sympy.sqrt(n)*(sympy.sqrt(2)*expression(x)*projector.expression[n-1] - sympy.exp(1j*expression(theta))*sympy.sqrt(n-1)*projector.expression[n-2])
        projector = projector @ projector.Dagger()
        projector.name = name
        return projector
    #----------------------------------------------------------
    def _AnnihilationProjector(self, alpha):
        """
        Computes the projection operator onto an eigenstate of the annihilation
        operator,i.e., a coherent state with amplitude alpha.
        """
        #--------------------------------------------
        def expression(x):
            try:
                return x.expression
            except:
                return x
        #--------------------------------------------
        name = "\\hat{\\Pi}_{\\alpha_{%s}}(%s)"%(self.name, expression(alpha))
        projector = copy(self.Vacuum()).Displace(alpha).state.density_operator
        projector.name = name
        return projector
    #----------------------------------------------------------
    def _NumberProjector(self, n):
        """
        Computes the projection operator onto an eigenstate of the number
        operator,i.e., a number state with number n.
        """
        #--------------------------------------------
        def expression(x):
            try:
                return x.expression
            except:
                return x
        #--------------------------------------------
        name = "\\hat{\\Pi}_{n_{%s}}(%s)"%(self.name, expression(n))
        projector = copy(self.NumberState(n)).state.density_operator
        projector.name = name
        return projector    
    #----------------------------------------------------------  
    def _DisplacementOperator(self, alpha):
        """
        Returns the displacement operator with parameter alpha
        """
        try:
            #assume alpha is of type ParameterSymbolic
            D = (self.a.Dagger()*alpha-self.a*alpha.Conjugate()).ExpTruncated(10)
            D.name = "\\hat{\\mathcal{D}}_{%s}\\left(%s\\right)"%(self.name, alpha.expression.__str__())
        except:
            D = (self.a.Dagger()*alpha-self.a*sympy.conjugate(alpha)).ExpTruncated(10)     
            D.name = "\\hat{\\mathcal{D}}_{%s}\\left(%s\\right)"%(self.name, alpha)
        return D
    #----------------------------------------------------------  
    def _SqueezingOperator(self, zeta):
        """
        Returns the squeezing operator with parameter zeta
        """
        try:
            #assume zeta is of type ParameterSymbolic
            S = (-(self.a.Dagger()**2*zeta-self.a**2*zeta.Conjugate())*sympy.sympify(1/2)).ExpTruncated(10)
            S.name = "\\hat{\\mathcal{S}}_{%s}\\left(%s\\right)"%(self.name, zeta.expression.__str__())
        except:
            S = (-(self.a.Dagger()**2*zeta-self.a**2*sympy.conjugate(zeta))*sympy.sympify(1/2)).ExpTruncated(10)
            S.name = "\\hat{\\mathcal{S}}_{%s}\\left(%s\\right)"%(self.name, zeta)
        return S
    #---------------------------------------------------------- 
    def _RotationOperator(self, theta):
        """
        Returns the rotation operator with parameter theta
        """
        try:
            #assume theta is of type ParameterSymbolic
            R = (self.n*sympy.sympify(1j)*theta).ExpTruncated(10)
            R.name = "\\hat{\\mathcal{R}}_{%s}\\left(%s\\right)"%(self.name, theta.expression.__str__())
        except:
            R = (self.n*sympy.sympify(1j*float(theta))).ExpTruncated(10)
            R.name = "\\hat{\\mathcal{R}}_{%s}\\left(%s\\right)"%(self.name, theta)
        return R
    #---------------------------------------------------------- 
    def Vacuum(self):
        """
        Sets the system to the vacuum state
        """
        x = copy(self)
        x.state.Vacuum()
        return x
    #----------------------------------------------------------
    def NumberState(self, n):
        """
        Sets the system to the n-th number state
        """
        assert n < self.hilbert_space.dimension, \
            "n=%d must be lower than the Hilbert space dimension %d"\
                %(n, self.hilbert_space.dimension)
        x = copy(self)
        x.state.NumberState(n)
        return x
    #----------------------------------------------------------
    def Displace(self, alpha):
        """
        Displaces the system, by applying the unitary displacement operator
        """
        x = copy(self)
        D = x._DisplacementOperator(alpha)
        x.state._density_operator = D @ self.state._density_operator @ D.Dagger()
        x.state.density_operator.name = "%s\\left(%s\\right)"%(D.name, self.state.density_operator.name)
        x.state.WignerFunction()
        return x
    #----------------------------------------------------------
    def Squeeze(self, zeta):
        """
        Squeezes the system, by applying the unitary squeezing operator
        """
        x = copy(self)
        S = x._SqueezingOperator(zeta)
        x.state._density_operator = S @ self.state._density_operator @ S.Dagger()
        x.state.density_operator.name = "%s\\left(%s\\right)"%(S.name, self.state.density_operator.name)
        x.state.WignerFunction()
        return x
    #----------------------------------------------------------
    def Rotate(self, theta):
        """
        Rotates the system, by applying the unitary rotation operator
        """
        x = copy(self)
        R = x._RotationOperator(theta)
        x.state._density_operator = R @ self.state._density_operator @ R.Dagger()
        x.state.density_operator.name = "%s\\left(%s\\right)"%(R.name, self.state.density_operator.name)
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
        x = copy(self)
        x._state = self._state
        U = (H*x.hbar*sympy.sympify(-sympy.I)).ExpTruncated(10)
        x.state._density_operator = U @ x.state._density_operator @ U.Dagger()
        x.state.density_operator.name = "%s\\left(%s\\right)"%(U.name, self.state.density_operator.name)
        x.state.WignerFunction()
        return x
    #---------------------------------------------------------
    def Annihilate(self):
        """
        Applies the annihilation operator to the quantum state, and renormalizes
        it.
        """
        #e0 = self.hilbert_space.canonical_basis[0]
        #vacuum =  e0 @ e0.T()
        x = copy(self)
        x.state._density_operator = x.a @ self.state._density_operator @ x.a.Dagger()
        x.state._density_operator /= x.state._density_operator.Trace()
        x.state.WignerFunction()
        return x
     #---------------------------------------------------------   
    def Create(self):
         """
         Applies the creation operator to the quantum state, and renormalizes
         it.
         """
         #e0 = self.hilbert_space.canonical_basis[0]
         #vacuum =  e0 @ e0.T()
         x = copy(self)
         x.state._density_operator = x.a.Dagger() @ self.state._density_operator @ x.a
         x.state._density_operator /= x.state._density_operator.Trace()
         x.state.WignerFunction()
         return x     
# In[symbolic simple harmonic oscillator]
class SimpleHarmonicOscillatorNumeric: 
    """
    This class describes a Numeric simple harmonic oscillator.
    """
    #----------------------------------------------------------
    def __init__(self, truncated_dimension, name="A", hbar=1):
        assert hbar in ACCEPTED_HBAR_VALUES
        self._name = name
        self.symbol = sympy.symbols(names=self.name)
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
        #Hamiltonian operator
        self._H = (self.n + 1/2)*self.hbar
        self.H.name = "\\hat{\\mathcal{H}}_{%s}"%self.name
        #Canonical variables
        self._q = (self.a + self.a.Dagger())*self.hbar**(1/2)/numpy.sqrt(2)
        self.q.name = "\\hat{q}_{%s}"%self.name
        
        self._p = (self.a - self.a.Dagger())*self.hbar**(1/2)/(numpy.sqrt(2)*1j)
        self.p.name = "\\hat{p}_{%s}"%self.name
    #----------------------------------------------------------
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
    def _GeneralizeQuadratureProjector(self, x, theta):
        """
        Computes, the homodyne detection POVM in the Fock basis, up
        to order "N" (i.e, in a reduced-dimension Hilbert space). 
        
        For a quadrature value x, the (n, m) element of the POVM matrix is
        <X|m>*<X|n>. The expression of <X|k> depends on the associated LO phase
        "theta"
        
        n_max: positive integer
        x: scalar
        theta: scalar
        """
        #--------------------------------------------
        def value(x):
            try:
                return x.value
            except:
                return x
        #--------------------------------------------
        #For each element of x, a POVM matrix is computed
        N = self.hilbert_space.dimension - 1
        name = "\\hat{\\Pi}_{q_{{%s}_{%.3f\\pi}}}(%s)"%(self.name, value(theta)/numpy.pi, x)
        projector = MatrixNumeric(name=name)
        for n in numpy.arange(N):
            if n==0:
                projector.value[n] = 1/(numpy.sqrt(numpy.sqrt(numpy.pi)))*numpy.exp(-0.5*x**2)
            elif n==1:
                projector.value[n] = x*numpy.sqrt(2)*numpy.exp(1j*theta) * projector.value[0]
            else:
                projector.value[n] = numpy.exp(1j*theta)/numpy.sqrt(n)*(numpy.sqrt(2)*x*projector.value[n-1] - numpy.exp(1j*theta)*numpy.sqrt(n-1)*projector.value[n-2]) 
        projector = projector @ projector.Dagger() 
        projector.name = name
        return projector

    #----------------------------------------------------------
    def _AnnihilationProjector(self, alpha):
        """
        Computes the projection operator onto an eigenstate of the annihilation
        operator,i.e., a coherent state with amplitude alpha.
        """
        #--------------------------------------------
        def value(x):
            try:
                return x.value
            except:
                return x
        #--------------------------------------------
        name = "\\hat{\\Pi}_{\\alpha_{%s}}(%s)"%(self.name, value(alpha))
        projector = copy(self.Vacuum()).Displace(alpha).state.density_operator
        projector.name = name
        return projector
    #----------------------------------------------------------
    def _NumberProjector(self, n):
        """
        Computes the projection operator onto an eigenstate of the number
        operator,i.e., a number state with number n.
        """
        #--------------------------------------------
        def value(x):
            try:
                return x.value
            except:
                return x
        #--------------------------------------------
        name = "\\hat{\\Pi}_{n_{%s}}(%s)"%(self.name, value(n))
        projector = copy(self.NumberState(n)).state.density_operator
        projector.name = name
        return projector    
    #----------------------------------------------------------    
    def _DisplacementOperator(self, alpha):
        """
        Returns the displacement operator with parameter alpha
        """
        try:
            #assume alpha is of type ParameterNumeric
            D = (self.a.Dagger()*alpha-self.a*alpha.Conjugate()).Exp()
            D.name = "\\hat{\\mathcal{D}}_{%s}\\left(%s\\right)"%(self.name, alpha.name)
        except:
            D = (self.a.Dagger()*alpha-self.a*numpy.conjugate(alpha)).Exp()
            D.name = "\\hat{\\mathcal{D}}_{%s}\\left(%s\\right)"%(self.name, str(alpha))
        return D
    #----------------------------------------------------------  
    def _SqueezingOperator(self, zeta):
        """
        Returns the squeezing operator with parameter zeta
        """
        try:
            #assume zeta is of type ParameterNumeric
            S = (-(self.a.Dagger()**2*zeta-self.a**2*zeta.Conjugate())*0.5).Exp()
            S.name = "\\hat{\\mathcal{S}}_{%s}\\left(%s\\right)"%(self.name, zeta.name)
        except:
            S = (-(self.a.Dagger()**2*zeta-self.a**2*numpy.conjugate(zeta))*0.5).Exp()
            S.name = "\\hat{\\mathcal{S}}_{%s}\\left(%s\\right)"%(self.name, str(zeta))
        return S
    #---------------------------------------------------------- 
    def _RotationOperator(self, theta):
        """
        Returns the rotation operator with parameter theta
        """
        R = (self.n*1j*theta).Exp()
        R.name = "\\hat{\\mathcal{R}}_{%s}\\left(%.3f\\right)"%(self.name, theta)
        return R
    #---------------------------------------------------------- 
        
    def Vacuum(self):
        """
        Sets the system to the vacuum state
        """
        x = copy(self)
        x.state.Vacuum()
        return x
    #----------------------------------------------------------
    def NumberState(self, n):
        """
        Sets the system to the n-th number state
        """
        assert n < self.hilbert_space.dimension, \
            "n=%d must be lower than the Hilbert space dimension %d"\
                %(n, self.hilbert_space.dimension)
        x = copy(self)
        x.state.NumberState(n)
        return x
    #----------------------------------------------------------
    def Displace(self, alpha):
        """
        Displaces the system, by applying the unitary displacement operator
        """
        x = copy(self)
        D = x._DisplacementOperator(alpha)
        x.state._density_operator = D @ x.state.density_operator @ D.Dagger()
        x.state._density_operator = x.state.density_operator.Hermitian()
        x.state.density_operator.name = "%s\\left(%s\\right)"%(D.name, self.state.density_operator.name)
        return x
    #----------------------------------------------------------
    def Squeeze(self, zeta):
        """
        Squeezes the system, by applying the unitary squeezing operator
        """
        x = copy(self)
        S = x._SqueezingOperator(zeta)
        x.state._density_operator = S @ x.state.density_operator @ S.Dagger()
        x.state._density_operator = x.state.density_operator.Hermitian()
        x.state.density_operator.name = "%s\\left(%s\\right)"%(S.name, self.state.density_operator.name)
        return x
    #----------------------------------------------------------
    def Rotate(self, theta):
        """
        Rotates the system, by applying the unitary rotation operator
        """
        x = copy(self)
        R = x._RotationOperator(theta)
        x.state._density_operator = R @ x.state.density_operator @ R.Dagger()
        x.state._density_operator = x.state.density_operator.Hermitian()
        x.state.density_operator.name = "%s\\left(%s\\right)"%(R.name, self.state.density_operator.name)
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
        x = copy(self)
        U = (H*x.hbar*(-1j)).Exp()
        x.state._density_operator = U @ self.state._density_operator @ U.Dagger()
        x.state._density_operator = x.state.density_operator.Hermitian()
        x.state.density_operator.name = "%s\\left(%s\\right)"%(U.name, self.state.density_operator.name)
        return x
    #---------------------------------------------------------
    def Annihilate(self):
        """
        Applies the annihilation operator to the quantum state, and renormalizes
        it.
        """
        #e0 = self.hilbert_space.canonical_basis[0]
        #vacuum =  e0 @ e0.T()
        x = copy(self)
        x.state._density_operator = x.a @ self.state._density_operator @ x.a.Dagger()
        x.state._density_operator /= x.state._density_operator.Trace()
        x.state._density_operator = x.state.density_operator.Hermitian()
        return x
    #---------------------------------------------------------   
    def Create(self):
         """
         Applies the creation operator to the quantum state, and renormalizes
         it.
         """
         #e0 = self.hilbert_space.canonical_basis[0]
         #vacuum =  e0 @ e0.T()
         x = copy(self)
         x.state._density_operator = x.a.Dagger() @ self.state._density_operator @ x.a
         x.state._density_operator /= x.state._density_operator.Trace()
         x.state._density_operator = x.state.density_operator.Hermitian()
         return x
    #--------------------------------------------------------- 
    def ProjectiveMeasurement(self, measurable, ntimes=1, return_all_systems=False, **kwargs):
        """
        Peforms a projective measurement of a measurable quantity associated
        to a linear operator following the generalized Born rule.
        It repeats the measurement 'ntimes' times (assuming 'ntimes' identical
                                                   copies of the system exist)
        """
        assert measurable in MEASURABLES,\
        "%s is not supported as a measurable quantity. \n It should be one of these: %s"\
            %(measurable, MEASURABLES)
        system = copy(self)
        def _generalized_born_rule(projector):
            system0 = copy(self)
            p = (system0.state.density_operator @ projector).Trace() #outcome probability density
            system0._state = QuantumStateFockNumeric(truncated_dimension=system0.hilbert_space.dimension, \
                                            name=system0.state.name)
            system0.state._density_operator = projector @ system0.state._density_operator \
                                      @ projector
            system0.state._density_operator /= p
            return system0, p
        #-------------------
        if measurable == "n":
            def Projector(n):
                en = system.hilbert_space.CanonicalBasisVector(n).numeric
                return en @ en.T()
            #Calculate the probabilities of outcomes
            values = numpy.arange(system.hilbert_space.dimension)
            p = numpy.zeros((len(values), ))           
            for j in numpy.arange(len(values)):
                n = values[j]
                projector = Projector(n) 
                p[j] = _generalized_born_rule(projector)[1].value
            p = numpy.abs(p)
            p /= numpy.sum(p)
            #Sample according to the calculated probabilities
            outcomes = numpy.random.choice(values, size=(ntimes,), p=p)
            if return_all_systems:
                #Apply the generalized born rule to the all measurements
                projector = []
                post_measurement_system = []
                for j in range(ntimes):
                    projector.append(Projector(outcomes[j]))
                    post_measurement_system.append(_generalized_born_rule(projector[j])[0])
            else:
                #Apply the generalized born rule to the last measurement
                projector = Projector(outcomes[-1])
                post_measurement_system = _generalized_born_rule(projector)[0]
        #-------------------
        elif measurable == "a":
            def Projector(alpha):
                return \
                    system.Displace(alpha_values[i]).state.density_operator
            #Consider a range of values that spans a few standard deviations beyond
            #the mean value of the number operator, which defines the energy
            #of the harmonic oscillator
            max_alpha = system.state.Mean(system.a).value \
                + system.state.Std(system.a).value*5
            #Calculate the probabilities of outcomes
            n_values = 300
            q_values, p_values = [2*numpy.sqrt(system.hbar.value/2)\
                                  *numpy.linspace(-max_alpha, max_alpha, n_values)\
                                      for j in range(2)]
            alpha_values = numpy.zeros((n_values*n_values,)) 
            p = numpy.zeros((n_values*n_values,))
            for j in numpy.arange(p.shape[0]):
                for k in numpy.arange(p.shape[1]):
                    i = j*p.shape[1] + k
                    alpha_values[i] = numpy.sqrt(system.hbar.value/2) * \
                        (q_values[j] + 1j*p_values[k])
                    projector = Projector(alpha_values[j])
                    p[i] = _generalized_born_rule(projector)[1].value
            p = numpy.abs(p)
            p /= numpy.sum(p)
            #Sample according to the calculated probabilities
            outcomes = numpy.random.choice(alpha_values, size=(ntimes,), p=p)
            if return_all_systems:
                #Apply the generalized born rule to the all measurements
                projector = []
                post_measurement_system = []
                for j in range(ntimes):
                    projector.append(Projector(outcomes[j]))
                    post_measurement_system.append(_generalized_born_rule(projector[j])[0])
            else:
                #Apply the generalized born rule to the last measurement
                projector = Projector(outcomes[-1])
                post_measurement_system = _generalized_born_rule(projector)[0]
            values = alpha_values
        #-------------------
        elif measurable == "x":
            def Projector(x, theta):
                proj = MatrixNumeric(name=name)
                proj.value = numpy.matrix(numpy.zeros((system.hilbert_space.dimension, 1)))    
                for n in numpy.arange(N):
                    if n==0:
                        proj.value[n] = 1/(numpy.sqrt(numpy.sqrt(numpy.pi)))*numpy.exp(-0.5*x**2)
                    elif n==1:
                        proj.value[n] = x*numpy.sqrt(2)*numpy.exp(1j*theta) * proj.value[0]
                    else:
                        proj.value[n] = numpy.exp(1j*theta)/numpy.sqrt(n)*(numpy.sqrt(2)*x*proj.value[n-1] - numpy.exp(1j*theta)*numpy.sqrt(n-1)*proj.value[n-2]) 
                return proj @ proj.Dagger()    
            theta = kwargs["theta"]    
            N = system.hilbert_space.dimension - 1
            #Consider a range of values that spans a few standard deviations beyond
            #the mean value of the number operator, which defines the energy
            #of the harmonic oscillator
            q_theta = system.q*numpy.cos(theta) + system.p*numpy.sin(theta)
            max_x = (system.state.Mean(q_theta).value + system.state.Std(q_theta).value*5)
            n_values = 300
            x_values = numpy.linspace(-max_x, max_x, n_values)
            p = numpy.zeros((n_values,))
            for j in range(n_values):
                x = x_values[j]
                name = "\\hat{\\Pi}_{%.1f}(%0.3f)"%(theta, x)
                projector = Projector(x, theta)
                p[j] = _generalized_born_rule(projector)[1].value
            p = numpy.abs(p)
            p /= numpy.sum(p) 
            from matplotlib import pyplot
            pyplot.figure()
            pyplot.plot(x_values, p)
            #Sample according to the calculated probabilities
            outcomes = numpy.random.choice(x_values, size=(ntimes,), p=p)
            if return_all_systems:
                #Apply the generalized born rule to the all measurements
                projector = []
                post_measurement_system = []
                for j in range(ntimes):
                    projector.append(Projector(outcomes[j], theta))
                    post_measurement_system.append(_generalized_born_rule(projector[j])[0])
            else:
                #Apply the generalized born rule to the last measurement
                projector = Projector(outcomes[-1], theta)
                post_measurement_system = _generalized_born_rule(projector)[0]
            values = x_values
        return outcomes, values, p, post_measurement_system
    #--------------------------------------------------------- 
    def Expand(self, n):
        """
        Returns a replica of the system with Hilbert space dimension increased
        by the integer n

        Parameters
        ----------
        n : int
            increase in the Hilbert space dimension
        Returns
        -------
        the new system
        """
        if n <= 0:
            return copy(self)
        #Initialize new system
        system = SimpleHarmonicOscillatorNumeric(\
                                                 truncated_dimension=self.hilbert_space.dimension+n,\
                                                     name=self.name,\
                                                         hbar=self.hbar.value)
        #Expand the quantum state
        system._state = self.state.Expand(n)
        return system
    #--------------------------------------------------------- 
    def Truncate(self, order):
        """
        Returns a replica of the quantum state, belonging to a Hilbert space
        with input dimension. If 'dimension' < self.hilbert_space.dimension, then
        the truncation operator of order 'dimension' is applied to the density operator.
        If 'dimension' > self.hilbert_space.dimension, then the density operator
        is extended via direct sum with a null operator of dimension
            dimension - self.hilbert_space.dimension
        All other parameters of the quantum state are simply reset.
        
        INPUTS
        ---------------
        dimension : int
            dimension of the new quantum state's Hilbert space
        """
        if order >= self.hilbert_space.dimension-1:
            return copy(self)
        #Initialize new system
        system = SimpleHarmonicOscillatorNumeric(\
                                                 truncated_dimension=order+1,\
                                                     name=self.name,\
                                                         hbar=self.hbar.value)
        #Expand the quantum state
        system._state = self.state.Truncate(order)
        return system
    #--------------------------------------------------------- 
    def Resize(self, dimension):
        """
        Returns a replica of the quantum state, belonging to a Hilbert space
        with input dimension. If 'dimension' < self.hilbert_space.dimension, then
        the truncation operator of order 'dimension' is applied to the density operator.
        If 'dimension' > self.hilbert_space.dimension, then the density operator
        is extended via direct sum with a null operator of dimension
            dimension - self.hilbert_space.dimension
        All other parameters of the quantum state are simply reset.
        
        INPUTS
        ---------------
        dimension : int
            dimension of the new quantum state's Hilbert space
        """
        #Initialize new system
        system = SimpleHarmonicOscillatorNumeric(\
                                                 truncated_dimension=dimension,\
                                                     name=self.name,\
                                                         hbar=self.hbar.value)
        #Expand the quantum state
        system._state = self.state.Resize(dimension)
        return system