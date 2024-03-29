"""
This module describes a HilberSpace
"""
#%%
import sympy, numpy
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix import Matrix
from Iaji.Mathematics.Parameter import Parameter, ParameterSymbolic, ParameterNumeric
from Iaji.Utilities import strutils
#%%
print_separator = "---------------------------------------------------------"
ACCEPTED_TYPES = {sympy.sets.fancysets.Reals, sympy.sets.fancysets.Complexes}
NUMBER_TYPES = [int, numpy.int64, float, numpy.float64, complex, numpy.complex64, numpy.complex128]
#%%
class HilbertSpace:
    """
    This class describes a Hilbert space on the field of complex numbers.
    """
    #---------------------------------------------------------------
    def __init__(self, dimension, scalars=sympy.Complexes, name="H"):
        """
        INPUTS
        -------------
            dimension : in {int>0, numpy.inf}
                the dimension of the Hilbert space
            scalars : in {sympy.Reals, sympy.Complexes}
                the ring of scalars
            name : str
        """
        self._dimension = dimension
        if self.isFiniteDimensional():
            if "int" in str(type(dimension)):
                if dimension < 0 :
                    raise ValueError("The value "+dimension.__str__()+" for the dimension of Hilber space "+self.name.__str__()+" is not valid.")
            else:
                raise ValueError("The value "+dimension.__str__()+" for the dimension of Hilber space "+self.name.__str__()+" is not valid.")
            self._vectors = scalars**dimension
        self.name = name
        self.symbol = sympy.symbols(names=self.name)
        if type(scalars) not in ACCEPTED_TYPES:
            raise TypeError("The type of scalars "+type(scalars)+" for Hilber space "+self.name.__str__()+" is not valid. \n"\
                            +"Accepted types are: "+ACCEPTED_TYPES.__str__())
        self._scalars = scalars
        #Define the inner product
        self.SetInnerProduct()
    #---------------------------------------------------------------
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @name.deleter
    def name(self):
        del self._name
    #---------------------------------------------------------------
    @property
    def symbol(self):
        return self._symbol

    @symbol.setter
    def symbol(self, symbol):
        self._symbol = symbol

    @symbol.deleter
    def symbol(self):
        del self._symbol
    # ---------------------------------------------------------------
    @property
    def dimension(self):
        return self._dimension


    @dimension.deleter
    def dimension(self):
        del self._dimension
    # ---------------------------------------------------------------
    @property
    def scalars(self):
        return self._scalars

    @scalars.deleter
    def scalars(self):
        del self._scalars
    # ---------------------------------------------------------------
    @property
    def vectors(self):
        return self._vectors

    @vectors.deleter
    def vectors(self):
        del self._vectors
    # ---------------------------------------------------------------
    @property
    def inner_product(self):
        return self._inner_product

    @inner_product.deleter
    def inner_product(self):
        del self._inner_product
    # ---------------------------------------------------------------
    @property
    def canonical_basis(self):
        return self._canonical_basis

    @canonical_basis.deleter
    def canonical_basis(self):
        del self._canonical_basis
    # ---------------------------------------------------------------
    def isFiniteDimensional(self):
        """
        This function returns True if an only if the Hilber space has finite dimension
        """
        return not(self.dimension == numpy.inf or self.dimension == sympy.oo)
    # ---------------------------------------------------------------
    def isTensorProduct(self):
        return "otimes" in self.symbol.name
    # ---------------------------------------------------------------
    def CanonicalBasis(self):
        """
        Sets and returns the canonical basis.
        """
        if self.isFiniteDimensional():
            self._canonical_basis = [None for j in range(self.dimension)]
            for j in range(self.dimension):
                expression = sympy.zeros(*(self.dimension, 1))
                expression[j] = 1
                self._canonical_basis[j] = Matrix(name="e_{"+(j+1).__str__()+"}")
                self._canonical_basis[j].symbolic.expression = expression
                self._canonical_basis[j].numeric.value = self._canonical_basis[j].symbolic.expression_lambda()
                    
        else:
           # raise NotImplementedError("Infinite-dimensional Hilbert spaces are not yet handeled.")
            self._canonical_basis = None
            
        return self.canonical_basis
    # ---------------------------------------------------------------
    def CanonicalBasisVector(self, n):
        """
        Sets and returns the n-th canonical basis vector.
        Vectors are numbered from 0.
        """
        vector = Matrix(name="e_{"+(n+1).__str__()+"}")
        expression = sympy.zeros(*(self.dimension, 1))
        expression[n] = 1
        vector.symbolic.expression = expression
        vector.numeric.value = vector.symbolic.expression_lambda()
        return vector
    # ---------------------------------------------------------------
    def __str__(self):
        """
        Returns a string with the summary of the interesting properties
        """
        s = super().__str__()
        s += "HILBERT SPACE: \n"\
             +"name: "+self.name.__str__()+"\n"\
             +"symbol: "+self.symbol.__str__()+"\n"\
             +"scalars: "+self.scalars.__str__()+"\n"\
             +"vectors: "+self.vectors.__str__()+"\n"\
             +"dimension: "+self.dimension.__str__()+"\n"
        return s
    # ---------------------------------------------------------------
    def Otimes(self, other):
        """
        Tensor product of Hilbert spaces
        """
        if self.scalars != other.scalars:
            raise TypeError("The ring of scalars of Hilber space %s is %s while the one of Hilbers space of %s is %s"\
                            %(self.name, self.scalars, other.name, other.scalars))
            return None
        else:
            name = "\\left(%s\\otimes\\;%s\\right)"%(self.name, other.name)
            return HilbertSpace(dimension=self.dimension*other.dimension, scalars=self.scalars, name=name)
    # ---------------------------------------------------------------
    def SetInnerProduct(self):     
        if not self.isFiniteDimensional():
           #print("WARNING in HilbertSpace.SetInnerProduct(): Infinite-dimensional Hilbert spaces are not yet handeled.")
           self._inner_product = None 
        else:
            def InnerProduct(v1, v2):
                """
                Returns the inner product between two vectors of the Hilber space
                INPUTS
                -------------
                    v1, v2 : in [Iaji Matrix, Iaji MatrixSymbolic, Iaji MatrixNumeric]
                        the two vectors of which the inner product is evaluated.
                
                OUTPUT
                ------------
                
                """
                import Iaji
                ACCEPTED_TYPES = [Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix.Matrix, \
                                  Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix.MatrixSymbolic,\
                                  Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix.MatrixNumeric]
                if type(v1) not in ACCEPTED_TYPES or type(v2) not in ACCEPTED_TYPES:
                    raise TypeError("unsupported operand type(s) for InnerProduct: %s and %s"%(type(v1), type(v2)))
                    return None
                else:
                    if type(v1) != type(v2):
                        raise TypeError("The type of the first operand is %s but the second is %s"%(type(v1), type(v2)))
                        return None
                    else:
                        result = v1.Dagger() @ v2
                        result.name = "\langle{"+v1.name.__str__()+"|"+v2.name.__str__()+"}\\rangle"
                        type_string_last = str(type(v1)).split("'")[1].split(".")[-1]
                        if  type_string_last == "Matrix":
                            result_parameter = Parameter(name=result.name)
                            result_parameter.symbolic.expression = result.symbolic.expression[0, 0]
                            result_parameter.numeric.value = result.numeric.value[0, 0]
                        elif type_string_last == "MatrixSymbolic":
                            result_parameter = ParameterSymbolic(name=result.name)
                            result_parameter.expression = result.expression[0, 0]
                        else:
                            result_parameter = ParameterNumeric(name=result.name)
                            result_parameter.value = result.value[0, 0]
                        return result_parameter
            self._inner_product = InnerProduct
    @classmethod
    def TensorProduct(cls, H_list):
        """
        Tensor product of a list of hilbert spaces
        """
        scalars = H_list[0].scalars
        dimension = numpy.prod([H.dimension for H in H_list])
        name = H_list[0].name
        for H in H_list[1:]:
            name += "\\otimes\\;%s"%(H.name)
        return HilbertSpace(dimension=dimension, scalars=scalars, name=name)

# In[Fock space]
class FockSpace(HilbertSpace):
    """
    This class describes a special Fock space, i.e., that of dimension equal to
    the cardinality of the natural numbers, on the field of complex numbers.
    It can describe the Hilbert space of a quantum harmonic oscillator and a
    single mode of a quantum field in second quantization theory.
    This class is only made for symbolic manipulations.
    """
    #---------------------------
    def __init__(self, name="A"):
        try:
            super().__init__(dimension=sympy.oo, scalars=sympy.Complexes, name="\\mathcal{F}_{%s}"%name)
        except ValueError: #HilbertSpace.SetInnerProduct() will complain about the space being infinite-dimensional
            pass
        self._vectors = self.symbol
    #---------------------------
    def CanonicalBasisVector(self, n):
        """
        Sets and returns the n-th canonical basis vector.
        Vectors are numbered from 0.
        """
        vector = ParameterSymbolic(name="\\left|%s\\right\\rangle"%n, type="vector")  
        return vector
    # ---------------------------------------------------------------
# In[Kets]
class Ket(ParameterSymbolic):
    """
    This class defines a ket, i.e., a symbolic ray in a projective Hilbert space 
    on the field of complex numbers.
    """
    #----------------------------------------------------------
    def __init__(self, name="\\Psi"):
        super().__init__(name="\\left\\vert%s\\right\\rangle"%name, type="vector")
        self.name = name
        self.symbol = self.expression
    #----------------------------------------------------------
    @property
    def expression(self):
        return self._expression
    
    @expression.setter
    def expression(self, expression):
        self._expression = expression
        if expression is not None:
            try:
                #Construct the lambda function associated to the symbolic expression
                self.expression_symbols = sorted(list(expression.free_symbols), key=lambda x: x.name)
                """
                If an expression has symbols whose names have proper latex
                math formatting, sympy.lambdify will complain. So, convert
                all symbol names from lateX to python variable friendly names.
                """
                expression_symbols_non_latex_names = []
                for s in self.expression_symbols:
                    name = strutils.de_latexify(s.name) #convert from lateX name to python-friendly name
                    expression_symbols_non_latex_names.append(\
                    sympy.symbols(names=name, real=s.is_real, nonnegative=s.is_nonnegative))
                expression_non_latex = strutils.de_latexify(str(expression))
                self.expression_lambda = sympy.lambdify(expression_symbols_non_latex_names,\
                                                        expression_non_latex, modules="numpy")
            except AttributeError:
                self.expression_lambda = None
            self.expression_changed.emit()  # emit expression changed signal
        else:
            self.expression_symbols = None
            self.expression_lambda = None
    
    @expression.deleter
    def expression(self):
        del self._expression
    # ----------------------------------------------------------
    def __add__(self, other):
        assert "Ket" in str(type(other)), \
        TypeError("unsupported operand type(s) for +: %s and %s" % (type(self.expression, other.expression)))
        other = self.prepare_other(other)
        x = Ket(name="\\left(%s+%s\\right)"%(self.name, other.name))
        x.expression = self.expression + other.expression
        x.symbol = x.expression
        return x
    # ----------------------------------------------------------
    def __mul__(self, other):
        other = self.prepare_other(other)
        x = Ket(name="\\left(%s%s\\right)"%(self.name, other.name))
        x.expression = other.expression * self.expression 
        x.symbol = x.expression
        return x
    # ----------------------------------------------------------
    def __sub__(self, other):
        assert "Ket" in str(type(other)), \
        TypeError("unsupported operand type(s) for +: %s and %s" % (type(self.expression, other.expression)))
        other = self.prepare_other(other)
        x = Ket(name="\\left(%s-%s\\right)"%(self.name, other.name))
        x.expression = self.expression - other.expression
        x.symbol = x.expression
        return x
    # ----------------------------------------------------------
    def __truediv__(self, other):
        other = self.prepare_other(other)
        x = Ket(name="\\left(\\frac{1}{%s}%s\\right)"%(other.name, self.name))
        x.expression = self.expression/other.expression 
        x.symbol = x.expression
        return x
    # ----------------------------------------------------------
    def Dagger(self):
        return Bra(self.name)
    # ----------------------------------------------------------
    def prepare_other(self, other):
        """
        Checks if the other operand is of the same type as self and, in case not
        returns a compatible type object
        """
        if "Iaji" in str(type(other)):
            return other
        elif "sympy" in str(type(other)): 
            #assuming other is a sympy symbol or expression
            is_real = other.is_real is True
            is_nonnegative = other.is_real is True and other.is_nonnegative is True
            other_temp = ParameterSymbolic(name=str(other), \
                                real=is_real, nonnegative=is_nonnegative)
            other_temp.expression = other
        else:
            #Assuming other is a primitive numerical type
            if type(other) in NUMBER_TYPES:
                if "int" in str(type(other)):
                    other = float(other)
                is_real = numpy.isclose(numpy.imag(other), 0)
                is_nonnegative = is_real is True and (other >= 0)
                other_temp = ParameterSymbolic(name=str(other), \
                                    real=is_real, nonnegative=is_nonnegative)
                other_temp.expression = sympy.sympify(other)
            else:
                raise TypeError("Incompatible operand types (%s. %s)"%(type(self), type(other)))
        return other_temp
# In[Bra]
class Bra(ParameterSymbolic):
    """
    This class defines a bra, i.e., a symbolic linear application from a projective Hilbert space 
    into the complex numbers.
    """
    #----------------------------------------------------------
    def __init__(self, name="\\Psi"):
        super().__init__(name="\\left\\langle%s\\right\\vert"%name, type="vector")
        self.name = name
        self.symbol = self.expression
    #----------------------------------------------------------
    @property
    def expression(self):
        return self._expression
    
    @expression.setter
    def expression(self, expression):
        self._expression = expression
        if expression is not None:
            try:
                #Construct the lambda function associated to <the symbolic expression
                self.expression_symbols = sorted(list(expression.free_symbols), key=lambda x: x.name)
                """
                If an expression has symbols whose names have proper latex
                math formatting, sympy.lambdify will complain. So, convert
                all symbol names from lateX to python variable friendly names.
                """
                expression_symbols_non_latex_names = []
                for s in self.expression_symbols:
                    name = strutils.de_latexify(s.name) #convert from lateX name to python-friendly name
                    expression_symbols_non_latex_names.append(\
                    sympy.symbols(names=name, real=s.is_real, nonnegative=s.is_nonnegative))
                expression_non_latex = strutils.de_latexify(str(expression))
                self.expression_lambda = sympy.lambdify(expression_symbols_non_latex_names,\
                                                        expression_non_latex, modules="numpy")
            except AttributeError:
                self.expression_lambda = None
            self.expression_changed.emit()  # emit expression changed signal
        else:
            self.expression_symbols = None
            self.expression_lambda = None
    
    @expression.deleter
    def expression(self):
        del self._expression
    # ----------------------------------------------------------
    def __add__(self, other):
        assert "Bra" in str(type(other)), \
        TypeError("unsupported operand type(s) for +: %s and %s" % (type(self.expression, other.expression)))
        other = self.prepare_other(other)
        x = Bra(name="\\left(%s+%s\\right)"%(self.name, other.name))
        x.expression = self.expression + other.expression
        x.symbol = x.expression
        return x
    # ----------------------------------------------------------
    def __mul__(self, other):
        other = self.prepare_other(other)
        x = Bra(name="\\left(%s%s\\right)"%(self.name, other.name))
        x.expression = other.expression * self.expression 
        x.symbol = x.expression
        return x
    # ----------------------------------------------------------
    def __sub__(self, other):
        assert "Bra" in str(type(other)), \
        TypeError("unsupported operand type(s) for +: %s and %s" % (type(self.expression, other.expression)))
        other = self.prepare_other(other)
        x = Bra(name="\\left(%s-%s\\right)"%(self.name, other.name))
        x.expression = self.expression - other.expression
        x.symbol = x.expression
        return x
    # ----------------------------------------------------------
    def __truediv__(self, other):
        other = self.prepare_other(other)
        x = Bra(name="\\left(\\frac{1}{%s}%s\\right)"%(other.name, self.name))
        x.expression = self.expression/other.expression 
        x.symbol = x.expression
        return x
    # ----------------------------------------------------------
    def Dagger(self):
        return Ket(self.name)
    # ----------------------------------------------------------
    def prepare_other(self, other):
        """
        Checks if the other operand is of the same type as self and, in case not
        returns a compatible type object
        """
        if "Iaji" in str(type(other)):
            return other
        elif "sympy" in str(type(other)): 
            #assuming other is a sympy symbol or expression
            is_real = other.is_real is True
            is_nonnegative = other.is_real is True and other.is_nonnegative is True
            other_temp = ParameterSymbolic(name=str(other), \
                                real=is_real, nonnegative=is_nonnegative)
            other_temp.expression = other
        else:
            #Assuming other is a primitive numerical type
            if type(other) in NUMBER_TYPES:
                if "int" in str(type(other)):
                    other = float(other)
                is_real = numpy.isclose(numpy.imag(other), 0)
                is_nonnegative = is_real is True and (other >= 0)
                other_temp = ParameterSymbolic(name=str(other), \
                                    real=is_real, nonnegative=is_nonnegative)
                other_temp.expression = sympy.sympify(other)
            else:
                raise TypeError("Incompatible operand types (%s. %s)"%(type(self), type(other)))
        return other_temp