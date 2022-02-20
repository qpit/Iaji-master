"""
This module describes a HilberSpace
"""
#%%
import sympy, numpy
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix import Matrix
from Iaji.Mathematics.Parameter import Parameter, ParameterSymbolic, ParameterNumeric
#%%
print_separator = "---------------------------------------------------------"
ACCEPTED_TYPES = {sympy.sets.fancysets.Reals, sympy.sets.fancysets.Complexes}
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
        if type(dimension) == int:
            if dimension < 0 :
                raise ValueError("The value "+dimension.__str__()+" for the dimension of Hilber space "+self.name.__str__()+" is not valid.")
        elif not self.isFiniteDimensional():
            raise NotImplementedError("Infinite-dimensional Hilbert spaces are not yet handeled.")
        else:
            raise ValueError("The value "+dimension.__str__()+" for the dimension of Hilber space "+self.name.__str__()+" is not valid.")
        self.name = name
        self.symbol = sympy.symbols(names=self.name)
        if type(scalars) not in ACCEPTED_TYPES:
            raise TypeError("The type of scalars "+type(scalars)+" for Hilber space "+self.name.__str__()+" is not valid. \n"\
                            +"Accepted types are: "+ACCEPTED_TYPES.__str__())
        self._scalars = scalars
        self._vectors = scalars**dimension
        #Define the canonical basis
        self._canonical_basis = self.CanonicalBasis()
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
            raise NotImplementedError("Infinite-dimensional Hilbert spaces are not yet handeled.")
            self._canonical_basis = None
            
        return self.canonical_basis
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
    def otimes(self, other):
        """
        Tensor product of Hilbert spaces
        """
        if self.scalars != other.scalars:
            raise TypeError("The ring of scalars of Hilber space %s is %s while the one of Hilbers space of %s is %s"\
                            %(self.name, self.scalars, other.name, other.scalars))
            return None
        else:
            name = "\\left(%s\\otimes%s\\right)"%(self.name, other.name)
            return HilbertSpace(dimension=self.dimension+other.dimension, scalars=self.scalars, name=name)
    
    # ---------------------------------------------------------------
    def SetInnerProduct(self):     
        if not self.isFiniteDimensional():
           raise NotImplementedError("Infinite-dimensional Hilbert spaces are not yet handeled.")
           self._inner_product = None 
        else:
            def InnerProduct(v1, v2):
                """
                Returns the inner product between two elements of the Hilber space
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
                        result = v1.Conjugate().Transpose() @ v2
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
                        
               
               


