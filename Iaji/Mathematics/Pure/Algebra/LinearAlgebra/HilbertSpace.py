"""
This module describes a HilberSpace
"""
#%%
import sympy, numpy
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix import Matrix
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
                value = numpy.zeros((self.dimension, 1))
                value[j] = 1
                
                expression = sympy.zeros(*(self.dimension, 1))
                expression[j] = 1
                
                self._canonical_basis[j] = Matrix(name="e_"+(j+1).__str__())
                self._canonical_basis[j].numeric.value = value
                self._canonical_basis[j].symbolic.expression = expression
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
    def __mul__(self, other):
        """
        Tensor product of Hilbert spaces
        """
        if self.scalars != other.scalars:
            raise TypeError("The ring of scalars of Hilber space %s is %s while the one of Hilbers space of %s is %s"\
                            %(self.name, self.scalars, other.name, other.scalars))
            return None
        else:
            return HilbertSpace(dimension=self.dimension+other.dimension, scalars=self.scalars, name="%s*%s"%(self.name, other.name))
            
        
        


