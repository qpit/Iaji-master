"""
This module describes a HilberSpace
"""
#%%
import sympy, numpy
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
            vectors : sympy.sets.sets.ProductSet
                the modulus of vectors
            name : str
        """
        if type(dimension) == int:
            if dimension < 0 :
                raise ValueError("The value "+dimension.__str__()+" for the dimension of Hilber space "+self.name.__str__()+" is not valid.")
        elif type(dimension) == numpy.inf:
            raise NotImplementedError("Infinite-dimensional Hilbert spaces are not yet handeled.")
        else:
            raise ValueError("The value "+dimension.__str__()+" for the dimension of Hilber space "+self.name.__str__()+" is not valid.")
        self.dimension = dimension
        self.name = name
        if type(scalars) not in ACCEPTED_TYPES:
            raise TypeError("The type of scalars "+type(scalars)+" for Hilber space "+self.name.__str__()+" is not valid. \n"\
                            +"Accepted types are: "+ACCEPTED_TYPES.__str__())
        self._scalars = scalars
        self._vectors = scalars**dimension
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
    # ---------------------------------------------------------------
    @property
    def dimension(self):
        return self._dimension

    @dimension.setter
    def dimension(self, dimension):
        self._dimension = dimension

    @dimension.deleter
    def dimension(self):
        del self._dimension
    # ---------------------------------------------------------------
    @property
    def dimension(self):
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


