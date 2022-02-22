"""
This module describes a density matrix
"""
#%%
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix import MatrixSymbolic, MatrixNumeric, Matrix
from .Exceptions import TestFailedError, InconsistentShapeError
from Iaji.Mathematics.Parameter import Parameter, ParameterSymbolic, ParameterNumeric
import numpy, sympy
#%%
print_separator = "-----------------------------------------------"
#%%
class DensityMatrix(Matrix):
    """
    This class describes a density matrix as a parameter.
    """
    # ----------------------------------------------------------
    def __init__(self, name="\\rho", value=None, real=False, nonnegative=False):
        super().__init__(name, value, real, nonnegative)
        self._symbolic = DensityMatrixSymbolic(name=name)
        self._numeric = DensityMatrixNumeric(name=name, value=value)
        # Connect property changed signals to chech functions
   #     self.numeric.value_changed.connect(self.check_shapes)
   #     self.symbolic.expression_changed.connect(self.check_shapes)
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
    def check_shapes(self, **kwargs):
        if self.numeric.shape is not None and self.symbolic.shape is not None:
            if self.numeric.shape != self.symbolic.shape:
                raise InconsistentShapeError(
                    "The shape of my value is " + self.numeric.shape.__str__() + \
                    ". while the shape of my symbolic expression is " + self.symbolic.shape.__str__())
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def __str__(self):
        """
        This function returns a string with the summary of the interesting p
        """
        s = "DENSITY MATRIX: \n" + "name: " + self.name.__str__() + "\n" + print_separator+"\n"\
        +self.numeric.__str__() + "\n" + print_separator + "\n" \
        + self.symbolic.__str__() + "\n" + print_separator
        return s
    # ----------------------------------------------------------
    # ----------------------------------------------------------
# In[]:
class DensityMatrixSymbolic(MatrixSymbolic):
    """
    This class describes a symbolic density matrix.
    """
    # ----------------------------------------------------------
    def __init__(self, name="Rho"):
        super().__init__(name=name, real=True, nonnegative=True)
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, expression):
        self._expression = expression
        if expression is not None:
            try:
                self.expression_symbols = sorted(list(expression.free_symbols), key=lambda x: x.name)
                self.expression_lambda = sympy.lambdify(self.expression_symbols, expression, modules="numpy")
            except AttributeError:
                self.expression_lambda = None
            self._expression = sympy.Matrix(sympy.Array(self._expression))
            self._shape = self._expression.shape
            try:
                if not self.isDensityMatrix()[0]:
                    error_message = "Expression of matrix "+self.name.__str__()+" cannot represent a density matrix\n"+self.isDensityMatrix()[1].__str__()
                    self._expression = None
                    self._shape = None
                    raise TypeError(error_message)
                self.expression_changed.emit()  # emit expression changed signal
            except TestFailedError:
                warning_message = "Could not verify that matrix "+self.name.__str__()+" is a density matrix.\n" + self.__str__()
                # raise TestFailedWarning(warning_message)
                print("WARNING: " + warning_message)
        else:
            self.expression_symbols = None
            self.expression_lambda = None
            self._shape = None
        self._eigenvalues, self._rank  = [None for j in range(2)]
        self._trace = ParameterSymbolic(name="Tr\\left(%s\\right)"%self.name)
        self._determinant = ParameterSymbolic(name="\\left|%s\\right|"%self.name)
        self.expression_changed.emit()  # emit value changed signal
    @expression.deleter
    def expression(self):
        del self._expression
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def __str__(self):
        s = super().__str__()
        return s.replace("MATRIX", "DENSITY MATRIX")
    # ----------------------------------------------------------
# In[]:
class DensityMatrixNumeric(MatrixNumeric):
    """
    This class describes a numerical density matrix.
    """
    # ----------------------------------------------------------
    def __init__(self, name="Rho", value=None):
        super().__init__(name=name, value=value)
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value is not None:
            self._value = numpy.matrix(value)
            self._shape = self._value.shape
            if not self.isDensityMatrix()[0]:
                error_message = "Value of matrix "+self.name.__str__()+" cannot represent a density matrix\n"+self.isDensityMatrix()[1].__str__()
                self._value = None
                self._shape = None
                self._eigenvalues, self._rank, self._trace, self._determinant = [None for j in range(4)]
                raise TypeError(error_message)
        else:
            self._value = None
            self._shape = None
        self._eigenvalues, self._rank  = [None for j in range(2)]
        self._trace = ParameterNumeric(name="Tr\\left(%s\\right)"%self.name)
        self._determinant = ParameterNumeric(name="\\left|%s\\right|"%self.name)
        self.value_changed.emit()  # emit value changed signal

    @value.deleter
    def value(self):
        del self._value
        # ----------------------------------------------------------
    # ----------------------------------------------------------
    def __str__(self):
        s = super().__str__()
        return s.replace("MATRIX", "DENSITY MATRIX")
    # ----------------------------------------------------------





