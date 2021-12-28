"""
This module describes a density matrix
"""
#%%
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix import MatrixSymbolic, MatrixNumeric
from .Exceptions import TestFailedError, TestFailedWarning
import numpy, sympy
#%%
print_separator = "-----------------------------------------------"
#%%
class DensityMatrix:
    """
    This class describes a density matrix as a physical parameter.
    """
    # ----------------------------------------------------------
    def __init__(self, name="Rho", value=None, real=False, nonnegative=False):
        self.name = name
        self.type = "vector"
        self.symbolic = DensityMatrixSymbolic(name=name)
        self.numeric = DensityMatrixNumeric(name=name, value=value)
        # Connect property changed signals to chech functions
        self.numeric.value_changed.connect(self.check_shapes)
        self.symbolic.expression_changed.connect(self.check_shapes)
    # ----------------------------------------------------------
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
                    error_message = "Value of matrix "+self.name.__str__()+" cannot represent a density matrix, because it is not square.\n"+self.__str__()
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
        self._eigenvalues, self._rank, self._trace, self._determinant = [None for j in range(4)]

    @expression.deleter
    def expression(self):
        del self._expression
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def __str__(self):
        s = super().__str__()
        return s.replace("MATRIX", "DENSITY MATRIX")
    # ----------------------------------------------------------



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
                error_message = "Value of matrix "+self.name.__str__()+" does not represent a density matrix. \n"+self.__str__()
                self._value = None
                self._shape = None
                self._eigenvalues, self._rank, self._trace, self._determinant = [None for j in range(4)]
                raise TypeError(error_message)
        else:
            self._value = None
            self._shape = None
        self._eigenvalues, self._rank, self._trace, self._determinant = [None for j in range(4)]
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





