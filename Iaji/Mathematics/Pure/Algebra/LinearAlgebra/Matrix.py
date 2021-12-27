"""
This module describes a matrix as a parameter
"""
#%%
import numpy, sympy
import uncertainties
from uncertainties import unumpy
import numpy.linalg
from Iaji.Mathematics.Parameter import Parameter, ParameterSymbolic, ParameterNumeric
from Iaji.Exceptions import InvalidArgumentError, InconsistentArgumentsError, MissingArgumentsError, MethodNotImplementedError
from .Exceptions import InconsistentShapeError
#%%
ACCEPTED_VALUE_TYPES = [numpy.matrix, numpy.ndarray, uncertainties.unumpy.core.matrix]
ACCEPTED_SHAPE_TYPES = [tuple, list, numpy.array, numpy.ndarray]
print_separator = "-----------------------------------------------"
#%%
class Matrix:
    """
    This class describes a matrix as a physical parameter.
    """
    # ----------------------------------------------------------
    def __init__(self, name="M", value=None, real=False, nonnegative=False):
        self.name = name
        self.type = "vector"
        self.symbolic = MatrixSymbolic(name=name, real=False, nonnegative=False)
        self.numeric = MatrixNumeric(name=name, value=value)
        # Connect property changed signals to chech functions
        self.numeric.value_changed.connect(self.check_shapes)
        self.symbolic.expression_changed.connect(self.check_shapes)
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
        s = "MATRIX: \n" + "name: " + self.name.__str__() + "\n" + print_separator+"\n"\
        +self.numeric.__str__() + "\n" + print_separator + "\n" \
        + self.symbolic.__str__() + "\n" + print_separator
        return s
    # ----------------------------------------------------------
    # ----------------------------------------------------------


class MatrixSymbolic(ParameterSymbolic):
    """
    This class describes a symbolic matrix.
    """
    # ----------------------------------------------------------
    def __init__(self, name="M", real=False, nonnegative=False):
        super().__init__(name=name, type="vector", real=real, nonnegative=nonnegative)
        self._eigenvalues, self._rank, self._trace, self._determinant = [None for j in range(4)]
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def shape(self):
        return self._shape

    @shape.deleter
    def shape(self):
        del self._shape
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def eigenvalues(self):
        return self._eigenvalues

    @eigenvalues.deleter
    def eigenvalues(self):
        del self._eigenvalues
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def rank(self):
        return self._rank

    @rank.deleter
    def rank(self):
        del self._rank
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def trace(self):
        return self._trace

    @trace.deleter
    def trace(self):
        del self._trace
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def determinant(self):
        return self._determinant

    @determinant.deleter
    def determinant(self):
        del self._determinant
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
            if self.type == "vector":
                self._expression = sympy.Matrix(sympy.Array(self._expression))
                self._shape = self._expression.shape
            self.expression_changed.emit()  # emit expression changed signal
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
        s += "\n"\
             +"shape: "+self.shape.__str__()+"\n"\
             +"eigenvalues: "+self.eigenvalues.__str__() +"\n"\
             +"rank: " +self.rank.__str__() +"\n"\
             +"trace: "+self.trace.__str__()+"\n"\
             +"determinant: "+self.determinant.__str__()
        return s.replace("PARAMETER", "MATRIX")
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def Eigenvalues(self):
        """
        This function computes the eigenvalues of the matrix

        OUTPUTS
        -----------------
            The eigenvalues and their algebric multiplicity.
        """
        if self.expression is None:
            raise TypeError("Cannot compute the eigenvalues because the symbolic expression of matrix "+self.name+" is None")
        else:
            self._eigenvalues = sympy.simplify(sympy.Matrix(self.expression).eigenvals())
            return self.eigenvalues
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def Rank(self):
        """
        This function computes the rank of the matrix

        OUTPUTS
        -------------
            The rank of the matrix
        """
        if self.expression is None:
            raise TypeError("Cannot compute the rank because the symbolic expression of matrix "+self.name+" is None")
        else:
            self._rank = sympy.simplify(sympy.Matrix(self.expression).rank())
            return self.rank
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def Trace(self):
        """
        This function computes the trace of the matrix

        OUTPUTS
        -------------
            The trace of the matrix
        """
        if self.expression is None:
            raise TypeError("Cannot compute the trace because the symbolic expression of matrix "+self.name+" is None")
        else:
            self._trace = sympy.simplify(sympy.Matrix(self.expression).trace())
            return self.trace
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def Determinant(self):
        """
        This function computes the determinant of the matrix

        OUTPUTS
        -------------
            The determinant of the matrix
        """
        if self.expression is None:
            raise TypeError("Cannot compute the determinant because the symbolic expression of matrix "+self.name+" is None")
        else:
            if self.eigenvalues is None:
                self.Eigenvalues()
            self._determinant = sympy.simplify(sympy.prod(sympy.Array(self.eigenvalues.keys())))
            return self.determinant
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isPositiveDefinite(self):
        """
        This function checks whether the input matrix is positive definite, i.e., whether all its eigenvalues are greater than zero.

        -------
        is_positive_definite: boolean
            it is true if and only if the input matrix is positive definite
        """
        raise MethodNotImplementedError
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isPositiveSemiDefinite(self):
        """
        This function checks whether the input matrix is positive semidefinite, i.e., whether all its eigenvalues are not lower than zero.
um
        OUTPUTS
        -------
        is_positive_definite: boolean
            it is true if and only if the input matrix is positive semidefinite
        """
        raise MethodNotImplementedError

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isCovarianceMatrix(self):
        """
        This function checks whether the input square matrix is a covariance matrix, i.e., it satisfies the following:
            - It is positive semidefinite (i.e., all is eigenvalue are not lower than zero);
            - It is symmetric (i.e., it is equal to its transpose);
            - It has an inverse (i.e., the determinant is not close to zero);

        OUTPUTS
        -------
        is_covariance_matrix : boolean
            it is true if an only if the input matrix is a covariance matrix

        """
        raise MethodNotImplementedError

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isDensityMatrix(self):
        """
        This function checks whether the input matrix is a density matrix.
        rho is a density matrix if and only if:

            - self is Hermitian
            - self is positive-semidefinite
            - self has trace 1
            - self^2 has trace <= 1
        """
        raise MethodNotImplementedError
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isPositiveSemidefinite(self, tolerance=1e-10):
        """
        This function checks whether the input Hermitian matrix is positive-semidefinite.
        M is positive-semidefinite if an only if all its eigenvalues are nonnegative

        INPUTS
        -------------
            tolerance : float
                An absolute margin on the check that all eigenvalues are nonnegative


        OUTPUTS
        -----------
        True if self is positive-semidefinite
        """
        raise MethodNotImplementedError
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isHermitian(self):
        """
        This function checks whether the input matrix is Hermitian.
        A matrix is hermitian if an only if all of its eigenvalues are real

        OUTPUTS
        -----------
        True if M is is Hermitian
        """
        raise MethodNotImplementedError
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isTraceNormalized(self):
        """
        This function checks whether the input matrix has trace equal to 1.

        OUTPUTS
        -----------
        True if M has trace equal to 1.
        """
        if self.trace is None:
            self.Trace()
        return self.trace == 1
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isTraceConvex(self):
        """
        This function checks whether the input matrix has trace(M^2) <= 1

        OUTPUTS
        -----------
        True if M has trace(M^2) <= 1
        """
        raise MethodNotImplementedError
    # ----------------------------------------------------------



class MatrixNumeric(ParameterNumeric):
    """
    This class describes a numerical matrix.
    """
    # ----------------------------------------------------------
    def __init__(self, name="M", value=None):
        super().__init__(name=name, type="vector", value=value)
        self._eigenvalues, self._rank, self._trace, self._determinant = [None for j in range(4)]
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def shape(self):
        return self._shape

    @shape.deleter
    def shape(self):
        del self._shape

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def eigenvalues(self):
        return self._eigenvalues

    @eigenvalues.deleter
    def eigenvalues(self):
        del self._eigenvalues

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def rank(self):
        return self._rank

    @rank.deleter
    def rank(self):
        del self._rank

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def trace(self):
        return self._trace

    @trace.deleter
    def trace(self):
        del self._trace

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def determinant(self):
        return self._determinant

    @determinant.deleter
    def determinant(self):
        del self._determinant

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
        s += "\n" \
             + "shape: " + self.shape.__str__() + "\n" \
             +"eigenvalues: "+self.eigenvalues.__str__()+"\n"\
             +"rank: " +self.rank.__str__()+"\n"\
             +"trace: "+self.trace.__str__()+"\n"\
             +"determinant: "+self.determinant.__str__()
        return s.replace("PARAMETER", "MATRIX")
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def Eigenvalues(self):
        """
        This function computes the eigenvalues of the matrix

        OUTPUTS
        -----------------
            The eigenvalues and their algebric multiplicity.
        """
        if self.value is None:
            raise TypeError("Cannot compute the eigenvalues because the value of matrix "+self.name+" is None")
        else:
            self._eigenvalues = numpy.linalg.eigvals(self.value)
        return self.eigenvalues
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def Rank(self):
        """
        This function computes the rank of the matrix

        OUTPUTS
        -------------
            The rank of the matrix
        """

        if self.value is None:
            raise TypeError("Cannot compute the rank because the value of matrix " + self.name + " is None")
        else:
            self._rank = numpy.linalg.matrix_rank(self.value)
        return self.rank
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def Trace(self):
        """
        This function computes the trace of the matrix

        OUTPUTS
        -------------
            The trace of the matrix
        """

        if self.value is None:
            raise TypeError("Cannot compute the trace because the value of matrix " + self.name + " is None")
        else:
            self._trace = numpy.trace(self.value)
        return self.trace
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def Determinant(self):
        """
        This function computes the determinant of the matrix

        OUTPUTS
        -------------
            The determinant of the matrix
        """
        if self.value is None:
            raise TypeError("Cannot compute the determinant because the value of matrix "+self.name+" is None")
        else:
            self._determinant = numpy.linalg.det(self.value)
        return self.determinant
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isPositiveDefinite(self):
        """
        This function checks whether the input matrix is positive definite, i.e., whether all its eigenvalues are greater than zero.

        -------
        is_positive_definite: boolean
            it is true if and only if the input matrix is positive definite
        """
        if self.eigenvalues is None:
            self.Eigenvalues()
        return numpy.all(self.eigenvalues > 0)
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isPositiveSemiDefinite(self):
        """
        This function checks whether the input matrix is positive semidefinite, i.e., whether all its eigenvalues are not lower than zero.
um
        OUTPUTS
        -------
        is_positive_definite: boolean
            it is true if and only if the input matrix is positive semidefinite
        """
        if self.eigenvalues is None:
            self.Eigenvalues()
        return numpy.all(self.eigenvalues >= 0)

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isCovarianceMatrix(self):
        """
        This function checks whether the input square matrix is a covariance matrix, i.e., it satisfies the following:
            - It is positive semidefinite (i.e., all is eigenvalue are not lower than zero);
            - It is symmetric (i.e., it is equal to its transpose);
            - It has an inverse (i.e., the determinant is not close to zero);

        OUTPUTS
        -------
        is_covariance_matrix : boolean
            it is true if an only if the input matrix is a covariance matrix

        """
        if self.eigenvalues is None:
            self.Eigenvalues()
        if self.determinant is None:
            self.Determinant()
        min_eigenvalue = numpy.min(numpy.abs(self.eigenvalues))  # modulus of the minimum eigenvalue of the input matrix
        tolerance = 1 / 100 * min_eigenvalue
        return self.isPositiveSemiDefinite() \
               and numpy.allclose(self.value, self.value.T, atol=tolerance) \
               and not numpy.isclose(self.determinant, 0, atol=tolerance)

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isDensityMatrix(self):
        """
        This function checks whether the input matrix is a density matrix.
        rho is a density matrix if and only if:

            - self is Hermitian
            - self is positive-semidefinite
            - self has trace 1
            - self^2 has trace <= 1
        """

        check = False
        check_details = {}
        check_details['is Hermitian'] = self.isHermitian()
        check_details['is positive-semidefinite'] = self.isPositiveSemidefinite()
        check_details['has trace 1'] = self.isTraceNormalized()
        check_details['square has trace <= 1'] = self.isTraceConvex()
        check = check_details['is Hermitian'] and check_details['is positive-semidefinite'] \
                and check_details['has trace 1'] and check_details['square has trace <= 1']
        return check, check_details
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isPositiveSemidefinite(self, tolerance=1e-10):
        """
        This function checks whether the input Hermitian matrix is positive-semidefinite.
        M is positive-semidefinite if an only if all its eigenvalues are nonnegative

        INPUTS
        -------------
            tolerance : float
                An absolute margin on the check that all eigenvalues are nonnegative


        OUTPUTS
        -----------
        True if self is positive-semidefinite
        """
        if not self.isHermitian():
            M = (self.value + numpy.transpose(numpy.conj(self.value))) / 2  # take the Hermitian part of M
        return numpy.all(numpy.real(M) >= 0 - tolerance)
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isHermitian(self):
        """
        This function checks whether the input matrix is Hermitian.
        A matrix is hermitian if an only if all of its eigenvalues are real

        OUTPUTS
        -----------
        True if M is is Hermitian
        """
        if self.eigenvalues is None:
            self.Eigenvalues()
        return numpy.all(numpy.isclose(numpy.imag(self.eigenvalues), 0))  # check that all eigenvalues are approximately real
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isTraceNormalized(self):
        """
        This function checks whether the input matrix has trace equal to 1.

        OUTPUTS
        -----------
        True if M has trace equal to 1.
        """
        if self.trace is None:
            self.Trace()
        return numpy.isclose(self.trace, 1)
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isTraceConvex(self):
        """
        This function checks whether the input matrix has trace(M^2) <= 1

        OUTPUTS
        -----------
        True if M has trace(M^2) <= 1
        """
        return numpy.abs(numpy.trace(self.value @ self.value)) <= 1
    # ----------------------------------------------------------




