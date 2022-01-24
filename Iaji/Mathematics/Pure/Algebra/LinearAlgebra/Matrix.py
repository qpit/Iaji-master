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
from .Exceptions import InconsistentShapeError, TestFailedError
#%%
ACCEPTED_VALUE_TYPES = [numpy.matrix, numpy.ndarray, uncertainties.unumpy.core.matrix]
ACCEPTED_SHAPE_TYPES = [tuple, list, numpy.array, numpy.ndarray]
print_separator = "-----------------------------------------------"
#%%
class Matrix:
    """
    This class describes a matrix as a parameter.
    """
    # ----------------------------------------------------------
    def __init__(self, name="M", value=None, real=False, nonnegative=False):
        self._symbolic = MatrixSymbolic(name=name, real=False, nonnegative=False)
        self._numeric = MatrixNumeric(name=name, value=value)
        self.name = name
        self.type = "vector"
        # Connect property changed signals to chech functions
        self.numeric.value_changed.connect(self.check_shapes)
        self.symbolic.expression_changed.connect(self.check_shapes)
    
    # ----------------------------------------------------------
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name
        self.numeric.name = name
        self.symbolic.name = name

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
    def check_shapes(self, **kwargs):
        if self.numeric.shape is not None and self.symbolic.shape is not None:
            if self.numeric.shape != self.symbolic.shape:
                raise InconsistentShapeError(
                    "The shape of my value is " + self.numeric.shape.__str__() + \
                    ". while the shape of my symbolic expression is " + self.symbolic.shape.__str__())
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
    #Elementwise addition
    def __add__(self, other):
        x = Matrix(name=self.name.__str__() + "+" + other.name.__str__())
        x._symbolic = self.symbolic + other.symbolic
        x._numeric = self.numeric + other.numeric
        return x
    # ----------------------------------------------------------
    #Elementwise multiplication
    def __mul__(self, other):
        x = Matrix(name=self.name.__str__() + "*" + other.name.__str__())
        x._symbolic = self.symbolic * other.symbolic
        x._numeric = self.numeric * other.numeric
        return x
    # ----------------------------------------------------------
    #Matrix multiplication
    def __matmul__(self, other):
        x = Matrix(name=self.name.__str__() + "@" + other.name.__str__())
        x._symbolic = self.symbolic @ other.symbolic
        x._numeric = self.numeric @ other.numeric
        return x
    # ----------------------------------------------------------
    #Direct sum
    def DirectSum(self, other):
        x = Matrix(name=self.name.__str__() + " \\bigoplus " + other.name.__str__())
        x._symbolic = self.symbolic.DirectSum(other.symbolic)
        x._numeric = self.numeric.DirectSum(other.numeric)
        return x
    # ----------------------------------------------------------
    def HermitianConjugate(self):
        """
        Hermitian conjugate
        """
        x = Matrix(name="%s^\\dagger"%self.name)
        x._numeric = self.numeric.HermitianConjugate()
        x._symbolic = self.symbolic.HermitianConjugate()
        return x
    # ----------------------------------------------------------
    def Transpose(self):
        """
        Transpose
        """
        x = Matrix(name="%s^T"%self.name)
        x._numeric = self.numeric.Transpose()
        x._symbolic = self.symbolic.Transpose()
        return x
    # ----------------------------------------------------------
    def Inverse(self):
        """
        Moore-Penrose inverse
        """
        x = Matrix(name="%s^{-1}"%self.name)
        x._numeric = self.numeric.Inverse()
        x._symbolic = self.symbolic.Inverse()
        return x



class MatrixSymbolic(ParameterSymbolic):
    """
    This class describes a symbolic matrix.
    """
    # ----------------------------------------------------------
    def __init__(self, name="M", real=False, nonnegative=False):
        super().__init__(name=name, type="vector", real=real, nonnegative=nonnegative)
        self._eigenvalues, self._rank, self._trace, self._determinant = [None for j in range(4)] 
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
            raise TypeError("Cannot compute the eigenvalues because the symbolic expression of matrix "+self.name+" is None\n"+self.__str__())
        elif not self.isSquare():
            raise TypeError("Cannot compute the eigenvalues because the symbolic expression of matrix "+self.name+" is not square\n"+self.__str__())
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
            raise TypeError("Cannot compute the rank because the symbolic expression of matrix "+self.name+" is None\n"+self.__str__())
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
            raise TypeError("Cannot compute the eigenvalues because the symbolic expression of matrix "+self.name+" is None\n"+self.__str__())
        elif not self.isSquare():
            raise TypeError("Cannot compute the eigenvalues because the symbolic expression of matrix "+self.name+" is not square\n"+self.__str__())
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
        return self.isPositiveSemiDefinite() and not self.isSingular()
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isPositiveSemidefinite(self):
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
        if not self.isHermitian():
            raise TypeError("Cannot test definiteness because the symbolic expression of matrix "+self.name+" is not Hermitian.\n"+self.__str__())
        elif not self.isSymmetric():
            raise TypeError("Cannot test definiteness because the symbolic expression of matrix "+self.name+" is not symmetric.\n"+self.__str__())
        else:
            #Check that all the eigenvalues are nonnegative
            condition = True
            for eigenvalue in sympy.Array(self.eigenvalues.keys()):
                check = eigenvalue >= 0
                if type(check) not in [bool, sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse]:
                    raise TestFailedError("Could not test definiteness because the relation "+check.__str__()\
                                          +" could not be verified for all values of the variables.")
                else:
                    condition = condition and check
                    if condition is False:
                        break
            return condition
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
        check_details = {}
        check_details["is square"] = self.isSquare()
        check_details["is symmetric"] = self.isSymmetric()
        check_details["is positive-semidefinite"] = self.isPositiveSemidefinite()
        check_details["is singular"] = self.isSingular()
        check = check_details["is square"]\
            and check_details["is symmetric"]\
            and check_details["is positive-semidefinite"]\
            and not check_details["is singular"]
        return check, check_details
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isSingular(self):
        """
        This function returns True if and only if the matrix has determinant zero
        """
        if self.determinant is None:
            self.Determinant()
        return self.determinant == 0
        # ----------------------------------------------------------
        # ----------------------------------------------------------

    def isSymmetric(self):
        """
        This function returns True if and only if the matrix is symmetric
        """
        if self.expression is None:
            raise TypeError("Cannot test simmetry because the symbolic expression of matrix "+self.name+" is None\n"+self.__str__())
        elif not self.isSquare():
            raise TypeError("Cannot test simmetry because the symbolic expression of matrix "+self.name+" is not square\n"+self.__str__())
        else:
            return sympy.simplify(self.expression - self.expression.T) == sympy.zeros(*self.shape)
        # ----------------------------------------------------------
        # ----------------------------------------------------------

    def isSquare(self):
        """
        This function returns True if and only if the matrix is square
        """
        if self.expression is None:
            raise TypeError("Cannot test shape because the symbolic expression of matrix "+self.name+" is None\n"+self.__str__())
        else:
            return self.shape[0] == self.shape[1]
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
        check_details = {}
        check_details['is Hermitian'] = self.isHermitian()
        check_details['is positive-semidefinite'] = self.isPositiveSemidefinite()
        check_details['has trace 1'] = self.isTraceNormalized()
        check = check_details['is Hermitian'] and check_details['is positive-semidefinite'] \
                and check_details['has trace 1']
        return check, check_details
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isHermitian(self):
        """
        This function checks whether the input matrix is Hermitian.
        A matrix is hermitian if an only if all of its eigenvalues are real
        """
        if self.eigenvalues is None:
            self.Eigenvalues()
        if not self.isSymmetric():
            raise TypeError("Cannot test Hermitianity because the symbolic expression of matrix "+self.name+" is not symmetric.\n"+self.__str__())
        else:
            #Check that all the eigenvalues are real
            condition = True
            for eigenvalue in sympy.Array(self.eigenvalues.keys()):
                eigenvalue_im = sympy.simplify(sympy.im(eigenvalue))
                #Often sympy does not recognize a null imaginary part,
                #because it does not see that atan2(0, ...) = 0.
                #So I enforce it by hard substitution
                eigenvalue_im = sympy.sympify(eigenvalue_im.__str__().replace("atan2(0,", "atan2(0, 1)*(")\
                                              .replace("^", "").replace("{", "").replace("}", ""))
                check = eigenvalue_im == 0
                if type(check) not in [bool, sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse]:
                    raise TestFailedError("Could not test Hermitianity because the relation "+check.__str__()\
                                          +" could not be verified for all values of the variables.")
                else:
                    condition = condition and check
                    if condition is False:
                        break
            return condition
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
        This function checks whether the matrix has trace(self.expression^2) <= 1
        """
        if self.expression is None:
            raise TypeError("Cannot test trace convexity because the symbolic expression of matrix "+self.name+" is None\n"+self.__str__())
        elif not self.isSquare():
            raise TypeError("Cannot test trace convexity because the symbolic expression of matrix "+self.name+" is not square\n"+self.__str__())
        elif not self.isHermitian():
            raise TypeError(
                "Cannot test trace convexity because the symbolic expression of matrix " + self.name + " is not Hermitian.\n" + self.__str__())
        else:
            condition = sympy.simplify((sympy.Matrix(self.expression)*sympy.Matrix(self.expression)).trace())-1 <= 0
            if type(condition) not in [bool, sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse]:
                raise TestFailedError(
                    "Could not test definiteness because the relation " + condition.__str__() \
                    + " could not be verified for all values of the variables.")
            else:
                return condition
    # ----------------------------------------------------------
    # Elementwise addition
    def __add__(self, other):
        x = MatrixSymbolic(name=self.name.__str__() + "+" + other.name.__str__())
        self_expression = self.expression
        other_expression = other.expression
        if self_expression is None or other_expression is None:
            raise TypeError("unsupported operand type(s) for +: %s and %s" % (type(self_expression, other_expression)))
        else:
            x.expression = self_expression + other_expression
        return x
    # ----------------------------------------------------------
    # Elementwise multiplication
    def __mul__(self, other):
        x = MatrixSymbolic(name=self.name.__str__() + "*" + other.name.__str__())
        self_expression = self.expression
        other_expression = other.expression
        if self_expression is None or other_expression is None:
            raise TypeError("unsupported operand type(s) for *: %s and %s" % (type(self_expression, other_expression)))
        else:
            x.expression = sympy.matrix_multiply_elementwise(self_expression, other_expression)
        return x
    # ----------------------------------------------------------
    # Matrix multiplication
    def __matmul__(self, other):
        x = MatrixSymbolic(name=self.name.__str__() + "@" + other.name.__str__())
        self_expression = self.expression
        other_expression = other.expression
        if self_expression is None or other_expression is None:
            raise TypeError("unsupported operand type(s) for @: %s and %s" % (type(self_expression), type(other_expression)))
        else:
            x.expression = self_expression * other_expression
        return x
    # ----------------------------------------------------------
    # Matrix determinant in operator syntax
    def __abs__(self):
        if self.expression is None:
            return None
        else:
            return self.Determinant()
    # ----------------------------------------------------------
    # Matrix direct sum
    def DirectSum(self, other):
        x = MatrixSymbolic(name=self.name.__str__() + " bigoplus " + other.name.__str__())
        self_expression = self.expression
        other_expression = other.expression
        if self_expression is None:
            self_expression = numpy.matrix([])
        if other_expression is None:
            self_expression = numpy.matrix([])
        x.expression = sympy.zeros(*(numpy.array(self_expression.shape) + numpy.array(other_expression.shape)))
        x.expression[0:self_expression.shape[0], 0:self_expression.shape[1]] = self_expression
        x.expression[self_expression.shape[0]:, self_expression.shape[1]:] = other_expression
        return x
    # ----------------------------------------------------------
    def HermitianConjugate(self):
        """
        Hermitian conjugate
        """
        x = self.Conjugate().Transpose()
        x.name = self.name.__str__()+"^\dagger"
        return x
    # ----------------------------------------------------------
    def Transpose(self):
        """
        Transpose
        """
        x = MatrixSymbolic(name = self.name.__str__()+"^T")
        if self.expression is None:
            raise TypeError("unsupported operand type for Transpose: %s" % (type(self.expression)))
        else:
            x.expression = self.expression.T
        return x
    # ----------------------------------------------------------
    def Conjugate(self):
        """
        Complex conjugate
        """
        x = MatrixSymbolic(name = self.name.__str__()+"^*")
        if self.expression is None:
            raise TypeError("unsupported operand type for Conjugate: %s" % (type(self.expression)))
        else:
            x.expression = sympy.conjugate(self.expression)
        return x
    # ----------------------------------------------------------
    def Inverse(self):
        """
        Moore-Penrose inverse
        """
        x = MatrixSymbolic(name = self.name.__str__()+"^{-1}")
        if self.expression is None:
            raise TypeError("unsupported operand type for Inverse: %s" % (type(self.expression)))
        else:
            x.expression = self.expression.inv()
        return x






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
            self._value = numpy.matrix(value, dtype=numpy.complex64)
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
            raise TypeError("Cannot compute the eigenvalues because the value of matrix "+self.name+" is None\n"+self.__str__())
        elif not self.isSquare():
            raise TypeError("Cannot compute the eigenvalues because the value of matrix "+self.name+" is not square\n"+self.__str__())
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
            raise TypeError("Cannot compute the rank because the value of matrix " + self.name + " is None\n"+self.__str__())
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
            raise TypeError("Cannot compute the eigenvalues because the value of matrix "+self.name+" is None\n"+self.__str__())
        elif not self.isSquare():
            raise TypeError("Cannot compute the eigenvalues because the value of matrix "+self.name+" is not square\n"+self.__str__())
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
        if self.eigenvalues is None:
            self.Eigenvalues()
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
        check_details = {}
        check_details["is square"] = self.isSquare()
        check_details["is symmetric"] = self.isSymmetric(tolerance=tolerance)
        check_details["is positive-semidefinite"] = self.isPositiveSemidefinite()
        check_details["is singular"] = self.isSingular(tolerance=tolerance)
        check = check_details["is square"]\
            and check_details["is symmetric"]\
            and check_details["is positive-semidefinite"]\
            and not check_details["is singular"]
        return check, check_details

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isSingular(self, tolerance=1/100):
        """
        This function returns True if and only if the matrix has determinant zero

        INPUTS
        ------------
            tolerance : float (>0)
                An absolute tolerance on the condition under test
        """
        if self.determinant is None:
            self.Determinant()
        return numpy.isclose(self.determinant, 0, atol=tolerance)
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isSymmetric(self, tolerance=1/100):
        """
        This function returns True if and only if the matrix is symmetric
        """
        if self.value is None:
            raise TypeError(
                "Cannot test simmetry because the value of matrix " + self.name + " is None\n" + self.__str__())
        elif not self.isSquare():
            raise TypeError(
                "Cannot test simmetry because the value of matrix " + self.name + " is not square\n" + self.__str__())
        else:
            min_eigenvalue = numpy.min(
                numpy.abs(self.eigenvalues))  # modulus of the minimum eigenvalue of the input matrix
            tolerance = tolerance * min_eigenvalue
            return numpy.allclose(self.value, self.value.T, atol=tolerance)
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isSquare(self):
        """
        This function returns True if and only if the matrix is square

        INPUTS
        ------------
            tolerance : float (>0)
                An absolute tolerance on the condition under test
        """
        if self.value is None:
            raise TypeError("Cannot check shape because the value of matrix " + self.name + " is None\n"+self.__str__())
        else:
            return self.shape[0] == self.shape[1]

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isDensityMatrix(self, tolerance=1/100):
        """
        This function checks whether the matrix is a density matrix.
        self is a density matrix if and only if:

            - self.value is Hermitian
            - self.value is positive-semidefinite
            - self.value has trace 1
        """
        check_details = {}
        check_details['is Hermitian'] = self.isHermitian(tolerance)
        check_details['is positive-semidefinite'] = self.isPositiveSemidefinite(tolerance)
        check_details['has trace 1'] = self.isTraceNormalized(tolerance)
        check = check_details['is Hermitian'] and check_details['is positive-semidefinite'] \
                and check_details['has trace 1']
        return check, check_details
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isPositiveSemidefinite(self, tolerance=1e-10):
        """
        This function checks whether the input Hermitian matrix is positive-semidefinite.
        M is positive-semidefinite if an only if all its eigenvalues are nonnegative

        INPUTS
        -------------
            tolerance : float (>0)
                An absolute tolerance on the condition under test


        OUTPUTS
        -----------
        True if self is positive-semidefinite
        """
        if self.eigenvalues is None:
            self.Eigenvalues()
        if not self.isHermitian(tolerance):
            raise TypeError(
                "Cannot test definiteness because the value of matrix " + self.name + " is not Hermitian.\n" + self.__str__())
        elif not self.isSymmetric(tolerance):
            raise TypeError(
                "Cannot test definiteness because the value of matrix " + self.name + " is not symmetric.\n" + self.__str__())
        else:
            return numpy.all(numpy.real(self.value) >= 0)
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isHermitian(self, tolerance=1/100):
        """
        This function checks whether the input matrix is Hermitian.
        A matrix is hermitian if an only if all of its eigenvalues are real
        """
        if self.eigenvalues is None:
            self.Eigenvalues()
        if not self.isSymmetric(tolerance=tolerance):
            raise TypeError(
                "Cannot test Hermitianity because the value of matrix " + self.name + " is not symmetric.\n" + self.__str__())
        else:
            min_eigenvalue = numpy.min(numpy.abs(self.eigenvalues))  # modulus of the minimum eigenvalue of the input matrix
            tolerance = tolerance * min_eigenvalue
            return numpy.all(numpy.isclose(numpy.imag(self.eigenvalues), 0, atol=tolerance))  # check that all eigenvalues are approximately real
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def isTraceNormalized(self, tolerance=1/100):
        """
        This function checks whether the matrix has trace equal to 1.
        """
        if self.trace is None:
            self.Trace()
        if self.eigenvalues is None:
            self.Eigenvalues()
        min_eigenvalue = numpy.min(numpy.abs(self.eigenvalues))  # modulus of the minimum eigenvalue of the input matrix
        tolerance = tolerance * min_eigenvalue
        return numpy.isclose(self.trace, 1, atol=tolerance)
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
    #Elementwise addition
    def __add__(self, other):
        x = MatrixNumeric(name=self.name.__str__() + "+" + other.name.__str__())
        self_value = self.value
        other_value = other.value
        if self_value is None or other_value is None:
            raise TypeError("unsupported operand type(s) for +: %s and %s" % (type(self_value, other_value)))
        else:
            x.value = self_value + other_value
        return x
    # ----------------------------------------------------------
    #Elementwise multiplication
    def __mul__(self, other):
        x = MatrixNumeric(name=self.name.__str__() + "*" + other.name.__str__())
        self_value = self.value
        other_value = other.value
        if self_value is None or other_value is None:
            raise TypeError("unsupported operand type(s) for *: %s and %s" % (type(self_value, other_value)))
        else:
            x.value = self_value * other_value
        return x
    # ----------------------------------------------------------
    #Matrix multiplication
    def __matmul__(self, other):
        x = MatrixNumeric(name=self.name.__str__() + "@" + other.name.__str__())
        self_value = self.value
        other_value = other.value
        if self_value is None or other_value is None:
            raise TypeError("unsupported operand type(s) for @: %s and %s" % (type(self_value, other_value)))
        else:
            x.value = self_value @ other_value
        return x

    # ----------------------------------------------------------
    # Matrix determinant in operator syntax
    def __abs__(self):
        if self.value is None:
            return None
        else:
            return self.Determinant()
    # ----------------------------------------------------------
    #Matrix direct sum
    def DirectSum(self, other):
        x = MatrixNumeric(name=self.name.__str__() + "\\bigoplus" + other.name.__str__())
        self_value = self.value
        other_value = other.value
        if self_value is None:
            self_value = numpy.matrix([])
        if other_value is None:
            self_value = numpy.matrix([])
        x.value = numpy.zeros(tuple(numpy.array(self_value.shape) + numpy.array(other_value.shape)))
        x.value[0:self_value.shape[0], 0:self_value.shape[1]] = self_value
        x.value[self_value.shape[0]:, self_value.shape[1]:] = other_value
        return x
    # ----------------------------------------------------------
    def HermitianConjugate(self):
        """
        Hermitian conjugate
        """
        x = self.Conjugate().Transpose()
        x.name = self.name.__str__()+"^\\dagger"
        return x
    # ----------------------------------------------------------
    def Transpose(self):
        """
        Transpose
        """
        x = MatrixNumeric(name = self.name.__str__()+"^T")
        if self.value is None:
            raise TypeError("unsupported operand type for Transpose: %s" % (type(self.value)))
        else:
            x.value = self.value.T
        return x
    # ----------------------------------------------------------
    def Conjugate(self):
        """
        Complex conjugate
        """
        x = MatrixNumeric(name = self.name.__str__()+"^*")
        if self.value is None:
            raise TypeError("unsupported operand type for Conjugate: %s" % (type(self.value)))
        else:
            x.value = numpy.conjugate(self.value)
        return x
    # ----------------------------------------------------------
    def Inverse(self):
        """
        Moore-Penrose inverse 
        """
        x = MatrixNumeric(name = self.name.__str__()+"^{-1}")
        if self.value is None:
            raise TypeError("unsupported operand type for Inverse: %s" % (type(self.value)))
        else:
            x.value = numpy.linalg.inv(self.value)
        return x
    





