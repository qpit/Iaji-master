"""
This module describes a matrix as a parameter
"""
#%%
import numpy as np
from uncertainties.unumpy import umatrix
from numpy import matrix
import numpy.linalg
from Iaji.Physics.Theory.Parameter import Parameter
from Iaji.Exceptions import InvalidArgumentError, UnmatchingArgumentsError, MissingArgumentsError
#%%
ACCEPTED_VALUE_TYPES = [numpy.matrix, numpy.ndarray, uncertainties.unumpy.core.matrix]
ACCEPTED_SHAPE_TYPES = [tuplel, list, numpy.array, numpy.ndarray]
class Matrix(Parameter):
    """
    This class describes a matrix as a physical parameter.
    """
    def __init__(self, shape=None, value=None, name="M"):
        """
        INPUTS
        ------------
            shape : tuple (shape int {(2,), (,2)})
                shape of the matrix
            value : in [numpy.matrix, numpy.ndarray, uncertainties.unumpy.core.matrix]
                value of the matrix
        """
        if not (shape or value):
            raise MissingArgumentsError(error_message="Either shape or value must be specified")
        elif value and not shape:
            if type(value) not in ACCEPTED_VALUE_TYPES:
                raise InvalidArgumentError(error_message="value must belong to one of the following classes: "+str(ACCEPTED_VALUE_TYPES))
            self.value = value
            self.shape = self.value.shape
        elif shape and not value:
            if shape not in ACCEPTED_SHAPE_TYPES:
                raise InvalidArgumentError(error_message="shape must belong to one of the following classes: "+str(ACCEPTED_SHAPE_TYPES))


def isPositiveDefinite(matrix):
    """
    This function checks whether the input matrix is positive definite, i.e., whether all its eigenvalues are greater than zero.

    INPUTS
    ----------
    matrix : 2D array-like of floats
        input square matrix.

    OUTPUTS
    -------
    is_positive_definite: boolean
        it is true if and only if the input matrix is positive definite
    """

    return np.all(np.linalg.eigvals(matrix) > 0)


def isPositiveSemiDefinite(matrix):
    """
    This function checks whether the input matrix is positive semidefinite, i.e., whether all its eigenvalues are not lower than zero.

    INPUTS
    ----------
    matrix : 2D array-like of floats
        input square matrix.

    OUTPUTS
    -------
    is_positive_definite: boolean
        it is true if and only if the input matrix is positive semidefinite
    """
    try:
        return np.all(np.linalg.eigvals(matrix) >= 0)
    except:
        raise TypeError


def isCovarianceMatrix(matrix):
    """
    This function checks whether the input square matrix is a covariance matrix, i.e., it satisfies the following:
        - It is positive semidefinite (i.e., all is eigenvalue are not lower than zero);
        - It is symmetric (i.e., it is equal to its transpose);
        - It has an inverse (i.e., the determinant is not close to zero);

    INPUTS
    ----------
    matrix : array-like of float
        input square matrix

    OUTPUTS
    -------
    is_covariance_matrix : boolean
        it is true if an only if the input matrix is a covariance matrix

    """
    try:
        matrix = np.matrix(matrix)
        min_eigenvalue = np.min(
            np.abs(np.linalg.eigvals(matrix)))  # modulus of the minimum eigenvalue of the input matrix
        tolerance = 1 / 100 * min_eigenvalue
        return isPositiveSemiDefinite(matrix) and np.allclose(matrix, matrix.T, atol=tolerance) and not np.isclose(
            np.linalg.det(matrix), 0, atol=tolerance)
    except:
        raise TypeError


def isDensityMatrix(rho):
    """
    This function checks whether the input matrix is a density matrix.
    rho is a density matrix if and only if:

        - rho is Hermitian
        - rho is positive-semidefinite
        - rho has trace 1
        - rho^2 has trace <= 1
    """

    check = False
    check_details = {}
    check_details['is Hermitian'] = isHermitian(rho)
    check_details['is positive-semidefinite'] = isPositiveSemidefinite(rho)
    check_details['has trace 1'] = isTraceNormalized(rho)
    check_details['square has trace <= 1'] = isTraceConvex(rho)
    check = check_details['is Hermitian'] and check_details['is positive-semidefinite'] \
            and check_details['has trace 1'] and check_details['square has trace <= 1']
    return check, check_details


def isPositiveSemidefinite(M, tolerance=1e-10):
    """
    This function checks whether the input Hermitian matrix is positive-semidefinite.
    M is positive-semidefinite if an only if all its eigenvalues are nonnegative

    INPUTS
    -------------
        M : 2-D array-like of complex
            The input matrix.

    OUTPUTS
    -----------
    True if M is positive-semidefinite
    """
    if not isHermitian(M):
        M = (M + np.transpose(np.conj(M))) / 2  # take the Hermitian part of M
    eigenvalues, _ = np.linalg.eig(M)
    return np.all(np.real(np.array(eigenvalues)) >= 0 - tolerance)


def isHermitian(M):
    """
    This function checks whether the input matrix is Hermitian.
    A matrix is hermitian if an only if all of its eigenvalues are real

    INPUTS
    -------------
        M : 2-D array-like of complex
            The input matrix.

    OUTPUTS
    -----------
    True if M is is Hermitian
    """

    eigenvalues, _ = np.linalg.eig(M)
    eigenvalues_imaginary = np.array([np.imag(e) for e in eigenvalues])  # imaginary part of all eigenvalues
    return np.all(np.isclose(eigenvalues_imaginary, 0))  # check that all eigenvalues are approximately real


def isTraceNormalized(M):
    """
    This function checks whether the input matrix has trace equal to 1.

    INPUTS
    -------------
        M : 2-D array-like of complex
            The input matrix.

    OUTPUTS
    -----------
    True if M has trace equal to 1.
    """

    return np.isclose(np.real(np.trace(M)), 1)


def isTraceConvex(M):
    """
    This function checks whether the input matrix has trace(M^2) <= 1

    INPUTS
    -------------
        M : 2-D array-like of complex
            The input matrix.

    OUTPUTS
    -----------
    True if M has trace(M^2) <= 1
    """

    return np.real(np.trace(M @ M)) <= 1


# %%
# Define exceptions
class InvalidParametersError(Exception):
    def __init__(self, error_message):
        print(error_message)


class InvalidFunctionTypeError(Exception):
    def __init__(self, error_message):
        print(error_message)