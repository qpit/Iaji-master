"""
This script tests the module Parameter.py
"""

from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix import Matrix
import sympy, numpy
sympy.init_printing()

xi, mu, nu, alpha = sympy.symbols(names="xi, mu, nu, alpha", real=True, nonnegative=True)
value = numpy.matrix([[1, 0], [2, -1j]])
expression = sympy.Matrix([[xi**2, mu+nu], [mu-nu, alpha**2]])
M = Matrix(name="Sigma")
M.symbolic.expression = expression
M.numeric.value = value
print("Variables of matrix %s:"%M.name)
print(M)
