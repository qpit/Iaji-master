"""
This script tests the module CovarianceMatrix.py
"""

from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.CovarianceMatrix import CovarianceMatrix
import sympy, numpy
sympy.init_printing()

v_1, v_2, c = sympy.symbols(names="v_1, v_2, c", real=True, positive=True)
value = numpy.matrix([[3, 0.5], [0.5, 3]])
expression = sympy.Matrix([[v_1, c], [c, v_2]])
V = CovarianceMatrix(name="V")
V.symbolic.expression = expression
V.numeric.value = value
print("Variables of covariance matrix %s:"%V.name)
print(V)
