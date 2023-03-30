"""
This script tests the module DensityMatrix.py
"""

from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.DensityMatrix import DensityMatrix
import sympy, numpy
sympy.init_printing()

alpha, beta, gamma = sympy.symbols(names="alpha, beta, gamma", real=True, positive=True)
value = 1/2.9999*numpy.matrix(numpy.eye(3,3))
expression = sympy.Matrix([[alpha, 0, 0], [0, beta, 0], [0, 0, gamma]])
expression /= expression.trace()
rho = DensityMatrix(name="rho")
rho.numeric.value = value
rho.symbolic.expression = expression
print("Variables of density matrix %s:"%rho.name)
print(rho)
