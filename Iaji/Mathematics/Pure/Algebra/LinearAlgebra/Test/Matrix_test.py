"""
This script tests the module Parameter.py
"""

from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix import Matrix
import sympy, numpy
sympy.init_printing()

xi, mu, nu, alpha = sympy.symbols(names="xi, mu, nu, alpha", real=True, nonnegative=True)
expression1 = xi*sympy.eye(3)
expression2 = (mu-nu)*sympy.eye(3)
expression2[0, 2] = expression2[2, 0] = xi**3
#Define aM first matrix
M1 = Matrix(name="M_1")
M1.symbolic.expression = expression1
M1.numeric.value = M1.symbolic.expression_lambda(2)
print(M1)
#Define a second matrix
M2 = Matrix(name="M_2")
M2.symbolic.expression = expression2
M2.numeric.value = M2.symbolic.expression_lambda(1, -1, 2)
print(M2)
#Test operations between matrices
print("Elementwise sum")
M = M1 + M2
print(M)

print("Elementwise multiplication")
M = M1 * M2
print(M)

print("Matrix multiplication")
M = M1 @ M2
print(M)

print("Direct sum")
M = M1.DirectSum(M2)
print(M)
