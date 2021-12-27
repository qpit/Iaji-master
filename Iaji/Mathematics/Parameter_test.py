"""
This script tests the module Parameter.py
"""

from Iaji.Mathematics.Parameter import Parameter
import sympy, numpy
sympy.init_printing()

xi, mu, nu, alpha = sympy.symbols(names="xi, mu, nu, alpha", real=True, nonnegative=True)
#Scalar parameter
p_scalar_value = 1j
p_scalar_symbolic_expression = xi*sympy.exp(-mu) + alpha*sympy.exp(nu)
p_scalar = Parameter(name="p_scalar", value=p_scalar_value, real=True, nonnegative=False)
p_scalar.symbolic.expression = p_scalar_symbolic_expression
print("Variables of p_scalar:")
print(p_scalar)
#Matrix parameter
p_matrix_symbolic_expression = sympy.Matrix([[xi**2, mu+nu], [mu-nu, alpha**2]])
p_matrix_value = numpy.matrix(numpy.ones((2, 2)))
p_matrix = Parameter(name="p_matrix", type="vector", value=p_matrix_value, real=True, nonnegative=False)
p_matrix.symbolic.expression = p_matrix_symbolic_expression
print("Variables of p_matrix:")
print(p_matrix)
