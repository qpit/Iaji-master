"""
This module defines a parameter.
"""
#%%
import sympy
from uncertainties import ufloat
import numpy as np
from Iaji.Mathematics.Parameter import Parameter as MathParameter
#%%
class Parameter(MathParameter):
    """
    This class defines a parameter.
    It consists of:
        - name
        - symbol
        - symbolic expression in terms of other symbols
        - a python function drawn from the symbolic expression
        - value
    """

    def __init__(self, name="x", value=ufloat(nominal_value=0, std_dev=0), real=True, nonnegative=False):
        super().__init__(name=name, value=value)
        self.symbol = sympy.symbols(name, real=real, nonnegative=nonnegative)
        self.expression_symbolic = self.symbol
        self.expression_lambda = sympy.lambdify(self.name, self.expression_symbolic)


