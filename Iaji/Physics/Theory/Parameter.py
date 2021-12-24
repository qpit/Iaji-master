"""
This module defines a parameter.
"""
#%%
import sympy
import numpy as np
import uncertainties
#%%
class Parameter:
    """
    This class defines a parameter.
    It consists of:
        - name
        - symbol
        - symbolic expression in terms of other symbols
        - a python function drawn from the symbolic expression
        - value
    """

    def __init__(self, name="x", value=uncertainties.ufloat(nominal_value=0, std_dev=0), real=True, nonnegative=False):
        self.name = name
        self.value = value
        self.symbol = sympy.symbols(name=self.name, real=real, nonnegative=nonnegative)
        self.expression_symbolic = self.symbol
        self.expression_lambda = sympy.lambdify(self.name, self.expression_symbolic)


