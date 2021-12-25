"""
This module defines a parameter.
"""
#%%
import sympy
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

    def __init__(self, name="x", value=0):
        self.name = name
        self.value = value
        self.symbol = sympy.symbols(name=self.name)
        self.expression_symbolic = self.symbol
        self.expression_lambda = sympy.lambdify(self.name, self.expression_symbolic)


