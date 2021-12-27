"""
This module defines a parameter.
"""
#%%
import sympy
import numpy
from ..Exceptions import InconsistentShapeError
from signalslot.signal import Signal
#%%
print_separator = "-----------------------------------------------"
_ACCEPTED_TYPES = ["scalar", "vector"]

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
    # ----------------------------------------------------------
    def __init__(self, name="x", type="scalar", value=None, real=False, nonnegative=False):
        self.name = name
        self._numeric = ParameterNumeric(name=name, type=type, value=value)
        self._symbolic = ParameterSymbolic(name=name, type=type, real=real, nonnegative=nonnegative)
        self.type = type
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def __str__(self):
        """
        This function returns a string with the summary of the interesting properties of this object.
        """
        s = "PARAMETER: \n"+"name: "+self.name.__str__()+"\n"+print_separator+"\n"\
            +self.numeric.__str__()+"\n"+print_separator+"\n"\
            +self.symbolic.__str__()+"\n"+print_separator

        return s
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @name.deleter
    def name(self):
        del self._name
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        if type not in _ACCEPTED_TYPES:
            raise TypeError("Parameter type is invalid. Valid parameter types are: "+_ACCEPTED_TYPES.__str__())
        else:
            self._type = type

    @type.deleter
    def type(self):
        del self._type
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def symbolic(self):
        return self._symbolic
    
    @symbolic.deleter
    def symbolic(self):
        del self._symbolic
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def numeric(self):
        return self._numeric

    @numeric.deleter
    def numeric(self):
        del self._numeric
    # ----------------------------------------------------------
    #----------------------------------------------------------



class ParameterSymbolic:
    """
    This class defines a symbolic parameter.
    It consists of:
        - a symbol
        - a symbolic expression
        - a list of expression symbols
    """
    # ----------------------------------------------------------
    def __init__(self, name="x", type="scalar", real=False, nonnegative=False, expression=None):
        """
        INPUTS
        ----------
            symbol_name : str
                Name of the symbol
            expression : sympy expression
                Symbolic expression in terms of other symbols
        """
        self.name = name
        self.type = type
        self.expression_changed = Signal()
        self.symbol = sympy.symbols(names=name, real=real, nonnegative=nonnegative)
        self.expression = expression
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def __str__(self):
        """
        This function returns a string with the summary of the interesting properties of this object.
        """
        s = "SYMBOLIC PARAMETER: \n"\
            +"name: " + self.name.__str__() + "\n" \
            + "symbol: " + self.symbol.__str__() + "\n" \
            + "symbolic expression: " + self.expression.__str__() + "\n" \
            + "lambda expression: " + self.expression_lambda.__str__()
        return s

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @name.deleter
    def name(self):
        del self._name
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        if type not in _ACCEPTED_TYPES:
            raise TypeError("Parameter type is invalid. Valid parameter types are: " + _ACCEPTED_TYPES.__str__())
        else:
            self._type = type

    @type.deleter
    def type(self):
        del self._type

    # ----------------------------------------------------------
    #----------------------------------------------------------
    @property
    def symbol(self):
        return self._symbol

    @symbol.setter
    def symbol(self, symbol):
        self._symbol = symbol

    @symbol.deleter
    def symbol(self):
        del self._symbol
    #----------------------------------------------------------
    #----------------------------------------------------------
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
            if self.type == "vector":
                self._expression = sympy.Array(self._expression)
            self.expression_changed.emit()  # emit expression changed signal
        else:
            self.expression_symbols = None
            self.expression_lambda = None

    @expression.deleter
    def expression(self):
        del self._expression
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def expression_symbols(self):
        return self._expression_symbols

    @expression_symbols.setter
    def expression_symbols(self, expression_symbols):
        self._expression_symbols = expression_symbols

    @expression_symbols.deleter
    def expression_symbols(self):
        del self._expression_symbols
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def expression_lambda(self):
        return self._expression_lambda

    @expression_lambda.setter
    def expression_lambda(self, l):
        self._expression_lambda = l

    @expression_lambda.deleter
    def expression_lambda(self):
        del self._expression_lambda
    # ----------------------------------------------------------




class ParameterNumeric:
    """
    This class describes a numerical parameter.
    Its consists of:
        - a name.
        - a value. It can be associated to an uncertainty
    """
    # ----------------------------------------------------------
    def __init__(self, name="x", type="scalar", value=None):
        self.name = name
        self.type = type
        self.value_changed = Signal()
        self.value = value
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def has_uncertainty(self):
        return hasattr(self.value, "std_dev") or hasattr(self.value, "std_devs")
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def __str__(self):
        """
        This function returns a string with the summary of the interesting properties of this object.
        """
        s = "NUMERIC PARAMETER\n"\
            +"name: " + self.name.__str__() + "\n" \
            +"value: " + self.value.__str__()
        return s
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, name):
        self._name = name
    @name.deleter
    def name(self):
        del self._name
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        if type not in _ACCEPTED_TYPES:
            raise TypeError("Parameter type is invalid. Valid parameter types are: " + _ACCEPTED_TYPES.__str__())
        else:
            self._type = type

    @type.deleter
    def type(self):
        del self._type

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, value):
        if self.type == "vector":
            self._value = numpy.asarray(value)
        else:
            self._value = value
        self.value_changed.emit() #emit value changed signal
    @value.deleter
    def value(self):
        del self._value
    # ----------------------------------------------------------
    # ----------------------------------------------------------









