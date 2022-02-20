"""
This module defines a parameter.
"""
#%%
import sympy
import numpy
from ..Exceptions import InconsistentShapeError
from signalslot.signal import Signal
import Iaji
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
    def __add__(self, other):
        try:
            if other.type == "vector":
                return other + self
        except:
            other_temp = self.prepare_other(other)
            new_type = "scalar" * (self.type == "scalar" and other_temp.type == "scalar") \
                       + "vector" * (self.type == "vector" or other_temp.type == "vector")
            name = "\\left(%s+%s\\right)"%(self.name, other_temp.name)
            x = Parameter(name=name, type=new_type)
            x._symbolic = self.symbolic + other_temp.symbolic
            x._numeric = self.numeric + other_temp.numeric
            return x
    # ----------------------------------------------------------
    def __mul__(self, other):
        try:
            
            if other.type == "vector":
                return other + self
        except:
            other_temp = self.prepare_other(other)
            new_type = "scalar" * (self.type == "scalar" and other_temp.type == "scalar") \
                       + "vector" * (self.type == "vector" or other_temp.type == "vector")
            name = "\\left(%s%s\\right)"%(self.name, other_temp.name)
            x = Parameter(name=name, type=new_type)
            x._symbolic = self.symbolic * other_temp.symbolic
            x._numeric = self.numeric * other_temp.numeric
            return x
    # ----------------------------------------------------------
    def __truediv__(self, other):
        try:
            
            if other.type == "vector":
                return other + self
        except:
            other_temp = self.prepare_other(other)
            new_type = "scalar" * (self.type == "scalar" and other_temp.type == "scalar") \
                       + "vector" * (self.type == "vector" or other_temp.type == "vector")
            name = "\\left(\\frac{%s}{%s}\\right)"%(self.name, other_temp.name)
            x = Parameter(name=name, type=new_type)
            x._symbolic = self.symbolic / other_temp.symbolic
            x._numeric = self.numeric / other_temp.numeric
            return x
    # ----------------------------------------------------------
    def __pow__(self, y):
        name = "\\left(%s^{%.1f}\\right)"%(self.name, y)
        x = Parameter(name=name, type=self.type)
        x._symbolic = self.symbolic**y
        x._numeric = self.numeric**y
        return x
    # ----------------------------------------------------------
    def prepare_other(self, other):
        """
        Checks if the other operand is of the same type as self and, in case not
        returns a compatible type object
        """
        try:
            #Assuming other is of type Parameter
            if other.type == "vector":
                return other
            elif other.type == "scalar":
                is_real = other.symbol.is_real is True
                is_nonnegative = other.symbol.is_nonnegative is True
                other_temp = Parameter(name=other.name, type="scalar",\
                                    real=is_real, nonnegative=is_nonnegative)
                other_temp.symbolic.expression = other.symbolic.expression*sympy.ones(*self.symbolic.shape)
                other_temp.numeric.value = other.numeric.value*numpy.ones(self.numeric.shape)
                return other_temp
        except:
            if type(other) in [int, float, complex]:
                if type(other) is int:
                    other = float(other)
                is_real = numpy.isclose(numpy.imag(other), 0)
                is_nonnegative = is_real and (other >= 0)
                other_temp = Parameter(name=str(other), type="scalar",\
                                    real=is_real, nonnegative=is_nonnegative)
                other_temp.symbolic.expression = other*sympy.ones(*self.symbolic.shape)
                other_temp.numeric.value = other*numpy.ones(self.numeric.shape)
            else:
                raise ValueError("Incompatible operand types (%s. %s)"%(type(self), type(other)))
            return other_temp
# In[]
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
        self.symbol = sympy.symbols(names=self.name)
    
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
                self._expression = sympy.Matrix(sympy.Array(self._expression))
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
    def __add__(self, other):
        try:
            
            if other.type == "vector":
                return other + self
        except:
            other_temp = self.prepare_other(other)
            new_type = "scalar" * (self.type == "scalar" and other_temp.type == "scalar") \
                       + "vector" * (self.type == "vector" or other_temp.type == "vector")
            name = "\\left(%s+%s\\right)"%(self.name, other_temp.name)
            x = ParameterSymbolic(name=name, type=new_type)
            self_expression = self.expression
            other_temp_expression = other_temp.expression
            if self_expression is None or other_temp_expression is None:
                raise TypeError("unsupported operand type(s) for +: %s and %s" % (type(self_expression, other_temp_expression)))
            else:
                if new_type == "vector":
                    if self.type == "scalar":
                        self_expression *= sympy.ones(*other_temp_expression.shape)
                    elif other_temp.type == "scalar":
                        other_temp_expression *= sympy.ones(*self_expression.shape)
                    else:
                        pass
            x.expression = self_expression + other_temp_expression
            return x
    # ----------------------------------------------------------
    def __mul__(self, other):
        try:
            if other.type == "vector":
                return other * self
        except:
            other_temp = self.prepare_other(other)
            new_type = "scalar"*(self.type=="scalar" and other_temp.type=="scalar")\
                     + "vector"*(self.type=="vector" or other_temp.type=="vector")
            name = "\\left(%s%s\\right)"%(self.name, other_temp.name)         
            x = ParameterSymbolic(name=name, type=new_type)
            self_expression = self.expression
            other_temp_expression = other_temp.expression
            if self_expression is None or other_temp_expression is None:
                raise TypeError("unsupported operand type(s) for *: %s and %s"%(type(self_expression, other_temp_expression)))
            else:
                if new_type == "vector":
                    if self.type == "scalar":
                        self_expression *= sympy.ones(*other_temp_expression.shape)
                    elif other_temp.type == "scalar":
                        other_temp_expression *= sympy.ones(*self_expression.shape)
                    else:
                        pass
            x.expression = self_expression * other_temp_expression
            return x
    # ----------------------------------------------------------
    def __truediv__(self, other):
        try:     
            if other.type == "vector":
                return other / self
        except:
            other_temp = self.prepare_other(other)
            new_type = "scalar"*(self.type=="scalar" and other_temp.type=="scalar")\
                     + "vector"*(self.type=="vector" or other_temp.type=="vector")
            name = "\\left(\\frac{%s}{%s}\\right)"%(self.name, other_temp.name)         
            x = ParameterSymbolic(name=name, type=new_type)
            self_expression = self.expression
            other_temp_expression = other_temp.expression
            if self_expression is None or other_temp_expression is None:
                raise TypeError("unsupported operand type(s) for *: %s and %s"%(type(self_expression, other_temp_expression)))
            else:
                if new_type == "vector":
                    if self.type == "scalar":
                        self_expression *= sympy.ones(*other_temp_expression.shape)
                    elif other_temp.type == "scalar":
                        other_temp_expression *= sympy.ones(*self_expression.shape)
                    else:
                        pass
            x.expression = self_expression / other_temp_expression
            return x
    # ----------------------------------------------------------
    def __pow__(self, y):
        """
        Power
        """
        name = "\\left(%s^{%.1f}\\right)"%(self.name, y)
        x = ParameterSymbolic(name=name, type=self.type)
        x.expression = self.expression
        if self.type == "scalar":
            x.expression **= y
        else:
            raise TypeError("unsupported operant type for **: %s"%(type(x)))
        return x
    # ----------------------------------------------------------
    def prepare_other(self, other):
        """
        Checks if the other operand is of the same type as self and, in case not
        returns a compatible type object
        """
        try:
            is_real = other.symbol.is_real is True
            is_nonnegative = other.symbol.is_nonnegative is True
            other_temp = ParameterSymbolic(name=other.name, type="scalar",\
                                real=is_real, nonnegative=is_nonnegative)
            other_temp.expression = other.expression
            return other_temp
        except:
            try:
                #assuming other is a sympy symbol or expression
                is_real = other.is_real is True
                is_nonnegative = other.is_real is True and other.is_nonnegative is True
                other_temp = ParameterSymbolic(name=str(other), type="scalar",\
                                    real=is_real, nonnegative=is_nonnegative)
                other_temp.expression = other
            except:
                #Assuming other is a primitive numerical type
                if type(other) in [int, float, complex]:
                    if type(other) is int:
                        other = float(other)
                    is_real = numpy.isclose(numpy.imag(other), 0)
                    is_nonnegative = is_real and (other >= 0)
                    other_temp = ParameterSymbolic(name=str(other), type="scalar",\
                                        real=is_real, nonnegative=is_nonnegative)
                    other_temp.expression = other
                else:
                    raise ValueError("Incompatible operand types (%s. %s)"%(type(self), type(other)))
            return other_temp
                
# In[]
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
    def __add__(self, other):
        try:
            
            if other.type == "vector":
                return other + self
        except:
            other_temp = self.prepare_other(other)
            new_type = "scalar" * (self.type == "scalar" and other_temp.type == "scalar") \
                       + "vector" * (self.type == "vector" or other_temp.type == "vector")
            name = "\\left(%s+%s\\right)"%(self.name, other_temp.name)
            x = ParameterNumeric(name=name, type=new_type)
            self_value = self.value
            other_temp_value = other_temp.value
            if self_value is None or other_temp_value is None:
                raise TypeError("unsupported operand type(s) for +: %s and %s" % (type(self_value, other_temp_value)))
            else:
                if new_type == "vector":
                    if self.type == "scalar":
                        self_value *= sympy.ones(*other_temp_value.shape)
                    elif other_temp.type == "scalar":
                        other_temp_value *= sympy.ones(*self_value.shape)
                    else:
                        pass
            x.value = self_value + other_temp_value
            return x
    # ----------------------------------------------------------
    def __mul__(self, other):
        try:
            
            if other.type == "vector":
                return other * self
        except:
            other_temp = self.prepare_other(other)
            new_type = "scalar"*(self.type=="scalar" and other_temp.type=="scalar")\
                     + "vector"*(self.type=="vector" or other_temp.type=="vector")
            name = "\\left(%s%s\\right)"%(self.name, other_temp.name)
            x = ParameterNumeric(name=name, type=new_type)
            self_value = self.value
            other_temp_value = other_temp.value
            if self_value is None or other_temp_value is None:
                raise TypeError("unsupported operand type(s) for *: %s and %s"%(type(self_value, other_temp_value)))
            else:
                if new_type == "vector":
                    if self.type == "scalar":
                        self_value *= sympy.ones(*other_temp_value.shape)
                    elif other_temp.type == "scalar":
                        other_temp_value *= sympy.ones(*self_value.shape)
                    else:
                        pass
            x.value = self_value * other_temp_value
            return x
    # ----------------------------------------------------------
    def __truediv__(self, other):
        try:
            
            if other.type == "vector":
                return other / self
        except:
            other_temp = self.prepare_other(other)
            new_type = "scalar"*(self.type=="scalar" and other_temp.type=="scalar")\
                     + "vector"*(self.type=="vector" or other_temp.type=="vector")
            name = "\\left(\\frac{%s}{%s}\\right)"%(self.name, other_temp.name)
            x = ParameterNumeric(name=name, type=new_type)
            self_value = self.value
            other_temp_value = other_temp.value
            if self_value is None or other_temp_value is None:
                raise TypeError("unsupported operand type(s) for *: %s and %s"%(type(self_value, other_temp_value)))
            else:
                if new_type == "vector":
                    if self.type == "scalar":
                        self_value *= sympy.ones(*other_temp_value.shape)
                    elif other_temp.type == "scalar":
                        other_temp_value *= sympy.ones(*self_value.shape)
                    else:
                        pass
            x.value = self_value / other_temp_value
            return x
    # ----------------------------------------------------------
    def __pow__(self, y):
        """
        Power
        """
        name = "\\left(%s^{%.1f}\\right)"%(self.name, y)
        x = ParameterNumeric(name=name, type=self.type)
        x.value = self.value
        if self.type == "scalar":
            x.value **= y
        else:
            raise TypeError("unsupported operant type for **: %s"%(type(x)))
        return x
    # ----------------------------------------------------------
    def prepare_other(self, other):
        """
        Checks if the other operand is of the same type as self and, in case not
        returns a compatible type object
        """
        try:
            if other.type == "vector":
                return other
            elif other.type == "scalar":
                other_temp = ParameterNumeric(name=other.name, type="scalar")
                other_temp.value = other.value*numpy.ones(self.shape)
                return other_temp
        except:
            if type(other) in [int, float, complex]:
                if type(other) is int:
                    other = float(other)
                other_temp = ParameterNumeric(name=str(other), type="scalar")
                other_temp.value = other*numpy.ones(self.shape)
            else:
                raise ValueError("Incompatible operand types (%s. %s)"%(type(self), type(other)))
            return other_temp






