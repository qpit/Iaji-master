"""
This module defines a parameter.
TODO: correctly generalize a parameter to handle vector operation, 
without recurring to the Matrix type (although I prefer using Matrix for every vector operation).
"""
#%%
import sympy
import numpy
from ..Exceptions import InconsistentShapeError
from signalslot.signal import Signal
import Iaji
from Iaji.Utilities import strutils
#%%
print_separator = "-----------------------------------------------"
_ACCEPTED_TYPES = ["scalar", "vector"]
NUMBER_TYPES = [int, numpy.int64, float, numpy.float64, complex, numpy.complex64, numpy.complex128]
# In[parameter]
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
            name = "%s%s"%(self.name, other_temp.name)
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
            name = "\\frac{%s}{%s}"%(self.name, other_temp.name)
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
    def __abs__(self):
        name = "\\left|%s\\right|"%self.name
        x = Parameter(name=name, type=self.type)
        x._symbolic = abs(self.symbolic)
        x._numeric = abs(self.numeric)
        return x
    # ----------------------------------------------------------
    def Conjugate(self):
        name = "\\left(%s\\right)^*"%self.name
        x = Parameter(name=name, type=self.type)
        x._symbolic = self.symbolic.Conjugate()
        x._numeric = self.numeric.Conjugate()
        return x
    # ----------------------------------------------------------
    def Angle(self):
        name = "arg\\left(%s\\right)"%self.name
        x = Parameter(name=name, type=self.type)
        x._symbolic = self.symbolic.Angle()
        x._numeric = self.numeric.Angle()
        return x
    # ----------------------------------------------------------
    def Exp(self):
        name = "e^{%s}"%self.name
        x = Parameter(name=name, type=self.type)
        x._symbolic = self.symbolic.Exp()
        x._numeric = self.numeric.Exp()
        return x
    # ----------------------------------------------------------
    def Sin(self):
        name = "\\sin\\left(%s\\right)"%self.name
        x = Parameter(name=name, type=self.type)
        x._symbolic = self.symbolic.Sin()
        x._numeric = self.numeric.Sin()
        return x
    # ----------------------------------------------------------
    def Arcsin(self):
        name = "\\arcsin\\left(%s\\right)"%self.name
        x = Parameter(name=name, type=self.type)
        x._symbolic = self.symbolic.Arcsin()
        x._numeric = self.numeric.Arcsin()
        return x 
    # ----------------------------------------------------------
    def Cos(self):
        name = "\\cos\\left(%s\\right)"%self.name
        x = Parameter(name=name, type=self.type)
        x._symbolic = self.symbolic.Cos()
        x._numeric = self.numeric.Cos()
        return x
    # ----------------------------------------------------------
    def Arccos(self):
        name = "\\arccos\\left(%s\\right)"%self.name
        x = Parameter(name=name, type=self.type)
        x._symbolic = self.symbolic.Arccos()
        x._numeric = self.numeric.Arccos()
        return x
    # ----------------------------------------------------------
    def Tan(self):
        name = "\\tan\\left(%s\\right)"%self.name
        x = Parameter(name=name, type=self.type)
        x._symbolic = self.symbolic.Tan()
        x._numeric = self.numeric.Tan()
        return x
    # ----------------------------------------------------------
    def Cosh(self):
        name = "\\cosh\\left(%s\\right)"%self.name
        x = Parameter(name=name, type=self.type)
        x._symbolic = self.symbolic.Cosh()
        x._numeric = self.numeric.Cosh()
        return x
    # ----------------------------------------------------------
    def Sinh(self):
        name = "\\sinh\\left(%s\\right)"%self.name
        x = Parameter(name=name, type=self.type)
        x._symbolic = self.symbolic.Sinh()
        x._numeric = self.numeric.Sinh()
        return x
    # ----------------------------------------------------------
    def Tanh(self):
        name = "\\tanh\\left(%s\\right)"%self.name
        x = Parameter(name=name, type=self.type)
        x._symbolic = self.symbolic.Tanh()
        x._numeric = self.numeric.Tanh()
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
            if type(other) in NUMBER_TYPES:
                if "int" in str(type(other)):
                    other = float(other)
                is_real = numpy.isclose(numpy.imag(other), 0)
                is_nonnegative = is_real and (other >= 0)
                other_temp = Parameter(name=str(other), type="scalar",\
                                    real=is_real, nonnegative=is_nonnegative)
                if self.type == "scalar":
                    other_temp.symbolic.expression = sympy.sympify(other)
                    other_temp.numeric.value = other
                else:
                    other_temp.symbolic.expression = other*sympy.ones(*self.symbolic.shape[0])
                    other_temp.numeric.value = other*numpy.ones(self.numeric.shape)
            else:
                raise ValueError("Incompatible operand types (%s. %s)"%(type(self), type(other)))
            return other_temp
# In[symbolic parameter]
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
        self.type = type
        self.expression_changed = Signal()
        self.symbol = sympy.symbols(names=name, real=real, nonnegative=nonnegative)
        self._name = name
        if expression is not None:
            self.expression = expression
        else:
            try:
                self.expression = self.symbol
            except Exception as e:
                print(e)
                self._expression = '0'
                self._expression_symbols = []
                self._expression_lambda = sympy.lambdify((), self.expression, modules="numpy")
            
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
        self._symbol = sympy.symbols(names=name, real=self.symbol.is_real, nonnegative=self.symbol.is_nonnegative)
    
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
                #Construct the lambda function associated to the symbolic expression
                self.expression_symbols = sorted(list(expression.free_symbols), key=lambda x: x.name)
                """
                If an expression has symbols whose names have proper latex
                math formatting, sympy.lambdify will complain. So, convert
                all symbol names from lateX to python variable friendly names.
                """
                expression_symbols_non_latex_names = []
                for s in self.expression_symbols:
                    name = strutils.de_latexify(s.name) #convert from lateX name to python-friendly name
                    expression_symbols_non_latex_names.append(\
                    sympy.symbols(names=name, real=s.is_real, nonnegative=s.is_nonnegative))
                expression_non_latex = strutils.de_latexify(str(expression))
                self.expression_lambda = sympy.lambdify(expression_symbols_non_latex_names,\
                                                        expression_non_latex, modules="numpy")
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
            name = "\\left(%s+%s\\right)"%(self.name, other_temp.name)
            x = ParameterSymbolic(name=name)
            self_expression = self.expression
            other_expression = other_temp.expression
            if self_expression is None or other_expression is None:
                raise TypeError("unsupported operand type(s) for +: %s and %s" % (type(self_expression, other_expression)))
            else:
                x.expression = self_expression + other_expression
            return x
    # ----------------------------------------------------------
    def __sub__(self, other):
        try:
            if other.type == "vector":
                return other - self
        except:
            other_temp = self.prepare_other(other)
            name = "\\left(%s-%s\\right)"%(self.name, other_temp.name)
            x = ParameterSymbolic(name=name)
            self_expression = self.expression
            other_expression = other_temp.expression
            if self_expression is None or other_expression is None:
                raise TypeError("unsupported operand type(s) for -: %s and %s" % (type(self_expression, other_expression)))
            else:
                x.expression = self_expression - other_expression
            return x
    # ----------------------------------------------------------
    #Elementwise multiplication
    def __mul__(self, other):
        try:
            if other.type == "vector":
                return other * self
        except:
            other_temp = self.prepare_other(other)
            print(type(self.expression))
            name = "%s*%s"%(self.name, other_temp.name)
            x = ParameterSymbolic(name=name)
            self_expression = self.expression
            other_expression = other_temp.expression
            if self_expression is None or other_expression is None:
                raise TypeError("unsupported operand type(s) for *: %s and %s" % (type(self_expression, other_expression)))
            else:
                x.expression = self_expression * other_expression
            return x
    # ----------------------------------------------------------
    def __truediv__(self, other):
        """
        Elementwise division
        """
        try:
            if other.type == "vector":
                return other / self
        except:
            other_temp = self.prepare_other(other)
            name = "\\frac{%s}{%s}"%(self.name, other_temp.name)
            x = ParameterSymbolic(name=name)
            self_expression = self.expression
            other_expression = other_temp.expression
            if self_expression is None or other_expression is None:
                raise TypeError("unsupported operand type(s) for *: %s and %s" % (type(self_expression, other_expression)))
            else:
                x.expression = self_expression / other_expression
            return x
    # ----------------------------------------------------------
    def __pow__(self, y):
        """
        Power
        """
        name = "\\left(%s\\right)^{%.1f}"%(self.name, y)
        x = ParameterSymbolic(name=name, type=self.type)
        x.expression = self.expression
        if self.type == "scalar":
            x.expression = sympy.simplify(x.expression**y)
        else:
            raise TypeError("unsupported operant type for **: %s"%(type(x)))
        return x
    # ----------------------------------------------------------
    def __abs__(self):
        """
        Power
        """
        name = "\\left|%s\\right|"%self.name
        x = ParameterSymbolic(name=name, type=self.type)
        x.expression = self.expression
        if self.type == "scalar":
            x.expression = sympy.simplify(sympy.Abs(x.expression))
        else:
            raise TypeError("unsupported operant type for **: %s"%(type(x)))
        return x
    # ----------------------------------------------------------
    def Sqrt(self):
        """
        Power
        """
        name = "\\sqrt{%s}"%self.name
        x = ParameterSymbolic(name=name, type=self.type)
        x.expression = self.expression
        if self.type == "scalar":
            x.expression = sympy.simplify(sympy.sqrt(x.expression))
        else:
            raise TypeError("unsupported operant type for **: %s"%(type(x)))
        return x
    # ----------------------------------------------------------
    def Conjugate(self):
        """
        Complex conjugate
        """
        name = "\\left(%s\\right)^*"%self.name
        x = ParameterSymbolic(name=name)
        if self.expression is None:
            raise TypeError("unsupported operand type for Conjugate: %s" % (type(self.expression)))
        else:
            x.expression = sympy.simplify(sympy.conjugate(self.expression))
        return x
    # ----------------------------------------------------------
    def Angle(self):
        """
        Complex argument
        """
        name = "arg\\left(%s\\right)"%self.name
        x = ParameterSymbolic(name=name)
        if self.expression is None:
            raise TypeError("unsupported operand type for Conjugate: %s" % (type(self.expression)))
        else:
            x.expression = sympy.atan2(sympy.im(self.expression), sympy.re(self.expression))
            if "atan2(0" in x.expression.__str__():
                x.expression = 0
        return x
    # ----------------------------------------------------------
    def Exp(self):
        """
        cosine
        """
        name = "e^{%s}"%self.name
        x = ParameterSymbolic(name=name)
        if self.expression is None:
            raise TypeError("unsupported operand type for Exp: %s" % (type(self.expression)))
        else:
            x.expression = sympy.exp(self.expression)
        return x
    # ----------------------------------------------------------
    def Cos(self):
        """
        cosine
        """
        name = "\\cos\\left(%s\\right)"%self.name
        x = ((self*1j).Exp() + (self*(-1j)).Exp())/2
        x.name = name
        return x
    # ----------------------------------------------------------
    def Arccos(self):
        """
        arc cosine
        """
        name = "\\arccos\\left(%s\\right)"%self.name
        x = ParameterSymbolic(name=name)
        x.expression_symbolic = sympy.acos(x)
        return x
    # ----------------------------------------------------------
    def Sin(self):
        """
        sine
        """
        name = "\\sin\\left(%s\\right)"%self.name
        x = ParameterSymbolic(name=name)
        x = ((self*1j).Exp() - (self*(-1j)).Exp())/(2j)
        x.name = name
        return x
    # ---------------------------------------------------------- 
    def Arcsin(self):
        """
        arc sine
        """
        name = "\\arcsin\\left(%s\\right)"%self.name
        x = ParameterSymbolic(name=name)
        x.expression_symbolic = sympy.asin(x)
        return x
    # ----------------------------------------------------------
    def Tan(self):
        """
        tangent
        """
        name = "\\tan\\left(%s\\right)"%self.name
        x = ParameterSymbolic(name=name)
        x = self.Sin()/self.Cos()
        x.name = name
        return x
    # ----------------------------------------------------------  
    def Cosh(self):
        """
        hyperbolic cosine
        """
        name = "\\cosh\\left(%s\\right)"%self.name
        x = ParameterSymbolic(name=name)
        x = (self.Exp() + self.Exp())/(2)
        x.name = name
        return x
    # ----------------------------------------------------------
    def Sinh(self):
        """
        hyperbolic sine
        """
        name = "\\sinh\\left(%s\\right)"%self.name
        x = ParameterSymbolic(name=name)
        x = (self.Exp() - self.Exp())/(2)
        x.name = name
        return x
    # ---------------------------------------------------------- 
    def Tanh(self):
        """
        hyperbolic tangent
        """
        name = "\\tanh\\left(%s\\right)"%self.name
        x = ParameterSymbolic(name=name)
        x = self.Sinh()/self.Cosh()
        x.name = name
        return x
    # ---------------------------------------------------------- 
    def prepare_other(self, other):
        """
        Checks if the other operand is of the same type as self and, in case not
        returns a compatible type object
        """
        try:
            if other.type == "scalar":
                return other
        except:
            try:
                #assuming other is a sympy symbol or expression
                is_real = other.is_real is True
                is_nonnegative = other.is_real is True and other.is_nonnegative is True
                other_temp = ParameterSymbolic(name=str(other), \
                                    real=is_real, nonnegative=is_nonnegative)
                other_temp.expression = other
            except:
                #Assuming other is a primitive numerical type
                if type(other) in NUMBER_TYPES:
                    if "int" in str(type(other)):
                        other = float(other)
                    is_real = numpy.isclose(numpy.imag(other), 0)
                    is_nonnegative = is_real is True and (other >= 0)
                    other_temp = ParameterSymbolic(name=str(other), \
                                        real=is_real, nonnegative=is_nonnegative)
                    other_temp.expression = sympy.sympify(other)
                else:
                    raise TypeError("Incompatible operand types (%s. %s)"%(type(self), type(other)))
            return other_temp
# In[numeric parameter]
class ParameterNumeric:
    """
    This class describes a numerical parameter.
    Its consists of:
        - a name.
        - a value. It can be associated to an uncertainty
    """
    # ----------------------------------------------------------
    def __init__(self, name="x", type="scalar", value=None):
        self.symbol = sympy.symbols(names=name)
        self._name = name
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
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, name):
        self._name = name
        self._symbol = sympy.symbols(names=name, real=self.symbol.is_real, nonnegative=self.symbol.is_nonnegative)
    @name.deleter
    def name(self):
        del self._name
    # ----------------------------------------------------------
    @property
    def symbol(self):
        return self._symbol

    @symbol.setter
    def symbol(self, symbol):
        self._symbol = symbol

    @symbol.deleter
    def symbol(self):
        del self._symbol
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
        other_temp = self.prepare_other(other)
        name = "\\left(%s+%s\\right)"%(self.name, other_temp.name)
        x = ParameterNumeric(name=name)
        self_value = self.value
        other_value = other_temp.value
        if self_value is None or other_value is None:
            raise TypeError("unsupported operand type(s) for +: %s and %s" % (type(self_value, other_value)))
        else:
            x.value = self_value + other_value
        return x
    # ----------------------------------------------------------
    def __sub__(self, other):
        other_temp = self.prepare_other(other)
        name = "\\left(%s-%s\\right)"%(self.name, other_temp.name)
        x = ParameterNumeric(name=name)
        self_value = self.value
        other_value = other_temp.value
        if self_value is None or other_value is None:
            raise TypeError("unsupported operand type(s) for -: %s and %s" % (type(self_value, other_value)))
        else:
            x.value = self_value - other_value
        return x
    # ----------------------------------------------------------
    #Elementwise multiplication
    def __mul__(self, other):
        other_temp = self.prepare_other(other)
        name = "%s*%s"%(self.name, other_temp.name)
        x = ParameterNumeric(name=name)
        self_value = self.value
        other_value = other_temp.value
        if self_value is None or other_value is None:
            #raise TypeError("unsupported operand type(s) for *: %s and %s" % (type(self_value, other_value)))
            pass
        else:
            x.value = self_value * other_value
        return x
    # ----------------------------------------------------------
    def __truediv__(self, other):
        """
        Elementwise division
        """
        other_temp = self.prepare_other(other)
        name = "\\frac{%s}{%s}"%(self.name, other_temp.name)
        x = ParameterNumeric(name=name)
        self_value = self.value
        other_value = other_temp.value
        if self_value is None or other_value is None:
            raise TypeError("unsupported operand type(s) for *: %s and %s" % (type(self_value, other_value)))
        else:
            x.value = self_value / other_value
        return x
    # ----------------------------------------------------------
    def __pow__(self, y):
        """
        Power
        """
        name = "\\left(%s\\right)^{%.1f}"%(self.name, y)
        x = ParameterNumeric(name=name, type=self.type)
        x.value = self.value
        if self.type == "scalar":
            x.value **= y
        else:
            raise TypeError("unsupported operant type for **: %s"%(type(x)))
        return x
    # ----------------------------------------------------------
    def __abs__(self):
        """
        Power
        """
        name = "\\left|%s\\right|"%self.name
        x = ParameterNumeric(name=name, type=self.type)
        x.value = self.value
        if self.type == "scalar":
            x.value = numpy.abs(x.value)
        else:
            raise TypeError("unsupported operant type for **: %s"%(type(x)))
        return x
    # ----------------------------------------------------------
    def Conjugate(self):
        """
        Complex conjugate
        """
        name = "\\left(%s\\right)^*"%self.name
        x = ParameterNumeric(name=name)
        if self.value is None:
            raise TypeError("unsupported operand type for Conjugate: %s" % (type(self.value)))
        else:
            x.value = numpy.conjugate(self.value)
        return x
    # ----------------------------------------------------------
    def Angle(self):
        """
        Complex argument
        """
        name = "arg\\left(%s\\right)"%self.name
        x = ParameterNumeric(name=name)
        if self.value is None:
            raise TypeError("unsupported operand type for Conjugate: %s" % (type(self.value)))
        else:
            x.value = numpy.angle(self.value)
        return x
    # ----------------------------------------------------------
    def Exp(self):
        """
        cosine
        """
        name = "e^{%s}"%self.name
        x = ParameterNumeric(name=name)
        if self.value is None:
            raise TypeError("unsupported operand type for Exp: %s" % (type(self.value)))
        else:
            x.value = numpy.exp(self.value)
        return x
    # ----------------------------------------------------------
    def Cos(self):
        """
        cosine
        """
        name = "\\cos\\left(%s\\right)"%self.name
        x = ((self*1j).Exp() + (self*(-1j)).Exp())/2
        x.name = name
        return x
    # ----------------------------------------------------------
    def Arccos(self):
        """
        arc cosine
        """
        name = "\\arccos\\left(%s\\right)"%self.name
        x = ParameterNumeric(name=name)
        x.value = numpy.acos(x)
        return x
    # ----------------------------------------------------------
    def Sin(self):
        """
        sine
        """
        name = "\\sin\\left(%s\\right)"%self.name
        x = ParameterNumeric(name=name)
        x = ((self*1j).Exp() - (self*(-1j)).Exp())/(2j)
        x.name = name
        return x
    # ----------------------------------------------------------  
    def Arcsin(self):
        """
        arc sine
        """
        name = "\\arcsin\\left(%s\\right)"%self.name
        x = ParameterNumeric(name=name)
        x.value = numpy.asin(x)
        return x
    # ----------------------------------------------------------
    def Tan(self):
        """
        tangent
        """
        name = "\\tan\\left(%s\\right)"%self.name
        x = ParameterNumeric(name=name)
        x = self.Sin()/self.Cos()
        x.name = name
        return x
    # ----------------------------------------------------------  
    def Cosh(self):
        """
        hyperbolic cosine
        """
        name = "\\cosh\\left(%s\\right)"%self.name
        x = ParameterNumeric(name=name)
        x = (self.Exp() + self.Exp())/(2)
        x.name = name
        return x
    # ----------------------------------------------------------
    def Sinh(self):
        """
        hyperbolic sine
        """
        name = "\\sinh\\left(%s\\right)"%self.name
        x = ParameterNumeric(name=name)
        x = (self.Exp() - self.Exp())/(2)
        x.name = name
        return x
    # ---------------------------------------------------------- 
    def Tanh(self):
        """
        hyperbolic tangent
        """
        name = "\\tanh\\left(%s\\right)"%self.name
        x = ParameterNumeric(name=name)
        x = self.Sinh()/self.Cosh()
        x.name = name
        return x
    # ---------------------------------------------------------- 
    def prepare_other(self, other):
        """
        Checks if the other operand is of the same type as self and, in case not
        returns a compatible type object
        """
        try:
            if other.type == "scalar":
                return other
        except:
            if type(other) in NUMBER_TYPES:
                if "int" in str(type(other)):
                    other = float(other)
                other_temp = ParameterNumeric(name=str(other))
                other_temp.value = other
            else:
                raise TypeError("Incompatible operand types (%s. %s)"%(type(self), type(other)))
            return other_temp





