"""
This module describes a mixed quantum state or a quantum harmonic oscillator
"""
# In[imports]
from Iaji.Mathematics.Parameter import ParameterSymbolic, ParameterNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.DensityMatrix import DensityMatrixSymbolic, \
     DensityMatrixNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.CovarianceMatrix import CovarianceMatrixSymbolic, \
     CovarianceMatrixNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.HilbertSpace import HilbertSpace
from Iaji.Utilities import strutils

import sympy, numpy
from sympy import assoc_laguerre
# In[GUI imports]
import matplotlib
from matplotlib import pyplot
from matplotlib import font_manager
from matplotlib import cm
# In[]
print_separator = "-----------------------------------------------"
#%%
#General plot settings
default_marker = ""
default_figure_size = (14.5, 10)
default_fontsize = 40
title_fontsize = default_fontsize
title_font = font_manager.FontProperties(family='Times New Roman',
                                   weight='bold',
                                   style='normal', size=title_fontsize)
axis_font = font_manager.FontProperties(family='Times New Roman',
                                   weight='normal',
                                   style='normal', size=title_fontsize*0.8)
legend_font = font_manager.FontProperties(family='Times New Roman',
                                   weight='normal',
                                   style='normal', size=int(numpy.floor(0.9*title_fontsize)))
ticks_fontsize = axis_font.get_size()*0.8
#%%
c1 = [(0.,'#ffffff'), (1/3.,'#FEFEFE'), (1,'#CC0000')]
c2 = [(0.,'#ffffff'), (0.1,'#0C50B7'), (0.2,'#2765C2'), (0.3,'#5889D3'), \
      (0.4,'#A2BEE8'), (0.49,'#FFFFFF'), (0.51,'#FFFFFF'), (0.6,'#E8A2A2'), \
      (0.7,'#D35858'), (0.8,'#C22727'), (0.9,'#B70C0C'), (1.,'#B20000')]
c3 = [(0.,'#ffffff'), (0.1,'#0C50B7'), (0.2,'#2765C2'), (0.3,'#5889D3'), \
      (0.4,'#A2BEE8'), (0.49,'#F9F9F9'), (0.51,'#F9F9F9'), (0.6,'#E8A2A2'), \
      (0.7,'#D35858'), (0.8,'#C22727'), (0.9,'#B70C0C'), (1.,'#B20000')]
cmwig1 = matplotlib.colors.LinearSegmentedColormap.from_list('cmwig1',c2)
cmwig2 = matplotlib.colors.LinearSegmentedColormap.from_list('cmwig2',c3)
# In[quantum state]
class QuantumState:
    """
    This class describes a quantum state.
    It consists of:
        - an associated Hilbert space on the complex numbers
        - a density operator
        - a Wigner function
    """
    #------------------------------------------------------------------------   
    #------------------------------------------------------------
    def __init__(self, name="A"):
        self.name = name
        self._numeric = QuantumStateNumeric(name=name)
        self._symbolic = QuantumStateSymbolic(name=name)
    #----------------------------------------------------------
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
    @property
    def symbolic(self):
        return self._symbolic

    @symbolic.deleter
    def symbolic(self):
        del self._symbolic
    # ----------------------------------------------------------
    @property
    def numeric(self):
        return self._numeric

    @numeric.deleter
    def numeric(self):
        del self._numeric
    # ----------------------------------------------------------
    def __str__(self):
        s = "QUANTUM STATE: \n" + "name: " + self.name.__str__() + "\n" + print_separator+"\n"\
            + self.symbolic.__str__() + "\n" + print_separator+"\n"\
            + self.numeric.__str__() + "\n" + print_separator
        return s
    # ---------------------------------------------------------- 
    #Tensor product
    def Otimes(self, other):
        x = QuantumState(name=self.name.__str__() + " \\Otimes\\; " + other.name.__str__())
        x._symbolic = self.symbolic.Otimes(other.symbolic)
        x._numeric = self.numeric.Otimes(other.numeric)
        return x
    # ----------------------------------------------------------
# In[symbolic Wigner function]
class WignerFunctionSymbolic(ParameterSymbolic):
    """
    This class describes a symbolic Wigner function.
    The only purpose of this re-definition is to give a precise ordering
    to the symbols that might be contained in the wigner function expression
    and related lambda function.
    First come all the 'q' and 'p' symbols, and then all the extra parameters
    in alphabetic order. This might make it easier to handle a symbolic
    Wigner function in simulation codes.
    """
    def __init__(self, name="x"):
        super().__init__(name=name, type="scalar", real=True, nonnegative=False, expression=None)
    # ----------------------------------------------------------    
    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, expression):
        self._expression = expression
        if expression is not None:
            try:
                #Construct the lambda function associated to the symbolic expression
                #First come all the 'q' and 'p' symbols, and then all the extra parameters
                #in alphabetic order. This might make it easier to handle a symbolic
                #Wigner function in simulation codes.
                p_symbols = [x for x in expression.free_symbols if x.name[0]=="p"]
                q_symbols = [x for x in expression.free_symbols if x.name[0]=="q"]
                #Other symbols are made of all symbols minus the union of q and p symbols
                other_symbols = sorted(list(set(expression.free_symbols)-(set(q_symbols)|set(p_symbols))), key=lambda x: x.name)
                #Put the symbols in order
                self.expression_symbols = q_symbols + p_symbols + other_symbols
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
        
# In[symbolic quantum state]
class QuantumStateSymbolic:
    """
    This class describes a symbolic quantum state.
    It consists of:
        - an associated Hilbert space on the complex numbers
        - a density operator
        - a Wigner function
    """
    #------------------------------------------------------------
    def __init__(self, name="A"):
        self.symbol = sympy.symbols(names=name)
        self._name = name      
        self._hilbert_space = None
        self._density_operator = DensityMatrixSymbolic(name="\\rho_{%s}"%self.name)
        self._wigner_function = ParameterSymbolic(name="W_{%s}"%self.name)
        self._covariance_matrix = CovarianceMatrixSymbolic(name="V_{%s}"%self.name)
    #------------------------------------------------------------
    @property
    def hilbert_space(self):
        return self._hilbert_space
    #----------------------------------------------------------
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
    #----------------------------------------------------------
    @property 
    def density_operator(self):
        return self._density_operator
    @density_operator.deleter
    def density_operator(self):
        del self._density_operator
    # ---------------------------------------------------------- 
    @property 
    def wigner_function(self):
        return self._wigner_function
    @wigner_function.deleter
    def wigner_function(self):
        del self._wigner_function
    # ---------------------------------------------------------- 
    @property 
    def covariance_matrix(self):
        return self._covariance_matrix
    @covariance_matrix.deleter
    def covariance_matrix(self):
        del self._covariance_matrix
    # ---------------------------------------------------------- 
    def __str__(self):
        s = "SYMBOLIC QUANTUM STATE: \n" + "name: " + self.name.__str__() + "\n" + print_separator+"\n"\
            + "Hilbert space: \n" + self.hilbert_space.__str__()+ "\n" + print_separator+"\n"\
            +"density operator: \n" + self.density_operator.__str__() + "\n" + print_separator+"\n"\
            +"Wigner function: \n" + self.wigner_function.__str__() + "\n" + print_separator + "\n"\
            + "covariance matrix: \n" + self.covariance_matrix.__str__() + "\n" + print_separator
        return s
    # ----------------------------------------------------------
    def Otimes(self, other):
        """
        Tensor product
        """
        name = "\\left(%s\\Otimes\\;%s\\right)"%(self.name, other.name)
        x = QuantumStateSymbolic(name=name)
        x._hilbert_space = self.hilbert_space.Otimes(other.hilbert_space)
        x._density_operator = self.density_operator.Otimes(other.density_operator)
        x._wigner_function = self.wigner_function * other.wigner_function
        #x._covariance_matrix = self.covariance_matrix.oplus(other.covariance_matrix)
        return x
    # ----------------------------------------------------------
    def Fidelity(self, other):
        X = self.density_operator.SqrtTruncated(5) @ other.density_operator.SqrtTruncated(5)
        return X.TraceDistance(0)**2
    # ----------------------------------------------------------
    def isTensorProduct(self):
        return "Otimes" in self.hilbert_space.symbol.name
    # ----------------------------------------------------------
    def PlotWignerFunction(self, q, p, parameters=(), alpha=0.5, colormap=cmwig1, plot_name='untitled', plot_contour_on_3D=True):

         assert self.hilbert_space is not None, \
             "This quantum state is not associated with any Hilbert space"
         assert not self.isTensorProduct(),\
            "Cannot plot the Wigner function of a composite system is not supported"
         assert len(self.wigner_function.expression_symbols) != 0, \
            "Cannot plot the Wigner function of the null state %s"%self.name
         assert len(parameters) == len(self.wigner_function.expression_symbols)-2, \
             "Not enough input parameters to plot the wigner function. They should be %s"%\
             self.wigner_function.expression_symbols.__str__()     
         Q, P = numpy.meshgrid(q, p)
         W = self.wigner_function.expression_lambda(Q, P, *parameters)
         W *= numpy.pi
         W_max = numpy.max(numpy.abs(W))
         #Define the x and y axes lines as a 2D function
         xy_2D = numpy.zeros((len(P), len(Q)))
         xy_2D[numpy.where(numpy.logical_or(P==0, Q==0))] = 1
         #3D plot
         figure_3D = pyplot.figure(num="Wigner function - "+plot_name+" - 3D", figsize=(11, 8))
         axis_3D = figure_3D.add_subplot(111,  projection='3d')
         #3D Wigner function
         W = W.astype(float)
         Q = Q.astype(float)
         P = P.astype(float)
         axis_3D.plot_surface(Q, P, W, alpha=alpha, cmap=colormap, norm=matplotlib.colors.Normalize(vmin=-W_max, vmax=W_max))
         pyplot.pause(0.5)
         #Plot the contour of the xy projection
         axis_3D.contour(Q, P, W, zdir='z', offset=numpy.min(W)-0.1*W_max, cmap=colormap, norm=matplotlib.colors.Normalize(vmin=-W_max, vmax=W_max))
         #Plot the Q axis
         axis_3D.plot([numpy.min(Q), numpy.max(Q)], [0, 0], zs=numpy.min(W)-0.1*W_max, color='grey', alpha=alpha)
         #Plot the P axis
         axis_3D.plot([0, 0], [numpy.min(P), numpy.max(P)],zs=numpy.min(W)-0.1*W_max, color='grey', alpha=alpha)
         axis_3D.grid(False)
         axis_3D.set_zlim([numpy.min(W)-0.1*W_max, W_max])
         axis_3D.set_xlabel('q (SNU)', fontsize=axis_font.get_size()*0.6, fontfamily=axis_font.get_family())
         axis_3D.set_ylabel('p (SNU)', fontsize=axis_font.get_size()*0.6, fontfamily=axis_font.get_family())
         axis_3D.set_zlabel('$\pi$ W(q, p)', fontsize=axis_font.get_size()*0.6, fontfamily=axis_font.get_family())
         #2D plot
         #Set the color of the plot to white, when the Wigner function is close to 0
         #W[numpy.where(numpy.isclose(W, 0, rtol=1e-3))] = numpy.nan
         #colormap.set_bad('w')
         figure_2D = pyplot.figure(num="Wigner function - "+plot_name+" - 2D", figsize=default_figure_size)
         axis_2D = figure_2D.add_subplot(111)
         axis_2D.set_aspect('equal')
         #2D plot
         _contourf = axis_2D.contourf(Q, P, W, alpha=alpha, cmap=colormap, norm=matplotlib.colors.Normalize(vmin=-W_max, vmax=W_max))
         #Plot the Q axis
         axis_2D.plot([numpy.min(Q), numpy.max(Q)], [0, 0], color='grey', alpha=alpha)
         #Plot the P axis
         axis_2D.plot([0, 0], [numpy.min(P), numpy.max(P)], color='grey', alpha=alpha)
         axis_2D.grid(False)
         axis_2D.set_xlabel('q (SNU)', font=axis_font)
         axis_2D.set_ylabel('p (SNU)', font=axis_font)
         colorbar = figure_2D.colorbar(_contourf)
         colorbar.set_label('$\pi$ W(q, p)', fontsize=axis_font.get_size(), fontfamily=axis_font.get_family())
         pyplot.pause(.05)
         axis_2D.set_xticklabels(axis_2D.get_xticks(), fontsize=axis_font.get_size(), fontfamily=axis_font.get_family())
         axis_2D.set_yticklabels(axis_2D.get_yticks(), fontsize=axis_font.get_size(), fontfamily=axis_font.get_family())
       #  colorbar.set_ticklabels(colorbar.get_ticks())
         #axis_PSD.legend(prop=legend_font)
         pyplot.pause(.05)
         axis_3D.set_xticklabels(axis_3D.get_xticks(), fontsize=axis_font.get_size()*0.4, fontfamily=axis_font.get_family())
         axis_3D.set_yticklabels(axis_3D.get_yticks(), fontsize=axis_font.get_size()*0.4, fontfamily=axis_font.get_family())
         axis_3D.set_zticklabels(numpy.round(axis_3D.get_zticks(), 1), fontsize=axis_font.get_size()*0.4, fontfamily=axis_font.get_family())
         
         figures = {'2D': figure_2D, '3D': figure_3D}
         return figures
     # ----------------------------------------------------------
    def Mean(self, operator):
        """
        Compute the average value of the input linear operator
        given the quantum state
        """
        result = (self.density_operator @ operator).Trace()
        result.name = "\\left\\langle%s\\right\\rangle_{%s}"\
                      %(operator.name, self.density_operator.name)
        return result
    # ----------------------------------------------------------
    def Var(self, operator):
        """
        Compute the variance of the input linear operator
        given the quantum state
        """
        #return self.Mean(x.Anticommutator(x))*0.5
        result = self.RMS(operator) - self.Mean(operator)**2
        result.name = "Var\\left(%s\\right)_{%s}" \
            %(operator.name, self.density_operator.name)
        return result
    # ----------------------------------------------------------
    def Std(self, operator):
        """
        Compute the standard deviation of the input linear operator
        given the quantum state
        """
        result = self.Var(operator)**(0.5)
        result.name = "Std\\left(%s\\right)_{%s}" \
            %(operator.name, self.density_operator.name)
        return result
    # ----------------------------------------------------------
    def RMS(self, operator):
        """
        Compute the RMS value of the input linear operator
        given the quantum state
        """
        return self.Mean(operator**2)
# In[numeric quantum state]
class QuantumStateNumeric:
    """
    This class describes a numeric quantum state.
    It consists of:
        - an associated Hilbert space on the complex numbers
        - a density operator
        - a Wigner function
    """
    #------------------------------------------------------------
    def __init__(self, name="A"):
        self.symbol = sympy.symbols(names=name)
        self._name = name      
        self._hilbert_space = None
        self._density_operator = DensityMatrixNumeric(name="\\rho_{%s}"%self.name)
        self._wigner_function = ParameterNumeric(name="W_{%s}"%self.name)
        self._covariance_matrix = CovarianceMatrixNumeric(name="V_{%s}"%self.name)
    #------------------------------------------------------------
    @property
    def hilbert_space(self):
        return self._hilbert_space
    #----------------------------------------------------------
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
    #----------------------------------------------------------
    @property 
    def density_operator(self):
        return self._density_operator
    @density_operator.deleter
    def density_operator(self):
        del self._density_operator
    # ---------------------------------------------------------- 
    @property 
    def wigner_function(self):
        return self._wigner_function
    @wigner_function.deleter
    def wigner_function(self):
        del self._wigner_function
    # ---------------------------------------------------------- 
    @property 
    def covariance_matrix(self):
        return self._covariance_matrix
    @covariance_matrix.deleter
    def covariance_matrix(self):
        del self._covariance_matrix
    # ---------------------------------------------------------- 
    def __str__(self):
        s = "NUMERIC QUANTUM STATE: \n" + "name: " + self.name.__str__() + "\n" + print_separator+"\n"\
            + "Hilbert space: \n" + self.hilbert_space.__str__()+ "\n" + print_separator+"\n"\
            +"density operator: \n" + self.density_operator.__str__() + "\n" + print_separator+"\n"\
            +"Wigner function: \n" + self.wigner_function.__str__() + "\n" + print_separator + "\n" \
            + "covariance matrix: \n" + self.covariance_matrix.__str__() + "\n" + print_separator
        return s
    # ----------------------------------------------------------
    # Tensor product
    def Otimes(self, other):
        name = "\\left(%s\\Otimes\\;%s\\right)"%(self.name, other.name)
        x = QuantumStateSymbolic(name=name)
        x._hilbert_space = self.hilbert_space.Otimes(other.hilbert_space)
        x._density_operator = self.density_operator.Otimes(other.density_operator)
        x._wigner_function = self.wigner_function * other.wigner_function
        #x._covariance_matrix = self.covariance_matrix.oplus(other.covariance_matrix)
        return x
    # ----------------------------------------------------------
    def Fidelity(self, other):
        """
        Fidelity between two quantum states
        
        """
        x = (((self.density_operator.Sqrt() @ other.density_operator @ self.density_operator.Sqrt()).Sqrt()).Trace())**2
        x.name = "\\mathcal{F}\\left(%s\\;%s\\right)"\
            %(self.density_operator.name, other.density_operator.name)
        return x
    # ----------------------------------------------------------
    def isTensorProduct(self):
        return "Otimes" in self.hilbert_space.symbol.name
    # ----------------------------------------------------------
    def Mean(self, operator):
        """
        Compute the average value of the input linear operator
        given the quantum state
        """
        result = (self.density_operator @ operator).Trace()
        result.name = "\\left\\langle%s\\right\\rangle_{%s}"\
                      %(operator.name, self.density_operator.name)
        return result
    # ----------------------------------------------------------
    def Var(self, operator):
        """
        Compute the variance of the input linear operator
        given the quantum state
        """
        #return self.Mean(x.Anticommutator(x))*0.5
        result = self.RMS(operator) - self.Mean(operator)**2
        result.name = "Var\\left(%s\\right)_{%s}" \
            %(operator.name, self.density_operator.name)
        return result
    # ----------------------------------------------------------
    def Std(self, operator):
        """
        Compute the standard deviation of the input linear operator
        given the quantum state
        """
        result = self.Var(operator)**(0.5)
        result.name = "Std\\left(%s\\right)_{%s}" \
            %(operator.name, self.density_operator.name)
        return result
    # ----------------------------------------------------------
    def RMS(self, operator):
        """
        Compute the RMS value of the input linear operator
        given the quantum state
        """
        return self.Mean(operator**2)