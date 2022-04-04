#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 23:50:58 2022

@author: jiedz

This module describes the quantum state of a simple harmonic oscillator in the 
number states (Fock) basis.
"""
# In[]
from Iaji.Mathematics.Parameter import ParameterSymbolic, ParameterNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.Matrix import MatrixNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.DensityMatrix import DensityMatrixSymbolic, \
                                                                      DensityMatrixNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.CovarianceMatrix import CovarianceMatrixSymbolic, \
     CovarianceMatrixNumeric
from Iaji.Mathematics.Pure.Algebra.LinearAlgebra.HilbertSpace import HilbertSpace
from Iaji.Physics.Theory.QuantumMechanics.QuantumState import QuantumStateSymbolic, \
                                                              WignerFunctionSymbolic, \
                                                              QuantumStateNumeric, \
                                                              QuantumState
import sympy, numpy, scipy
from sympy import assoc_laguerre
from copy import deepcopy as copy
# In[GUI imports]
import matplotlib
from matplotlib import pyplot
from matplotlib import font_manager
#%%
#General plot settings
default_marker = ""
default_figure_size = (14.5, 10)
default_fontsize = 30
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
# In[]
class QuantumStateFock(QuantumState):
    """
    This class describes the quantum state of a simple harmonic oscillator
    in the number states (Fock) basis. 
    TODONOTE: infinite-dimensional Hilbert spaces are not handeled yet, so the
    dimension of the Hilbert space is truncated up to a finite integer.
    """    
    def __init__(self, truncated_dimension, name="A"):
        super().__init__(name=name)
        self._numeric = QuantumStateFockNumeric(truncated_dimension=truncated_dimension, name=name)
        self._symbolic = QuantumStateFockSymbolic(truncated_dimension=truncated_dimension, name=name)
# In[]
class QuantumStateFockSymbolic(QuantumStateSymbolic):
    """
    This class describes the symbolic quantum state of a simple harmonic oscillator
    in the number states (Fock) basis. 
    TODONOTE: infinite-dimensional Hilbert spaces are not handeled yet, so the
    dimension of the Hilbert space is truncated up to a finite integer.
    """
    #------------------------------------------------------------
    def __init__(self, truncated_dimension, name="A"):
        super().__init__(name=name)
        self._hilbert_space = \
            HilbertSpace(dimension=truncated_dimension, name="H_{%s}"%self.name)
        self.Vacuum()
    # ---------------------------------------------------------- 
    def Vacuum(self):
         """
         Sets the quantum state to the vacuum
         Returns
         -------
         self
    
         """
         e0 = self.hilbert_space.CanonicalBasisVector(0).symbolic
         rho = e0 @ e0.T()
         self._density_operator = DensityMatrixSymbolic() 
         self.density_operator.expression = rho.expression
         self.density_operator.name = "\\left|%d\\right\\rangle\\left\\langle%d\\right|_{%s}"\
             %(0, 0, self.name)
         self.WignerFunction()
         return self
    # ---------------------------------------------------------- 
    def NumberState(self, n):
          """
          Sets the quantum state to the n-th number state
          Returns
          -------
          self
     
          """
          assert n == int(n)
          en= self.hilbert_space.CanonicalBasisVector(int(n)).symbolic
          rho = en @ en.T()
          self._density_operator = DensityMatrixSymbolic() 
          self.density_operator.expression = rho.expression
          self.density_operator.name = "\\left|%d\\right\\rangle\\left\\langle%d\\right|_{%s}"\
              %(int(n), int(n), self.name)
          self.WignerFunction()
          return self
    #----------------------------------------------------------
    def WignerFunction(self):
        """
        Calculates the Wigner function in the number states basis from the
        density matrix
        """
        q, p = sympy.symbols("q_{%s}, p_{%s}"%(self.name, self.name), real=True)
        self._wigner_function = WignerFunctionSymbolic(name="W_{%s}"%self.name)
        self._wigner_function.expression = 0
        N = self.hilbert_space.dimension - 1 
        W_nm = sympy.zeros(N+1, N+1) #expansion coefficients of the Wigner function
        x = 2*(q**2 + p**2)
        #Compute the lower triangle of the matrix W_nm
        for n in numpy.arange(N+1):
            for m in numpy.arange(n+1):
                k=m
                alpha = float(n-m)
                W_nm[n, m] = 1/sympy.pi * sympy.exp(-0.5*x) * (-1)**m \
                                   *(q-sympy.I*p)**(n-m) * sympy.sqrt( 2**(n-m)*sympy.factorial(m)/sympy.factorial(n) )\
                                   * assoc_laguerre(k, alpha, x)     
        #Compute the upper triangle without the diagonal
        for m in numpy.arange(N+1):
            for n in numpy.arange(m):
                W_nm[n, m] = sympy.conjugate(W_nm[m, n])
        
        #Compute the Wigner function
        for n in numpy.arange(N+1):
            for m in numpy.arange(N+1):
                self.wigner_function.expression += W_nm[n, m] * self.density_operator.expression[n, m]
        #Normalize the Wigner function
        self.wigner_function.expression /= \
            sympy.integrate(self.wigner_function.expression, (q, -sympy.oo, sympy.oo), (p, -sympy.oo, sympy.oo))
        return self.wigner_function
    # ----------------------------------------------------------
    def PlotDensityOperator(self, parameters=(), alpha=0.5, colormap=cmwig1, plot_name='untitled'):
         assert self.hilbert_space is not None, \
             "This quantum state is not associated with any Hilbert space"
         assert self.hilbert_space.isFiniteDimensional(),\
             "Cannot plot the density operator of an infinite-dimensional Hilbert space"
         assert not self.isTensorProduct(),\
            "Cannot plot the density operator of a composite system is not supported"
         assert len(self.wigner_function.expression_symbols) != 0, \
           "Cannot plot the density operator of the null state %s"%self.name
         assert len(parameters) == len(self.density_operator.expression_symbols), \
             "Not enough input parameters to plot the density operator. They should be %s"%\
             self.density_operator.expression_symbols.__str__() 
         #Basis enumerate vector
         n = numpy.arange(0, self.hilbert_space.dimension)
         #Compute the density operator
         rho = self.density_operator.expression_lambda(*parameters)
         #Define the figure
         figure = pyplot.figure(num=plot_name, figsize=(11, 8))
         axis = figure.add_subplot(111)
         #Define the maximum modulus of the density operator
         rho_max = numpy.max(numpy.abs(rho))
         #Plot
         axis.imshow(numpy.abs(rho), cmap=cmwig1, alpha=alpha, norm=matplotlib.colors.Normalize(vmin=-rho_max, vmax=rho_max))    
         axis.set_xlabel("n", font=axis_font)
         axis.set_ylabel("m", font=axis_font) 
         pyplot.pause(.05)
         axis.set_xticks(n)
         axis.set_yticks(n)
         axis.set_xticklabels(n, fontsize=ticks_fontsize)
         axis.set_yticklabels(n, fontsize=ticks_fontsize)
         axis.grid(True, color="grey", alpha=0.2)
    #----------------------------------------------------------
    def PlotNumberDistribution(self, parameters=(), alpha=0.7, color="tab:blue", plot_name='untitled'):
       assert self.hilbert_space is not None, \
           "This quantum state is not associated with any Hilbert space"
       assert self.hilbert_space.isFiniteDimensional(),\
           "Cannot plot the number distribution of an infinite-dimensional Hilbert space"
       assert not self.isTensorProduct(),\
          "Cannot plot the number distribution of a composite system is not supported"
       assert len(self.wigner_function.expression_symbols) != 0, \
         "Cannot plot the number distribution of the null state %s"%self.name
       assert len(parameters) == len(self.density_operator.expression_symbols), \
           "Not enough input parameters to plot the number distribution. They should be %s"%\
           self.density_operator.expression_symbols.__str__() 
       #Basis enumerate vector
       n = numpy.arange(0, self.hilbert_space.dimension)
       #Compute the density operator
       rho = self.density_operator.expression_lambda(*parameters)
       #Define figure
       figure = pyplot.figure(num=plot_name, figsize=(11, 8))
       axis = figure.add_subplot(111)
       axis.set_xlabel("number", font=axis_font)
       axis.set_ylabel("probability", font=axis_font)
       photon_number_dsitribution = [numpy.abs(rho[j, j]) for j in range(rho.shape[0])]
       #Plot
       axis.bar(n, photon_number_dsitribution, color=color, alpha=alpha)
       pyplot.pause(.05)
       axis.set_xticklabels(axis.get_xticklabels(), fontsize=ticks_fontsize)
       axis.set_yticklabels(axis.get_yticklabels(), fontsize=ticks_fontsize)
    #----------------------------------------------------------
    def _InitFigure(self, figure_name):
        if figure_name is None:
            figure = pyplot.figure(num="Quantum State - $%s$ "%self.name, figsize=(13, 9))
        else:
            figure = pyplot.figure(num="Quantum State - $%s$ "%figure_name, figsize=(13, 9))
        if len(figure.axes) == 0:
            figure.add_subplot(2, 2, 1,  projection='3d') #Wigner function 3D
            figure.add_subplot(2, 2, 2) #Wigner function 2D
            figure.add_subplot(2, 2, 3) #Density operator
            figure.add_subplot(2, 2, 4) #Boson number Distribution
        return figure
# In[]
class QuantumStateFockNumeric(QuantumStateNumeric):
    """
    This class describes the numeric quantum state of a simple harmonic oscillator
    in the number states (Fock) basis. 
    TODONOTE: infinite-dimensional Hilbert spaces are not handeled yet, so the
    dimension of the Hilbert space is truncated up to a finite integer.
    """
    #------------------------------------------------------------
    def __init__(self, truncated_dimension, name="A"):
        super().__init__(name=name)
        self._hilbert_space = \
            HilbertSpace(dimension=truncated_dimension, name="H_{%s}"%self.name)
        self.Vacuum()
        self.figure_name = None
        self.resized.connect(self._InitFigure)
    # ---------------------------------------------------------- 
    def Vacuum(self):
         """
         Sets the quantum state to the vacuum
         Returns
         -------
         """
         e0 = self.hilbert_space.CanonicalBasisVector(0).numeric
         self._density_operator = e0 @ e0.T()
         self.density_operator.name = "\\left|%d\\right\\rangle\\left\\langle%d\\right|_{%s}"\
             %(int(0), int(0), self.name)
         self._wigner_function = ParameterNumeric(name="W_{%s}"%self.name)
         return copy(self)
    # ---------------------------------------------------------- 
    def NumberState(self, n):
          """
          Sets the quantum state to the n-th number state
          Returns
          -------
          self
     
          """
          assert n == int(n)
          en = self.hilbert_space.CanonicalBasisVector(int(n)).numeric
          self._density_operator = en @ en.T()
          self.density_operator.name = "\\left|%d\\right\\rangle\\left\\langle%d\\right|_{%s}"\
              %(int(n), int(n), self.name)
          self._wigner_function = ParameterNumeric(name="W_{%s}"%self.name)
          return copy(self)
    #----------------------------------------------------------
    def Resize(self, dimension):
        """
        Returns a replica of the quantum state, belonging to a Hilbert space
        with input dimension. If 'dimension' < self.hilbert_space.dimension, then
        the truncation operator of order 'dimension' is applied to the density operator.
        If 'dimension' > self.hilbert_space.dimension, then the density operator
        is extended via direct sum with a null operator of dimension
            dimension - self.hilbert_space.dimension
        All other parameters of the quantum state are simply reset.
        
        INPUTS
        ---------------
        dimension : int
            dimension of the new quantum state's Hilbert space
        """
        if dimension == self.hilbert_space.dimension:
            return copy(self)
        elif dimension < self.hilbert_space.dimension:
            return self.Truncate(dimension-1)
        else:
            return self.Expand(dimension-self.hilbert_space.dimension)
    #----------------------------------------------------------
    def Truncate(self, order):
        """
        Returns a replica of the quantum state, where the truncation operator 
        of order 'n' is applied to the density operator.
        
        INPUTS
        ------------
        order : int
            The truncation order. It must be 'order' < self.hilbert_space.dimension
        
        OUTPUTS
        ------------
        The new quantum state
        """
        if order >= self.hilbert_space.dimension-1:
            return copy(self)
        #Construct truncation operator
        N = self.hilbert_space.dimension
        T = MatrixNumeric.Zeros((N, N))
        for n in range(order+1):
            proj = self.hilbert_space.CanonicalBasisVector(n).numeric
            proj = proj @ proj.Dagger()
            T += proj*numpy.math.factorial(order)/(numpy.math.factorial(order-n)*order**n)
        #Apply the truncation operator to the current quantum state
        rho = copy(self).density_operator
        rho = T @ rho @ T.Dagger()
        rho.value = rho.value[0:order+1, 0:order+1]
        #Define new quantum state
        state = QuantumStateFockNumeric(truncated_dimension=order+1, \
                                        name=self.name)
        state._density_operator = rho
        state._density_operator /= state.density_operator.Trace()
        #Copy the old name for the density operator
        state.density_operator.name = self.density_operator.name
        state.resized.emit()
        return state
    #----------------------------------------------------------
    def Expand(self, n):
        """
        Returns a replica of the quantum state, where the density operator
        is extended via direct sum with a null operator of dimension
            dimension - self.hilbert_space.dimension
        All other parameters of the quantum state are simply reset.
        
        INPUTS
        ----------
        n : int
            increase in the Hilbert space's dimension. 
            
        OUTPUTS
        -------
        The new quantum state
        """
        if n <= 0:
            return copy(self)
        #Define new quantum state
        state = QuantumStateFockNumeric(truncated_dimension=self.hilbert_space.dimension+n, \
                                        name=self.name)
        #Extend the density operator
        state._density_operator = MatrixNumeric\
            .DirectSum([copy(self).density_operator, \
                        MatrixNumeric.Zeros((n, n))])
        #Copy the old name for the density operator
        state.density_operator.name = self.density_operator.name
        state.resized.emit()
        return state
    # ----------------------------------------------------------
    def WignerFunction(self, q, p):
        """
        Calculates the Wigner function in the number states basis from the
        density matrix
        """
        P, Q = numpy.meshgrid(numpy.atleast_1d(p), numpy.atleast_1d(q))
        W = numpy.zeros((len(p), len(q)), dtype=complex)
        N = self.hilbert_space.dimension - 1 #dimension of the (truncated) Hilbert space of rho
        W_nm = numpy.zeros((len(p), len(q), N+1, N+1), dtype=complex) #expansion coefficients of the Wigner function
        X = 2*(Q**2 + P**2)
        #Compute the lower triangle of the matrix W_nm
        for n in numpy.arange(N+1):
            for m in numpy.arange(n+1):
                k=m
                alpha = float(n-m)
                W_nm[:, :, n, m] = 1/numpy.pi * numpy.exp(-0.5*X) * (-1)**m \
                                   *(Q-1j*P)**(n-m) * numpy.sqrt( 2**(n-m)*scipy.special.gamma(m+1)/scipy.special.gamma(n+1) )\
                                   * scipy.special.assoc_laguerre(x=X, n=k, k=alpha)     
        #Compute the upper triangle without the diagonal
        for m in numpy.arange(N+1):
            for n in numpy.arange(m):
                W_nm[:, :, n, m] = numpy.conj(W_nm[:, :, m, n])
        
        #Compute the Wigner function
        for n in numpy.arange(N+1):
            for m in numpy.arange(N+1):
                W += W_nm[:, :, n, m] * self.density_operator.value[n, m]
        #Get rid of nan values
        shape = W.shape
        W = W.flatten()
        W[numpy.where(numpy.isnan(W))] = 0
        W = W.reshape(shape)
        #Normalize in L1
        dq = q[1] - q[0]
        dp = p[1] - p[0]
        W /= numpy.sum(numpy.sum(W) * dq*dp)
        self.wigner_function.value = W
        return Q, P, W
    #----------------------------------------------------------
    def PlotWignerFunction(self, q, p, alpha=0.5, colormap=cmwig1, plot_name = None, plot_contour_on_3D=True):
         assert self.hilbert_space is not None, \
             "This quantum state is not associated with any Hilbert space"
         assert not self.isTensorProduct(),\
            "Cannot plot the Wigner function of a composite system is not supported"
         Q, P, W = self.WignerFunction(q, p)
         W *= numpy.pi
         W_max = numpy.max(numpy.abs(W))
         #Define the x and y axes lines as a 2D function
         xy_2D = numpy.zeros((len(P), len(Q)))
         xy_2D[numpy.where(numpy.logical_or(P==0, Q==0))] = 1
         #3D plot
         figure = self._InitFigure(plot_name)
         #3D Wigner function
         W = numpy.real(W).astype(float)
         Q = Q.astype(float)
         P = P.astype(float)
         axis_3D = figure.axes[0]
         axis_3D.clear()
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
         #figure_2D = pyplot.figure(num="Wigner function - "+plot_name+" - 2D", figsize=default_figure_size)
         axis_2D = figure.axes[1]
         axis_2D.clear()
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
         colorbar = figure.colorbar(_contourf)
         colorbar.set_label('$\pi$ W(q, p)', fontsize=axis_font.get_size(), fontfamily=axis_font.get_family())
         pyplot.pause(.05)
         #axis_2D.set_xticklabels(axis_2D.get_xticks(), fontsize=axis_font.get_size(), fontfamily=axis_font.get_family())
         #axis_2D.set_yticklabels(axis_2D.get_yticks(), fontsize=axis_font.get_size(), fontfamily=axis_font.get_family())
       #  colorbar.set_ticklabels(colorbar.get_ticks())
         #axis_PSD.legend(prop=legend_font)
         pyplot.pause(.05)
         axis_3D.set_xticklabels(axis_3D.get_xticks(), fontsize=axis_font.get_size()*0.4, fontfamily=axis_font.get_family())
         axis_3D.set_yticklabels(axis_3D.get_yticks(), fontsize=axis_font.get_size()*0.4, fontfamily=axis_font.get_family())
         axis_3D.set_zticklabels(numpy.round(axis_3D.get_zticks(), 1), fontsize=axis_font.get_size()*0.4, fontfamily=axis_font.get_family())
     # ----------------------------------------------------------
    def PlotDensityOperator(self, alpha=0.5, colormap=cmwig1, plot_name=None):
        assert self.hilbert_space is not None, \
            "This quantum state is not associated with any Hilbert space"
        assert self.hilbert_space.isFiniteDimensional(),\
            "Cannot plot the density operator of an infinite-dimensional Hilbert space"
        assert not self.isTensorProduct(),\
           "Cannot plot the density operator of a composite system is not supported"
        #Basis enumerate vector
        n = numpy.arange(0, self.hilbert_space.dimension)
        #Compute the density operator
        rho = self.density_operator.value
        #Define the figure
        figure = self._InitFigure(plot_name)
        axis = figure.axes[2]
        #Define the maximum modulus of the density operator
        rho_max = numpy.max(numpy.abs(rho))
        #Plot
        _imshow = axis.imshow(numpy.abs(rho), cmap=cmwig1, alpha=alpha, norm=matplotlib.colors.Normalize(vmin=-rho_max, vmax=rho_max))    
        axis.set_xlabel("n", font=axis_font)
        axis.set_ylabel("m", font=axis_font) 
        colorbar = figure.colorbar(_imshow)
        colorbar.set_label('$\\hat{\\rho}$', fontsize=axis_font.get_size(), fontfamily=axis_font.get_family())
        pyplot.pause(.05)
        axis.set_xticks(n)
        axis.set_yticks(n)
        axis.set_xticklabels(n, fontsize=ticks_fontsize)
        axis.set_yticklabels(n, fontsize=ticks_fontsize)
        axis.grid(True, color="grey", alpha=0.2)
    #----------------------------------------------------------
    def PlotNumberDistribution(self, parameters=(), alpha=0.7, color="tab:blue", plot_name=None):
       assert self.hilbert_space is not None, \
           "This quantum state is not associated with any Hilbert space"
       assert self.hilbert_space.isFiniteDimensional(),\
           "Cannot plot the number distribution of an infinite-dimensional Hilbert space"
       assert not self.isTensorProduct(),\
          "Cannot plot the number distribution of a composite system is not supported"
       #Basis enumerate vector
       n = numpy.arange(0, self.hilbert_space.dimension)
       #Compute the density operator
       rho = self.density_operator.value
       #Define figure
       figure = self._InitFigure(plot_name)
       axis = figure.axes[3]
       axis.set_xlabel("quantum number", font=axis_font)
       axis.set_ylabel("probability", font=axis_font)
       photon_number_dsitribution = [numpy.abs(rho[j, j]) for j in range(rho.shape[0])]
       #Plot
       axis.bar(n, photon_number_dsitribution, color=color, alpha=alpha)
       pyplot.pause(.05)
       axis.set_xticklabels(axis.get_xticklabels(), fontsize=ticks_fontsize)
       axis.set_yticklabels(axis.get_yticklabels(), fontsize=ticks_fontsize)
    #----------------------------------------------------------
    def _InitFigure(self, figure_name=None, **kwargs):
        if figure_name is None:
            if self.figure_name is None:
                self.figure_name = "Quantum State - $%s$ "%self.name
        else:
            self.figure_name = "Quantum State - $%s$ "%figure_name
        figure = pyplot.figure(num=self.figure_name, figsize=(13, 9))
        figure.clear()
        if len(figure.axes) == 0:
            figure.add_subplot(2, 2, 1,  projection='3d') #Wigner function 3D
            figure.add_subplot(2, 2, 2) #Wigner function 2D
            figure.add_subplot(2, 2, 3) #Density operator
            figure.add_subplot(2, 2, 4) #Boson number Distribution
        return figure