"""
This module defines some utility classes and functions for GUI programming with Qt
"""
# In[imports]
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar)

from matplotlib import pyplot
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QRect
from PyQt5.Qt import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDateTimeEdit,
    QDial,
    QDoubleSpinBox,
    QFontComboBox,
    QHBoxLayout,
    QLabel,
    QLCDNumber,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTimeEdit,
    QVBoxLayout,
    QBoxLayout,
    QGridLayout,
    QWidget,
)

import numpy
# In[Pyplot widget]
class PyplotWidget(QWidget):
    """
    This class describes a Qt widget that contains a matplotlib.pyplot figure
    """
    #-------------------------------------
    def __init__(self, figure: Figure = None, name="Plot"):
        """
        :param figure: matplotlib.figure.Figure
            figure
        :param name: str
            name of the plot
        """
        super().__init__()
        self.figure = figure
        if self.figure is None:
            figure = pyplot.figure()
        self.canvas = FigureCanvas(self.figure)
        self.name = name
        #Window title
        self.setWindowTitle(self.name)
        #Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        # Matplotlib navigation toolbar
        self.navigation_toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.navigation_toolbar)
        #Figure widget
        self.layout.addWidget(self.canvas)
    # -------------------------------------
    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        excluded_strings = ["layout", "callback", "clicked", "toggled", "changed", "edited", "checked"]
        for widget_type in ["label", "button", "tabs"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if
                       widget_type in name and not any_in_string(excluded_strings, name)]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])
    # -------------------------------------
    def update(self):
        self.canvas.draw_idle()
'''
class PyplotWidget(FigureCanvasQTAgg):
    """
    This class describes a Qt widget that contains a matplotlib.pyplot figure
    """
    #-------------------------------------
    def __init__(self, name="Plot"):
        """
        :param figure: matplotlib.figure.Figure
            figure
        :param name: str
            name of the plot
        """
        super().__init__()
        self.name = name
        #Window title
        self.setWindowTitle(self.name)
        #Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
    # -------------------------------------
    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        excluded_strings = ["layout", "callback", "clicked", "toggled", "changed", "edited", "checked"]
        for widget_type in ["label", "button", "tabs"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if
                       widget_type in name and not any_in_string(excluded_strings, name)]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])
    # -------------------------------------
    #def update(self):
   #     self.canvas.draw_idle()
'''



