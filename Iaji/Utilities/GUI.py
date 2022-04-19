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


class WidgetStyle:
    def __init__(self):
        self.widget_types = ["main", "button", "label", "slider", "tabs", "radiobutton", "doublespinbox", \
                             "linedit", "checkbox", "combobox"]
        self.theme_types = ["dark", "light"]
        self.style_sheets = {}
        for widget_type in self.widget_types:
            self.style_sheets[widget_type] = dict(zip(self.theme_types, [{} for j in range(len(self.theme_types))]))
        #Set default dark theme
        self.style_sheets["main"]["dark"] = """
                                            background-color: #082032; 
                                            color: #EEEEEE;
                                            """
        self.style_sheets["label"]["dark"] = """
                                            QLabel
                                            {
                                            background-color : #082032; 
                                            color: #EEEEEE;
                                            border-color: #EEEEEE; 
                                            border: 2px;
                                            font-family: Times New Roman ;
                                            font-size: 18pt;
                                            max-width : 400px;
                                            max-height :  50px;
                                            }
                                            """
        self.style_sheets["button"]["dark"] = """
                                              QPushButton
                                              {
                                              background-color: #2C394B; 
                                              color: #EEEEEE; 
                                              border: 1.5px solid #C4C4C3;
                                              border-color: #EEEEEE;
                                              border-top-left-radius: 4px;
                                              border-top-right-radius: 4px;
                                              border-bottom-left-radius: 4px;
                                              border-bottom-right-radius: 4px;
                                              max-width : 200px;
                                              max-height :  30px;
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              }
                                              QPushButton:pressed
                                              {
                                              background-color: #EEEEEE; 
                                              color: #2C394B;
                                              }
                                              QPushButton:hover
                                              {
                                              background-color: #EEEEEE; 
                                              color: #2C394B;
                                              }
                                              """
        self.style_sheets["radiobutton"]["dark"] = """
                                              QRadioButton
                                              {
                                              background-color: #2C394B; 
                                              color: #EEEEEE; 
                                              border: 1.5px solid #C4C4C3;
                                              border-color: #EEEEEE;
                                              border-top-left-radius: 4px;
                                              border-top-right-radius: 4px;
                                              border-bottom-left-radius: 4px;
                                              border-bottom-right-radius: 4px;
                                              max-width : 200px;
                                              max-height :  30px;
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              }
                                              QPushButton:pressed
                                              {
                                              background-color: #EEEEEE; 
                                              color: #2C394B;
                                              }
                                              QPushButton:hover
                                              {
                                              background-color: #EEEEEE; 
                                              color: #2C394B;
                                              }
                                              """
        self.style_sheets["tabs"]["dark"] ="""
                                            QTabBar::tab 
                                            {
                                            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                                        stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,
                                                                        stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);
                                            border: 2px solid #C4C4C3;
                                            border-bottom-color: #C2C7CB; /* same as the pane color */
                                            border-top-left-radius: 4px;
                                            border-top-right-radius: 4px;
                                            min-width: 8ex;
                                            width : 250px;
                                            height: 30px;
                                            padding: 2px;
                                            color: black;
                                            font-family: Times New roman;
                                            font-size: 13pt;
                                                }
                                            """
        self.style_sheets["slider"]["dark"] = """
                                              QSlider
                                              {
                                              background-color: #2C394B;
                                              color: #EEEEEE;
                                              border: 1px solid #C4C4C3;
                                              border-color: #EEEEEE;
                                              max-width : 200px;
                                              max-height :  20px;
                                              }
                                              QSlider::handle
                                              {
                                              color : #EEEEEE;
                                              background-color : #EEEEEE;
                                              width : 18px;
                                              height: 35px;
                                              border-radius : 4px;
                                              border: 1px solid #C4C4C3;
                                              border-color: #2C394B; 
                                              }
                                              QSlider::groove
                                              {
                                              background-color: #2C394B;
                                              color: #EEEEEE;
                                              }
                                              """
        self.style_sheets["checkbox"]["dark"] = """
                                              QCheckBox
                                              {
                                              background-color: #2C394B; 
                                              color: #EEEEEE; 
                                              border-color: #EEEEEE;
                                              border: 1.5px solid #C4C4C3;
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              max-width : 200px;
                                              max-height :  20px;
                                              }
                                              """
        self.style_sheets["linedit"]["dark"] = """
                                                 QLineEdit
                                                 {
                                                 background-color : #2C394B;
                                                 border-color:#EEEEEE;
                                                 font-family: Times New Roman;
                                                 font-size: 12pt;
                                                 }
                                                 """
        self.style_sheets["doublespinbox"]["dark"] = """
                                                    QDoubleSpinBox
                                                    {
                                                    background-color : #2C394B;
                                                     border-color:#EEEEEE;
                                                     font-family: Times New Roman;
                                                     font-size: 12pt;
                                                    }
                                                    """
        self.style_sheets["combobox"]["dark"] = """
                                              QComboBox
                                              {
                                              background-color: #2C394B; 
                                              color: #EEEEEE; 
                                              border-color: #EEEEEE;
                                              border: 1.5px solid #C4C4C3;
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              max-width : 200px;
                                              max-height :  20px;
                                              }
                                              """

