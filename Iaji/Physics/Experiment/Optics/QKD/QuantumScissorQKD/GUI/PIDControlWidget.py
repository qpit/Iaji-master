"""
This module defines a PID control widget. It is by default not connected to any functionality.
"""

#%%
from PyQt5.QtCore import Qt, QRect
from PyQt5.Qt import QFont, QFrame
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDateTimeEdit,
    QDial,
    QDoubleSpinBox,
    QFontComboBox,
    QLabel,
    QLCDNumber,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QHBoxLayout,
    QBoxLayout,
    QGridLayout,
    QWidget,
)

from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.WidgetStyles import PIDControlWidgetStyle
from Iaji.Utilities.strutils import any_in_string
#%%

class PIDControlWidget(QWidget):
    def __init__(self, pid, name="PID Control"):
        super().__init__()
        self.pid = pid
        self.name = name
        # Define a layout for the PID parameters
        self.layout = QHBoxLayout()
        self.parameters = ["p", "i", "ival"]
        # Set up PID widgets
        for parameter in self.parameters:
            setattr(self, parameter + "_layout", QHBoxLayout())  # create a layout for the PID parameter
            setattr(self, parameter + "_label", QLabel())  # create a label to show the value
            setattr(self, parameter + "_doublespinbox", QDoubleSpinBox())  # create a slider to set the value
            getattr(self, parameter + "_label").setText(parameter + ":")  # Set the initial text to the label
            getattr(self, parameter + "_layout").addWidget(getattr(self, parameter + "_label"))  # Add the label widget to the current layout
            getattr(self, parameter + "_layout").addWidget(getattr(self, parameter + "_doublespinbox"))  # Add the slider widget to the current layout
            self.layout.addLayout(getattr(self, parameter + "_layout"))  # Add the current layout to the PID layout
            #Set current values
            getattr(self, parameter+"_doublespinbox").setValue(getattr(self.pid, parameter))
            #Set range
            getattr(self, parameter + "_doublespinbox").setRange(-2e3, 2e3)
            #Set precision step
            getattr(self, parameter + "_doublespinbox").setSingleStep(2e-3)
            #Set number of displayed decimals
            getattr(self, parameter + "_doublespinbox").setDecimals(4)
            #Set callback function
            getattr(self, parameter + "_doublespinbox").valueChanged.connect(getattr(self, parameter + "_doublespinbox_changed"))
        self.setLayout(self.layout)

        self.style_sheets = PIDControlWidgetStyle().style_sheets
        self.set_style(theme="dark")

    def set_parameter_changed_callback(self, parameter, callback):
        getattr(self, parameter+"_doublespinbox").valueChanged.connect(callback)

    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        excluded_strings = ["layout", "callback", "clicked", "toggled", "changed", "edited", "checked"]
        for widget_type in ["label", "doublespinbox"]:
            widgets= [getattr(self, name) for name in list(self.__dict__.keys()) if
                       widget_type in name and not any_in_string(excluded_strings, name)]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])

    def p_doublespinbox_changed(self):
        self.pid.p = self.p_doublespinbox.value()

    def i_doublespinbox_changed(self):
        self.pid.i = self.i_doublespinbox.value()

    def ival_doublespinbox_changed(self):
        self.pid.ival = self.ival_doublespinbox.value()