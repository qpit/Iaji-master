"""
This module defines the GUI of the LecroyOscilloscope module.
#TODO
"""
#%%
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
    QBoxLayout,
    QGridLayout,
    QWidget,
)

from Iaji.InstrumentsControl.LecroyOscilloscope import LecroyOscilloscope
import numpy as np

#%%
class LecroyOscilloscopeWidget(QWidget):
    """
    This class describes a lecroy oscilloscope widget.
    It consists of:
        - A button that acquires and saves a trace
        - A button that sets the trace saving path
    """

    def __init__(self, scope, name="LecroyOscilloscope Widget"):
        super().__init__()
        self.scope = scope
        self.name = name
        self.setWindowTitle(name)
        #Define the layout
        self.layout = QVBoxLayout()
        #Define phase control layout
        self.control_layout = QVBoxLayout()
        #Set title to layout
        # Define widget title label
        self.name_label = QLabel()
        self.name_label.setText(self.scope.name)
        self.control_layout.addWidget(self.name_label, Qt.AlignCenter)
        #Define push buttons
        #Button that saves a trace
        self.button_acquire = QPushButton("acquire")
        #TODO: define a simple plot widget to display the acquired traces

        self.setLayout(self.layout)
        self.style_sheets = PhaseControllerWidgetStyle().style_sheets
        self.set_style(theme="dark")

    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        for widget_type in ["label", "slider", "button", "radiobutton"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if widget_type in name and "layout" not in name and "callback" not in name]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])
















