"""
This module defines the GUI of the TwoModeSqueezedVacuumGenerator module.
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
    QTabWidget,
)

from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.TwoModeSqueezedVacuumGenerator import TwoModeSqueezedVacuumGenerator
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.PumpControllerWidget import PumpControllerWidget
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.OPOControllerWidget import OPOControllerWidget
import numpy as np

class TwoModeSqueezedVacuumGeneratorWidget(QWidget):
    """
    This class describes the widget of a Two-mode Squeezed Vacuum Generator.
    Documentation to be completed.
    """
    def __init__(self, TMSV_generator, name="Two-mode Squeezed Vacuum Generator Widget"):
        super().__init__()
        self.TMSV_generator = TMSV_generator
        self.setWindowTitle(name)
        self.layout = QVBoxLayout()
        #Define style parameters
        self.label_font = QFont("Times New Roman", pointSize=18)
        #Define widget title label
        self.name_label = QLabel()
        self.name_label.setText(self.TMSV_generator.name)
        self.name_label.setFont(self.label_font)
        self.layout.addWidget(self.name_label, Qt.AlignCenter)
        #Define the pump controller widget
        self.pump_controller_widget = PumpControllerWidget(TMSV_generator.pump_controller, name=TMSV_generator.pump_controller.name)
        #Define the OPO controller widget
        self.OPO_controller_widget = OPOControllerWidget(TMSV_generator.OPO_controller, name=TMSV_generator.OPO_controller.name)
        #Define a tab widget with two tabs (cavity, gain lock)
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.pump_controller_widget, "Pump Controller")
        self.tab_widget.addTab(self.OPO_controller_widget, "OPO Controller")
        self.layout.addWidget(self.tab_widget)

        self.setLayout(self.layout)

