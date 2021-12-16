"""
This module defines the GUI of the PumpController module.
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

from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.PumpController import PumpController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.CavityLockWidget import CavityLockWidget
import numpy as np

class PumpControllerWidget(QWidget):
    """
    This class describes the widget of a Pump Controller.
    Documentation to be completed.
    """
    def __init__(self, pump_controller, name="Pump Controller Widget"):
        super().__init__()
        self.pump_controller = pump_controller
        self.setWindowTitle(name)
        self.layout = QVBoxLayout()
        #Define style parameters
        self.button_style_sheet = "QPushButton {background-color:#78909C; color:white}"
        self.label_font = QFont("Times New Roman", pointSize=18)
        self.button_font = QFont("Times New Roman", pointSize=13)
        #Define widget title label
        self.name_label = QLabel()
        self.name_label.setText(self.pump_controller.name)
        self.name_label.setFont(self.label_font)
        self.layout.addWidget(self.name_label, Qt.AlignCenter)
        #Insert the SHG cavity lock widget
        self.SHG_cavity_lock_widget = CavityLockWidget(pump_controller.SHG_cavity_lock, name=pump_controller.SHG_cavity_lock.name)
        self.layout.addWidget(self.SHG_cavity_lock_widget)



        self.setLayout(self.layout)

