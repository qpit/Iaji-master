"""
This module defines the GUI of the OPOController module.
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

from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.OPOController import OPOController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.CavityLockWidget import CavityLockWidget
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.GainLockWidget import GainLockWidget
import numpy as np
from Iaji.Utilities.strutils import any_in_string

class OPOControllerWidget(QWidget):
    """
    This class describes the widget of a OPO Controller.
    Documentation to be completed.
    """
    def __init__(self, OPO_controller, name="OPO Controller Widget"):
        super().__init__()
        self.OPO_controller = OPO_controller
        self.setWindowTitle(name)
        self.layout = QVBoxLayout()
        #Define style parameters
        self.button_style_sheet = "QPushButton {background-color:#78909C; color:white}"
        self.label_font = QFont("Times New Roman", pointSize=18)
        self.button_font = QFont("Times New Roman", pointSize=13)
        #Define widget title label
        self.name_label = QLabel()
        self.name_label.setText(self.OPO_controller.name)
        self.name_label.setFont(self.label_font)
        self.layout.addWidget(self.name_label, Qt.AlignCenter)
        #Define the OPO cavity lock widget
        self.cavity_lock_widget = CavityLockWidget(OPO_controller.cavity_lock, name=OPO_controller.cavity_lock.name)
        #Define the OPO gain lock widget
        self.gain_lock_widget = GainLockWidget(OPO_controller.gain_lock, name=OPO_controller.gain_lock.name)
        #Define a tab widget with two tabs (cavity, gain lock)
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.cavity_lock_widget, "OPO Cavity Lock")
        self.tab_widget.addTab(self.gain_lock_widget, "OPO Gain Lock")
        self.layout.addWidget(self.tab_widget)

        self.setLayout(self.layout)

