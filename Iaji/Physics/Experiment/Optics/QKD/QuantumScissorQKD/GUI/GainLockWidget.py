"""
This module defines the GUI of the GainLock module.
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

from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GainLock import GainLock
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.PhaseControllerWidget import PhaseControllerWidget
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.WidgetStyles import GainLockWidgetStyle
import numpy as np

class GainLockWidget(QWidget):
    """
    This class describes the widget of a homodyne detector controller.
    Documentation to be completed.
    """
    def __init__(self, gain_lock, name="Gain Lock Widget"):
        super().__init__()
        self.gain_lock = gain_lock
        self.setWindowTitle(name)
        # Define style parameters
        self.layout = QVBoxLayout()
        # Define widget title label
        self.name_label = QLabel()
        self.name_label.setText(self.gain_lock.name)
        self.layout.addWidget(self.name_label, Qt.AlignCenter)
        #Add phase controller widget
        self.phase_controller_widget = PhaseControllerWidget(self.gain_lock.phase_controller)
        self.layout_phase_controller = QVBoxLayout()
        self.layout_phase_controller.addWidget(self.phase_controller_widget)
        self.layout.addLayout(self.layout_phase_controller)

        self.setLayout(self.layout)

        self.style_sheets = GainLockWidgetStyle().style_sheets
        self.set_style(theme="dark")


    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        self.name_label.setStyleSheet(self.style_sheets["label"][theme])



