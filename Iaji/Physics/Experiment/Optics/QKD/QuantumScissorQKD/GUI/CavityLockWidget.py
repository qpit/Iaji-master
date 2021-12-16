"""
This module defines the GUI of the CavityLock module.
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
    QBoxLayout,
    QGridLayout,
    QWidget,
)

from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.CavityLock import CavityLock
import numpy as np

class CavityLockWidget(QWidget):
    """
    This class describes the widget of a cavity lock.
    Documentation to be completed.
    """
    def __init__(self, cavity_lock, name="Cavity Lock Widget"):
        super().__init__()
        self.cavity_lock = cavity_lock
        self.setWindowTitle(name)
        self.layout = QVBoxLayout()
        #Define style parameters
        self.button_style_sheet = "QPushButton {background-color:#78909C; color:white}"
        self.label_font = QFont("Times New Roman", pointSize=18)
        self.button_font = QFont("Times New Roman", pointSize=13)
        # Define widget title label
        self.name_label = QLabel()
        self.name_label.setText(self.cavity_lock.name)
        self.name_label.setFont(self.label_font)
        self.layout.addWidget(self.name_label, Qt.AlignCenter)
        #Add a high-finesse control panel
        self.high_finesse_label = QLabel()
        self.high_finesse_label.setLineWidth(5)
        self.high_finesse_label.setText(self.cavity_lock.high_finesse_lock.name)
        self.high_finesse_label.setFont(self.label_font)
        self.layout.addWidget(self.high_finesse_label, Qt.AlignCenter)
        self.high_finesse_control_buttons_layout = QGridLayout()
        self.high_finesse_button_names = ["scan", "lock", "unlock", "calibrate",
                             "set demodulation phase", "flip demodulation phase"]
        self.high_finesse_button_callbacks = dict(
            zip(self.high_finesse_button_names, [self.high_finesse_button_scan_callback, self.high_finesse_button_lock_callback, self.high_finesse_button_unlock_callback, \
                                    self.high_finesse_button_calibrate_callback, \
                                    self.high_finesse_button_set_demodulation_phase_callback, self.high_finesse_button_flip_phase_callback]))
        self.high_finesse_buttons = {}
        n_rows = 2
        for j in range(len(self.high_finesse_button_names)):
            name = self.high_finesse_button_names[j]
            button = QPushButton(name)
            button.clicked.connect(self.high_finesse_button_callbacks[name])
            button.setStyleSheet(self.button_style_sheet)
            button.setFont(self.button_font)
            self.high_finesse_control_buttons_layout.addWidget(button, int(j / n_rows), int(np.mod(j, n_rows)))
            self.high_finesse_buttons[name] = button
        self.layout.addLayout(self.high_finesse_control_buttons_layout)
        #Define a layout for the lockbox module PID parameters
        # Define a monitor scope layout and widget
        self.scope_layout = QVBoxLayout()
        self.scope_widget = cavity_lock.pyrpl_obj.rp.scope._module_widget
        self.scope_layout.addWidget(self.scope_widget)
        self.layout.addLayout(self.scope_layout)


        self.setLayout(self.layout)

    def high_finesse_button_scan_callback(self):
        self.cavity_lock.scan()

    def high_finesse_button_lock_callback(self):
        self.cavity_lock.lock()

    def high_finesse_button_unlock_callback(self):
        self.cavity_lock.unlock()

    def high_finesse_button_calibrate_callback(self):
        self.cavity_lock.calibrate_high_finesse()

    def high_finesse_button_set_demodulation_phase_callback(self):
        self.cavity_lock.set_demodulation_phase()

    def high_finesse_button_flip_phase_callback(self):
        self.cavity_lock.flip_phase()
