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
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.WidgetStyles import CavityLockWidgetStyle
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
        # Define widget title label
        self.name_label = QLabel()
        self.name_label.setText(self.cavity_lock.name)
        self.layout.addWidget(self.name_label, Qt.AlignCenter)
        #Add a high-finesse control panel
        self.control_label = QLabel()
        self.control_label.setLineWidth(5)
        self.control_label.setText("Control Panel")
        self.layout.addWidget(self.control_label, Qt.AlignCenter)
        #Add a radio button to select the lock type
        self.button_high_finesse = QRadioButton("High-finesse cavity")
        self.button_high_finesse.toggled.connect(self.button_high_finesse_clicked)
        self.layout.addWidget(self.button_high_finesse)
        #Add buttons for control panel
        self.control_buttons_layout = QGridLayout()
        self.control_button_names = ["scan", "lock", "unlock", "calibrate",
                             "set demodulation phase", "flip demodulation phase"]
        self.control_button_callbacks = dict(
            zip(self.control_button_names, [self.control_button_scan_callback, self.control_button_lock_callback, self.control_button_unlock_callback, \
                                    self.control_button_calibrate_callback, \
                                    self.control_button_set_demodulation_phase_callback, self.control_button_flip_phase_callback]))
        self.control_buttons = {}
        n_rows = 2
        for j in range(len(self.control_button_names)):
            name = self.control_button_names[j]
            button = QPushButton(name)
            button.clicked.connect(self.control_button_callbacks[name])
            self.control_buttons_layout.addWidget(button, int(j / n_rows), int(np.mod(j, n_rows)))
            self.control_buttons[name] = button
        self.layout.addLayout(self.control_buttons_layout)
        #Define a layout for the lockbox module PID parameters
        # Define a monitor scope layout and widget
        self.scope_layout = QVBoxLayout()
        self.scope_widget = cavity_lock.pyrpl_obj.rp.scope._module_widget
        self.scope_layout.addWidget(self.scope_widget)
        self.layout.addLayout(self.scope_layout)


        self.setLayout(self.layout)

        self.style_sheets = CavityLockWidgetStyle().style_sheets
        self.set_style(theme="dark")

    def control_button_scan_callback(self):
        self.cavity_lock.scan()

    def control_button_lock_callback(self):
        self.cavity_lock.lock()

    def control_button_unlock_callback(self):
        self.cavity_lock.unlock()

    def control_button_calibrate_callback(self):
        self.cavity_lock.calibrate()

    def control_button_set_demodulation_phase_callback(self):
        self.cavity_lock.set_demodulation_phase()

    def control_button_flip_phase_callback(self):
        self.cavity_lock.flip_phase()

    def button_high_finesse_clicked(self):
        if self.button_high_finesse.isChecked():
            lock_type = "high finesse"
        else:
            lock_type = "regular"
        self.cavity_lock.set_lock_type(lock_type)

    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        self.name_label.setStyleSheet(self.style_sheets["label"][theme])
        self.control_label.setStyleSheet(self.style_sheets["label"][theme])
        for name in self.control_button_names:
            self.control_buttons[name].setStyleSheet(self.style_sheets["button"][theme])
        self.button_high_finesse.setStyleSheet(self.style_sheets["radiobutton"][theme])

