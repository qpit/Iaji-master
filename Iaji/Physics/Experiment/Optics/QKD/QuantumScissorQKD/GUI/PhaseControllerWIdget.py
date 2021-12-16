"""
This module defines the GUI of the HomodyneDetectorController module.
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

from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.PhaseController import PhaseController
import numpy as np

#%%
class PhaseControllerWidget(QWidget):
    """
    This class describes a phase controller widget.
    It consists of:

    - A button for scanning the phase
    - A button for locking the phase
    - A button to unlock the phase
    -  A button for removing the offset from phase_controller.PID_DC
    - A button for calibrating
    - A button for setting the demodulation phase
    - A button for setting the iq factor

    Documentation to be completed.
    """

    def __init__(self, phase_controller, name="Phase Controller Widget"):
        super().__init__()
        self.phase_controller = phase_controller
        self.name = name
        self.setWindowTitle(name)
        #Define style parameters
        self.button_style_sheet = "QPushButton {background-color:#2196F3; color:white}"
        self.button_font = QFont("Times New Roman", pointSize=13)
        self.label_font = QFont("Times New Roman", pointSize=18)
        #Define the layout
        self.layout = QVBoxLayout()
        #Define phase control layout
        self.control_layout = QVBoxLayout()
        #Set title to layout
        # Define widget title label
        self.name_label = QLabel()
        self.name_label.setText(self.phase_controller.name)
        self.name_label.setFont(self.label_font)
        self.control_layout.addWidget(self.name_label, Qt.AlignCenter)
        #Define push buttons
        self.control_buttons_layout = QGridLayout()
        self.button_names = ["scan", "lock", "unlock", "calibrate", "remove offset from DC error signal", "set demodulation phase", "set iq factors"]
        self.button_callbacks = dict(zip(self.button_names, [self.button_scan_callback, self.button_lock_callback, self.button_unlock_callback, \
                                 self.button_calibrate_callback, self.button_remove_offset_pid_DC_callback, \
                                 self.button_set_demodulation_phase_callback, self.button_set_iq_qfactor_callback]))
        self.buttons = {}
        n_rows = 3
        for j in range(len(self.button_names)):
            name = self.button_names[j]
            button = QPushButton(name)
            button.clicked.connect(self.button_callbacks[name])
            button.setStyleSheet(self.button_style_sheet)
            button.setFont(self.button_font)
            self.control_buttons_layout.addWidget(button, int(j/n_rows), int(np.mod(j, n_rows)))
            self.buttons[name] = button
        self.control_layout.addLayout(self.control_buttons_layout)
        # Define a label to show the phase
        self.label_phase = QLabel()
        self.label_phase.setText("Phase: %0.2f deg"%(self.phase_controller.phase * 180 / np.pi))
        self.label_phase.setFont(QFont("Times New Roman", pointSize=15))
        self.control_layout.addWidget(self.label_phase)
        # Define a slider to set the LO phase
        self.slider_set_phase = QSlider(Qt.Horizontal)
        self.slider_set_phase.setRange(0, 180)
        self.slider_set_phase.setSingleStep(0.01)
        self.slider_set_phase.valueChanged.connect(self.slider_set_phase_value_changed_callback)
        self.control_layout.addWidget(self.slider_set_phase)
        self.layout.addLayout(self.control_layout)
        #Define a monitor scope layout and widget
        self.scope_layout = QVBoxLayout()
        self.scope_widget = self.phase_controller.redpitaya.scope._module_widget
        self.scope_layout.addWidget(self.scope_widget)
        self.layout.addLayout(self.scope_layout)

        self.setLayout(self.layout)


    def button_scan_callback(self):
        self.phase_controller.scan()

    def button_lock_callback(self):
        self.phase_controller.lock()

    def button_unlock_callback(self):
        self.phase_controller.unlock()

    def button_calibrate_callback(self):
        self.phase_controller.calibrate()

    def button_remove_offset_pid_DC_callback(self):
        self.phase_controller.remove_offset_pid_DC()

    def button_set_demodulation_phase_callback(self):
        self.phase_controller.set_iq_phase()

    def button_set_iq_qfactor_callback(self):
        self.phase_controller.set_iq_qfactor()

    def slider_set_phase_value_changed_callback(self, value):
        self.phase_controller.set_phase(value)
        self.label_phase.setText("Phase: %0.2f deg"%(self.phase_controller.phase * 180 / np.pi))












