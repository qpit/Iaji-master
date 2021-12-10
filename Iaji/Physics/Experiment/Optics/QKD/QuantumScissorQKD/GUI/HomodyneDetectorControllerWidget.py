"""
This module defines the GUI of the HomodyneDetectorController module.
"""
#%%
from PyQt5.QtCore import Qt
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
    QWidget,
)

from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.HomodyneDetectorController import HomodyneDetectorController
import numpy as np

class HomodyneDetectorControllerWidget(QWidget):
    """
    This class describes the widget of a homodyne detector controller.
    Documentation to be completed.
    """
    def __init__(self, homodyne_detector_controller, name="Homodyne Detector Controller Widget"):
        super().__init__()
        self.hd_controller = homodyne_detector_controller
        self.setWindowTitle(name)
        self.layout = QVBoxLayout()
        #Add phase controller widget
        self.phase_controller_widget = PhaseControllerWidget(self.hd_controller.phase_controller)
        self.layout_phase_controller = QVBoxLayout()
        self.layout_phase_controller.addWidget(self.phase_controller_widget)
        self.layout.addLayout(self.layout_phase_controller)

        self.setLayout(self.layout)


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
        self.setWindowTitle(name)
        #Define the layout
        self.layout = QVBoxLayout()
        #Define phase control layout
        #Define push buttons
        self.layout_control = QVBoxLayout()
        self.button_names = ["scan", "lock", "unlock", "calibrate", "remove offset from DC error signal", "set demodulation phase", "set iq factors"]
        self.button_callbacks = dict(zip(self.button_names, [self.button_scan_callback, self.button_lock_callback, self.button_unlock_callback, \
                                 self.button_calibrate_callback, self.button_remove_offset_pid_DC_callback, \
                                 self.button_set_demodulation_phase_callback, self.button_set_iq_qfactor_callback]))
        self.buttons = {}
        for name in self.button_names:
            button = QPushButton(name)
            button.clicked.connect(self.button_callbacks[name])
            self.layout_control.addWidget(button)
            self.buttons[name] = button
        self.layout.addLayout(self.layout_control)
        # Define a label to show the phase
        self.label_phase = QLabel()
        self.label_phase.setText("Phase: %0.2f deg"%(self.phase_controller.phase * 180 / np.pi))
        self.layout_control.addWidget(self.label_phase)
        # Define a slider to set the LO phase
        self.slider_set_phase = QSlider(Qt.Horizontal)
        self.slider_set_phase.setRange(0, 180)
        self.slider_set_phase.setSingleStep(0.01)
        self.slider_set_phase.valueChanged.connect(self.slider_set_phase_value_changed_callback)
        self.layout_control.addWidget(self.slider_set_phase)
        #Define a monitor scope layout and widget
        self.layout_scope = QVBoxLayout()
        self.scope_widget = self.phase_controller.redpitaya.scope._module_widget
        self.layout_scope.addWidget(self.scope_widget)
        self.layout.addLayout(self.layout_scope)


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












