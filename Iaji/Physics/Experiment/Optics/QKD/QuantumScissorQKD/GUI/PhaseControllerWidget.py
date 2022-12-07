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
    QHBoxLayout,
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
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.PIDControlWidget import PIDControlWidget
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.WidgetStyles import PhaseControllerWidgetStyle
import numpy as np
from Iaji.Utilities.strutils import any_in_string
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

    def __init__(self, phase_controller:PhaseController, name="Phase Controller Widget"):
        super().__init__()
        self.phase_controller = phase_controller
        self.name = name
        self.setWindowTitle(name)
        #Define the layout
        self.layout = QVBoxLayout()
        #Define phase control layout
        self.control_layout = QVBoxLayout()
        #Set title to layout
        # Define widget title label
        self.name_label = QLabel()
        self.name_label.setText(self.phase_controller.name)
        self.control_layout.addWidget(self.name_label, Qt.AlignCenter)
        #Define push buttons
        self.control_buttons_layout = QGridLayout()
        control_button_names = ["scan", "lock", "unlock", "flip_iq_phase", "calibrate", "set_iq_qfactor"]
        control_button_callbacks = dict(
            zip(control_button_names, [getattr(self, "control_button_"+name+"_callback") for name in control_button_names]))
        n_rows = 2
        for j in range(len(control_button_names)):
            name = control_button_names[j]
            button = QPushButton(name.replace("_"," "))
            button.clicked.connect(control_button_callbacks[name])
            self.control_buttons_layout.addWidget(button, int(j / n_rows), int(np.mod(j, n_rows)))
            setattr(self, "control_button_"+name, button)
        self.control_layout.addLayout(self.control_buttons_layout)
        #PID autotune lock checkbox
        self.pid_autotune_checkbox = QCheckBox("PID autotune")
        self.pid_autotune_checkbox.setChecked(self.phase_controller.pid_autotune)
        self.pid_autotune_checkbox.toggled.connect(self.pid_autotune_checkbox_toggled)
        self.control_layout.addWidget(self.pid_autotune_checkbox)
        #Phase selection layout
        self.set_phase_layout = QHBoxLayout()
        self.control_layout.addLayout(self.set_phase_layout)
        #Label to show the current phase
        self.label_phase = QLabel()
        self.label_phase.setText("Phase: %0.2f deg" % (self.phase_controller.phase * 180 / np.pi))
        self.set_phase_layout.addWidget(self.label_phase)
        #Slider to set the phase
        self.slider_set_phase = QSlider(Qt.Horizontal)
        self.slider_set_phase.setRange(0, 180)
        self.slider_set_phase.setSingleStep(0.01)
        self.slider_set_phase.valueChanged.connect(self.slider_set_phase_value_changed_callback)
        self.set_phase_layout.addWidget(self.slider_set_phase, Qt.AlignLeft)
        self.layout.addLayout(self.control_layout, Qt.AlignLeft)
        #PID widget
        self.pid_widget = PIDControlWidget(self.phase_controller.pid_control)
        self.layout.addWidget(self.pid_widget)
        #Define a monitor scope layout and widget
        self.scope_layout = QVBoxLayout()
        self.scope_widget = self.phase_controller.redpitaya.scope._module_widget
        self.scope_layout.addWidget(self.scope_widget)
        self.layout.addLayout(self.scope_layout)
        self.setLayout(self.layout)
        self.style_sheets = PhaseControllerWidgetStyle().style_sheets
        self.set_style(theme="dark")

    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        excluded_strings = ["layout", "callback", "clicked", "toggled", "changed", "edited", "checked"]
        for widget_type in ["label", "button", "tabs", "slider", "checkbox"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if
                       widget_type in name and not any_in_string(excluded_strings, name)]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])
        #Set style to custom widgets
        self.pid_widget.style_sheets = self.style_sheets
        self.pid_widget.set_style(theme="dark")

    def control_find_transfer_function_callback(self):
        self.phase_controller.find_transfer_function()

    def control_button_scan_callback(self):
        self.phase_controller.scan()

    def control_button_lock_callback(self):
        self.phase_controller.lock()

    def control_button_unlock_callback(self):
        self.phase_controller.unlock()

    def control_button_calibrate_callback(self):
        self.phase_controller.calibrate()

    def control_button_remove_offset_pid_DC_callback(self):
        self.phase_controller.remove_offset_pid_DC()

    def control_button_set_demodulation_phase_callback(self):
        self.phase_controller.set_demodulation_phase()

    def control_button_set_iq_qfactor_callback(self):
        self.phase_controller.set_iq_qfactor()

    def control_button_flip_iq_phase_callback(self):
        self.phase_controller.flip_iq_phase()

    def slider_set_phase_value_changed_callback(self, value):
        self.phase_controller.set_phase(value)
        self.label_phase.setText("Phase: %0.2f deg"%(self.phase_controller.phase * 180 / np.pi))

    def pid_autotune_checkbox_toggled(self):
        self.phase_controller.pid_autotune = self.pid_autotune_checkbox.isChecked()












