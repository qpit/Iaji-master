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
        #Demodulation frequency choice
        self.demodulation_layout = QHBoxLayout()
        self.label_demod = QLabel()
        self.label_demod.setText("Demodulation phase:")
        self.demodulation_layout.addWidget(self.label_demod)
        self.calibration_frequency_checkbox = QCheckBox("%.2f kHz" %(self.phase_controller.calibration_frequency/1e3))
        self.calibration_frequency_checkbox.setChecked(self.phase_controller.calibration_frequency_on)
        self.calibration_frequency_checkbox.toggled.connect(self.calibration_frequency_checkbox_toggled)
        self.demodulation_layout.addWidget(self.calibration_frequency_checkbox)
        self.measurement_frequency_checkbox = QCheckBox("%.2f kHz" %(self.phase_controller.measurement_frequency/1e3))
        self.measurement_frequency_checkbox.setChecked(self.phase_controller.measurement_frequency_on)
        self.measurement_frequency_checkbox.toggled.connect(self.measurement_frequency_checkbox_toggled)
        self.demodulation_layout.addWidget(self.measurement_frequency_checkbox)
        self.layout.addLayout(self.demodulation_layout)
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
        #Checkbox layout
        self.checkbox_layout = QVBoxLayout()
        #PID autotune lock checkbox
        self.pid_autotune_checkbox = QCheckBox("PID autotune")
        self.pid_autotune_checkbox.setChecked(self.phase_controller.pid_autotune)
        self.pid_autotune_checkbox.toggled.connect(self.pid_autotune_checkbox_toggled)
<<<<<<< HEAD
        self.checkbox_layout.addWidget(self.pid_autotune_checkbox)
        #Enable output modulation checkbox
=======
        self.control_layout.addWidget(self.pid_autotune_checkbox)
        # Checkbox layout
        self.checkbox_layout = QVBoxLayout()
        # Enable output modulation checkbox
>>>>>>> origin/qpitlab_folder
        self.enable_output_modulation_checkbox = QCheckBox("Enable output modulation")
        self.enable_output_modulation_checkbox.setChecked(self.phase_controller.modulation_output_enabled)
        self.enable_output_modulation_checkbox.toggled.connect(self.enable_output_modulation_checkbox_toggled)
        self.checkbox_layout.addWidget(self.enable_output_modulation_checkbox)
        self.control_layout.addLayout(self.checkbox_layout)
        #Phase selection layout
        self.set_phase_layout = QHBoxLayout()
        self.control_layout.addLayout(self.set_phase_layout)
        #Label to show the current phase
        self.label_phase = QLabel()
        self.label_phase.setText("Phase:")
        self.set_phase_layout.addWidget(self.label_phase)
        '''
        #Slider to set the phase
        self.slider_set_phase = QSlider(Qt.Horizontal)
        self.slider_set_phase.setRange(0, 180)
        self.slider_set_phase.setSingleStep(0.01)
        self.slider_set_phase.valueChanged.connect(self.slider_set_phase_value_changed_callback)
        self.set_phase_layout.addWidget(self.slider_set_phase, Qt.AlignLeft)
        self.layout.addLayout(self.control_layout, Qt.AlignLeft)
        '''
        #Phase selection linedit
        self.set_phase_linedit = QLineEdit(str(0))
        self.set_phase_linedit.textEdited.connect(self.set_phase_linedit_changed)
        self.set_phase_layout.addWidget(self.set_phase_linedit)
        self.control_layout.addLayout(self.set_phase_layout)
        self.layout.addLayout(self.control_layout)
        #PID widget
        self.pid_widget = PIDControlWidget(self.phase_controller.pid_control)
        self.layout.addWidget(self.pid_widget)
        #Define a monitor scope layout and widget
        self.scope_layout = QVBoxLayout()
        self.scope = self.phase_controller.redpitaya.scope
        try:
            self.scope_widget = self.scope._module_widget
        except:
            raise Exception("Problem with scope widget, refer to 'Red Pitaya troubleshooting' in QPIT Knowledge Base. \nPitaya with the problem:", self.phase_controller.name)
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
        self.pid_widget.p_doublespinbox.setValue(self.phase_controller.pid_control.p)
        self.pid_widget.i_doublespinbox.setValue(self.phase_controller.pid_control.i)

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

    def set_phase_linedit_changed(self, value):
        self.phase_controller.set_phase(value)
        self.label_phase.setText("Phase: %.2f"%(self.phase_controller.phase*180/np.pi))

    def pid_autotune_checkbox_toggled(self):
        self.phase_controller.pid_autotune = self.pid_autotune_checkbox.isChecked()

    def enable_output_modulation_checkbox_toggled(self):
        is_on = self.enable_output_modulation_checkbox.isChecked()
        if is_on:
            self.phase_controller.modulation_output_enabled = True
            self.phase_controller.setup_iq()
        else:
            self.phase_controller.modulation_output_enabled = False
            self.phase_controller.setup_iq()
        print("Output modulation is " + "on"*(self.phase_controller.modulation_output_enabled) + "off"*(not self.phase_controller.modulation_output_enabled))
<<<<<<< HEAD
=======

    def calibration_frequency_checkbox_toggled(self):
        self.measurement_frequency_checkbox.setChecked(False)
        self.phase_controller.calibration_frequency_on = True
        self.phase_controller.measurement_frequency_on = False
        self.phase_controller.modulation_frequency = self.phase_controller.calibration_frequency
        self.phase_controller.setup_iq()

    def measurement_frequency_checkbox_toggled(self):
        self.calibration_frequency_checkbox.setChecked(False)
        self.phase_controller.calibration_frequency_on = False
        self.phase_controller.measurement_frequency_on = True
        self.phase_controller.modulation_frequency = self.phase_controller.measurement_frequency
        self.phase_controller.setup_iq()
>>>>>>> origin/qpitlab_folder










