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
    QHBoxLayout,
    QBoxLayout,
    QGridLayout,
    QWidget,
)

from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.CavityLock import CavityLock
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.WidgetStyles import CavityLockWidgetStyle
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.PIDControlWidget import PIDControlWidget
import numpy as np
from Iaji.Utilities.strutils import any_in_string

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
        #Add a radio button to select the lock type
        self.radiobutton_high_finesse = QRadioButton("High-finesse cavity")
        self.radiobutton_high_finesse.toggled.connect(self.radiobutton_high_finesse_clicked)
        self.layout.addWidget(self.radiobutton_high_finesse)
        #Add buttons for control panel
        self.control_buttons_layout = QGridLayout()
        control_button_names = ["scan", "lock", "unlock", "calibrate",
                             "set_demodulation_phase", "flip_phase"]
        control_button_callbacks = dict(
            zip(control_button_names, [getattr(self, "control_button_"+name+"_callback") for name in control_button_names]))
        n_rows = 2
        for j in range(len(control_button_names)):
            name = control_button_names[j]
            button = QPushButton(name)
            button.clicked.connect(control_button_callbacks[name])
            self.control_buttons_layout.addWidget(button, int(j / n_rows), int(np.mod(j, n_rows)))
            setattr(self, "control_button_"+name, button)
        self.layout.addLayout(self.control_buttons_layout)
        # Create the PID widget
        self.PID_layout = QHBoxLayout()
        self.PID_widget = PIDControlWidget(self.cavity_lock.pid_coarse)
        self.high_finesse_PID_widget = HighFinessePIDWidget(self.cavity_lock.pid_fine)
        '''
        for stage in ["coarse", "fine"]:
            getattr(self.high_finesse_PID_widget, "PID_"+stage+"_widget").P_doublespinbox.setRange(-2e3, 2e3)
            getattr(self.high_finesse_PID_widget, "PID_" + stage + "_widget").I_doublespinbox.setRange(-1e4, 1e4)
            getattr(self.high_finesse_PID_widget, "PID_" + stage + "_widget").I_value_doublespinbox.setRange(-4, 4)
        for parameter in ["P", "I", "I_value"]:
            self.high_finesse_PID_widget.PID_coarse_widget.set_parameter_changed_callback(parameter, \
                                getattr(self, "high_finesse_PID_widget_coarse_"+parameter+"_doublespinbox_value_changed"))
            self.high_finesse_PID_widget.PID_fine_widget.set_parameter_changed_callback(parameter, \
                        getattr(self, "high_finesse_PID_widget_fine_" + parameter + "_doublespinbox_value_changed"))
        for stage in ["coarse", "fine"]:
            widget = getattr(self.high_finesse_PID_widget, "PID_"+stage+"_widget")
            for parameter in ["P", "I", "I_value"]:
                if parameter == "I_value":
                    parameter_lower_case = "ival"
                else:
                    parameter_lower_case = parameter.lower()
                getattr(widget, parameter+"_doublespinbox").setValue(getattr(getattr(self.cavity_lock.high_finesse_lock, "pid_"+stage), parameter_lower_case))
                getattr(widget, parameter + "_doublespinbox").setSingleStep(2e-4)
                getattr(widget, parameter + "_doublespinbox").setDecimals(4)
        '''
        self.PID_layout.addWidget(self.PID_widget)
        self.PID_layout.addWidget(self.high_finesse_PID_widget)
        if self.cavity_lock.lock_type == "high finesse":
            self.PID_widget.hide()
        else:
            self.high_finesse_PID_widget.hide()
        self.layout.addLayout(self.PID_layout)
        # Define a monitor scope layout and widget
        self.scope_layout = QVBoxLayout()
        self.scope_widget = self.cavity_lock.pyrpl_obj.rp.scope._module_widget
        self.scope_layout.addWidget(self.scope_widget)
        self.layout.addLayout(self.scope_layout)

        self.style_sheets = CavityLockWidgetStyle().style_sheets
        self.set_style(theme="dark")

        self.setLayout(self.layout)

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

    def radiobutton_high_finesse_clicked(self):
        if self.radiobutton_high_finesse.isChecked():
            lock_type = "high finesse"
            self.high_finesse_PID_widget.show()
            self.PID_widget.hide()
        else:
            lock_type = "regular"
            self.high_finesse_PID_widget.hide()
            self.PID_widget.show()

        self.cavity_lock.set_lock_type(lock_type)

    def high_finesse_PID_widget_coarse_P_doublespinbox_value_changed(self, value):
        self.cavity_lock.high_finesse_lock.pid_coarse.p = value
        self.high_finesse_PID_widget.PID_coarse_widget.P_label.setText("P: %0.4f"%value)

    def high_finesse_PID_widget_coarse_I_doublespinbox_value_changed(self, value):
        self.cavity_lock.high_finesse_lock.pid_coarse.i = value
        self.high_finesse_PID_widget.PID_coarse_widget.I_label.setText("I: %0.4f"%value)

    def high_finesse_PID_widget_coarse_I_value_doublespinbox_value_changed(self, value):
        self.cavity_lock.high_finesse_lock.pid_coarse.ival = value
        self.high_finesse_PID_widget.PID_coarse_widget.I_value_label.setText("I value: %0.4f"%value)

    def high_finesse_PID_widget_fine_P_doublespinbox_value_changed(self, value):
        self.cavity_lock.high_finesse_lock.pid_fine.p = value
        self.high_finesse_PID_widget.PID_fine_widget.P_label.setText("P: %0.4f"%value)

    def high_finesse_PID_widget_fine_I_doublespinbox_value_changed(self, value):
        self.cavity_lock.high_finesse_lock.pid_fine.i = value
        self.high_finesse_PID_widget.PID_fine_widget.I_label.setText("I: %0.4f"%value)

    def high_finesse_PID_widget_fine_I_value_doublespinbox_value_changed(self, value):
        self.cavity_lock.high_finesse_lock.pid_fine.ival = value
        self.high_finesse_PID_widget.PID_fine_widget.I_value_label.setText("I value: %0.4f"%value)

    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        for widget_type in ["label", "doublespinbox", "button", "radiobutton"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if widget_type in name and "layout" not in name and "callback" not in name]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])

class HighFinessePIDWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.PID_coarse_widget = PIDControlWidget()
        self.PID_coarse_label = QLabel()
        self.PID_coarse_label.setText("Coarse Stage")

        self.PID_fine_widget = PIDControlWidget()
        self.PID_fine_label = QLabel()
        self.PID_fine_label.setText("Fine Stage")

        self.layout.addWidget(self.PID_coarse_label)
        self.layout.addWidget(self.PID_coarse_widget)
        self.layout.addWidget(self.PID_fine_label)
        self.layout.addWidget(self.PID_fine_widget)


        self.setLayout(self.layout)
        self.style_sheets = CavityLockWidgetStyle().style_sheets
        self.set_style(theme="dark")

    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        excluded_strings = ["layout", "callback", "clicked", "toggled", "changed", "edited", "checked"]
        for widget_type in ["label", "button", "doublespinbox", "radiobutton", "tabs", "slider", "checkbox"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if
                       widget_type in name and not any_in_string(excluded_strings, name)]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])



