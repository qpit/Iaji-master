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
        self.PID_layout = QVBoxLayout()
        self.make_PID_widget()
        self.layout.addLayout(self.PID_layout)
        # Define a monitor scope layout and widget
        self.scope_layout = QVBoxLayout()
        self.scope_widget = self.cavity_lock.pyrpl_obj.rp.scope._module_widget
        self.scope_layout.addWidget(self.scope_widget)
        self.layout.addLayout(self.scope_layout)

        self.style_sheets = CavityLockWidgetStyle().style_sheets
        self.set_style(theme="dark")

        self.setLayout(self.layout)

    def make_PID_widget(self):
        self.PID_layout = QVBoxLayout()
        if self.cavity_lock.lock_type == "high finesse":
            self.PID_widget = HighFinessePIDWidget()
        else:
            self.PID_widget = PIDControlWidget()
        self.PID_layout.addWidget(self.PID_widget)
        self.update()


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
        else:
            lock_type = "regular"
        self.cavity_lock.set_lock_type(lock_type)
        self.make_PID_widget()

    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        for widget_type in ["label", "slider", "button", "radiobutton"]:
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



