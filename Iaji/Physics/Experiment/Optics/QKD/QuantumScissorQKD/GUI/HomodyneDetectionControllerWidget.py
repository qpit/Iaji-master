"""
This module defines the GUI of the HomodyneDetectionController module.
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
    QTabWidget,
    QTimeEdit,
    QVBoxLayout,
    QBoxLayout,
    QGridLayout,
    QWidget,
)

from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.HomodyneDetectionController import HomodyneDetectionController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.PhaseControllerWidget import PhaseControllerWidget
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.AcquisitionSystemWidget import AcquisitionSystemWidget
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.WidgetStyles import HomodyneDetectionControllerStyle
import numpy
from Iaji.Utilities.strutils import any_in_string

class HomodyneDetectionControllerWidget(QWidget):
    """
    This class describes the widget of a homodyne Detection controller.
    Documentation to be completed.
    """
    def __init__(self, homodyne_detection_controller, name="Homodyne Detection Controller Widget"):
        super().__init__()
        self.hd_controller = homodyne_detection_controller
        self.setWindowTitle(name)
        self.layout = QVBoxLayout()
        # Define widget title label
        self.name_label = QLabel()
        self.name_label.setText(self.hd_controller.name)
        self.layout.addWidget(self.name_label, Qt.AlignCenter)
        #Main measurement layout
        self.measurement_layout = QHBoxLayout()
        self.layout.addLayout(self.measurement_layout)
        ##Measure quadrature
        self.measure_quadrature_button = QPushButton("measure quadrature")
        self.measurement_layout.addWidget(self.measure_quadrature_button)
        self.measure_quadrature_button.clicked.connect(self.measure_quadrature_button_clicked)
        ##Measure vacuum
        self.measure_vacuum_button = QPushButton("measure vacuum quadrature")
        self.measurement_layout.addWidget(self.measure_vacuum_button)
        self.measure_vacuum_button.clicked.connect(self.measure_vacuum_button_clicked)
        ##Measure electronic noise
        self.measure_electronic_noise_button = QPushButton("measure electronic noise")
        self.measurement_layout.addWidget(self.measure_electronic_noise_button)
        self.measure_electronic_noise_button.clicked.connect(self.measure_electronic_noise_button_clicked)
        #Add phase controller widget
        self.phase_controller_layout = QVBoxLayout()
        self.phase_controller_widget = PhaseControllerWidget(self.hd_controller.phase_controller)
        #self.phase_controller_layout.addWidget(self.phase_controller_widget)
        self.layout.addLayout(self.phase_controller_layout)
        #Add acquisition system widget
        self.acquisition_system_layout = QVBoxLayout()
        self.layout.addLayout(self.acquisition_system_layout)
        self.acquisition_system_widget = AcquisitionSystemWidget(self.hd_controller.acquisition_system)
        #self.acquisition_system_layout.addWidget(self.acquisition_system_widget)
        #Organize widgets in tabs
        self.tabs_layout = QHBoxLayout()
        self.layout.addLayout(self.tabs_layout)
        self.tabs = QTabWidget()
        self.tabs_layout.addWidget(self.tabs)
        self.tabs.addTab(self.phase_controller_widget, "Phase Controller")
        self.tabs.addTab(self.acquisition_system_widget, "Acquisition System")
        #Set style
        self.style_sheets = HomodyneDetectionControllerStyle().style_sheets
        self.set_style(theme="dark")

        self.phase_controller_widget.style_sheets = self.style_sheets
        self.phase_controller_widget.set_style(theme="dark")

        self.acquisition_system_widget.style_sheets = self.style_sheets
        self.acquisition_system_widget.set_style(theme="dark")
        self.setLayout(self.layout)


    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        excluded_strings = ["layout", "callback", "clicked", "toggled", "changed", "edited", "checked"]
        for widget_type in ["label", "button", "tabs"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if widget_type in name and not any_in_string(excluded_strings, name)]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])

    def measure_quadrature_button_clicked(self):
        self.hd_controller.measure_quadrature(self.hd_controller.phase_controller.phase*180/numpy.pi)

    def measure_vacuum_button_clicked(self):
        self.hd_controller.measure_vacuum()

    def measure_electronic_noise_button_clicked(self):
        self.hd_controller.measure_electronic_noise()


