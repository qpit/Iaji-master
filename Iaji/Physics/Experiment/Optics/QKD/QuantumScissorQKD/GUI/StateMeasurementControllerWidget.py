"""
This module defines the GUI of the StateMeasurementController module.
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
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateMeasurementController\
    import StateMeasurementController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.HomodyneDetectionControllerWidget \
    import HomodyneDetectionControllerWidget as HDWidget
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.WidgetStyles import StateMeasurementControllerStyle
from Iaji.Utilities.strutils import any_in_string

class StateMeasurementControllerWidget(QWidget):
    #-------------------------------------------
    def __init__(self, state_measurement:StateMeasurementController, name = "State Measurement Controller Widget"):
        super().__init__()
        self.setWindowTitle(name)
        self.name = name
        self.state_measurement = state_measurement
        #Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        #Title
        self.title_label = QLabel()
        self.layout.addWidget(self.title_label)
        self.title_label.setText(self.state_measurement.name)
        #State Measurement Layout
        self.state_measurement_widget = QWidget()
        ##State measurement buttons
        self.state_measurement_layout = QVBoxLayout()
        self.state_measurement_widget.setLayout(self.state_measurement_layout)
        self.state_measurement_buttons_layout = QHBoxLayout()
        self.state_measurement_layout.addLayout(self.state_measurement_buttons_layout)
        self.layout.addLayout(self.state_measurement_layout)
        button_names = ["tomography", "scanned_measurement"]
        for name in button_names:
            setattr(self, name+"_button",  QPushButton(name.replace("_", " ")))
            button = getattr(self, name+"_button")
            self.state_measurement_buttons_layout.addWidget(button)
            button.clicked.connect(getattr(self, name+"_button_clicked"))
        # Homodyne detection controller
        self.hd_controller_widget = HDWidget(self.state_measurement.hd_controller)
        self.state_measurement_layout.addWidget(self.hd_controller_widget)
        #State analysis layout
        self.state_analysis_widget = QWidget()
        self.state_analysis_layout = QVBoxLayout()
        self.state_analysis_widget.setLayout(self.state_analysis_layout)
        self.layout.addLayout(self.state_analysis_layout)
        #State measurement and analysis tabs
        self.state_tabs = QTabWidget()
        self.layout.addWidget(self.state_tabs)
        self.state_tabs.addTab(self.state_measurement_widget, "State Measurement")
        self.state_tabs.addTab(self.state_analysis_widget, "State Analysis")
        # Set style
        self.style_sheets = StateMeasurementControllerStyle().style_sheets
        self.set_style(theme="dark")
    # -------------------------------------------
    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        excluded_strings = ["layout", "callback", "clicked", "toggled", "changed", "edited", "checked"]
        for widget_type in ["label", "button", "tabs"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if
                       widget_type in name and not any_in_string(excluded_strings, name)]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])
        #Set style of custom widgets
        self.hd_controller_widget.style_sheets = self.style_sheets
        self.hd_controller_widget.set_style(theme="dark")
    #-------------------------------------------
    def tomography_button_clicked(self):
        self.state_measurement.tomography_measurement(phases=[0, 30, 60, 90, 120, 150])
    # -------------------------------------------
    def scanned_measurement_button_clicked(self): #TODO
        self.state_measurement.scanned_measurement()


