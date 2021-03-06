"""
This module defines the GUI of the StateGenerator module.
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
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateGenerator import StateGenerator
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.StateMeasurementControllerWidget import StateMeasurementControllerWidget
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.WidgetStyles import StateGeneratorWidgetStyle
from Iaji.Utilities.GUI import PyplotWidget
from Iaji.Utilities.strutils import any_in_string
from matplotlib import pyplot
import numpy
#In []
class StateGeneratorWidget(QWidget):
    """
    """
    #-------------------------------------------
    def __init__(self, state_generator: StateGenerator, name = "State Generator Widget"):
        '''
        :param state_generator: Iaji StateGenerator
        :param name: str
        '''
        super().__init__()
        self.setWindowTitle(name)
        self.state_generator = state_generator
        self.name = name
        #Main Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        #Title
        self.title_label = QLabel()
        self.title_label.setText(name)
        self.layout.addWidget(self.title_label)
        #Tabs
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        #State Measurement Widget
        self.state_measurement_widget = StateMeasurementControllerWidget(state_generator.state_measurement, \
                                                                         name=state_generator.state_measurement.name)
        self.tabs.addTab(self.state_measurement_widget, self.state_measurement_widget.name)
        self.make_calibration_widget()
        # Set style
        self.style_sheets = StateGeneratorWidgetStyle().style_sheets
        self.set_style(theme="dark")
    # -------------------------------------------
    def make_calibration_widget(self):
        #Make widget and layout
        self.calibration_widget = QWidget()
        self.calibration_layout = QGridLayout()
        self.calibration_widget.setLayout(self.calibration_layout)
        #Add tab
        self.tabs.addTab(self.calibration_widget, "Calibration")
        #Calibration button
        self.devices = ["aoms", "phase_eom", "amplitude_eom"]
        self.state_generator.voltage_ranges = [(0, 100e-3), (-1, 1), (-1, 1)]
        for col in range(len(self.devices)):
            device = self.devices[col]
            voltage_range = self.state_generator.voltage_ranges[col]
            #Title
            setattr(self, "%s_label"%device, QLabel())
            title_label = getattr(self, "%s_label"%device)
            title_label.setText(device.replace("_", " "))
            self.calibration_layout.addWidget(title_label, 1, col+1)
            #Calibration button
            setattr(self, "%s_button"%device, QPushButton("Calibrate"))
            button = getattr(self, "%s_button"%device)
            button.clicked.connect(getattr(self, "%s_button_clicked"%device))
            self.calibration_layout.addWidget(button, 2, col + 1)
            #Voltage ranges
            setattr(self, "%s_voltage_range_layout"%device, QHBoxLayout())
            setattr(self, "%s_min_voltage_linedit"%device, QLineEdit(str(voltage_range[0])))
            setattr(self, "%s_max_voltage_linedit" % device, QLineEdit(str(voltage_range[1])))
            getattr(self, "%s_voltage_range_layout"%device).addWidget(getattr(self, "%s_min_voltage_linedit"%device))
            getattr(self, "%s_voltage_range_layout" % device).addWidget(getattr(self, "%s_max_voltage_linedit" % device))
            self.calibration_layout.addLayout(getattr(self, "%s_voltage_range_layout"%device), 3, col + 1)
            #Plot widget
            #setattr(self, "%s_plot"%device, pyqtgraph.PlotWidget())
            figure = pyplot.figure()
            axis = figure.add_subplot(111)
            metric = "$|\\alpha|$" * (device != "phase_aom") + "$Arg(\\alpha)$" * (device == "phase_eom")
            axis.set_xlabel("input voltage (V)", fontdict={"size":12, "family":"Times New Roman"})
            axis.set_ylabel(metric, fontdict={"size": 12, "family": "Times New Roman"})
            axis.grid(True)
            axis.plot(0, 0, color="green")
            plot_name = "%s_plot"%device
            setattr(self, plot_name, PyplotWidget(figure=figure, name=plot_name))
            self.calibration_layout.addWidget(getattr(self, plot_name), 4, col+1)
    # -------------------------------------------
    def make_generation_widget(self):
        '''

        :return:
        '''

    # -------------------------------------------
    def aoms_button_clicked(self):
        '''
        :return:
        '''
        #Dummy test behavior
        axis = self.aoms_plot.figure.axes[0]
        #Calibrate the aoms
        min_voltage = float(self.aoms_min_voltage_linedit.text())
        max_voltage = float(self.aoms_max_voltage_linedit.text())
        x, y = self.state_generator.calibrate_aoms(voltage_range=[min_voltage, max_voltage])
        axis.plot(x, y, linestyle="None", marker="o", markersize=10)
        self.aoms_plot.update()
    # -------------------------------------------
    def phase_eom_button_clicked(self):
        '''

        :return:
        '''
    # -------------------------------------------
    def amplitude_eom_button_clicked(self):
        '''

        :return:
        '''
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
        self.state_measurement_widget.style_sheets = self.style_sheets
        self.state_measurement_widget.set_style(theme="dark")