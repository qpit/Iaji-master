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
import datetime
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
        #State calibration tab widget
        self.make_calibration_and_tomography_widget()
        self.tabs.addTab(self.calibration_and_tomography_widget, "State Calibration and Tomography")
        # Set style
        self.style_sheets = StateGeneratorWidgetStyle().style_sheets
        self.set_style(theme="dark")

    # -------------------------------------------
    def make_calibration_and_tomography_widget(self):
        #Compound tab widget
        self.calibration_and_tomography_widget = QWidget()
        self.calibration_and_tomography_layout = QVBoxLayout()
        self.calibration_and_tomography_widget.setLayout(self.calibration_and_tomography_layout)
        #Stop calibration button
        self.stop_calibration_button = QPushButton("stop (not working)")
        self.stop_calibration_button.clicked.connect(self.stop_calibration_button_clicked)
        self.calibration_and_tomography_layout.addWidget(self.stop_calibration_button)
        #Voltages widget
        self.make_voltages_widget()
        self.calibration_and_tomography_layout.addWidget(self.voltages_widget)
        # Make tomography and calibration widgets
        self.calibration_tabs = QTabWidget()
        self.calibration_and_tomography_layout.addWidget(self.calibration_tabs)
        ##Calibration
        self.make_calibration_widget()
        self.calibration_tabs.addTab(self.calibration_widget, "Calibrations")
        ##Tomography
        self.make_tomography_widget()
        self.calibration_tabs.addTab(self.tomography_widget, "Tomography")
    # -------------------------------------------
    def make_voltages_widget(self):
        # Fixed AOM and EOM voltages
        self.voltages_widget = QWidget()
        self.voltages_layout = QHBoxLayout()
        self.voltages_widget.setLayout(self.voltages_layout)
        self.devices = ["aoms", "amplitude_eom", "phase_eom"]
        default_voltages = dict(zip(self.devices, [0.1, 5, 0]))
        for device in self.devices:
            # Label
            setattr(self, "%s_voltage_label" % device, QLabel())
            label = getattr(self, "%s_voltage_label" % device)
            label.setText("%s voltage [V]" % device.replace("_", " "))
            self.voltages_layout.addWidget(label)
            # Linedit
            setattr(self, "%s_voltage_linedit" % device, QLineEdit(str(default_voltages[device])))
            linedit = getattr(self, "%s_voltage_linedit" % device)
            linedit.textEdited.connect(getattr(self, '%s_voltage_linedit_changed' % device))
            getattr(self, '%s_voltage_linedit_changed' % device)(str(default_voltages[device]))
            self.voltages_layout.addWidget(linedit)
    # -------------------------------------------
    def make_tomography_widget(self):
        #Define the widget
        self.tomography_widget = QWidget()
        # Layout for homodyne tomography
        self.tomography_layout = QVBoxLayout()
        self.tomography_widget.setLayout(self.tomography_layout)
        # Tomography button
        self.tomography_button = QPushButton("calibration tomography")
        self.tomography_button.clicked.connect(self.tomography_button_clicked)
        self.tomography_layout.addWidget(self.tomography_button)
        # Hilbert space dimension
        self.tomography_dimension_layout = QHBoxLayout()
        self.tomography_layout.addLayout(self.tomography_dimension_layout)
        ## Dimension slider
        self.tomography_dimension_slider = QSlider(Qt.Horizontal)
        self.tomography_dimension_slider.setRange(1, 50)
        self.tomography_dimension_slider.setSingleStep(1)
        if self.state_generator.state_measurement.quantum_state is not None:
            self.tomography_dimension_slider.setValue(self.state_generator.state_measurement.quantum_state.hilbert_space.dimension)
        self.tomography_dimension_slider.valueChanged.connect(self.tomography_dimension_slider_changed)
        self.tomography_dimension_layout.addWidget(self.tomography_dimension_slider)
        ##Dimension label
        self.tomography_dimension_label = QLabel()
        label = "dimension: %d" % self.tomography_dimension_slider.value()
        self.tomography_dimension_label.setText(label)
        self.tomography_dimension_layout.addWidget(self.tomography_dimension_label)
        # Plot
        figure = pyplot.figure(num=self.state_generator.state_measurement.quantum_state.figure_name)
        self.tomography_plot_widget = PyplotWidget(figure=figure)
        self.tomography_plot_widget.update()
        self.tomography_layout.addWidget(self.tomography_plot_widget)


    # -------------------------------------------
    def make_calibration_widget(self):
        #Make widget and layout
        self.calibration_widget = QWidget()
        #Layout with different calibrations
        self.calibration_layout = QGridLayout()
        self.calibration_widget.setLayout(self.calibration_layout)
        #Add tab
        #Calibration button
        self.devices = ["aoms", "amplitude_eom", "phase_eom"]
        self.state_generator.voltage_ranges = [(1e-2, 100e-3), (5, 0), (-1, 1)]
        for col in range(len(self.devices)):
            device = self.devices[col]
            voltage_range = self.state_generator.voltage_ranges[col]
            #Title
            setattr(self, "%s_label"%device, QLabel())
            title_label = getattr(self, "%s_label"%device)
            title_label.setText(device.replace("_", " "))
            self.calibration_layout.addWidget(title_label, 1, col+1)
            # Voltage ranges
            setattr(self, "%s_voltage_range_layout" % device, QHBoxLayout()) #horizontal layout
            setattr(self, "%s_voltage_range_label"%device, QLabel()) #title
            getattr(self, "%s_voltage_range_label"%device).setText("voltage range [V]") #title text
            setattr(self, "%s_min_voltage_linedit" % device, QLineEdit(str(voltage_range[0]))) #minimum voltage
            setattr(self, "%s_max_voltage_linedit" % device, QLineEdit(str(voltage_range[1]))) #maximum voltage
            ## Add the widgets to layout
            getattr(self, "%s_voltage_range_layout" % device).addWidget(
                getattr(self, "%s_voltage_range_label" % device))
            getattr(self, "%s_voltage_range_layout" % device).addWidget(
                getattr(self, "%s_min_voltage_linedit" % device))
            getattr(self, "%s_voltage_range_layout" % device).addWidget(
                getattr(self, "%s_max_voltage_linedit" % device))
            self.calibration_layout.addLayout(getattr(self, "%s_voltage_range_layout" % device), 2, col + 1)
            # Number of points per calibration
            setattr(self, "%s_n_points_layout" % device, QHBoxLayout())  # horizontal layout
            setattr(self, "%s_n_points_label" % device, QLabel())  # title
            getattr(self, "%s_n_points_label" % device).setText("number of points")  # title text
            setattr(self, "%s_n_points_linedit" % device, QLineEdit(str(6)))  # number of points
            self.calibration_layout.addLayout(getattr(self, "%s_n_points_layout" % device), 3, col + 1)
            ## Add the widgets to layout
            getattr(self, "%s_n_points_layout" % device).addWidget(
                getattr(self, "%s_n_points_label" % device))
            getattr(self, "%s_n_points_layout" % device).addWidget(
                getattr(self, "%s_n_points_linedit" % device))
            #Calibration button
            setattr(self, "%s_button"%device, QPushButton("Calibrate"))
            button = getattr(self, "%s_button"%device)
            button.clicked.connect(getattr(self, "%s_button_clicked"%device))
            self.calibration_layout.addWidget(button, 4, col + 1)
            #Plot widget
            #setattr(self, "%s_plot"%device, pyqtgraph.PlotWidget())
            figure = pyplot.figure()
            axis = figure.add_subplot(111)
            metric = "$|\\alpha|$" * (device != "phase_eom") + "$Arg(\\alpha)$ $(^\\circ)$" * (device == "phase_eom")
            axis.set_xlabel("input voltage (V)", fontdict={"size":12, "family":"Times New Roman"})
            axis.set_ylabel(metric, fontdict={"size": 12, "family": "Times New Roman"})
            axis.grid(True)
            axis.plot(0, 0, color="green")
            plot_name = "%s_plot"%device
            setattr(self, plot_name, PyplotWidget(figure=figure, name=plot_name, shape=(50, 50)))
            self.calibration_layout.addWidget(getattr(self, plot_name), 5, col+1)
    # -------------------------------------------
    def make_generation_widget(self):
        '''

        :return:
        '''

    # -------------------------------------------
    def aoms_voltage_linedit_changed(self, text):
        self.state_generator.aoms_voltage = float(text)
    # -------------------------------------------
    def amplitude_eom_voltage_linedit_changed(self, text):
        self.state_generator.amplitude_eom_voltage = float(text)
    # -------------------------------------------
    def phase_eom_voltage_linedit_changed(self, text):
        self.state_generator.phase_eom_voltage = float(text)
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
        n_points = int(self.aoms_n_points_linedit.text())
        x, y = self.state_generator.calibrate_aoms(voltage_range=[min_voltage, max_voltage], n_points=n_points)
        axis.plot(x, numpy.abs(y), linestyle="None", marker="o", markersize=10, \
                label=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        axis.legend(prop={"family":"Times New Roman"})
        self.aoms_plot.update()
    # -------------------------------------------
    def phase_eom_button_clicked(self):
        '''

        :return:
        '''
        axis = self.phase_eom_plot.figure.axes[0]
        # Calibrate the aoms
        min_voltage = float(self.phase_eom_min_voltage_linedit.text())
        max_voltage = float(self.phase_eom_max_voltage_linedit.text())
        n_points = int(self.phase_eom_n_points_linedit.text())
        x, y = self.state_generator.calibrate_phase_eom(voltage_range=[min_voltage, max_voltage], n_points=n_points)
        axis.plot(x, numpy.angle(y)*180/numpy.pi, linestyle="None", marker="o", markersize=10, \
                  label=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        axis.legend(prop={"family":"Times New Roman"})
        self.phase_eom_plot.update()
        numpy.savez(file=self.state_generator.state_measurement.hd_controller.acquisition_system.host_save_directory, \
                    voltages=x, displacements=y)
    # -------------------------------------------
    def amplitude_eom_button_clicked(self):
        '''

        :return:
        '''
        axis = self.amplitude_eom_plot.figure.axes[0]
        # Calibrate the aoms
        min_voltage = float(self.amplitude_eom_min_voltage_linedit.text())
        max_voltage = float(self.amplitude_eom_max_voltage_linedit.text())
        n_points = int(self.amplitude_eom_n_points_linedit.text())
        x, y = self.state_generator.calibrate_amplitude_eom(voltage_range=[min_voltage, max_voltage], n_points=n_points)
        axis.plot(x, numpy.abs(y), linestyle="None", marker="o", markersize=10, \
                  label=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        axis.legend(prop={"family":"Times New Roman"})
        self.amplitude_eom_plot.update()
    # -------------------------------------------
    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        excluded_strings = ["layout", "callback", "clicked", "toggled", "changed", "edited", "checked"]
        for widget_type in ["label", "button", "tabs", "slider", "linedit"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if
                       widget_type in name and not any_in_string(excluded_strings, name)]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])
        #Set style of custom widgets
        self.state_measurement_widget.style_sheets = self.style_sheets
        self.state_measurement_widget.set_style(theme="dark")
    # -------------------------------------------
    def tomography_button_clicked(self):
        '''
        - Make tomography measurement with calibration settings
        - Reconstruct and plot quantum state

        :return:
        '''
        self.state_generator.tomography_mesaurement_for_calibration(\
                                                                    aom_voltage = self.state_generator.aoms_voltage, \
                                                                    amplitude_eom_voltage = self.state_generator.amplitude_eom_voltage, \
                                                                    phase_eom_voltage = self.state_generator.phase_eom_voltage)
    # -------------------------------------------
    def tomography_dimension_slider_changed(self, value):
        label = "dimension: %d" % value
        self.tomography_dimension_label.setText(label)
        self.state_generator.state_measurement.quantum_state = self.state_generator.state_measurement.quantum_state.Resize(value)

    # -------------------------------------------
    def stop_calibration_button_clicked(self):
        self.state_generator.calibration_stopped = True