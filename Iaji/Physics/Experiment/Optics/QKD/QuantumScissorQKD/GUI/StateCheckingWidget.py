"""
This module defines the GUI of the StateGenerator module.
"""
#%%
import matplotlib.pyplot as plt
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
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateChecking import StateChecking
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.ABStateMeasurementControllerWidget import AliceStateMeasurementControllerWidget
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.ABStateMeasurementControllerWidget import BobStateMeasurementControllerWidget
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.HomodyneDetectionControllerWidget import HomodyneDetectionControllerWidget as HDWidget
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.PhaseControllerWidget import PhaseControllerWidget
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.WidgetStyles import StateGeneratorWidgetStyle
from Iaji.Utilities.GUI import PyplotWidget
from Iaji.Utilities.strutils import any_in_string
from matplotlib import pyplot
import numpy
import datetime
import time
import os
# In[]
fonts = {"title": {"fontsize": 26, "family": "Times New Roman"}, \
                 "axis": {"fontsize": 22, "family": "Times New Roman"}, \
                 "legend": {"size": 24, "family": "Times New Roman"}}
# In[]
class StateCheckingWidget(QWidget):
    """
    """
    #-------------------------------------------
    def __init__(self, state_generator: StateGenerator, state_checking: StateChecking, relay_lock,name = "State Checking Widget"):
        '''
        :param state_generator: Iaji StateGenerator
        :param name: str
        '''
        super().__init__()
        self.setWindowTitle(name)
        self.state_generator = state_generator
        self.state_checking = state_checking
        self.Blue_Velvet_state_measurement = self.state_checking.state_measurement
        self.name = name
        self.relay_lock = relay_lock
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
        self.state_measurement_widget = AliceStateMeasurementControllerWidget(state_generator.state_measurement, self.Blue_Velvet_state_measurement, state_generator)
        self.tabs.addTab(self.state_measurement_widget, self.state_measurement_widget.name)
        #State calibration tab widget
        self.make_calibration_and_tomography_widget()
        self.tabs.addTab(self.calibration_and_tomography_widget, "State Calibration and Tomography")
        # State generation tab widget
        self.make_state_generation_widget()
        self.tabs.addTab(self.state_generation_widget, "State Generation")
        # Set style
        self.style_sheets = StateGeneratorWidgetStyle().style_sheets
        self.set_style(theme="dark")
        self.state_generator.amplitude = 0
        self.state_generator.angle = 0
    # -------------------------------------------
    def make_calibration_and_tomography_widget(self):
        #Compound tab widget
        self.calibration_and_tomography_widget = QWidget()
        self.calibration_and_tomography_layout = QVBoxLayout()
        self.calibration_and_tomography_widget.setLayout(self.calibration_and_tomography_layout)
        self.buttons_widget = QWidget()
        self.buttons_layout = QHBoxLayout()
        self.calibration_and_tomography_layout.addWidget(self.buttons_widget)
        self.buttons_widget.setLayout(self.buttons_layout)
        #Stop calibration button
        self.stop_calibration_button = QPushButton("stop (not working)")
        self.stop_calibration_button.clicked.connect(self.stop_calibration_button_clicked)
        self.buttons_layout.addWidget(self.stop_calibration_button)
        #Send time-multiplexing signal checkbox
        self.time_multiplexing_signals_layout = QHBoxLayout()
        self.time_multiplexing_signals_widget = QWidget()
        self.time_multiplexing_signals_widget.setLayout(self.time_multiplexing_signals_layout)
        self.time_multiplexing_signals_checkbox = QCheckBox("generate/terminate \n time multiplexing signals")
        self.time_multiplexing_signals_checkbox.setChecked(False)
        self.time_multiplexing_signals_checkbox.toggled.connect(self.time_multiplexing_signals_checkbox_toggled)
        self.time_multiplexing_signals_layout.addWidget(self.time_multiplexing_signals_checkbox)

        self.two_time_multiplexing_signals_checkbox = QCheckBox("generate/terminate two-level \n time multiplexing signals")
        self.two_time_multiplexing_signals_checkbox.setChecked(False)
        self.two_time_multiplexing_signals_checkbox.toggled.connect(self.two_time_multiplexing_signals_checkbox_toggled)
        self.time_multiplexing_signals_layout.addWidget(self.two_time_multiplexing_signals_checkbox)
        self.calibration_and_tomography_layout.addWidget(self.time_multiplexing_signals_widget)
        #Find AOMS high button
        self.find_aoms_high_button = QPushButton("Find AOMS high")
        self.find_aoms_high_button.clicked.connect(self.find_aoms_high_button_clicked)
        self.buttons_layout.addWidget(self.find_aoms_high_button)
        #Close pop-up graphs button
        self.close_pop_up_graphs_button = QPushButton("Close pop-up graphs")
        self.close_pop_up_graphs_button.clicked.connect(self.close_pop_up_graphs_button_clicked)
        self.buttons_layout.addWidget(self.close_pop_up_graphs_button)
        #Voltages widget
        self.make_voltages_widget()
        self.calibration_and_tomography_layout.addWidget(self.voltages_widget)
        # Make tomography and calibration widgets
        self.calibration_tabs = QTabWidget()
        self.calibration_and_tomography_layout.addWidget(self.calibration_tabs)
        ##Calibration
        self.make_calibration_widget()
        self.calibration_tabs.addTab(self.calibration_widget, "Calibrations")
        ##Relay phase controller
        self.make_phase_calibration_widget()
        self.calibration_tabs.addTab(self.phase_calibration_widget,"Relay Phase Controller")
        ##Tomography
        self.make_tomography_widget()
        self.calibration_tabs.addTab(self.tomography_widget, "Tomography")
    # -------------------------------------------
    def make_state_generation_widget(self):
        # Compound tab widget
        self.state_generation_widget = QWidget()
        self.state_generation_layout = QVBoxLayout()
        self.state_generation_widget.setLayout(self.state_generation_layout)
        # Create state button
        self.state_generation_button = QPushButton('Create state')
        self.state_generation_layout.addWidget(self.state_generation_button)
        self.state_generation_button.clicked.connect(self.state_generation_button_clicked)
        # Set input state parameters
        ## Displacement
        self.state_parameters_layout = QHBoxLayout()
        self.state_generation_layout.addLayout(self.state_parameters_layout)
        self.displacement_label = QLabel()
        self.displacement_label.setText('$|\\alpha|$')
        self.state_parameters_layout.addWidget(self.displacement_label)
        self.displacement_linedit = QLineEdit()
        self.displacement_linedit.textEdited.connect(self.displacement_linedit_changed)
        self.state_parameters_layout.addWidget(self.displacement_linedit)
        ## Angle
        self.angle_label = QLabel()
        self.angle_label.setText('$\\phi$')
        self.state_parameters_layout.addWidget(self.angle_label)
        self.angle_linedit = QLineEdit()
        self.angle_linedit.textEdited.connect(self.angle_linedit_changed)
        self.state_parameters_layout.addWidget(self.angle_linedit)
        #Plot
        figure = pyplot.figure(num = 'State Generation Plot')
        self.state_generation_plot_widget = PyplotWidget(figure=figure)
        self.state_generation_layout.addWidget(self.state_generation_plot_widget)
        axis = self.state_generation_plot_widget.figure.add_subplot(111)
        axis.set_xlabel('q', fontdict=fonts["axis"])
        axis.set_ylabel('p', fontdict=fonts["axis"])
        axis.set_title('Phase Space Plot', fontdict=fonts["title"])
        axis.grid()
        axis.set_aspect('equal')
        axis.set_xlim([-5, 5])
        axis.set_ylim([-5, 5])
        self.state_generation_plot_widget.update()
    # -------------------------------------------
    def make_voltages_widget(self):
        # Fixed AOM and EOM voltages
        self.voltages_widget = QWidget()
        self.voltages_layout = QHBoxLayout()
        self.voltages_widget.setLayout(self.voltages_layout)
        # Set AOM high voltage
        self.aoms_high_voltage_label = QLabel()
        self.aoms_high_voltage_label.setText('aoms high voltage [V]')
        self.voltages_layout.addWidget(self.aoms_high_voltage_label)
        self.aoms_high_voltage_linedit = QLineEdit(str(self.state_generator.devices['aoms']['levels']['lock']))
        self.aoms_high_voltage_linedit.textEdited.connect(self.aoms_high_voltage_linedit_changed)
        self.voltages_layout.addWidget(self.aoms_high_voltage_linedit)
        # Default medium voltages
        self.devices = ["aoms"]#, "amplitude_eom"]#, "phase_eom"] # Changed
        aoms_default = self.state_generator.devices['aoms']['levels']['state_generation']
        amplitude_eom_default = self.state_generator.devices['amplitude_eom']['levels']['state_generation']
        default_voltages = dict(zip(self.devices, [aoms_default, amplitude_eom_default, 0]))
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
        self.devices = ["aoms"]#, "amplitude_eom"]#, "phase_eom"] # Changed
        self.state_generator.voltage_ranges = [(5e-3, 2e-2), (3, 5), (-2, 2)]
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
            setattr(self, "%s_n_points_linedit" % device, QLineEdit(str(10)))  # number of points
            self.calibration_layout.addLayout(getattr(self, "%s_n_points_layout" % device), 3, col + 1)
            ## Add the widgets to layout
            getattr(self, "%s_n_points_layout" % device).addWidget(
                getattr(self, "%s_n_points_label" % device))
            getattr(self, "%s_n_points_layout" % device).addWidget(
                getattr(self, "%s_n_points_linedit" % device))
            #Calibration layout
            setattr(self, "%s_calibr_layout" % device, QHBoxLayout())
            setattr(self, "%s_button"%device, QPushButton("calibrate"))
            button = getattr(self, "%s_button"%device)
            button.clicked.connect(getattr(self, "%s_button_clicked"%device))
            getattr(self, "%s_calibr_layout" % device).addWidget(button)
            #self.calibration_layout.addWidget(button, 4, col + 1)
            #Calibrate n times button
            setattr(self, "%s_button" % device, QPushButton("calibrate n times"))
            button = getattr(self, "%s_button" % device)
            button.clicked.connect(getattr(self, "%s_n_times_button_clicked" % device))
            getattr(self, "%s_calibr_layout" % device).addWidget(button)
            #self.calibration_layout.addWidget(button, 4, col + 1)
            setattr(self, "%s_iterations" % device, QLineEdit(str(5)))
            getattr(self, "%s_calibr_layout" %device).addWidget(
                getattr(self, "%s_iterations" %device))
            self.calibration_layout.addLayout(getattr(self, "%s_calibr_layout" % device), 4, col + 1)
            #Uncomment if using phase EOM
            '''
            if device == "phase_eom":
                setattr(self, "%s_scanned_calibration_layout" % device, QHBoxLayout()) #horizontal layout
                #Scanned calibration button
                setattr(self, "%s_scan_calibrate_button"%device, QPushButton("Scan calibrate"))
                button = getattr(self, "%s_scan_calibrate_button" % device)
                button.clicked.connect(getattr(self, "%s_scan_calibrate_button_clicked" % device))
                getattr(self, "%s_scanned_calibration_layout" % device).addWidget(button)
                #Scan checkbox
                setattr(self, "%s_scan_checkbox"%device, QCheckBox("Scan/unscan"))
                checkbox = getattr(self, "%s_scan_checkbox" % device)
                checkbox.toggled.connect(getattr(self, "%s_scan_checkbox_toggled" % device))
                getattr(self, "%s_scanned_calibration_layout" % device).addWidget(checkbox)
                self.calibration_layout.addLayout(getattr(self, "%s_scanned_calibration_layout" % device), 5, col + 1)
            '''
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
            self.calibration_layout.addWidget(getattr(self, plot_name), 6, col+1)
            # Clear graphs button
            setattr(self, "%s_clear_button" % device, QPushButton("clear graph"))
            button = getattr(self, "%s_clear_button" % device)
            button.clicked.connect(getattr(self, "%s_clear_button_clicked" % device))
            self.calibration_layout.addWidget(button, 7, col + 1)
    # -------------------------------------------
    def make_phase_calibration_widget(self):
        #Make widget and layout
        self.phase_calibration_widget = QWidget()
        self.phase_calibration_layout = QVBoxLayout()
        self.phase_calibration_widget.setLayout(self.phase_calibration_layout)
        #Phase controller
        self.relay_lock_widget = PhaseControllerWidget(self.relay_lock.hd_controller.phase_controller)
        #self.relay_lock_widget = HDWidget(self.relay_lock.hd_controller)
        self.phase_calibration_layout.addWidget(self.relay_lock_widget)
    # -------------------------------------------
    def make_generation_widget(self):
        '''

        :return:
        '''

    # -------------------------------------------
    def aoms_high_voltage_linedit_changed(self, text):
        self.state_generator.aoms_high = float(text)
        self.state_generator.devices['aoms']['instrument'].offset = float(text)/self.state_generator.devices['aoms']['amplification_gain'] - 1
        self.state_generator.devices['aoms']['levels']['lock'] = float(text)/self.state_generator.devices['aoms']['amplification_gain']
        self.state_generator.pyrpl_obj_aom.rp.asg0.output_direct = 'out1'
    # -------------------------------------------
    def aoms_voltage_linedit_changed(self, text):
        self.state_generator.devices['aoms']['levels']['state_generation'] = float(text)
    # -------------------------------------------
    def amplitude_eom_voltage_linedit_changed(self, text):
        self.state_generator.devices['amplitude_eom']['levels']['state_generation'] = float(text)
        if self.state_generator.eom:
            self.state_generator.devices['amplitude_eom']['instrument'].offset = float(text)/self.state_generator.devices['amplitude_eom']['amplification_gain'] - self.state_generator.devices['amplitude_eom']['instrument_offset']
            self.state_generator.pyrpl_obj_eom.rp.asg0.output_direct = 'out1'
    # -------------------------------------------
    '''
    def phase_eom_voltage_linedit_changed(self, text):
        self.state_generator.devices['phase_eom']['levels']['state_generation'] = float(text)
    '''
    # -------------------------------------------
    def displacement_linedit_changed(self, text):
        self.state_generator.amplitude = float(text)
    # -------------------------------------------
    def angle_linedit_changed(self, text):
        self.state_generator.angle = float(text)
    # -------------------------------------------
    def state_generation_button_clicked(self):
        '''
        self.state_generator.generate_state()
        for key, value in self.state_generator.calibrations.items():
            print(key)
        #Set amplitude
        self.state_generator.devices['amplitude_eom']['levels']['state_generation'] = \
        self.state_generator.calibrations["amplitude_eom"]["function"](self.state_generator.amplitude)
        self.amplitude_eom_voltage_linedit.setText("%.2f"%self.state_generator.devices['amplitude_eom']['levels']['state_generation'])
        print("Amplitude EOM voltage: %.2f" %self.state_generator.devices['amplitude_eom']['levels']['state_generation'])
        '''
        # Set amplitude
        # First, invert function
        def inverse(values, amp):
            sqrt = numpy.sqrt(values[1] ** 2 - 4 * values[0]*(values[2] - amp))
            return (sqrt - values[1])/(2*values[0])

        if len(self.state_generator.calibrations) == 0:
            print('No AOM calibration was made. Using default parameters.')
            params = numpy.ndarray([6.17637427e+02, 9.21515583e+00, -1.36959470e-02])
        else:
            print('DEBUG:', len(self.state_generator.calibrations), self.state_generator.calibrations)
            params = self.state_generator.calibrations['aoms']['parameters']
        if len(params) != 3:
            print('Function cannot be inverted!')
        else:
            voltage = inverse(params,self.state_generator.amplitude)
            self.state_generator.devices['aoms']['levels']['state_generation'] = voltage
            # I'm also considering vacuum period as a state generation period, so the input state is never vacuum when taking measurements
            self.state_generator.devices['aoms']['levels']['vacuum'] = voltage
        print('Voltage for desired amplitude is %.3f' %voltage)

        #Set angle
        # First time multiplexing needs to be off, so relay may be properly calibrated
        self.state_generator.turn_off_time_multiplexing_signals()

        # Angle is set by locking phase of relay
        phase_controller = self.relay_lock.hd_controller.phase_controller
        phase_controller.calibrate()
        phase_controller.set_phase(self.state_generator.angle)
        phase_controller.lock()

        # Sample-hold is turned on for desired voltages
        aom_levels = [self.state_generator.devices['aoms']['levels'][x] for x in
                      ['lock', 'state_generation', 'vacuum']]
        eom_levels = [self.state_generator.devices['amplitude_eom']['levels'][x] for x in
                      ['lock', 'state_generation', 'vacuum']]
        duty_cycles = self.state_generator.time_multiplexing['duty_cycles']
        frequency = self.state_generator.time_multiplexing['frequency']
        max_delay = self.state_generator.time_multiplexing['max_delay']
        self.state_generator._time_multiplexing_signals(max_delay=max_delay, aom_levels=aom_levels,
                                                        eom_levels=eom_levels, \
                                                        frequency=frequency, aom_duty_cycles=duty_cycles,
                                                        eom_duty_cycles=duty_cycles, gate_flag = False)

        # Plot figure
        axis = self.state_generation_plot_widget.figure.axes[0]

        p = self.state_generator.amplitude*numpy.sin(self.state_generator.angle*numpy.pi/180)
        q = self.state_generator.amplitude*numpy.cos(self.state_generator.angle*numpy.pi/180)
        shot_noise = 1/2

        circle = plt.Circle((q, p), shot_noise, color = 'r')
        axis.add_patch(circle)
    # -------------------------------------------
    def aoms_clear_button_clicked(self):
        axis = self.aoms_plot.figure.axes[0]
        axis.lines = []
        self.aoms_plot.update()
    # -------------------------------------------
    def amplitude_eom_clear_button_clicked(self):
        axis = self.amplitude_eom_plot.figure.axes[0]
        axis.lines = []
        self.amplitude_eom_plot.update()
    # -------------------------------------------
    '''
    def phase_eom_clear_button_clicked(self):
        axis = self.phase_eom_plot.figure.axes[0]
        axis.lines = []
        self.phase_eom_plot.update()
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
        n_points = int(self.aoms_n_points_linedit.text())
        self.time_multiplexing_signals_checkbox.setChecked(True)
        x, y, x_fit, y_fit, y_error = self.state_generator.calibrate_aoms(voltage_range=[min_voltage, max_voltage], n_points=n_points)
        axis.errorbar(x, numpy.abs(y), yerr = y_error, linestyle="None", marker="o", markersize=10, \
                label=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        axis.plot(x_fit, y_fit, label="polynomial fit", linewidth=3.5)
        axis.legend(loc="upper right", prop=fonts["legend"])
        axis.legend(prop={"family":"Times New Roman"})
        self.aoms_plot.update()

    # -------------------------------------------
    def aoms_n_times_button_clicked(self):
        '''
        :return:
        '''
        number_of_iterations = int(self.aoms_iterations.text())
        # Dummy test behavior
        axis = self.aoms_plot.figure.axes[0]
        # Calibrate the aoms
        min_voltage = float(self.aoms_min_voltage_linedit.text())
        max_voltage = float(self.aoms_max_voltage_linedit.text())
        n_points = int(self.aoms_n_points_linedit.text())
        self.time_multiplexing_signals_checkbox.setChecked(True)
        x, y, x_fit, y_fit, sdev = self.state_generator.calibrate_aoms_n_times(num_of_iterations=number_of_iterations, voltage_range=[min_voltage, max_voltage], n_points=n_points)
        axis.errorbar(x, numpy.abs(y), sdev, linestyle="None", marker="o", markersize=10, \
                  label=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        #axis.plot(x, numpy.abs(y), linestyle="None", marker="o", markersize=10, \
        #          label=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        axis.plot(x_fit, y_fit, label="polynomial fit", linewidth=3.5)
        axis.legend(loc="upper right", prop=fonts["legend"])
        axis.legend(prop={"family": "Times New Roman"})
        self.aoms_plot.update()
    # -------------------------------------------
    def amplitude_eom_n_times_button_clicked(self):
        '''
        :return:
        '''
        pass
    # -------------------------------------------
    """
    def phase_eom_button_clicked(self):
        '''

        :return:
        '''
        axis = self.phase_eom_plot.figure.axes[0]
        # Calibrate the phase eom
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
    """
    # -------------------------------------------
    '''
    def phase_eom_scan_calibrate_button_clicked(self):
        min_voltage = float(self.phase_eom_min_voltage_linedit.text())
        max_voltage = float(self.phase_eom_max_voltage_linedit.text())
        #Calibrate
        self.time_multiplexing_signals_checkbox.setChecked(False)
        self.phase_eom_scan_checkbox.setChecked(True)
        hd_scanned, hd_scanned_fitted, time = \
        self.state_generator.scan_calibrate_phase_eom(voltage_range = [min_voltage, max_voltage])
        fonts = {"title": {"fontsize": 26, "family": "Times New Roman"}, \
                 "axis": {"fontsize": 22, "family": "Times New Roman"}, \
                 "legend": {"size": 24, "family": "Times New Roman"}}
        axis = self.phase_eom_plot.figure.axes[0]
       # axis.clear()
       # self.amplitude_eom_plot.update()
        axis.plot(time * 1e6, hd_scanned, linestyle="None", marker="o", markersize=8)
        axis.set_xlabel("input voltage (V)", fontdict=fonts["axis"])
        axis.set_ylabel("HD output voltage (V)", fontdict=fonts["axis"])
        axis.set_title("scanned HD output", fontdict=fonts["title"])
        axis.grid(True)
        axis.plot(time * 1e6, hd_scanned_fitted, label="sinusoidal fit", linewidth=3.5)
        axis.legend(loc="upper right", prop=fonts["legend"])
        self.phase_eom_scan_checkbox.setChecked(False)
        self.phase_eom_plot.update()
    '''
    # -------------------------------------------
    '''
    def phase_eom_scan_checkbox_toggled(self):
        asg = self.state_generator.devices['phase_eom']['instrument']
        is_on = self.phase_eom_scan_checkbox.isChecked()
        if is_on:
            frequency = 2e2
            voltage_range = [float(self.phase_eom_min_voltage_linedit.text()), float(self.phase_eom_max_voltage_linedit.text())]
            self.state_generator.scan_phase_eom(frequency=frequency, voltage_range=voltage_range)
        else:
            self.state_generator.turn_off_phase_eom_scan()
    '''
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
        self.time_multiplexing_signals_checkbox.setChecked(True)
        x, y, x_fit, y_fit = self.state_generator.calibrate_amplitude_eom(voltage_range=[min_voltage, max_voltage], n_points=n_points)
        axis.plot(x, numpy.abs(y), linestyle="None", marker="o", markersize=10, \
                  label=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        axis.plot(x_fit, y_fit, label="sinusoidal fit", linewidth=3.5)
        axis.legend(loc="upper right", prop=fonts["legend"])
        axis.legend(prop={"family":"Times New Roman"})
        self.amplitude_eom_plot.update()
    # -------------------------------------------
    def time_multiplexing_signals_checkbox_toggled(self):
        was_on = not self.time_multiplexing_signals_checkbox.isChecked()
        if was_on:
            self.state_generator.turn_off_time_multiplexing_signals()
            #aom = self.state_generator.devices['aoms']['instrument']
            #eom = self.state_generator.devices['amplitude_eom']['instrument']
            #sample_hold = self.state_generator.pyrpl_obj_aom.rp.asg1

            #aom_value = float(self.aoms_high_voltage_linedit.text())
            #print(aom_value)
            #eom_value = float(self.amplitude_eom_voltage_linedit.text())
            #sample_hold_value = 1

            #aom.setup(trigger_source="immediately", amplitude=1, offset=aom_value)
            #eom.setup(trigger_source="immediately", amplitude=1, offset=eom_value)
            #sample_hold.setup(trigger_source="immediately", amplitude=1, offset=sample_hold_value)

            #for name in ['aoms', 'amplitude_eom']:
            #    self.devices[name]['instrument'].output_direct = 'off'
            #    n_points = 2**14
            #    self.devices[name]['instrument'].data = numpy.zeros(n_points)
            #    self.state_generator.devices[name]['instrument'].offset = self.state_generator.devices[name]['levels']['lock']
            #    self.state_generator.pyrpl_obj_aom.rp.asg0.output_direct = 'out1'
            #    self.state_generator.pyrpl_obj_eom.rp.asg0.output_direct = 'out1'
        else:
            aom_levels = [self.state_generator.devices['aoms']['levels'][x] for x in ['lock', 'state_generation', 'vacuum']]
            eom_levels = [self.state_generator.devices['amplitude_eom']['levels'][x] for x in ['lock', 'state_generation', 'vacuum']]
            duty_cycles = self.state_generator.time_multiplexing['duty_cycles']
            frequency = self.state_generator.time_multiplexing['frequency']
            max_delay = self.state_generator.time_multiplexing['max_delay']
            self.state_generator._time_multiplexing_signals(max_delay=max_delay, aom_levels=aom_levels, eom_levels=eom_levels, \
                                                            frequency=frequency, aom_duty_cycles=duty_cycles, eom_duty_cycles=duty_cycles)

    # -------------------------------------------
    def two_time_multiplexing_signals_checkbox_toggled(self):
        was_on = not self.time_multiplexing_signals_checkbox.isChecked()
        if was_on:
            self.state_generator.turn_off_time_multiplexing_signals()
            # aom = self.state_generator.devices['aoms']['instrument']
            # eom = self.state_generator.devices['amplitude_eom']['instrument']
            # sample_hold = self.state_generator.pyrpl_obj_aom.rp.asg1

            # aom_value = float(self.aoms_high_voltage_linedit.text())
            # print(aom_value)
            # eom_value = float(self.amplitude_eom_voltage_linedit.text())
            # sample_hold_value = 1

            # aom.setup(trigger_source="immediately", amplitude=1, offset=aom_value)
            # eom.setup(trigger_source="immediately", amplitude=1, offset=eom_value)
            # sample_hold.setup(trigger_source="immediately", amplitude=1, offset=sample_hold_value)

            # for name in ['aoms', 'amplitude_eom']:
            #    self.devices[name]['instrument'].output_direct = 'off'
            #    n_points = 2**14
            #    self.devices[name]['instrument'].data = numpy.zeros(n_points)
            #    self.state_generator.devices[name]['instrument'].offset = self.state_generator.devices[name]['levels']['lock']
            #    self.state_generator.pyrpl_obj_aom.rp.asg0.output_direct = 'out1'
            #    self.state_generator.pyrpl_obj_eom.rp.asg0.output_direct = 'out1'
        else:
            aom_levels = [self.state_generator.devices['aoms']['levels'][x] for x in
                          ['lock', 'state_generation', 'state_generation']]
            eom_levels = [self.state_generator.devices['amplitude_eom']['levels'][x] for x in
                          ['lock', 'state_generation', 'state_generation']]
            duty_cycles = self.state_generator.time_multiplexing['duty_cycles']
            frequency = self.state_generator.time_multiplexing['frequency']
            max_delay = self.state_generator.time_multiplexing['max_delay']
            self.state_generator._time_multiplexing_signals(max_delay=max_delay, aom_levels=aom_levels,
                                                            eom_levels=eom_levels, \
                                                            frequency=frequency, aom_duty_cycles=duty_cycles,
                                                            eom_duty_cycles=duty_cycles, gate_flag = False)
    # --------------------------------------------
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
        aoms_voltage, amplitude_eom_voltage, phase_eom_voltage = \
        [self.state_generator.devices[name]['levels']['state_generation'] for name in ['aoms', 'amplitude_eom', 'phase_eom']]
        self.state_generator.tomography_mesaurement_for_calibration(\
                                                                    aom_voltage = aoms_voltage, \
                                                                    amplitude_eom_voltage = amplitude_eom_voltage, \
                                                                    phase_eom_voltage = phase_eom_voltage)
    # -------------------------------------------
    def tomography_dimension_slider_changed(self, value):
        label = "dimension: %d" % value
        self.tomography_dimension_label.setText(label)
        self.state_generator.state_measurement.quantum_state = self.state_generator.state_measurement.quantum_state.Resize(value)

    # -------------------------------------------
    def stop_calibration_button_clicked(self):
        self.state_generator.calibration_stopped = True
    # -------------------------------------------
    def find_aoms_high_button_clicked(self):
        self.state_generator._find_aoms_high()
        self.state_generator.aoms_high
        self.aoms_high_voltage_linedit.clear()
        self.aoms_high_voltage_linedit.insert((str(self.state_generator.aoms_high)))
    # -------------------------------------------
    def close_pop_up_graphs_button_clicked(self):
        pyplot.close('all')