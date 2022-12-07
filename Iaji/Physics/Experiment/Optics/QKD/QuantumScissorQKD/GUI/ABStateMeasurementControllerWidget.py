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
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateChecking import StateChecking
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateGenerator import StateGenerator
from Iaji.Utilities.GUI import PyplotWidget
from Iaji.Utilities.strutils import any_in_string
import numpy


class AliceStateMeasurementControllerWidget(QWidget):
    #-------------------------------------------
    def __init__(self, Alice_state_measurement:StateMeasurementController, Bob_state_measurement:StateMeasurementController, state_generator: StateGenerator,name = "State Measurement Controller Widget"):
        super().__init__()
        self.setWindowTitle(name)
        self.name = name
        self.state_generator = state_generator
        self.Alice_state_measurement = Alice_state_measurement
        if Bob_state_measurement == None:
            self.Bob_HD = False
        else:
            self.Bob_HD = True
        self.Bob_state_measurement = Bob_state_measurement
        #Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        #Title
        self.title_label = QLabel()
        self.layout.addWidget(self.title_label)
        self.title_label.setText(self.Alice_state_measurement.name)
        #State Measurement Layout
        self.Alice_state_measurement_widget = QWidget()
        ##State measurement buttons
        self.Alice_state_measurement_layout = QVBoxLayout()
        self.Alice_state_measurement_widget.setLayout(self.Alice_state_measurement_layout)
        self.Alice_state_measurement_buttons_layout = QHBoxLayout()
        self.Alice_state_measurement_layout.addLayout(self.Alice_state_measurement_buttons_layout)
        self.layout.addLayout(self.Alice_state_measurement_layout)
        button_names = ["Alice_tomography_measurement", "Alice_scanned_measurement"]
        for name in button_names:
            setattr(self, name+"_button",  QPushButton(name.replace("_", " ")))
            button = getattr(self, name+"_button")
            self.Alice_state_measurement_buttons_layout.addWidget(button)
            button.clicked.connect(getattr(self, name+"_button_clicked"))
        # Homodyne detection controller
        self.hd_controller_widget = HDWidget(self.Alice_state_measurement.hd_controller)
        self.Alice_state_measurement_layout.addWidget(self.hd_controller_widget)
        #State measurement and analysis tabs
        self.state_tabs = QTabWidget()
        self.layout.addWidget(self.state_tabs)
        # Make Alice state measurement widget and add tab
        self.state_tabs.addTab(self.Alice_state_measurement_widget, "Alice State Measurement")
        print('DEBUG:', self.Bob_state_measurement)
        if self.Bob_HD:
            # Make Bob state measurement widget and add tab
            self.Bob_state_measurement_widget = BobStateMeasurementControllerWidget(self.Bob_state_measurement)
            self.state_tabs.addTab(self.Bob_state_measurement_widget, "Bob State Measurement")
            # Make state analysis widget and add tab
            self.state_analysis_widget = StateAnalysisWidget(self.Bob_state_measurement, self.state_generator)
            self.state_tabs.addTab(self.state_analysis_widget, "State Analysis")
            # Set style
            self.style_sheets = StateMeasurementControllerStyle().style_sheets
            self.set_style(theme="dark")
        else:
            pass
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
        if self.Bob_HD:
            self.Bob_state_measurement_widget.style_sheets = self.style_sheets
            self.Bob_state_measurement_widget.set_style(theme="dark")
            self.state_analysis_widget.style_sheets = self.style_sheets
            self.state_analysis_widget.set_style(theme="dark")
        else:
            pass
    #-------------------------------------------
    def Alice_tomography_measurement_button_clicked(self):
        self.Alice_state_measurement.tomography_measurement(phases=[0, 30, 60, 90, 120, 150])
    # -------------------------------------------
    def Alice_scanned_measurement_button_clicked(self): #TODO
        self.Alice_state_measurement.scanned_measurement()
# In[]
class BobStateMeasurementControllerWidget(QWidget):
    #-------------------------------------------
    def __init__(self, Bob_state_measurement:StateMeasurementController, name = "Bob State Measurement Controller Widget"):
        super().__init__()
        self.name = name
        self.Bob_state_measurement = Bob_state_measurement
        #Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        #Buttons Layout
        self.Bob_state_measurement_buttons_layout = QHBoxLayout()
        self.layout.addLayout(self.Bob_state_measurement_buttons_layout)
        button_names = ["Bob_tomography_measurement", "Bob_scanned_measurement"]
        for name in button_names:
            setattr(self, name+"_button",  QPushButton(name.replace("_", " ")))
            button = getattr(self, name+"_button")
            self.Bob_state_measurement_buttons_layout.addWidget(button)
            button.clicked.connect(getattr(self, name+"_button_clicked"))
        # Homodyne detection controller
        self.hd_controller_widget = HDWidget(self.Bob_state_measurement.hd_controller)
        self.layout.addWidget(self.hd_controller_widget)
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
    # -------------------------------------------
    def Bob_tomography_measurement_button_clicked(self):
        self.Bob_state_measurement.tomography_measurement(phases=[0, 30, 60, 90, 120, 150])
    # -------------------------------------------
    def Bob_scanned_measurement_button_clicked(self): #TODO
        self.Bob_state_measurement.scanned_measurement()
# In[]
class StateAnalysisWidget(QWidget):
    # -------------------------------------------
    def __init__(self, state_measurement: StateMeasurementController, state_generator: StateGenerator, name="State Analysis"):
        super().__init__()
        self.state_measurement = state_measurement
        self.name = name
        self.state_generator = state_generator
        self.state_checking = StateChecking(state_measurement=self.state_measurement, state_generator=self.state_generator)
        #Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        #Control panel
        self.control_layout = QVBoxLayout()
        self.layout.addLayout(self.control_layout)
        ##Single photon analysis button
        self.single_photon_layout = QHBoxLayout()
        self.control_layout.addLayout(self.single_photon_layout)
        self.single_photon_analysis_button = QPushButton("Single photon analysis")
        self.single_photon_analysis_button.clicked.connect(self.single_photon_analysis_button_clicked)
        self.single_photon_layout.addWidget(self.single_photon_analysis_button)
        ##Temporal mode function parameters
        #self.temporal_mode_function_label = QLabel()
        #self.temporal_mode_function_label.setText("TMF = $\\kappa \\exp(-\\gamma |t - t_0|) - \\gamma \\exp(-\\kappa |t - t_0|) $")
        #self.single_photon_layout.addWidget(self.temporal_mode_function_label)
        tmf_parameters = ["kappa", "gamma", "t0"]
        tmf_values = dict(zip(tmf_parameters, [2*numpy.pi*11e6, 2*numpy.pi*19e6, -220e-9]))
        tmf_values_str = dict(zip(tmf_parameters, ["2 pi %.0f E6"%(tmf_values["kappa"]/(2*numpy.pi*10**6)), \
                                                   "2 pi %.0f E6"%(tmf_values["gamma"]/(2*numpy.pi*10**6)), "%.0f E9"%(tmf_values["t0"]*10**9)]))
        for parameter in tmf_parameters:
            label = QLabel()
            label.setText(parameter)
            self.single_photon_layout.addWidget(label)
            setattr(self, "tmf_%s_linedit" % parameter, QLineEdit(tmf_values_str[parameter]))
            linedit = getattr(self, "tmf_%s_linedit" % parameter)
            linedit.textEdited.connect(getattr(self, "tmf_%s_linedit_changed" % parameter))
            getattr(self, "tmf_%s_linedit_changed" % parameter)(tmf_values_str[parameter])
            self.single_photon_layout.addWidget(linedit)
        ##Time arrival histogram button
        self.time_arrival_histogram_layout = QHBoxLayout()
        self.control_layout.addLayout(self.time_arrival_histogram_layout)
        self.time_arrival_histogram_button = QPushButton("Find SSPD time differences")
        self.time_arrival_histogram_button.clicked.connect(self.time_arrival_histogram_button_clicked)
        self.time_arrival_histogram_layout.addWidget(self.time_arrival_histogram_button)
        SSPD_differences = ["heralding_SSPD_C", "heralding_SSPD_D"]
        time_differences = dict(zip(SSPD_differences, [114e-9, 45e-9]))
        time_differences_str = dict(zip(SSPD_differences, ["%.0f ns"%(time_differences["heralding_SSPD_C"]*10**9), \
                                                           "%.0f ns"%(time_differences["heralding_SSPD_D"]*10**9)]))
        for SSPD in SSPD_differences:
            label = QLabel()
            label.setText(SSPD.replace("_", " "))
            self.time_arrival_histogram_layout.addWidget(label)
            setattr(self, "%s_linedit" % SSPD, QLineEdit(time_differences_str[SSPD]))
            linedit = getattr(self, "%s_linedit" % SSPD)
            linedit.textEdited.connect(getattr(self, "%s_linedit_changed" % SSPD))
            getattr(self, "%s_linedit_changed" % SSPD)(time_differences[SSPD])
            self.time_arrival_histogram_layout.addWidget(linedit)
        label = QLabel()
        label.setText("number of bins")
        self.time_arrival_histogram_layout.addWidget(label)
        self.bins_number = 100
        self.bins_number_linedit = QLineEdit(str(self.bins_number))
        self.bins_number_linedit.textEdited.connect(self.bins_number_linedit_changed)
        self.time_arrival_histogram_layout.addWidget(self.bins_number_linedit)
        ##Analysis button
        self.analyze_layout = QHBoxLayout()
        self.control_layout.addLayout(self.analyze_layout)
        self.analyze_button = QPushButton("Analyze quantum state")
        self.analyze_button.clicked.connect(self.analyze_button_clicked)
        self.analyze_layout.addWidget(self.analyze_button)
        ##Dimension slider
        self.dimension_slider = QSlider(Qt.Horizontal)
        self.dimension_slider.setRange(1, 50)
        self.dimension_slider.setSingleStep(1)
        if self.state_measurement.quantum_state is not None:
            self.dimension_slider.setValue(self.state_measurement.quantum_state.hilbert_space.dimension)
        self.dimension_slider.valueChanged.connect(self.dimension_slider_changed)
        self.analyze_layout.addWidget(self.dimension_slider)
        ##Dimension label
        self.dimension_label = QLabel()
        label = "dimension: %d" % self.dimension_slider.value()
        self.dimension_label.setText(label)
        self.analyze_layout.addWidget(self.dimension_label)
        #Plot
        figure = None
        self.plot_widget = PyplotWidget(figure)
        self.plot_widget.update()
        self.layout.addWidget(self.plot_widget)
    # -------------------------------------------
    def time_arrival_histogram_button_clicked(self):
        heralding_sspdc_coincidence_time, heralding_sspdd_coincidence_time = \
            self.state_checking.time_arrival_histogram(bins_number=self.bins_number)
        self.heralding_SSPD_C_linedit_changed(str(heralding_sspdc_coincidence_time) + " ns")
        self.heralding_SSPD_D_linedit_changed(str(heralding_sspdd_coincidence_time) + " ns")
    # -------------------------------------------
    def bins_number_linedit_changed(self, text):
        self.bins_number = float(text)
    # -------------------------------------------
    def heralding_SSPD_C_linedit_changed(self, text):
        pass
    # -------------------------------------------
    def heralding_SSPD_D_linedit_changed(self, text):
        pass
    # -------------------------------------------
    def single_photon_analysis_button_clicked(self):
        pass
    # -------------------------------------------
    def tmf_kappa_linedit_changed(self, text):
        """
        :param text: value of kappa/(2 pi 10**6)
        :return:
        """
        pass
    # -------------------------------------------
    def tmf_gamma_linedit_changed(self, text):
        """"
        :param text: value of gamma/(2 pi 10**6)
        :return:
        """
        pass
    # -------------------------------------------
    def tmf_t0_linedit_changed(self, text):
        """
        :param text: value of t0 in ns
        :return:
        """
        pass
    # -------------------------------------------
    def analyze_button_clicked(self):
        self.plot_widget = PyplotWidget(self.state_measurement.quantum_state.PlotWignerFunction(\
                q=self.state_measurement.q, p=self.state_measurement.p, plot_name=self.state_measurement.name))
        self.state_measurement.quantum_state.PlotDensityOperator(plot_name=self.state_measurement.name)
        self.state_measurement.quantum_state.PlotNumberDistribution(plot_name=self.state_measurement.name)
        self.plot_widget.update()
    # -------------------------------------------
    def dimension_slider_changed(self): #TODO
        label = "dimension: %d"%self.dimension_slider.value()
        self.dimension_label.setText(label)
        self.state_measurement.quantum_state.Resize(self.dimension_slider.value())
    # -------------------------------------------
    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        excluded_strings = ["layout", "callback", "clicked", "toggled", "changed", "edited", "checked"]
        for widget_type in ["label", "button", "tabs", "slider"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if
                       widget_type in name and not any_in_string(excluded_strings, name)]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])






