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
from Iaji.Utilities.GUI import PyplotWidget
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
        button_names = ["tomography_measurement", "scanned_measurement"]
        for name in button_names:
            setattr(self, name+"_button",  QPushButton(name.replace("_", " ")))
            button = getattr(self, name+"_button")
            self.state_measurement_buttons_layout.addWidget(button)
            button.clicked.connect(getattr(self, name+"_button_clicked"))
        # Homodyne detection controller
        self.hd_controller_widget = HDWidget(self.state_measurement.hd_controller)
        self.state_measurement_layout.addWidget(self.hd_controller_widget)
        #State measurement and analysis tabs
        self.state_tabs = QTabWidget()
        self.layout.addWidget(self.state_tabs)
        # Make state measurement widget and add tab
        self.state_tabs.addTab(self.state_measurement_widget, "State Measurement")
        #Make state analysis widget and add tab
        self.state_analysis_widget = StateAnalysisWidget(self.state_measurement)
        self.state_tabs.addTab(self.state_analysis_widget, "State Analysis")
        # Set style
        self.style_sheets = StateMeasurementControllerStyle().style_sheets
        self.set_style(theme="dark")
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
        self.state_analysis_widget.style_sheets = self.style_sheets
        self.state_analysis_widget.set_style(theme="dark")
    #-------------------------------------------
    def tomography_measurement_button_clicked(self):
        self.state_measurement.tomography_measurement(phases=[0, 30, 60, 90, 120, 150])
    # -------------------------------------------
    def scanned_measurement_button_clicked(self): #TODO
        self.state_measurement.scanned_measurement()
    # -------------------------------------------
    def state_analysis_button_clicked(self):
        pass
# In[]
class StateAnalysisWidget(QWidget):
    # -------------------------------------------
    def __init__(self, state_measurement: StateMeasurementController, name="State Analysis"):
        super().__init__()
        self.state_measurement = state_measurement
        self.name = name
        #Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        #Control panel
        self.control_layout = QHBoxLayout()
        self.layout.addLayout(self.control_layout)
        ##Analysis button
        self.analyze_button = QPushButton("analyze quantum state")
        self.analyze_button.clicked.connect(self.analyze_button_clicked)
        self.control_layout.addWidget(self.analyze_button)
        ##Dimension slider
        self.dimension_slider = QSlider(Qt.Horizontal)
        self.dimension_slider.setRange(1, 50)
        self.dimension_slider.setSingleStep(1)
        if self.state_measurement.quantum_state is not None:
            self.dimension_slider.setValue(self.state_measurement.quantum_state.hilbert_space.dimension)
        self.dimension_slider.valueChanged.connect(self.dimension_slider_changed)
        self.control_layout.addWidget(self.dimension_slider)
        ##Dimension label
        self.dimension_label = QLabel()
        label = "dimension: %d" % self.dimension_slider.value()
        self.dimension_label.setText(label)
        self.control_layout.addWidget(self.dimension_label)
        #Plot
        figure = None
        self.plot_widget = PyplotWidget(figure)
        self.plot_widget.update()
        self.layout.addWidget(self.plot_widget)
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
    # -------------------------------------------
    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        excluded_strings = ["layout", "callback", "clicked", "toggled", "changed", "edited", "checked"]
        for widget_type in ["label", "button", "tabs", "slider"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if
                       widget_type in name and not any_in_string(excluded_strings, name)]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])






