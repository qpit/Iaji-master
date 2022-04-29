"""
This module defines the widget for SigilentSignalGenerator module
"""
# In[imports]
from PyQt5.QtCore import Qt, QRect
from PyQt5.Qt import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDateTimeEdit,
    QDial,
    QDoubleSpinBox,
    QFileDialog,
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
from Iaji.InstrumentsControl.SigilentSignalGenerator import SigilentSignalGenerator, SigilentSignalGeneratorChannel
from Iaji.InstrumentsControl.GUI.WidgetStyles import SigilentSignalGeneratorWidgetStyle
from Iaji.Utilities.strutils import any_in_string
import signalslot
# In[signal generator widget]
class SigilentSignalGeneratorWidget(QWidget):
    """
    this class describes the widget of a Sigilent signal generator
    """
    def __init__(self, signal_generator: SigilentSignalGenerator, name=None):
        super().__init__()
        self.signal_generator = signal_generator
        if name is None:
            self.name = self.signal_generator.name
        else:
            self.name = name
        #Window title
        self.setWindowTitle(self.name)
        #Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        #name
        self.name_label = QLabel()
        self.layout.addWidget(self.name_label)
        self.name_label.setText(self.name)
        #Channel widgets
        for channel in list(self.signal_generator.channels.values()):
            setattr(self, "channel_%d_widget"%channel.number, SigilentSignalGeneratorChannelWidget(channel=channel))
            self.layout.addWidget(getattr(self, "channel_%d_widget"%channel.number))
        #Phase linking
        self.phase_lock_radiobutton = QRadioButton("lock phases")
        self.phase_lock_radiobutton.toggled.connect(self.phase_lock_radiobutton_toggled)
        self.layout.addWidget(self.phase_lock_radiobutton)
        #Set style
        self.style_sheets = SigilentSignalGeneratorWidgetStyle().style_sheets
        self.set_style(theme="dark")
    # -------------------------------------------
    def phase_lock_radiobutton_toggled(self, state):
        self.signal_generator.lock_phase(state)
        channel_names = list(self.signal_generator.channels.keys())
        if self.signal_generator.channels[channel_names[0]].waveform in ["SQUARE", "SINE"] \
            and self.signal_generator.channels[channel_names[1]].waveform in ["SQUARE", "SINE"]:
            self.channel_1_widget.waveform_widget.phase_changed.connect(self.channel_1_widget.waveform_widget.phase_doublespinbox_changed)
            self.channel_2_widget.waveform_widget.phase_changed.connect(
                self.channel_2_widget.waveform_widget.phase_doublespinbox_changed)
    # -------------------------------------------
    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        excluded_strings = ["layout", "callback", "clicked", "toggled", "changed", "edited", "checked"]
        for widget_type in ["label", "radiobutton"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if
                       widget_type in name and not any_in_string(excluded_strings, name)]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])
        # Set style to custom widgets
        for channel in list(self.signal_generator.channels.values()):
            getattr(self, "channel_%d_widget"%channel.number).style_sheets = self.style_sheets
            getattr(self, "channel_%d_widget"%channel.number).set_style(theme="dark")
# In[channel widget]
class SigilentSignalGeneratorChannelWidget(QWidget):
    """
    This class describes the widget of a Sigilent signal generator channel
    """
    # -------------------------------------------
    def __init__(self, channel: SigilentSignalGeneratorChannel, name=None):
        super().__init__()
        self.channel = channel
        if name is not None:
            self.name = name
        else:
            self.name = self.channel.name
        #Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        #Name
        self.name_label = QLabel()
        self.layout.addWidget(self.name_label)
        self.name_label.setText("Channel %s"%self.name)
        #Waveform type
        self.waveform_combobox = QComboBox()
        self.layout.addWidget(self.waveform_combobox)
        waveforms = (["dc", "square"])
        self.waveform_combobox.addItems(waveforms)
        self.waveform_combobox.currentIndexChanged.connect(self.waveform_combobox_changed)
        #Waveform panel
        self.waveform_layout = QHBoxLayout()
        self.layout.addLayout(self.waveform_layout)
        waveform = self.channel.get_parameter("waveform")
        index = waveforms.index(waveform.lower())
        self.waveform_combobox.setCurrentIndex(index)
    # -------------------------------------------
    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        excluded_strings = ["layout", "callback", "clicked", "toggled", "changed", "edited", "checked"]
        for widget_type in ["label", "combobox"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if
                       widget_type in name and not any_in_string(excluded_strings, name)]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])
    # -------------------------------------------
    def waveform_combobox_changed(self, selected_index):
        type = self.waveform_combobox.itemText(selected_index)
        self.channel.set_waveform(type.upper())
        # Clear waveform layout
        if self.waveform_layout.count() != 0:
            self.waveform_layout.itemAt(self.waveform_layout.count()-1).widget().setParent(None)
        if type == "dc":
            self.waveform_widget = DCWaveformPanel(channel=self.channel)
        elif type == "square":
            self.waveform_widget = SquareWaveformPanel(channel=self.channel)
        else:
            self.waveform_widget = None
        #Add waveform widget
        self.waveform_layout.addWidget(self.waveform_widget)
        self.waveform_widget.style_sheets = self.style_sheets
        self.waveform_widget.set_style(theme="dark")
    # -------------------------------------------
class DCWaveformPanel(QWidget):
    # -------------------------------------------
    def __init__(self, channel: SigilentSignalGeneratorChannel):
        super().__init__()
        self.channel = channel
        #Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        #Offset
        # ---------------
        self.offset_layout = QHBoxLayout()
        self.layout.addLayout(self.offset_layout)
        #Label
        self.offset_label = QLabel()
        self.offset_layout.addWidget(self.offset_label)
        self.offset_label.setText("offset [V]")
        #Slider
        self.offset_doublespinbox = QDoubleSpinBox()
        self.offset_layout.addWidget(self.offset_doublespinbox)
        self.offset_doublespinbox.setRange(-10, 10)
        self.offset_doublespinbox.setSingleStep(1e-3)
        self.offset_doublespinbox.valueChanged.connect(self.offset_doublespinbox_changed)
        self.offset_doublespinbox.setValue(self.channel.get_parameter("offset"))
    # -------------------------------------------
    def offset_doublespinbox_changed(self, value):
        self.channel.set_offset(value)
        #self.offset_label.setText("offset: %s V" % value)
    # -------------------------------------------
    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        excluded_strings = ["layout", "callback", "clicked", "toggled", "changed", "edited", "checked"]
        for widget_type in ["label", "slider", "doublespinbox"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if
                       widget_type in name and not any_in_string(excluded_strings, name)]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])
# -------------------------------------------
class SquareWaveformPanel(QWidget):
    # -------------------------------------------
    def __init__(self, channel: SigilentSignalGeneratorChannel):
        super().__init__()
        self.channel = channel
        #Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        #Frequency
        #---------------
        self.frequency_layout = QHBoxLayout()
        self.layout.addLayout(self.frequency_layout)
        #Label
        self.frequency_label = QLabel()
        self.frequency_layout.addWidget(self.frequency_label)
        self.frequency_label.setText("frequency [Hz]")
        #Slider
        self.frequency_doublespinbox = QDoubleSpinBox()
        self.frequency_layout.addWidget(self.frequency_doublespinbox)
        self.frequency_doublespinbox.setRange(0, 25e6)
        self.frequency_doublespinbox.setSingleStep(1)
        self.frequency_doublespinbox.valueChanged.connect(self.frequency_doublespinbox_changed)
        print(".")
        self.frequency_doublespinbox.setValue(self.channel.get_parameter("frequency"))
        # ---------------
        # low_level
        # ---------------
        self.low_level_layout = QHBoxLayout()
        self.layout.addLayout(self.low_level_layout)
        # Label
        self.low_level_label = QLabel()
        self.low_level_layout.addWidget(self.low_level_label)
        self.low_level_label.setText("low level [V]")
        # Slider
        self.low_level_doublespinbox = QDoubleSpinBox()
        self.low_level_layout.addWidget(self.low_level_doublespinbox)
        self.low_level_doublespinbox.setRange(-10, 10)
        self.low_level_doublespinbox.setSingleStep(1e-3)
        self.low_level_doublespinbox.valueChanged.connect(self.low_level_doublespinbox_changed)
        self.low_level_doublespinbox.setValue(self.channel.get_parameter("low level"))
        # ---------------
        # high_level
        # ---------------
        self.high_level_layout = QHBoxLayout()
        self.layout.addLayout(self.high_level_layout)
        # Label
        self.high_level_label = QLabel()
        self.high_level_layout.addWidget(self.high_level_label)
        self.high_level_label.setText("high level [V]")
        # Slider
        self.high_level_doublespinbox = QDoubleSpinBox()
        self.high_level_layout.addWidget(self.high_level_doublespinbox)
        self.high_level_doublespinbox.setRange(-10, 10)
        self.high_level_doublespinbox.setSingleStep(1e-3)
        self.high_level_doublespinbox.valueChanged.connect(self.high_level_doublespinbox_changed)
        self.high_level_doublespinbox.setValue(self.channel.get_parameter("high level"))
        # ---------------
        # phase
        # ---------------
        self.phase_layout = QHBoxLayout()
        self.layout.addLayout(self.phase_layout)
        # Label
        self.phase_label = QLabel()
        self.phase_layout.addWidget(self.phase_label)
        self.phase_label.setText("phase [degrees]")
        # Slider
        self.phase_doublespinbox = QDoubleSpinBox()
        self.phase_layout.addWidget(self.phase_doublespinbox)
        self.phase_doublespinbox.setRange(0, 360)
        self.phase_doublespinbox.setSingleStep(0.1)
        self.phase_doublespinbox.valueChanged.connect(self.phase_doublespinbox_changed)
        self.phase_doublespinbox.setValue(self.channel.get_parameter("phase"))
        #Phase changed signal
        self.phase_changed = signalslot.Signal()
        # ---------------
        # duty cycle
        # ---------------
        self.duty_cycle_layout = QHBoxLayout()
        self.layout.addLayout(self.duty_cycle_layout)
        # Label
        self.duty_cycle_label = QLabel()
        self.duty_cycle_layout.addWidget(self.duty_cycle_label)
        self.duty_cycle_label.setText("duty_cycle [%]")
        # Slider
        self.duty_cycle_doublespinbox = QDoubleSpinBox()
        self.duty_cycle_layout.addWidget(self.duty_cycle_doublespinbox)
        self.duty_cycle_doublespinbox.setRange(0.01, 100)
        self.duty_cycle_doublespinbox.setSingleStep(0.1)
        self.duty_cycle_doublespinbox.valueChanged.connect(self.duty_cycle_doublespinbox_changed)
        self.duty_cycle_doublespinbox.setValue(self.channel.get_parameter("duty cycle"))
        # ---------------
    # -------------------------------------------
    def frequency_doublespinbox_changed(self, value):
        self.channel.set_frequency(value)
        #self.frequency_label.setText("frequency: %s Hz" %value)
    # -------------------------------------------
    def low_level_doublespinbox_changed(self, value):
        self.channel.set_low_level(value)
        #self.low_level_label.setText("low value: %s V" % value)
    # -------------------------------------------
    def high_level_doublespinbox_changed(self, value):
        self.channel.set_high_level(value)
        #self.high_level_label.setText("high level: %s V" % value)
    # -------------------------------------------
    def phase_doublespinbox_changed(self, value, **kwargs):
        self.channel.set_phase(value)
        self.phase_changed.emit(value)
        #self.phase_label.setText("phase: %s degrees" % value)
    # -------------------------------------------
    def duty_cycle_doublespinbox_changed(self, value):
        self.channel.set_duty_cycle(value)
        #self.duty_cycle_label.setText("duty cycle: %s \\%" % value)
    # -------------------------------------------
    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        excluded_strings = ["layout", "callback", "clicked", "toggled", "changed", "edited", "checked"]
        for widget_type in ["label", "slider", "doublespinbox"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if
                       widget_type in name and not any_in_string(excluded_strings, name)]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])





