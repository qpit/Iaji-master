"""
This module defines the GUI of the LecroyOscilloscope module.
#TODO
"""
#%%
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
from Iaji.InstrumentsControl.GUI.WidgetStyles import LecroyOscilloscopeWidgetStyle
#%%
class LecroyOscilloscopeWidget(QWidget):
    """
    This class describes a lecroy oscilloscope widget.
    """

    def __init__(self, scope, name="Lecroy Oscilloscope Widget"):
        super().__init__()
        self.scope = scope
        self.name = name
        self.host_save_directory = "."
        self.setWindowTitle(name)
        #Define the layout
        #-------------------------------
        self.layout = QVBoxLayout()
        #Widget title label
        self.name_label = QLabel()
        self.name_label.setText(self.scope.name)
        self.layout.addWidget(self.name_label, Qt.AlignCenter)
        #Save settings layout
        #--------------------------------
        self.save_layout = QHBoxLayout()
        #Title label
        self.save_layout_title_label = QLabel()
        self.save_layout_title_label.setText("Save Settings")
        self.save_layout.addWidget(self.save_layout_title_label, Qt.AlignCenter)
        #Push buttons
        button_names = ["save_trace", "scope_save_path", "host_save_path"]
        button_callbacks = dict(
            zip(button_names,
                [getattr(self, "button_" + name + "_callback") for name in button_names]))
        for name in button_names:
            button = QPushButton(name.replace("_", " "))
            button.clicked.connect(button_callbacks[name])
            self.save_layout.addWidget(button)
            setattr(self, "button_" + name, button)
        self.layout.addLayout(self.save_layout)
        # --------------------------------
        #Scope setup layout
        self.scope_settings_layout = QVBoxLayout()
        self.scope_setting_title = QLabel()
        self.scope_setting_title.setText("Scope Settings")
        self.scope_settings_layout.addWidget(self.scope_setting_title)
        self.layout.addLayout(self.scope_settings_layout)
        ##Select scope channels
        ## --------------------------------
        self.select_scope_channels_layout = QHBoxLayout()
        self.select_scope_channels_title = QLabel()
        self.select_scope_channels_title.setText("Active Channels:")
        self.select_scope_channels_layout.addWidget(self.select_scope_channels_title)
        channel_names = list(scope.channels.keys())
        #channel_names = ["C1", "C2", "aooo", "dio"]
        for j in range(len(channel_names)):
            name = channel_names[j]
            widget_name = "checkbox_channel_%s"%(j+1)
            checkbox = QCheckBox("channel %s"%name)
            checkbox.toggled.connect(getattr(self, "%s_checked"%widget_name))
            self.select_scope_channels_layout.addWidget(checkbox)
            setattr(self, widget_name, checkbox)
        self.layout.addLayout(self.select_scope_channels_layout)
        ## --------------------------------
        ##File name for every channel
        ## --------------------------------
        self.filenames_layout = QHBoxLayout()
        self.layout.addLayout(self.filenames_layout)
        ###Title
        self.filenames_title = QLabel()
        self.filenames_title.setText("file names")
        self.filenames_layout.addWidget(self.filenames_title)
        ###Line edit boxes
        self.filenames = ["trace" for j in range(len(channel_names))]
        for j in range(len(channel_names)):
            name = channel_names[j]
            widget_name = "linedit_filename_channel_%s"%(j+1)
            linedit = QLineEdit(self.filenames[j])
            linedit.textChanged.connect(getattr(self, "%s_changed"%widget_name))
            self.filenames_layout.addWidget(linedit)
            setattr(self, widget_name, linedit)        
        ## --------------------------------
        # --------------------------------
        #TODO: define a simple plot widget to display the acquired traces
        #Set overall layout
        self.setLayout(self.layout)
        #Set style
        self.style_sheets = LecroyOscilloscopeWidgetStyle().style_sheets
        self.set_style(theme="dark")
    # --------------------------------
    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        for widget_type in ["label", "button", "line edit", "checkbox"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if widget_type in name and "layout" not in name and "callback" not in name]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])
    # --------------------------------
    def button_scope_save_path_callback(self):
        """
        Lets the user select the save directory in the scope's file system
        """
        #Open a file dialog in the current save directory on the scope, seen from the host PC (host_drive).
        title = "Select the saving directory in the scope's file system"
        directory = QFileDialog.getExistingDirectory(caption=title, directory=self.scope.host_drive+"\\"+self.scope.save_directory)
        #Remove host_drive from the save directory name
        directory = directory.replace(self.scope.host_drive+"/", "")
        self.scope.set_save_directory(directory)
        #print(self.scope.get_save_directory())
    # --------------------------------
    def button_host_save_path_callback(self):
        """
        Lets the user select the save directory in the host's file system
        """
        #Open a file dialog in the current save directory on the host
        title = "Select the saving directory on this machine"
        self.host_save_directory = QFileDialog.getExistingDirectory(caption=title, directory=self.host_save_directory)
    # --------------------------------
    def button_save_trace_callback(self):
        """
        Acquires and saves the selected scope traces.
        TODO: acquisition settings:
            1. option of acquiring from a subset of all the channels
            2. choosing filenames in separate text edit boxes
            3. option of loading the acquired traces
            4. option of adjusting the horizontal and vertical axes
        """
        self.scope.acquire(channel_names=list(self.scope.channels.keys()), filenames=self.filenames, save_directory=self.host_save_directory, \
                      load_traces=True, \
                      adapt_vertical_axis=False, adapt_horizontal_axis=False)
    # --------------------------------
    def checkbox_channel_1_checked(self):
        channel_name = list(self.scope.channels.keys())[0]
        self.scope.channels[channel_name].enable(self.checkbox_channel_1.isChecked())
    def checkbox_channel_2_checked(self):
        channel_name = list(self.scope.channels.keys())[1]
        self.scope.channels[channel_name].enable(self.checkbox_channel_2.isChecked())
    def checkbox_channel_3_checked(self):
        channel_name = list(self.scope.channels.keys())[2]
        self.scope.channels[channel_name].enable(self.checkbox_channel_3.isChecked())
    def checkbox_channel_4_checked(self):
        channel_name = list(self.scope.channels.keys())[3]
        self.scope.channels[channel_name].enable(self.checkbox_channel_4.isChecked())
    # --------------------------------
    def linedit_filename_channel_1_changed(self):
        self.filenames[0] = self.linedit_filename_channel_1.text()+".trc"
       # print(self.filenames[0])
    def linedit_filename_channel_2_changed(self):
        self.filenames[1] = self.linedit_filename_channel_2.text()+".trc"
       # print(self.filenames[1])
    def linedit_filename_channel_3_changed(self):
        self.filenames[2] = self.linedit_filename_channel_3.text()+".trc"
       # print(self.filenames[2])
    def linedit_filename_channel_4_changed(self):
        self.filenames[3] = self.linedit_filename_channel_4.text()+".trc"
      #  print(self.filenames[3])        
    
    
















