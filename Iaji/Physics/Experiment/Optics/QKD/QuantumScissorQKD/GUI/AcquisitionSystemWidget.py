#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:39:25 2022

@author: jiedz

This module defines the acquisition system widget.
"""
# In[GUI imports]
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
# In[imports]
from Iaji.InstrumentsControl.GUI.LecroyOscilloscopeWidget import LecroyOscilloscopeWidget \
    as ScopeWidget
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.WidgetStyles import \
    AcquisitionSystemWidgetStyle
from Iaji.Utilities.strutils import any_in_string
# In[acquisition system widget]
class AcquisitionSystemWidget(QWidget):
    '''
    This class describes a widget for the AcquisitionSystem module.
    '''
    # --------------------------------
    def __init__(self, acquisition_system, name="Acquisition System Widget"):
        super().__init__()
        self.acquisition_system = acquisition_system
        self.name = name
        self.setWindowTitle(self.name)
        #Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        ##Main title
        self.title_label = QLabel()
        self.title_label.setText(self.name)
        self.layout.addWidget(self.title_label)
        #Scope widget
        self.scope_widget = ScopeWidget(self.acquisition_system.scope, \
                                        name=self.acquisition_system.scope.name+" Widget")
        self.layout.addWidget(self.scope_widget)
        ##Overwrite the scope widget function that sets the host save directory
        ##to also update the acquisition system's host save directory
        self.scope_widget.save_layout.itemAt(2).widget().clicked.connect(self.button_host_save_path_callback_new)
        ##Overwrite the scope widget function that sets the trace file names
        ##to also update the acquisition system's trace file names
        channel_names = list(self.acquisition_system.scope.channels.keys())
        for j in range(len(channel_names)):
            name = channel_names[j]
            widget_name = "linedit_filename_channel_%s"%(j+1)
            linedit = getattr(self.scope_widget, widget_name)
            linedit.textChanged.connect(getattr(self, "%s_changed_new"%widget_name))
        #Set style
        self.style_sheets = AcquisitionSystemWidgetStyle().style_sheets
        self.set_style(theme="dark")
    # --------------------------------
    def set_style(self, theme):
        self.setStyleSheet(self.style_sheets["main"][theme])
        excluded_strings = ["layout", "callback", "clicked", "toggled", "changed", "edited", "checked"]
        for widget_type in ["label", "button", "tabs", "linedit", "checkbox"]:
            widgets = [getattr(self, name) for name in list(self.__dict__.keys()) if
                       widget_type in name and not any_in_string(excluded_strings, name)]
            for widget in widgets:
                widget.setStyleSheet(self.style_sheets[widget_type][theme])
        #Set the style of the custom sub-widgets
        self.scope_widget.style_sheets = self.style_sheets
        self.scope_widget.set_style(theme="dark")
    # --------------------------------
    def button_host_save_path_callback_new(self):
        self.acquisition_system.host_save_directory = self.scope_widget.host_save_directory
    # --------------------------------
    def linedit_filename_channel_1_changed_new(self):
        self.scope_widget.linedit_filename_channel_1_changed()
        channel_name = list(self.acquisition_system.scope.channels.keys())[0]
        self.acquisition_system.filenames[channel_name]\
            = self.scope_widget.filenames[channel_name]
    def linedit_filename_channel_2_changed_new(self):
        self.scope_widget.linedit_filename_channel_2_changed()
        channel_name = list(self.acquisition_system.scope.channels.keys())[1]
        self.acquisition_system.filenames[channel_name]\
            = self.scope_widget.filenames[channel_name]
    def linedit_filename_channel_3_changed_new(self):
        self.scope_widget.linedit_filename_channel_3_changed()
        channel_name = list(self.acquisition_system.scope.channels.keys())[2]
        self.acquisition_system.filenames[channel_name]\
            = self.scope_widget.filenames[channel_name]
    def linedit_filename_channel_4_changed_new(self):
        self.scope_widget.linedit_filename_channel_4_changed()
        channel_name = list(self.acquisition_system.scope.channels.keys())[3]
        self.acquisition_system.filenames[channel_name]\
            = self.scope_widget.filenames[channel_name]
        
        
        