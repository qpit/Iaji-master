"""
This module generates an input state for quantum scissor QKD.
"""
# In[imports]
import numpy
import time
import pyrpl
from Iaji.InstrumentsControl.SigilentSignalGenerator import SigilentSignalGenerator
from pyqtgraph.Qt import QtGui

class StateGenerator:
    """
    This class controls the AOMs and define amplitude and phase modulations for the EOMs.
    """
    # -------------------------------------------
    def __init__(self, redpitaya_config_filename, signal_generator:SigilentSignalGenerator, state_measurement, name = "State Generator"):
        '''
        :param red_pitaya_config_filename: str
            pitaya config file full path
        :param signal_generator: Iaji SigilentSignalGenerator
            signal generator object, controlling the state generation AOM driver
        :param state_measurement: Iaji StateMeasurementController
            state measurement controller object, for state calibration
        :param name: str
        '''
        self.redpitaya_config_filename = redpitaya_config_filename
        self.connect_to_redpitaya()
        self.signal_generator = signal_generator #signal generator controlling the state generator AOM driver
        self.state_measurement = state_measurement
        self.name = name
    # -------------------------------------------
    def connect_to_redpitaya(self, show_pyrpl_GUI=True):
        self.pyrpl_obj = pyrpl.Pyrpl(config=self.redpitaya_config_filename)
        if show_pyrpl_GUI:
            self.pyrpl_GUI = QtGui.QApplication.instance()
    # -------------------------------------------
    def calibrate_aoms(self, voltage_range=[0, 5]): #TODO
        '''
        Calibrates the state generation AOMs with respect to the induced amplitude variation.
        The aim is to draw a functional dependence of the induced amplitude variation on the voltage output from the
        signal generator that controls the AOM drivers.
        It iteratively inputs a voltage to the AOMs and measures the displacement of the coherent state
        from a homodyne detector.
        :param voltage_range: iterable of float (size=2)
            range of voltages output from the signal generator that feeds into the AOM driver [V]
        :return:
        '''
    # -------------------------------------------
    def calibrate_phase_eom(self, voltage_range=[-1, 1]): #TODO
        '''
         Calibrates the state generation phase EOM with respect to the induced phase rotation.
         The aim is to draw a functional dependence of the induced phase rotation on the voltage output from the
         RedPitaya that controls the phase EOM.
         It iteratively inputs a voltage to the EOM and measures the displacement of the coherent state
         from a homodyne detector.
         :param voltage_range: iterable of float (size=2)
             range of voltages output from the signal generator that feeds into the EOM driver [V]
         :return:
         '''
    # -------------------------------------------
    def calibrate_amplitude_eom(self, voltage_range=[-1, 1]): #TODO
        '''
         Calibrates the state generation amplitude EOM with respect to the induced amplitude variation.
         The aim is to draw a functional dependence of the induced amplitude variation on the voltage output from the
         RedPitaya that controls the amplitude EOM.
         It iteratively inputs a voltage to the EOM and measures the displacement of the coherent state
         from a homodyne detector.
         :param voltage_range: iterable of float (size=2)
             range of voltages output from the signal generator that feeds into the EOM driver [V]
         :return:
         '''
    def create_coherent_state(self, q, p): #TODO
        '''
        :param q: float
            amplitude quadrature in snu
        :param p: float
            phase quadrature in snu
        :return:
        '''