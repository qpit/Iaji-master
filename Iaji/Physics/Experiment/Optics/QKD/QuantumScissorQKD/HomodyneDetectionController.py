"""
This class describes a homodyne Detection used in the experiment.
It consists of:
    - A phase controller, associated to a RedPitaya that controls the interferometer's phase
    - An acquisition system, that acquires the output signals from the homodyne detector
"""
#%%
import pyrpl
from pyqtgraph.Qt import QtGui
from .Exceptions import ConnectionError, ResonanceNotFoundError
from .PhaseController import PhaseController
import numpy
import time
#%%
print_separator = "---------------------------------------------"
#%%
class HomodyneDetectionController:
    #-----------------------------------
    def __init__(self, phase_controller: PhaseController, acquisition_system, DC_channel=1, AC_channel=2, name="Homodyne Detection Controller"):
        """
        :param phase_controller: Iaji QuantumScissorQKD PhaseController
            object that controls the phase of the interferometer
        :param acquisition_system: Iaji QuantumScissorQKD AcquisitionSystem
            object that acquires the signals output from the homodyne detector
        :param name: str
            name of the homodyne detection controller
        """
        self.phase_controller = phase_controller
        self.acquisition_system = acquisition_system
        self.name = name
        #Set DC and AC channels
        self.DC_channel_number, self.AC_channel_number = (None, None)
        self.set_DC_channel(DC_channel)
        self.set_AC_channel(AC_channel)
    # -----------------------------------
    def set_DC_channel(self, channel_number):
        channel_names = list(self.acquisition_system.scope.channels.keys())
        # Reset the name of the old channel
        if self.DC_channel_number is not None:
            channel_name_old = channel_names[self.DC_channel_number-1].split("_DC")[0]
            print(channel_name_old)
            self.acquisition_system.set_channel_names(self.DC_channel_number, channel_name_old)
            # Disable the old channel
            self.acquisition_system.scope.channels[channel_name_old].enable(False)
        #Set the name of the new channel
        self.DC_channel_number = channel_number
        channel_names = list(self.acquisition_system.scope.channels.keys())
        channel_name_new = channel_names[self.DC_channel_number - 1] + "_DC"
        self.acquisition_system.set_channel_names(self.DC_channel_number, channel_name_new)
        self.acquisition_system.scope.channels[channel_name_new].enable(True)
    # -----------------------------------
    def set_AC_channel(self, channel_number):
        channel_names = list(self.acquisition_system.scope.channels.keys())
        # Reset the name of the old channel
        if self.AC_channel_number is not None:
            channel_name_old = channel_names[self.AC_channel_number-1].split("_AC")[0]
            self.acquisition_system.set_channel_names(self.AC_channel_number, channel_name_old)
            #Disable the old channel
            self.acquisition_system.scope.channels[channel_name_old].enable(False)
        # Set the name of the new channel
        self.AC_channel_number = channel_number
        channel_names = list(self.acquisition_system.scope.channels.keys())
        channel_name_new = channel_names[self.AC_channel_number - 1] + "_AC"
        self.acquisition_system.set_channel_names(self.AC_channel_number, channel_name_new)
        self.acquisition_system.scope.channels[channel_name_new].enable(True)
    # -----------------------------------
    def measure_quadrature(self, phase):
        """
        Peforms a single quadrature measurement
        :param phase: float
            phase of the interferometer, along which the quadrature measurement is made [deg].

        :return
            traces acquired by the acquisition system.
            The channels are specified by custom names and ordered from first to last.
        """
        phase = numpy.round(phase, 1)
        if not(numpy.isclose(numpy.round(self.phase_controller.phase*180/numpy.pi, 1), phase)):
            self.phase_controller.set_phase(phase)
        channel_names = list(self.acquisition_system.scope.channels.keys())
        channel_numbers = [self.AC_channel_number]
        channel_numbers.sort()
        filenames = []
        for channel_number in channel_numbers:
            filenames.append(self.acquisition_system.filenames[channel_names[channel_number - 1]])
            filenames[-1] += "_%d"%phase
        #Deactivate DC channel
        self.acquisition_system.scope.channels[channel_names[self.DC_channel_number - 1]].enable(False)
        #Lock the phase
        self.phase_controller.lock()
        #Acquire
        traces = self.acquisition_system.acquire(filenames)
        self.acquisition_system.scope.channels[channel_names[self.DC_channel_number - 1]].enable(True)
        return traces
    # -----------------------------------
    def measure_vacuum(self):
        """
        Peforms a single vacuum quadrature measurement
        :return
            traces acquired by the acquisition system.
            The channels are specified by custom names and ordered from first to last.
        """
        channel_names = list(self.acquisition_system.scope.channels.keys())
        channel_names = list(self.acquisition_system.scope.channels.keys())
        channel_numbers = [self.AC_channel_number]
        channel_numbers.sort()
        filenames = []
        for channel_number in channel_numbers:
            filenames.append(self.acquisition_system.filenames[channel_names[channel_number - 1]])
            filenames[-1] += "_vacuum"
            # Deactivate DC channel
            self.acquisition_system.scope.channels[channel_names[self.DC_channel_number - 1]].enable(False)
            # Acquire
            traces = self.acquisition_system.acquire(filenames)
            self.acquisition_system.scope.channels[channel_names[self.DC_channel_number - 1]].enable(True)
            return traces
    # -----------------------------------
    def measure_electronic_noise(self):
        """
        Peforms a single electronic noise measurement
        :return
            traces acquired by the acquisition system.
            The channels are specified by custom names and ordered from first to last.
        """
        channel_names = list(self.acquisition_system.scope.channels.keys())
        channel_names = list(self.acquisition_system.scope.channels.keys())
        channel_numbers = [self.AC_channel_number]
        channel_numbers.sort()
        filenames = []
        for channel_number in channel_numbers:
            filenames.append(self.acquisition_system.filenames[channel_names[channel_number - 1]])
            filenames[-1] += "_electronic-noise"
            # Deactivate DC channel
            self.acquisition_system.scope.channels[channel_names[self.DC_channel_number - 1]].enable(False)
            # Acquire
            traces = self.acquisition_system.acquire(filenames)
            self.acquisition_system.scope.channels[channel_names[self.DC_channel_number - 1]].enable(True)
            return traces









