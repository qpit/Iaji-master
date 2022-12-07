"""
This module checks the output state for quantum scissor QKD.
"""
# In[imports]
import matplotlib.pyplot as plt
import sys
import os
import re
import numpy
import time
from time import sleep

import numpy as np
import pyrpl
import PyQt5
from pyqtgraph.Qt import QtGui
from matplotlib import pyplot
from qopt import lecroy
from scipy import signal
from scipy import optimize
from signalslot import Signal
from Iaji.SignalProcessing.Signals1D.correlator import correlator
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.QuantumStateTomography import QuadratureTomographer as Tomographer
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.SimpleHarmonicOscillator import SimpleHarmonicOscillatorNumeric
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateMeasurementController import  StateMeasurementController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateGenerator import StateGenerator

class StateChecking:
    """
    """
    # -------------------------------------------
    def __init__(self, state_measurement: StateMeasurementController, state_generator: StateGenerator ,name="State Checking Widget"):
        self.state_measurement = state_measurement
        self.state_generator = state_generator
        self.name = name
        self.acquisition = self.state_measurement.hd_controller.acquisition_system
        self.phase_controller = self.state_measurement.hd_controller.phase_controller
        self.scope = self.acquisition.scope
    # -------------------------------------------
    def _setup_acquisition_calibration(self, channels):
        '''
        Enables and renames the acquisition channels used for calibration.

        :param channels: length-4 dict of int
            channels["hd"] : channel number of the hd output
            channels["heralding"] : channel number of the heralding SSPD
            channels["sspdc"] : channel number of SSPD C
            channels["sspdd"] : channel number of SSPD D
            channels["sample-hold"] : channel number for sample-hold signal
        :return:
        '''
        acq = self.state_measurement.hd_controller.acquisition_system
        # Reset all the channels of the acquisition system
        channel_numbers = list(channels.values())
        channel_names = list(channels.keys())
        # Turn off all channels
        for name in list(acq.scope.channels.keys()):
            acq.scope.channels[name].enable(False)
        # Set new names to the selected channels
        acq.set_channel_names(channel_numbers, channel_names)
        # Turn on the selected channels
        for name in channel_names:
            acq.scope.channels[name].enable(True)
    # -------------------------------------------
    def measure_quadratures(self, phase, acquisition_channels):
        '''
        Lock HD to specified phase and acquire data from desired channels

        :param phase: float
        HD phase [degrees]
        :param acquisition_channels: channels to be recorded: dict
        :return:
        '''
        attempt = 0
        run = 0
        while attempt == 0:
            run += 1
            print('trial: %d'%run)
            acq = self.state_measurement.hd_controller.acquisition_system
            phase_controller = self.state_measurement.hd_controller.phase_controller
            # Remove the DC offset
            phase_controller.remove_offset_pid_DC()
            #Set homodyne detection phase
            phase_controller.set_phase(phase)
            #Lock the phase
            phase_controller.lock()
            #Set the file names
            channel_names = list(acquisition_channels.keys())
            phase_str = "00"*(phase < 10) + "0"*(phase<100 and phase >= 10) + str(phase)
            for channel_name in channel_names:
                acq.filenames[channel_name] = channel_name + "_" + phase_str + "deg"
            #Acquire
            try:
                acq.acquire()
                attempt = 1
            except:
                attempt = 0
                print("Didn't work")
    # ----------------------------------------
    def measure_vacuum(self, phase, channels):
        attempt = 0
        run = 0
        while attempt == 0:
            run += 1
            print('trial: %d' % run)
            acq = self.state_measurement.hd_controller.acquisition_system
            self.state_generator.pyrpl_obj_calibr.rp.asg0.output_direct = 'off'
            self.state_generator.pyrpl_obj_calibr.rp.asg1.output_direct = 'off'
            # Set the file names
            channel_names = list(channels.keys())
            phase_str = "00" * (phase < 10) + "0" * (phase < 100 and phase >= 10) + str(phase)
            for channel_name in channel_names:
                acq.filenames[channel_name] = channel_name + "_" + phase_str + "vac"
            # Acquire
            try:
                acq.acquire()
                attempt = 1
            except:
                attempt = 0
                print("Didn't work")
    # ----------------------------------------
    def extract_quadrature_measurements(self, vac, hd):
        pass
    # ----------------------------------------
    def _select_directory(self):
        '''
        Select directory for saving data
        :return:
        '''
        print('Select directory for saving data.')
        saving_path = os.getcwd()
        file_dialog_title = "Select the directory where data will be saved"
        app = PyQt5.QtWidgets.QApplication(sys.argv)
        data_path = PyQt5.QtWidgets.QFileDialog.getExistingDirectory(caption=file_dialog_title, directory=saving_path)
    # -------------------------------------------
    def _trigger_on_heralding_click(self):
        #TODO
        '''
        Function should trigger in pattern style
                                CH2 lower than 0.5 V
                                EXT higher than 1 V
        Scale should be betwenn 20 ns and 100 ns per division
        500 samples per segment
        250 MS/s
        '''
        trigger_type = "Qualified First"
        trigger_source = "C2"
        self.scope.setup_trigger(trigger_type=trigger_type, trigger_source=trigger_source)
    # -------------------------------------------
    def select_teleportation_data(self, meta, sspd_data):
        pass
    # -------------------------------------------
    def peak_detect(self, meta, sspd_data):
        """
        Find peaks by finding the minimum in the SSPD measurements even it's below a specified threshold.
        :param meta: metadata for the heralding SSPD data: dict
        :param sspd_data: list with flattened SSPD data measurements [heralding, sspdc, sspdd]: list
        :return: list with all time differences between clicks: list
        """
        # Extract information from metadata
        Ts = meta['horiz_interval']  # acquisition sampling period [s]
        time_base = [int(s) for s in re.findall(r'\d+', meta['time_base']) if s.isdigit()][0] * 10 ** (-9) #Time per division ##ns
        points_per_segment = int(time_base * 10 / step)
        points = len(heralding)
        sequences = int(points / points_per_segment)

        _heralding, _sspdc, _sspdd = sspd_data
        # Peak detect
        heralding = [x - np.max(_heralding) for x in _heralding]
        sspdc = [x - np.max(_sspdc) for x in _sspdc]
        sspdd = [x - np.max(_sspdd) for x in _sspdd]
        peak_limit = -0.8

        heralding_sspdc_time_differences = []
        heralding_sspdd_time_differences = []
        keep_data = []

        for k in range(sequences):
            begin = k * points_per_segment
            end = (k + 1) * points_per_segment
            time = np.linspace(begin, end, points_per_segment)

            heralding_peaks = []
            peak_candidate = numpy.min(heralding[begin:end])
            if peak_candidate > peak_limit:
                pass
            else:
                heralding_peaks.append(numpy.where(heralding[begin:end] == peak_candidate)[0][0])

            sspdc_peaks = []
            sspdc_index = 0
            peak_candidate = numpy.min(sspdc[begin:end])
            if peak_candidate > peak_limit:
                pass
            else:
                sspdc_peaks.append(numpy.where(sspdc[begin:end] == peak_candidate)[0][0])
                sspdc_index = 1

            sspdd_peaks = []
            sspdd_index = 0
            peak_candidate = numpy.min(sspdd[begin:end])
            if peak_candidate > peak_limit:
                pass
            else:
                sspdd_peaks.append(numpy.where(sspdd[begin:end] == peak_candidate)[0][0])
                sspdd_index = 1

            for j in range(len(heralding_peaks)):
                for i in range(len(sspdc_peaks)):
                    heralding_sspdc_time_differences.append(sspdc_peaks[i] - heralding_peaks[j])
                for i in range(len(sspdd_peaks)):
                    heralding_sspdd_time_differences.append(sspdd_peaks[i] - heralding_peaks[j])

            if (sspdc_index == 1 and sspdd_index == 0) or (sspdc_index == 0 and sspdd_index == 1):
                keep_data.append(k)
        success_rate = len(keep_data) / len(heralding)
        print("Amount of data: %d" % len(heraldin))
        print("Amount of coincidences: %d" % len(keep_data))
        print("Success rate = %.2f" % success_rate)
        return heralding_sspdc_time_differences, heralding_sspdd_time_differences
    # -------------------------------------------
    def plot_histogram(self, time_differences, bins_number, title):
        histogram = numpy.histogram(time_differences, bins=bins_number)
        max_index = np.where(histogram[0] == np.max(histogram[0]))[0][0]
        coincidence_time = 1e9 * (histogram[1][max_index] + histogram[1][max_index + 1]) / 2
        print('%d coindidence peak at %.2f ns' %(title, coincidence_time))

        plt.figure()
        plt.hist(heralding_sspdc_time_differences, bins=bins_number)
        plt.grid()
        plt.title(title)
        plt.xlabel('Time differences (s)')
        plt.ylabel('Coincidences')
        plt.text(-7e-7, 800, 'Coincidence peak at %d ns' %coincidence_time, \
                 bbox=dict(boxstyle='round', facecolor='pink'))
        plt.show()
        return coincidence_time
    # -------------------------------------------
    def time_arrival_histogram(self, bins_number = 100, channels = {"heralding": 2, "sspdc": 3, "sspdd": 4}):
        """
        #TODO
        - Recognize with sample hold is on and turn it off
        - Set scope parameters from the code _trigger_on_heralding_click

        Save SSPD data and return histogram with click time differences.
        :param bins_number: number of bins of the histogram: int
        :param channels: SSPD channels: dictionary
        :return: Time difference in nanoseconds between heralding and SSPD C clicks and heralding and SSPD D: float
        """
        # Choose directory for saving data
        self._select_directory()
        # Set up acquisition system
        self._setup_acquisition_calibration(channels=channels)
        channel_names = list(channels.keys())
        #self._trigger_on_heralding_click()
        # Record channels
        print("Recording SSPD channels for histogram")
        traces = dict(zip(channel_names, [None for j in range(len(channel_names))]))
        for name in channel_names:
            self.acquisition.filenames[name] = name + "_time_arrival"
        attempt = 0
        while attempt == 0:
            try:
                self.acquisition.acquire()
                attempt = 1
            except:
                attempt = 0
                print("Didn't work")
        for name in channel_names:
            traces[name] = self.scope.traces[name]

        # Extract data
        meta = traces[channel_names[0]][0]
        heralding = traces["heralding"][2].flatten()
        sspdc = traces["sspdc"][2].flatten()
        sspdd = traces["sspdd"][2].flatten()
        sspd_data = [heralding, sspdc, sspdd]

        #Find peaks and plot histogram
        heralding_sspdc_time_differences, heralding_sspdd_time_differences = self.peak_detect(meta, sspd_data)
        heralding_sspdc_coincidence_time = self.plot_histogram(heralding_sspdc_time_differences, bins_number, title = 'Heralding - SSPD C')
        heralding_sspdd_coincidence_time = self.plot_histogram(heralding_sspdd_time_differences, bins_number, title = 'Heralding - SSPD D')

        return heralding_sspdc_coincidence_time, heralding_sspdd_coincidence_time
    # -------------------------------------------
    def teleportation_analysis(self, phases = 15+numpy.array([0, 30, 60, 90, 120, 150]), channels={"hd": 1, "heralding": 2, "sspdc": 3, "sspdd": 4}):
        # Choose directory for saving data
        self._select_directory()
        # Set up acquisition system
        self._setup_acquisition_calibration(channels=channels)
        channel_names = list(channels.keys())
        # self._trigger_on_heralding_click()
        #Lock to different phases and acquire
        vac_channel = {"hd": 1}
        for phase in phases:
            self.measure_vacuum(phase, vac_channel)
            traces[phase][vac_channel] = self.acquisition.scope.traces[vac_channel]
            self.measure_quadratures(phase, channels)
            for name in channel_names:
                traces[phase][name] = self.acquisition.scope.traces[name]
        meta = traces[phases[0]]["heralding"][0]
        hd = dict(zip(phases))
        heralding = traces["heralding"][2].flatten()
        sspdc = traces["sspdc"][2].flatten()
        sspdd = traces["sspdd"][2].flatten()
        sspd_data = [heralding, sspdc, sspdd]
