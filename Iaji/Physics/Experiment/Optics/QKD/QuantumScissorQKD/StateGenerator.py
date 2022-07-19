"""
This module generates an input state for quantum scissor QKD.
"""
# In[imports]
import numpy
import time
import pyrpl
from Iaji.InstrumentsControl.SigilentSignalGenerator import SigilentSignalGenerator
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateMeasurementController import  StateMeasurementController
from pyqtgraph.Qt import QtGui
from matplotlib import pyplot
from qopt import lecroy

class StateGenerator:
    """
    This class controls the AOMs and define amplitude and phase modulations for the EOMs.
    """
    # -------------------------------------------
    def __init__(self, modulation_redpitaya_config_filename, \
                 calibration_redpitaya_config_filename, \
                 signal_enabler:SigilentSignalGenerator, state_measurement: StateMeasurementController, name = "State Generator"):
        '''
        :param red_pitaya_config_filename: str
            pitaya config file full path
        :param signal_generator: Iaji SigilentSignalGenerator
            signal generator object, controlling the state generation AOM driver
        :param state_measurement: Iaji StateMeasurementController
            state measurement controller object, for state calibration
        :param name: str
        '''
        self.modulation_redpitaya_config_filename = modulation_redpitaya_config_filename
        self.calibration_redpitaya_config_filename = calibration_redpitaya_config_filename
        try:
            self.connect_to_redpitayas()
        except:
            print("Fanculo I couldn't connect to the state generation Pitayas!")
        self.signal_enabler = signal_enabler #signal generator controlling the state generator AOM driver
        self.state_measurement = state_measurement
        self.name = name
        #Add the signal generator to the state measurement controller for fast vacuum quadrature measurements
        self.state_measurement.signal_enabler = self.signal_enabler
    # -------------------------------------------
    def connect_to_redpitayas(self, show_pyrpl_GUI=True):
        # Connect to modulation Pitaya
        self.pyrpl_obj_mod = pyrpl.Pyrpl(config=self.modulation_redpitaya_config_filename)
        if show_pyrpl_GUI:
            self.pyrpl_GUI_mod = QtGui.QApplication.instance()
        # Connect to calibration Pitaya
        self.pyrpl_obj_calibr = pyrpl.Pyrpl(config=self.calibration_redpitaya_config_filename)
        if show_pyrpl_GUI:
            self.pyrpl_GUI_calibr = QtGui.QApplication.instance()
    # -------------------------------------------
    def _n_level_step_function(self, frequency, levels, duty_cycles, n_points, Ts):
        assert len(levels) == len(duty_cycles), \
            'The number of levels must be equal to the number of duty cycles'
        assert numpy.sum(duty_cycles) == 1, \
            'The duty cycles must add up to 100%%'
        n_levels = len(levels)
        period = 1 / frequency
        n_period = int(numpy.floor(period / Ts))
        function_period = []
        for j in range(n_levels):
            function_period = numpy.concatenate(
                (function_period, levels[j] * numpy.ones((int(duty_cycles[j] * n_period),)), \
                 ))
        # Determine the number of periods contained in the specified number of points
        n_periods = numpy.floor(n_points / n_period)
        n_leftover = n_points % n_period
        function = numpy.repeat(function_period, n_periods)
        function = numpy.concatenate((function, function_period[:n_leftover]))
        return function
    # ------------------------------------------
    def _time_multiplexing_signals(self, max_delay, aom_levels, eom_levels, frequency, aom_duty_cycles, eom_duty_cycles):
        '''
        Generate the 3 level sample hold for the AOM and 2 level for the EOM, with the delay optimization between them.
        :return:
        '''
        asg0 = self.pyrpl_obj_calibr.rp.asg0
        asg1 = self.pyrpl_obj_calibr.rp.asg1

        asg0.output_direct = "out1"
        asg1.output_direct = "out2"

        scope = self.pyrpl_obj_calibr.rp.scope
        n_points = 2 ** 14
        d = int(numpy.ceil(125e6/(frequency*n_points)))
        scope.decimation = 2**d
        Ts = scope.decimation / 125e6

        amplification_gain = 2.5
        rp_offset = 1
        levels_3 = numpy.array(aom_levels) / amplification_gain - rp_offset
        levels_2 = numpy.array(eom_levels) / amplification_gain - rp_offset
        asg0.data = n_level_step_function(frequency, levels_3, aom_duty_cycles, n_points, Ts)
        # time.sleep(0.1)
        # Optimize the generation of the two waveforms by requiring that their (random) time delay between be lower
        # than a certain target value
        # visualize and acquire the waveforms on the scope
        scope.input1 = "asg0"
        scope.input2 = "asg1"
        delay = 2*max_delay
        while delay > max_delay:
            print("delay: %.2f ms > %.2f ms" % (delay * 1e3, max_delay * 1e3))
            asg1.setup(trigger_source="immediately")
            asg1.data = n_level_step_function(frequency, levels_2, eom_duty_cycles, n_points, Ts)
            # acquire
            traces = scope.curve()
            scope.continuous()
            trace_3 = traces[0]
            trace_2 = traces[1]
            # Compute the gradients
            gradient_3 = numpy.gradient(trace_3)
            gradient_2 = numpy.gradient(trace_2)
            # Find the first positive peak the 3-level step function
            peak_3 = signal.find_peaks(-gradient_3)[0][0]
            # Find the first negative peak the 2-level step function
            peak_2 = signal.find_peaks(gradient_2)[0][0]
            delay = abs(Ts * (peak_3 - peak_2))

        scope.input1 = "out1"
        scope.input2 = "out2"
    # -----------------------------------------
    def measure_quadrature_calibration(self, phase, acquisition_channels):
        '''
        Lock HD to specified phase and acquire data from desired channels

        :param phase: float
        HD phase [degrees]
        :return:
        '''
        acq = self.state_measurement.hd_controller.acquisition_system
        phase_controller = self.state_measurement.hd_controller.phase_controller
        #Set homodyne detection phase
        phase_controller.set_phase(phase)
        #Remove the DC offset
        phase_controller.remove_offset_pid_DC()
        #Lock the phase
        phase_controller.lock()
        #Set the file names
        channel_names = list(acquisition_channels.keys())
        phase_str = str(numpy.round(phase/1e3, 3)).split(".")[1]
        for channel_name in channel_names:
            acq.filenames[channel_name] = channel_name + "_" + phase_str + "deg"
        #Acquire
        acq.acquire()
    # ----------------------------------------
    def _setup_acquisition_calibration(self, channels):
        '''
        Enables and renames the acquisition channels used for calibration.

        :param channels: length-2 dict of int
            channels["hd"] : channel number of the hd output
            channels["time multiplexing"] : channel number of the AOM time multiplexing signal
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
    # ----------------------------------------
    def calibrate_aoms(self, voltage_range=[0, 0.1], acquisition_channels={"hd": 1, "time-multiplexing": 2}): #TODO
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
        n_points = 10
        aom_high = 5
        aom_low = 0
        aom_medium_array = numpy.linspace(voltage_range[0], voltage_range[1], n_points)
        # EOM transmits when there is no voltage and blocks when there is voltage
        eom_high = 0
        eom_low = 5
        # Define properties of time multiplexing signals
        max_delay = 1e-4
        frequency = 60
        aom_duty_cycles = [0.6, 0.2, 0.2]
        eom_duty_cycles = [0.6, 0.4]
        #Set up acquisition system
        self._setup_acquisition_calibration(channels=acquisition_channels)
        channel_names = list(acquisition_channels.keys())
        #Calibrate phase controller
        self.state_measurement.hd_controller.phase_controller.calibrate()
        #define array of displacements
        self.displacements = []
        for i in range(n_points):
            # Generate the AOM and EOMtime multiplexing signals for calibration
            aom_levels = [aom_high, aom_medium_array[i], aom_low]
            eom_levels = [eom_high, eom_low]
            self._time_multiplexing_signals(max_delay, aom_levels, eom_levels, frequency, aom_duty_cycles,
                                       eom_duty_cycles)
            #Lock the homodyne detector and measure its output, together with the AOM time multiplexing signal
            #q quadrature
            self.measure_quadrature_calibration(phase=0, acquisition_channels=acquisition_channels)
            traces_q = {channel_names[0]: acq.scope.traces[channel_names[0]], \
                        channel_names[1]: acq.scope.traces[channel_names[1]]}
            #p quadrature
            self.measure_quadrature_calibration(phase=90, acquisition_channels=acquisition_channels)
            traces_p = {channel_names[0]: acq.scope.traces[channel_names[0]], \
                        channel_names[1]: acq.scope.traces[channel_names[1]]}
            #Extract acquisition time information
            Ts = traces_q[channel_names[0]][0]['horiz_interval'] #acquisition sampling period [s]
            #Analyze q quadrature
            vac, sig = self._extract_quadrature_measurements(hd_output=traces_q[channel_names[0]], \
                                                             time_multiplexing_signal=traces_q[channel_names[0]], \
                                                             Ts=Ts, dead_time=7e-4)
            vac_std = numpy.std(vac)
            q_mean = numpy.mean(sig)/vac_std
            #Analyze p quadrature
            vac, sig = self._extract_quadrature_measurements(hd_output=traces_p[channel_names[0]], \
                                                             time_multiplexing_signal=traces_p[channel_names[0]], \
                                                             Ts=Ts, dead_time=7e-4)
            vac_std = numpy.std(vac)
            p_mean = numpy.mean(sig) / vac_std
            #Compute and save displacement
            self.displacements.append((q_mean+1j*p_mean)/numpy.sqrt(2))
        return aom_medium_array, self.displacements

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
    # -------------------------------------------
    def create_coherent_state(self, q, p): #TODO
        '''
        :param q: float
            amplitude quadrature in snu
        :param p: float
            phase quadrature in snu
        :return:
        '''
    # -------------------------------------------
    def _extract_quadrature_measurements(self, hd_output, time_multiplexing_signal, Ts, dead_time=0, measurement_time=-1,
                                        plot=False):
        '''
        Extract signal and vacuum quadrature measurements from raw homodyne detector
        output samples, based on the simultaneously acquired time-multiplexing signal.
        The time multiplexing signal has three levels:

            - HIGH: phase locking
            - MEDIUM: signal quadrature measurment
            - LOW: vacuum quadrature measurement

        It is crucial to select the part of the MEDIUM state before the signal quadrature
        drifts significantly because of the absence of phase locking.
        '''
        # Smoothen the homodyne detector output
        b = signal.firwin(501, cutoff=100, fs=1 / Ts, pass_zero="lowpass")  # band-pass filter coefficients [a.u.]
        n_samples = len(hd_output)
        n_plot = numpy.min([n_samples, int(3e6)])
        smooth = signal.filtfilt(b, 1, time_multiplexing_signal)
        gradient = numpy.gradient(smooth)
        time = Ts * numpy.arange(n_samples)
        if plot:
            pyplot.figure()
            pyplot.plot(time[:n_plot], hd_output[:n_plot], label='HD output', color="tab:red", linestyle="None",
                        marker=".", linewidth=3)
            pyplot.plot(time[:n_plot], gradient[:n_plot], label='time multiplexing signal (smoothened and gradient)',
                        color="tab:green", linestyle="--", linewidth=3)
            pyplot.legend(loc='upper right')
            pyplot.xlabel("time (s)")
            pyplot.grid()
        # Detect all the egdes in the time-multiplexing signal
        positive_peak_indices, properties = signal.find_peaks(gradient, height=0.1 * numpy.max(gradient))
        positive_peak_heights = properties["peak_heights"]

        negative_peak_indices, properties = signal.find_peaks(-gradient, height=0.1 * numpy.max(-gradient))
        negative_peak_heights = -properties["peak_heights"]

        if plot:
            pyplot.plot(positive_peak_indices * Ts, positive_peak_heights, linestyle="None", marker="x",
                        color="tab:green", markersize=10)
            pyplot.plot(negative_peak_indices * Ts, negative_peak_heights, linestyle="None", marker="x",
                        color="tab:green", markersize=10)
        '''
        Periods of the 3-level time-multiplexing signal start with one positive peak
        For every positive peak, collect the signal and vacuum quadrature measurements, 
        taking care of correcting the offset of the signal based on the vacuum quadrature
        measured in the same period.
        '''
        vac, sig, time_vac, time_sig = ([], [], [], [])
        dead_samples = int(dead_time / Ts)
        for j in range(len(positive_peak_indices[1:-1])):
            start, end = (positive_peak_indices[j], positive_peak_indices[j + 1])
            peak1, peak2 = negative_peak_indices[numpy.where(numpy.logical_and( \
                negative_peak_indices > start, \
                negative_peak_indices < end))]
            vac_temp = hd_output[peak2 + dead_samples:end - dead_samples]
            if plot:
                time_vac_temp = time[peak2 + dead_samples:end - dead_samples]
                time_vac = numpy.concatenate((time_vac, time_vac_temp))

            if measurement_time == -1:
                measurement_stop = peak2 - dead_samples
            else:
                measurement_stop = int(measurement_time / Ts)
            sig_temp = hd_output[peak1 + dead_samples:measurement_stop]
            if plot:
                time_sig_temp = time[peak1 + dead_samples:measurement_stop]
                time_sig = numpy.concatenate((time_sig, time_sig_temp))
            # Correct for the offset
            vac_mean = numpy.mean(vac_temp)
            sig_temp -= vac_mean
            vac_temp -= vac_mean
            # Store the traces
            vac = numpy.concatenate((vac, vac_temp))
            sig = numpy.concatenate((sig, sig_temp))
        if plot:
            pyplot.plot(time_vac[:n_plot], vac[:n_plot], color="tab:blue", label="vacuum quadrature samples",
                        linestyle="None", marker=".", linewidth=3)
            pyplot.plot(time_sig[:n_plot], sig[:n_plot], color="tab:purple", label="signal quadrature samples",
                        linestyle="None", marker=".", linewidth=3)
            pyplot.legend(loc="upper right")
        return vac, sig