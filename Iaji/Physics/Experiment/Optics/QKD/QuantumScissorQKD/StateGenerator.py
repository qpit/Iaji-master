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
from scipy import signal
from signalslot import Signal
from Iaji.SignalProcessing.Signals1D.correlator import correlator
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.QuantumStateTomography import QuadratureTomographer as Tomographer
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.SimpleHarmonicOscillator import SimpleHarmonicOscillatorNumeric

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
        self.calibration_stopped = False
        self.amplitude_eom_low = None
    # -------------------------------------------
    def connect_to_redpitayas(self, show_pyrpl_GUI=True):
        # Connect to modulation Pitaya
        self.pyrpl_obj_mod = pyrpl.Pyrpl(config=self.modulation_redpitaya_config_filename)
        if show_pyrpl_GUI:
            self.pyrpl_GUI_mod = QtGui.QApplication.instance()
        for channel in [0, 1]:
            getattr(self.pyrpl_obj_mod.rp, "asg%d"%channel).setup(waveform='dc', offset=0, trigger_source='immediately')
            getattr(self.pyrpl_obj_mod.rp, "asg%d" % channel).output_direct = "off"*(channel==0) + "out%d"%(channel+1)*(channel==1)
        # Connect to AOMs and amplitude EOM bias calibration Pitaya
        self.pyrpl_obj_calibr = pyrpl.Pyrpl(config=self.calibration_redpitaya_config_filename)
        self.pyrpl_obj_calibr.rp.asg0.offset = 0
        self.pyrpl_obj_calibr.rp.asg1.offset = 0
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
        scope.decimation = int(numpy.ceil(125e6/(frequency*n_points)))
        Ts = scope.decimation / 125e6

        amplification_gain = 2.5
        rp_offset = 1
        aom_levels = numpy.array(aom_levels) / amplification_gain - rp_offset
        eom_levels = numpy.array(eom_levels) / amplification_gain - rp_offset
        asg0.data = self._n_level_step_function(frequency, aom_levels, aom_duty_cycles, n_points, Ts)
        # time.sleep(0.1)
        # Optimize the generation of the two waveforms by requiring that their (random) time delay between be lower
        # than a certain target value
        # visualize and acquire the waveforms on the scope
        scope.input1 = "asg0"
        scope.input2 = "asg1"
        delay = 2*max_delay
        print("Preparing the time-multiplexing signals for state calibration")
        while abs(delay) > max_delay or delay < 0:
            if self.calibration_stopped:
                self.calibration_stopped = False
                return
            #print("delay: %.2f ms > %.2f ms" % (delay * 1e3, max_delay * 1e3))
            asg1.setup(trigger_source="immediately")
            asg1.data = self._n_level_step_function(frequency, eom_levels, eom_duty_cycles, n_points, Ts)
            # acquire
            traces = scope.curve()
            scope.continuous()
            trace_aom = traces[0]
            trace_eom = traces[1]
            '''
            corr_obj = correlator(signal_1=numpy.gradient(trace_aom), signal_2=-numpy.gradient(trace_eom), sampling_period=Ts)
            delay = abs(corr_obj.getDelay(absolute_correlation=False))
            '''
            #Find the delay between the last samples of the aom and eom (either high or medium) levels
            for j in range(3):
                aom_index = numpy.where(numpy.isclose(trace_aom, aom_levels[j], atol=1e-3))[0][0]
                eom_index = numpy.where(numpy.isclose(trace_eom, eom_levels[j], atol=1e-3))[0][0]
                if aom_index != 0 and eom_index != 0:
                    delay = (eom_index - aom_index) * Ts
                    break


        scope.input1 = "out1"
        scope.input2 = "out2"
        print("delay: %.2f ms < %.2f ms" % (delay * 1e3, max_delay * 1e3))
        #Debug plotting
        '''
        pyplot.figure()
        pyplot.plot(corr_obj.lags, corr_obj.correlation_function, marker="o", linestyle="None")

        pyplot.plot(Ts*numpy.arange(len(corr_obj.signal_1)), corr_obj.signal_1, color="tab:orange", linestyle="--")
        pyplot.plot(Ts * numpy.arange(len(trace_aom)), trace_aom, color="tab:orange")

        pyplot.plot(Ts * numpy.arange(len(corr_obj.signal_2)), -corr_obj.signal_2, color="tab:green", linestyle="--")
        pyplot.plot(Ts * numpy.arange(len(trace_eom)), trace_eom, color="tab:green")
        '''
    # -----------------------------------------
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
        b = signal.firwin(501, cutoff=30, fs=1 / Ts, pass_zero="lowpass")  # band-pass filter coefficients [a.u.]
        tm_smooth = signal.filtfilt(b, 1, time_multiplexing_signal)
        #Align the HD output with the time multiplexing signal in time
        corr = correlator(signal_1=tm_smooth, signal_2=numpy.abs(hd_output), sampling_period=Ts)
        delay = corr.getDelay()
        tm_smooth, hd_output = self._delay(tm_smooth, hd_output, delay, Ts)
        n_samples = len(hd_output)
        time = Ts*numpy.arange(n_samples)

        n_plot = numpy.min([n_samples, int(5e6)])

        gradient = numpy.gradient(tm_smooth)
        if plot:
            pyplot.figure()
            pyplot.plot(time[:n_plot:10], hd_output[:n_plot:10], label='HD output', color="tab:red", linestyle="None",
                        marker=".", linewidth=3)
            pyplot.plot(time[:n_plot:10], tm_smooth[:n_plot:10], label='time multiplexing signal (smoothened)',
                        color="tab:green",  linestyle="None",
                        marker=".", linewidth=3)
            pyplot.plot(time[:n_plot:10], 1e3*gradient[:n_plot:10], label='time multiplexing signal (smoothened and gradient)',
                       color="tab:green", linestyle="--", linewidth=3)
            pyplot.legend(loc='upper right')
            pyplot.xlabel("time (s)")
            pyplot.grid()
        # Detect all the egdes in the time-multiplexing signal
        positive_peak_indices, properties = signal.find_peaks(gradient, height=0.01 * numpy.max(gradient))
        positive_peak_heights = properties["peak_heights"]
        good_indices = numpy.where(positive_peak_heights > .8*numpy.max(positive_peak_heights))
        positive_peak_indices = positive_peak_indices[good_indices]
        positive_peak_heights = positive_peak_heights[good_indices]

        negative_peak_indices, properties = signal.find_peaks(-gradient, height=0.01 * numpy.max(-gradient))
        negative_peak_heights = -properties["peak_heights"]
        good_indices = numpy.where(numpy.logical_or(negative_peak_heights < .8*numpy.min(negative_peak_heights), \
                                                     negative_peak_heights > .2*numpy.min(negative_peak_heights)))
        negative_peak_indices = negative_peak_indices[good_indices]
        negative_peak_heights = negative_peak_heights[good_indices]

        if plot:
            pyplot.plot(positive_peak_indices * Ts, 1e3*positive_peak_heights, linestyle="None", marker="x",
                        color="tab:green", markersize=10)
            pyplot.plot(negative_peak_indices * Ts, 1e3*negative_peak_heights, linestyle="None", marker="x",
                        color="tab:green", markersize=10)
        '''
        Periods of the 3-level time-multiplexing signal start with one positive peak
        For every positive peak, collect the signal and vacuum quadrature measurements, 
        taking care of correcting the offset of the signal based on the vacuum quadrature
        measured in the same period.
        '''
        vac, sig, time_vac, time_sig = ([], [], [], [])
        dead_samples = int(dead_time / Ts)
        end_dead_samples = int(1.5e-3/Ts)
        for j in range(len(positive_peak_indices[1:-1])):
            if self.calibration_stopped:
                self.calibration_stopped = False
                return
            start, end = (positive_peak_indices[j], positive_peak_indices[j + 1])
            neg_peak_indices =  negative_peak_indices[numpy.where(numpy.logical_and( \
                negative_peak_indices > start, \
                negative_peak_indices < end))]
            #Debug printing
           # print("(start time, end time) = (%f, %f)"%(start*Ts, end*Ts))
           # print("negative peak times: %s"%(numpy.array(neg_peak_indices)*Ts))
            peak1, peak2 = (neg_peak_indices[0], neg_peak_indices[1])
            vac_temp = hd_output[peak2 + dead_samples:end - end_dead_samples]
            if plot:
                time_vac_temp = time[peak2 + dead_samples:end - end_dead_samples]
                time_vac = numpy.concatenate((time_vac, time_vac_temp))

            if measurement_time == -1:
                measurement_stop = peak2
            else:
                measurement_stop = int(measurement_time / Ts)
            sig_temp = hd_output[peak1 + dead_samples:measurement_stop - end_dead_samples]
            if plot:
                time_sig_temp = time[peak1 + dead_samples:measurement_stop - end_dead_samples]
                time_sig = numpy.concatenate((time_sig, time_sig_temp))
            # Correct for the offset
            vac_mean = numpy.mean(vac_temp)
            sig_temp -= vac_mean
            vac_temp -= vac_mean
            # Store the traces
            vac = numpy.concatenate((vac, vac_temp))
            sig = numpy.concatenate((sig, sig_temp))
        if plot:
            pyplot.plot(time_vac[:n_plot:10], vac[:n_plot:10], color="tab:blue", label="vacuum quadrature samples",
                        linestyle="None", marker=".", linewidth=3)
            pyplot.plot(time_sig[:n_plot:10], sig[:n_plot:10], color="tab:purple", label="signal quadrature samples",
                        linestyle="None", marker=".", linewidth=3)
            pyplot.legend(loc="upper right")
        return vac, sig

    # -------------------------------------------
    def find_low_eom_transmission(self, voltage_range=[2.5, 5], n_points=100):
        '''
        Scan EOM voltage and return value with minimum state transmission

        :return:
        '''
        print("Finding the minimum transmission bias voltage for the amplitude EOM")
        asg_aom = self.pyrpl_obj_calibr.rp.asg0
        asg_eom = self.pyrpl_obj_calibr.rp.asg1

        #Set aom in a locking state
        amplification_gain = 2.5
        rp_offset = 1
        asg_aom.setup(waveform='dc', offset=5/amplification_gain-rp_offset, trigger_source='immediately')
        asg_aom.output_direct = "out1"
        #Find minimum transmission
        asg_eom.setup(waveform='dc', offset=2.5/amplification_gain-rp_offset, trigger_source='immediately')
        asg_eom.output_direct = "out2"
        voltages = numpy.linspace(*voltage_range, n_points)
        amplitudes = numpy.zeros((n_points,))
        #Scan LO phase
        self.state_measurement.hd_controller.phase_controller.scan()
        for j in numpy.arange(n_points):
            voltage = voltages[j]
            asg_eom.offset = voltage/amplification_gain-rp_offset
            amplitudes[j] = self.state_measurement.hd_controller.phase_controller.get_signal_amplitude()
        asg_eom.offset = 0
        #Find minimum transmission voltage
        self.state_measurement.hd_controller.phase_controller.turn_off_scan()
        self.amplitude_eom_low = voltages[numpy.argmin(amplitudes)]
        return self.amplitude_eom_low
    # -------------------------------------------
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
        phase_str = "00"*(phase < 10) + "0"*(phase<100 and phase >= 10) + str(phase)
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
    def calibrate_aoms(self, voltage_range=[1e-2, 0.1], n_points = 6, acquisition_channels={"hd": 1, "time-multiplexing": 2}): #TODO
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
        acq = self.state_measurement.hd_controller.acquisition_system
        aom_high = 5
        aom_low = 0
        aom_medium_array = numpy.linspace(voltage_range[0], voltage_range[1], n_points)
        # EOM transmits when there is no voltage and blocks when there is voltage
        eom_high = 2.5
        if not self.amplitude_eom_low:
            self.find_low_eom_transmission()
        eom_low = self.amplitude_eom_low
        eom_medium = self.amplitude_eom_voltage
        # Define properties of time multiplexing signals
        max_delay = 1e-4
        frequency = 60
        aom_duty_cycles = [0.6, 0.2, 0.2]
        eom_duty_cycles = [0.6, 0.2, 0.2]
        #Set up acquisition system
        self._setup_acquisition_calibration(channels=acquisition_channels)
        channel_names = list(acquisition_channels.keys())
        #Calibrate phase controller
        self.pyrpl_obj_calibr.rp.asg0.waveform = "dc"
        self.pyrpl_obj_calibr.rp.asg0.offset = 1
        self.state_measurement.hd_controller.phase_controller.calibrate()
        self.pyrpl_obj_calibr.rp.asg0.offset = 0
        #define array of displacements
        self.displacements = []
        for i in range(n_points):
            if self.calibration_stopped:
                self.calibration_stopped = False
                return
            # Generate the AOM and EOMtime multiplexing signals for calibration
            aom_levels = [aom_high, aom_medium_array[i], aom_low]
            eom_levels = [eom_high, eom_medium, eom_low]
            self._time_multiplexing_signals(max_delay, aom_levels, eom_levels, frequency, aom_duty_cycles,
                                       eom_duty_cycles)
            #Lock the homodyne detector and measure its output, together with the AOM time multiplexing signal
            phase = -45
            #generalized q quadrature
            self.measure_quadrature_calibration(phase=phase, acquisition_channels=acquisition_channels)
            traces_q = {channel_names[0]: acq.scope.traces[channel_names[0]], \
                        channel_names[1]: acq.scope.traces[channel_names[1]]}
            #generalized p quadrature
            self.measure_quadrature_calibration(phase=phase+90, acquisition_channels=acquisition_channels)
            traces_p = {channel_names[0]: acq.scope.traces[channel_names[0]], \
                        channel_names[1]: acq.scope.traces[channel_names[1]]}
            #Extract acquisition time information
            Ts = traces_q[channel_names[0]][0]['horiz_interval'] #acquisition sampling period [s]
            #Analyze generalized q quadrature
            vac, sig = self._extract_quadrature_measurements(hd_output=traces_q[channel_names[0]][2][0, :], \
                                                             time_multiplexing_signal=traces_q[channel_names[1]][2][0, :], \
                                                             Ts=Ts, dead_time=0.5e-3, plot = False)
            vac_std = numpy.std(vac)
            q_mean = numpy.mean(sig)/(vac_std / (1/numpy.sqrt(2)))
            #Analyze generalized p quadrature
            vac, sig = self._extract_quadrature_measurements(hd_output=traces_p[channel_names[0]][2][0, :], \
                                                             time_multiplexing_signal=traces_p[channel_names[1]][2][0, :], \
                                                             Ts=Ts, dead_time=0.5e-3, plot = False)
            vac_std = numpy.std(vac)
            p_mean = numpy.mean(sig) / (vac_std / (1/numpy.sqrt(2)))
            phase_rad = phase*numpy.pi/180
            #Compute and save displacement
            self.displacements.append(numpy.exp(1j*phase_rad)/numpy.sqrt(2)*(q_mean+1j*p_mean))
        return aom_medium_array, self.displacements

    # -------------------------------------------
    def calibrate_phase_eom(self, voltage_range=[-1, 1], n_points = 6, acquisition_channels={"hd": 1, "time-multiplexing": 2}): #TODO
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
        acq = self.state_measurement.hd_controller.acquisition_system
        aom_high = 5
        aom_low = 0
        aom_medium = self.aoms_voltage
        #Amplitude EOM
        ## EOM transmits when there is no voltage and blocks when there is voltage
        amplitude_eom_high = 2.5
        amplitude_eom_medium = self.amplitude_eom_voltage
        if not self.amplitude_eom_low:
            self.find_low_eom_transmission()
        amplitude_eom_low = self.amplitude_eom_low
        #Phase EOM
        phase_eom_level_array = numpy.linspace(*voltage_range, n_points)
        amplification_gain = 10
        rp_offset = 0
        # Define properties of time multiplexing signals
        max_delay = 1e-4
        frequency = 60
        aom_duty_cycles = [0.6, 0.2, 0.2]
        eom_duty_cycles = [0.6, 0.2, 0.2]
        #Set up acquisition system
        self._setup_acquisition_calibration(channels=acquisition_channels)
        channel_names = list(acquisition_channels.keys())
        #Calibrate phase controller
        self.pyrpl_obj_calibr.rp.asg0.waveform = "dc"
        self.pyrpl_obj_calibr.rp.asg0.offset = 1
        self.state_measurement.hd_controller.phase_controller.calibrate()
        self.pyrpl_obj_calibr.rp.asg0.offset = 0
        #define array of displacements
        self.displacements = []
        for i in range(n_points):
            if self.calibration_stopped:
                self.calibration_stopped = False
                return
            # Generate the AOM and EOMtime multiplexing signals for calibration
            aom_levels = [aom_high, aom_medium, aom_low]
            amplitude_eom_levels = [amplitude_eom_high, amplitude_eom_medium, amplitude_eom_low]
            self._time_multiplexing_signals(max_delay, aom_levels, amplitude_eom_levels, frequency, aom_duty_cycles,
                                       eom_duty_cycles)
            # Send voltage to phase modulator
            self.pyrpl_obj_mod.rp.asg1.offset = phase_eom_level_array[i]/amplification_gain - rp_offset
            #self.signal_enabler.channels["C1"].offset =  phase_eom_level_array[i]
            #Lock the homodyne detector and measure its output, together with the AOM time multiplexing signal
            phase = -45
            #generalized q quadrature
            self.measure_quadrature_calibration(phase=phase, acquisition_channels=acquisition_channels)
            traces_q = {channel_names[0]: acq.scope.traces[channel_names[0]], \
                        channel_names[1]: acq.scope.traces[channel_names[1]]}
            #generalized p quadrature
            self.measure_quadrature_calibration(phase=phase+90, acquisition_channels=acquisition_channels)
            traces_p = {channel_names[0]: acq.scope.traces[channel_names[0]], \
                        channel_names[1]: acq.scope.traces[channel_names[1]]}
            #Extract acquisition time information
            Ts = traces_q[channel_names[0]][0]['horiz_interval'] #acquisition sampling period [s]
            #Analyze generalized q quadrature
            vac, sig = self._extract_quadrature_measurements(hd_output=traces_q[channel_names[0]][2][0, :], \
                                                             time_multiplexing_signal=traces_q[channel_names[1]][2][0, :], \
                                                             Ts=Ts, dead_time=0.15e-3, plot = False)
            vac_std = numpy.std(vac)
            q_mean = numpy.mean(sig)/(vac_std / (1/numpy.sqrt(2)))
            #Analyze generalized p quadrature
            vac, sig = self._extract_quadrature_measurements(hd_output=traces_p[channel_names[0]][2][0, :], \
                                                             time_multiplexing_signal=traces_p[channel_names[1]][2][0, :], \
                                                             Ts=Ts, dead_time=0.15e-3, plot = False)
            vac_std = numpy.std(vac)
            p_mean = numpy.mean(sig) / (vac_std / (1/numpy.sqrt(2)))
            phase_rad = phase*numpy.pi/180
            #Compute and save displacement
            self.displacements.append(numpy.exp(1j*phase_rad)/numpy.sqrt(2)*(q_mean+1j*p_mean))
        return phase_eom_level_array, self.displacements
    # -------------------------------------------
    def calibrate_amplitude_eom(self, voltage_range = [5, 0], n_points = 6, acquisition_channels={"hd": 1, "time-multiplexing": 2}): #TODO
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
        acq = self.state_measurement.hd_controller.acquisition_system
        aom_high = 5
        aom_low = 0
        aom_medium = self.aoms_voltage
        # EOM transmits when there is no voltage and blocks when there is voltage
        eom_high = 2.5
        eom_medium_array = numpy.linspace(voltage_range[0], voltage_range[1], n_points)
        if not self.amplitude_eom_low:
            self.find_low_eom_transmission()
        eom_low = self.amplitude_eom_low
        # Define properties of time multiplexing signals
        max_delay = 1e-4
        frequency = 60
        aom_duty_cycles = [0.6, 0.2, 0.2]
        eom_duty_cycles = [0.6, 0.2, 0.2]
        #Set up acquisition system
        self._setup_acquisition_calibration(channels=acquisition_channels)
        channel_names = list(acquisition_channels.keys())
        #Calibrate phase controller
        self.pyrpl_obj_calibr.rp.asg0.waveform = "dc"
        self.pyrpl_obj_calibr.rp.asg0.offset = 1
        self.state_measurement.hd_controller.phase_controller.calibrate()
        self.pyrpl_obj_calibr.rp.asg0.offset = 0
        #define array of displacements
        self.displacements = []
        for i in range(n_points):
            if self.calibration_stopped:
                self.calibration_stopped = False
                return
            # Generate the AOM and EOM time multiplexing signals for calibration
            aom_levels = [aom_high, aom_medium, aom_low]
            eom_levels = [eom_high, eom_medium_array[i], eom_low]
            self._time_multiplexing_signals(max_delay, aom_levels, eom_levels, frequency, aom_duty_cycles,
                                       eom_duty_cycles)
            #Lock the homodyne detector and measure its output, together with the AOM time multiplexing signal
            phase = -45
            #generalized q quadrature
            self.measure_quadrature_calibration(phase=phase, acquisition_channels=acquisition_channels)
            traces_q = {channel_names[0]: acq.scope.traces[channel_names[0]], \
                        channel_names[1]: acq.scope.traces[channel_names[1]]}
            #generalized p quadrature
            self.measure_quadrature_calibration(phase=phase+90, acquisition_channels=acquisition_channels)
            traces_p = {channel_names[0]: acq.scope.traces[channel_names[0]], \
                        channel_names[1]: acq.scope.traces[channel_names[1]]}
            #Extract acquisition time information
            Ts = traces_q[channel_names[0]][0]['horiz_interval'] #acquisition sampling period [s]
            #Analyze generalized q quadrature
            vac, sig = self._extract_quadrature_measurements(hd_output=traces_q[channel_names[0]][2][0, :], \
                                                             time_multiplexing_signal=traces_q[channel_names[1]][2][0, :], \
                                                             Ts=Ts, dead_time=0.5e-3, plot = False)
            vac_std = numpy.std(vac)
            q_mean = numpy.mean(sig)/(vac_std / (1/numpy.sqrt(2)))
            #Analyze generalized p quadrature
            vac, sig = self._extract_quadrature_measurements(hd_output=traces_p[channel_names[0]][2][0, :], \
                                                             time_multiplexing_signal=traces_p[channel_names[1]][2][0, :], \
                                                             Ts=Ts, dead_time=0.5e-3, plot = False)
            vac_std = numpy.std(vac)
            p_mean = numpy.mean(sig) / (vac_std / (1/numpy.sqrt(2)))
            phase_rad = phase*numpy.pi/180
            #Compute and save displacement
            self.displacements.append(numpy.exp(1j*phase_rad)/numpy.sqrt(2)*(q_mean+1j*p_mean))
        return eom_medium_array, self.displacements
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
    def tomography_mesaurement_for_calibration(self, aom_voltage, amplitude_eom_voltage, phase_eom_voltage, phases=15+numpy.array([0, 30, 60, 90, 120, 150]), \
                                               acquisition_channels={"hd": 1, "time-multiplexing": 2}):
        '''
        Performs a homodyne tomography measurement with the time-multiplexing sch

        :param phases:
        :return:
        '''
        phases = numpy.array(phases)
        acq = self.state_measurement.hd_controller.acquisition_system
        aom_high = 5
        aom_low = 0
        aom_medium = aom_voltage
        # EOM transmits when there is no voltage and blocks when there is voltage
        amplitude_eom_high = 2.5
        if not self.amplitude_eom_low:
            self.find_low_eom_transmission()
        amplitude_eom_low = self.amplitude_eom_low
        amplitude_eom_medium = amplitude_eom_voltage
        # Define properties of time multiplexing signals
        max_delay = 1e-4
        frequency = 60
        aom_duty_cycles = [0.6, 0.2, 0.2]
        eom_duty_cycles = [0.6, 0.2, 0.2]
        #Set up acquisition system
        self._setup_acquisition_calibration(channels=acquisition_channels)
        channel_names = list(acquisition_channels.keys())
        #Calibrate phase controller
        self.pyrpl_obj_calibr.rp.asg0.waveform = "dc"
        self.pyrpl_obj_calibr.rp.asg0.offset = 1
        self.state_measurement.hd_controller.phase_controller.calibrate()
        self.pyrpl_obj_calibr.rp.asg0.offset = 0
        #Generate time-multiplexing signals for AOM and amplitude EOM
        aom_levels = [aom_high, aom_medium, aom_low]
        eom_levels = [amplitude_eom_high, amplitude_eom_medium, amplitude_eom_low]
        self._time_multiplexing_signals(max_delay, aom_levels, eom_levels, frequency, aom_duty_cycles,
                                        eom_duty_cycles)
        #Define data dictionaries
        vac, sig = [dict(zip(phases, [None for j in range(len(phases))])) for s in range(2)]
        traces = dict(zip(phases, [dict(zip(channel_names, [None for j in range(len(channel_names))])) for s in range(len(phases))]))
        print("Calibration tomography: measuring quadratures")
        for phase in phases:
            if self.calibration_stopped:
                self.calibration_stopped = False
                return
            self.measure_quadrature_calibration(phase=phase, acquisition_channels=acquisition_channels)
            traces[phase][channel_names[0]] = acq.scope.traces[channel_names[0]]
            traces[phase][channel_names[1]] = acq.scope.traces[channel_names[1]]
        # Extract acquisition time information
        Ts = traces[phases[0]][channel_names[0]][0]['horiz_interval']  # acquisition sampling period [s]
        # Analyze quadratures
        print("Calibration tomography: analyzing quadrature measurements")
        for phase in phases:
            vac[phase], sig[phase] = self._extract_quadrature_measurements(hd_output=traces[phase][channel_names[0]][2][0, :], \
                                                             time_multiplexing_signal=traces[phase][channel_names[1]][2][0,:], \
                                                             Ts=Ts, dead_time=0.5e-3, plot= False)

        #%%
        # Plot the raw power spectral densities
        figure_PSD_raw = pyplot.figure()

        figure_PSD_raw.subplots_adjust(wspace=0.4, hspace=0.7)
        axis = figure_PSD_raw.add_subplot(111)
        axis.set_xlabel("frequency (MHz)")
        axis.set_ylabel("power spectral density (dBm/Hz)")
        axis.grid(True)
        PSDs_raw = {}
        # Plot electronic and vacuum noise
        '''
        frequency, PSD = signal.welch(electronic_noise, fs=1/Ts, nperseg=len(electronic_noise)/100, noverlap=20)
        axis.plot(frequency*1e-6, 10*numpy.log10(abs(PSD)/1e-3), label='electronic noise', marker=default_marker)
        PSDs_raw['electronic noise'] = PSD
        '''
        frequency, PSD = signal.welch(vac[phases[0]], fs=1 / Ts, nperseg=len(vac[phases[0]]) / 100, noverlap=20)
        axis.plot(frequency * 1e-6, 10 * numpy.log10(abs(PSD) / 1e-3), label='vacuum quadrature', marker=".")
        PSDs_raw['vacuum quadrature'] = PSD

        PSDs_raw['frequency'] = frequency

        for phase in phases:
            trace = sig[phase]
            frequency, PSD = signal.welch(trace, fs=1 / Ts, nperseg=len(trace) / 100, noverlap=20)
            axis.plot(frequency * 1e-6, 10 * numpy.log10(abs(PSD) / 1e-3), label="phase = %.1f$^\\circ$" % phase,
                      marker=".")
            PSDs_raw[phase] = PSD

        axis.legend(loc='upper right')
        #%%

        # Do tomography
        ## Prefilter the data
        b = signal.firwin(501, cutoff=20e6, fs=1 / Ts, pass_zero="lowpass")
        for phase in phases:
            sig[phase] = signal.filtfilt(b, 1, sig[phase])
            vac[phase] = signal.filtfilt(b, 1, vac[phase])
        ## Prepare data in the form of an array
        n_samples_sig = numpy.min([sig[phase].size for phase in phases])  # take quadrature traces of the same length, at the cost of some samples
        n_samples_vac = numpy.min([vac[phase].size for phase in phases])
        n_samples = numpy.min([n_samples_vac, n_samples_sig])
        sig_array = numpy.zeros((n_samples, len(phases)))
        vac_array = numpy.zeros((n_samples, len(phases)))
        for j in range(len(phases)):
            phase = phases[j]
            sig_array[:, j] = sig[phase][:n_samples]
            vac_array[:, j] = vac[phase][:n_samples]
        vac = vac_array.flatten()
        ## Setup tomographer
        convergence_rule = 'fidelity of state'  # 'fidelity of iteration operator'#
        n_max = self.state_measurement.quantum_state.hilbert_space.dimension - 1
        self.state_measurement.tomographer = Tomographer(n_max=n_max, \
                                  convergence_rule=convergence_rule)
        #tomographer.temporal_mode_function = mode_function
        self.state_measurement.tomographer.setQuadratureData(quadratures=sig_array, vacuum=vac, phases=phases * numpy.pi / 180, dt=Ts,
                                      apply_mode_function=False)
        # %%
        self.state_measurement.tomographer.reconstruct(quadratures_range_increase=0, convergence_parameter=1e-8)
        # In[Create quantum state]
        self.state_measurement.quantum_state._density_operator.value = self.state_measurement.tomographer.rho
        self.state_measurement.quantum_state.density_operator.name = "\\hat{\\rho}"
        # In[Compute the displacement of the]
        harmonic_oscillator = SimpleHarmonicOscillatorNumeric(truncated_dimension=n_max + 1)
        displacement = (harmonic_oscillator.a @ self.state_measurement.quantum_state.density_operator).Trace()
       # displacement = (harmonic_oscillator.q + 1j*harmonic_oscillator.p)/numpy.sqrt(2)
        print("displacement modulus: %f"%abs(displacement).value)
        displacement_angle = (displacement.Angle()).value * 180/numpy.pi # rad
        print("displacement angle: %f degrees" % displacement_angle)
        # In[Plot wigner function]
        self.state_measurement.quantum_state.PlotWignerFunction(self.state_measurement.q, self.state_measurement.p, plot_name=self.state_measurement.quantum_state.figure_name)
        self.state_measurement.quantum_state.PlotDensityOperator(plot_name=self.state_measurement.quantum_state.figure_name)
        self.state_measurement.quantum_state.PlotNumberDistribution(plot_name=self.state_measurement.quantum_state.figure_name)

        # Check marginal probability densities
        from Iaji.Utilities.statistics import PDF_histogram
        n_bins = 100
        histograms = {}
        pyplot.figure()
        for phase in phases * numpy.pi / 180:
            histograms[phase] = PDF_histogram(self.state_measurement.tomographer.quadratures[phase],
                                              [numpy.min(self.state_measurement.tomographer.quadratures[phase]),
                                               numpy.max(self.state_measurement.tomographer.quadratures[phase])],
                                              n_bins=n_bins)
            pyplot.plot(histograms[phase][0], histograms[phase][1],
                        label="phase = %.1f$^\\circ$" % (phase * 180 / numpy.pi), marker='.')
        histogram_vac = PDF_histogram(self.state_measurement.tomographer.vacuum,
                                      [numpy.min(self.state_measurement.tomographer.vacuum),
                                       numpy.max(self.state_measurement.tomographer.vacuum)], n_bins=n_bins)
        pyplot.plot(histogram_vac[0], histogram_vac[1], label='vac', marker='.')
        pyplot.grid()
        pyplot.legend()

    def _delay(self, x1, x2, delay, Ts):
        '''
        Delays the two signals x1 and x2.

        :param x1:
        :param x2:
        :param delay:
        :param Ts:
        :return:
        '''

        n_samples = numpy.min([len(x1), len(x2)])
        delay_samples = abs(int(numpy.ceil(delay / Ts)))
        n_samples_recorrelated = n_samples - delay_samples
        if delay > 0:
            x1 = x1[delay_samples:]
            x2 = x2[0:n_samples_recorrelated]
        else:
            x2 = x2[delay_samples:]
            x1 = x1[0:n_samples_recorrelated]
        return x1, x2
