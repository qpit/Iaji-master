"""
This module generates an input state for quantum scissor QKD.
"""
# In[imports]
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import time
from time import sleep
import pyrpl
from Iaji.InstrumentsControl.SigilentSignalGenerator import SigilentSignalGenerator
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateMeasurementController import  StateMeasurementController
from pyqtgraph.Qt import QtGui
from matplotlib import pyplot
from qopt import lecroy
from scipy import signal
from scipy import optimize
from signalslot import Signal
from Iaji.SignalProcessing.Signals1D.correlator import correlator
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.QuantumStateTomography import QuadratureTomographer as Tomographer
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.SimpleHarmonicOscillator import SimpleHarmonicOscillatorNumeric

class StateGenerator:
    """
    This class controls the AOMs and define amplitude and phase modulations for the EOMs.
    """
    # -------------------------------------------
    def __init__(self, eom_redpitaya_config_filename, \
                 aom_redpitaya_config_filename, \
                 sample_hold_redpitaya_config_filename, \
                 #signal_enabler:SigilentSignalGenerator,
                 state_measurement: StateMeasurementController, name = "State Generator"):
        '''
        :param red_pitaya_config_filename: str
            pitaya config file full path
        :param signal_generator: Iaji SigilentSignalGenerator
            signal generator object, controlling the state generation AOM driver
        :param state_measurement: Iaji StateMeasurementController
            state measurement controller object, for state calibration
        :param name: str
        '''
        self.eom = False ## If change, edit line with string 20230713
        self.eom_redpitaya_config_filename = eom_redpitaya_config_filename
        self.aom_redpitaya_config_filename = aom_redpitaya_config_filename
        self.sample_hold_redpitaya_config_filename = sample_hold_redpitaya_config_filename
        self.state_measurement = state_measurement
        try:
            self.connect_to_redpitayas()
        except:
            print("Fanculo I couldn't connect to the state generation Pitayas!")
        if self.eom == True:
            self.lock_eom_bias(setpoint=0.01)
        # Trigger scope to C4, digital pause signal
        lecroy = self.state_measurement.hd_controller.acquisition_system.scope
        lecroy.setup_trigger(trigger_source='C4', trigger_level=1.36)
        #self.signal_enabler = signal_enabler #signal generator controlling the state generator AOM driver
        self.name = name
        #Add the signal generator to the state measurement controller for fast vacuum quadrature measurements
        #self.state_measurement.signal_enabler = self.signal_enabler
        self.calibration_stopped = False
        #Set up the dictionaries that contain devices configurations
        self.devices = {}
        ##Instruments
        self.devices['aoms'] = {'instrument': self.pyrpl_obj_aom.rp.asg0}
        self.devices['amplitude_eom'] = {'instrument': None} #self.pyrpl_obj_eom.rp.asg0} ## Edit here if using eom
        #self.devices['phase_eom'] = {'instrument': self.pyrpl_obj_mod.rp.asg1}
        ##Set up devices configurations
        ###Levels
        for name in ['aoms', 'amplitude_eom']:#, 'phase_eom']:
            self.devices[name]['levels'] = dict(zip(['lock', 'state_generation', 'vacuum'], [None, None, None]))
        ###Electronic output offsets
        self.devices['aoms']['instrument_offset'] = 1
        self.devices['aoms']['amplification_gain'] = 1
        self.devices['amplitude_eom']['instrument_offset'] = 1
        self.devices['amplitude_eom']['amplification_gain'] = 1 #Connect to an amplifier when using EOM
        # self.devices['phase_eom']['instrument_offset'] = 0
        self.aoms_high = 0.2
        self.eom_high = 1 #Change to None when using EOM
        self.amplitude_eom_low = 0
        #self.phase_eom_amplification_gain = 10
        self.devices['aoms']['levels']['lock'] = self.aoms_high
        self.devices['aoms']['levels']['state_generation'] = 0.1
        self.devices['amplitude_eom']['levels']['state_generation'] = 0
        self.devices['aoms']['levels']['vacuum'] = 0
        self.devices['amplitude_eom']['levels']['vacuum'] = self.amplitude_eom_low
        #self.devices['phase_eom']['levels']['lock'] = 0
        ###Amplification gains
        self.devices['amplitude_eom']['levels']['lock'] = self.eom_high
        #self.devices['phase_eom']['amplification_gain'] = self.phase_eom_amplification_gain
        self.time_multiplexing = {'instrument': self.pyrpl_obj_sh.rp.asg1,'duty_cycles': [0.6, 0.2, 0.2], 'frequency': 60, 'max_delay': 5e-4}
        self.devices['sspd_aom'] = {'instrument': self.pyrpl_obj_sh.rp.asg0}
        self.calibrations = {}
        self.phase_calibration = None
        self.calibrated = "no" #Flag if homodyne phase calibration has been performed
        #Sent voltage to AOM
        self.devices['aoms']['instrument'].offset = self.aoms_high/self.devices['aoms']['amplification_gain'] - self.devices['aoms']['instrument_offset']
    # -------------------------------------------
    #@property
    #def aoms_high(self):
    #    if self._aoms_high == None:
    #        self._find_aoms_high()
    #    return self._aoms_high
    #@aoms_high.setter
    #def aoms_high(self, value):
    #    self._aoms_high = value
    #    self.devices['aoms']['levels']['lock'] = self.aoms_high
    #@aoms_high.deleter
    #def aoms_high(self):
    #   del self._aoms_high
    # -------------------------------------------
    #@property
    #def aoms_amplification_gain(self):
    #    if self._aoms_amplification_gain == None:
    #        self._find_aoms_amplification_gain()
    #    return self._aoms_amplification_gain
    #@aoms_amplification_gain.setter
    #def aoms_amplification_gain(self, value):
    #    self._aoms_amplification_gain = value
    #    self.devices['aoms']['amplification_gain'] = self.aoms_amplification_gain
    #@aoms_amplification_gain.deleter
    #def aoms_amplification_gain(self):
    #    del self._aoms_amplification_gain
    # -------------------------------------------
    @property
    def eom_high(self):
        if self._eom_high == None:
            self._find_eom_high()
        return self._eom_high

    @eom_high.setter
    def eom_high(self, value):
        self._eom_high = value
        self.devices['amplitude_eom']['levels']['lock'] = self.eom_high

    @eom_high.deleter
    def eom_high(self):
        del self._eom_high
    # -------------------------------------------
    # @property
    # def amplitude_eom_low(self):
    #     if self._amplitude_eom_low is None:
    #         self.find_low_eom_transmission()
    #     return self._amplitude_eom_low
    # @amplitude_eom_low.setter
    # def amplitude_eom_low(self, value):
    #     self._amplitude_eom_low = value
    #     self.devices['amplitude_eom']['levels']['vacuum'] = value
    # @amplitude_eom_low.deleter
    # def amplitude_eom_low(self):
    #     del self._amplitude_eom_low
    # -------------------------------------------
    #@property
    #def phase_eom_amplification_gain(self):
    #    if self._phase_eom_amplification_gain is None:
    #        self._find_phase_eom_amplification_gain(frequency=50)
    #    return self._phase_eom_amplification_gain

    #@phase_eom_amplification_gain.setter
    #def phase_eom_amplification_gain(self, value):
    #    self._phase_eom_amplification_gain = value
    #    self.devices['phase_eom']['amplification_gain'] = value

    #@phase_eom_amplification_gain.deleter
    #def phase_eom_amplification_gain(self):
    #    del self._phase_eom_amplification_gain
    # -------------------------------------------
    def connect_to_redpitayas(self, show_pyrpl_GUI=True):
        # Connect to AOM Pitaya
        self.pyrpl_obj_aom = pyrpl.Pyrpl(config=self.aom_redpitaya_config_filename)
        if show_pyrpl_GUI:
            self.pyrpl_GUI_aom = QtGui.QApplication.instance()
        self.pyrpl_obj_aom.rp.asg0.offset = 0  # AOM MOD
        self.pyrpl_obj_aom.rp.asg0.output_direct = "out1"
        self.pyrpl_obj_aom.rp.asg1.offset = 1  # AOM Gate
        self.pyrpl_obj_aom.rp.asg1.output_direct = "out2"
        self.pyrpl_obj_aom.rp.scope.input1 = "out1"
        self.pyrpl_obj_aom.rp.scope.input2 = "out2"
        #for channel in [0, 1]:
        #    getattr(self.pyrpl_obj_aom.rp, "asg%d" % channel).setup(waveform='dc', offset=0,
        #                                                            trigger_source='immediately')
        #    getattr(self.pyrpl_obj_aom.rp, "asg%d" % channel).output_direct = "off" * (channel == 0) + "out%d" % (
        #            channel + 1) * (channel == 1)
        # Connect to sample hold pitaya
        self.pyrpl_obj_sh = pyrpl.Pyrpl(config=self.sample_hold_redpitaya_config_filename)
        self.pyrpl_obj_sh.rp.asg1.offset = 1  # Sample-hold signal
        self.pyrpl_obj_sh.rp.asg1.output_direct = "out2"
        self.pyrpl_obj_sh.rp.scope.input1 = "out1"
        self.pyrpl_obj_sh.rp.scope.input2 = "out2"
        if show_pyrpl_GUI:
            self.pyrpl_GUI_sh = QtGui.QApplication.instance()
        # Connect to EOM Pitaya
        if self.eom:
            self.pyrpl_obj_eom = pyrpl.Pyrpl(config=self.eom_redpitaya_config_filename)
            self.pyrpl_obj_eom.rp.asg0.offset = 0  # EOM RF
            self.pyrpl_obj_eom.rp.asg0.output_direct = "off"
            self.pyrpl_obj_eom.rp.asg1.offset = 0  # EOM BIAS
            self.pyrpl_obj_eom.rp.asg1.output_direct = "off"
            self.pyrpl_obj_eom.rp.scope.input1 = "out1"
            self.pyrpl_obj_eom.rp.scope.input2 = "in1"
            if show_pyrpl_GUI:
                self.pyrpl_GUI_eom = QtGui.QApplication.instance()
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
        #count_0 = function.tolist().count(levels[0])
        #count_1 = function.tolist().count(levels[1])
        #count_2 = function.tolist().count(levels[2])
        #print('Duty cycles:', count_0/len(function), count_1/len(function), count_2/len(function))
        return function
    # ------------------------------------------
    def lock_eom_bias(self, setpoint):
        pitaya_eom = self.pyrpl_obj_eom.rp
        scope = pitaya_eom.scope

        pitaya_eom.pid0.input = "in1"
        pitaya_eom.pid0.setpoint = setpoint
        pitaya_eom.pid0.p = -0.03
        pitaya_eom.pid0.i = -100
        '''
        pitaya_eom.asg1.waveform = 'dc'
        pitaya_eom.asg1.offset = 0
        pitaya_eom.asg1.output_direct = "out2"
        mean = [setpoint + 1]
        sign = 'p'
        last = None
        while mean[-1] > setpoint:
            _trace = scope.curve()
            scope.continuous()
            trace = _trace[1]

            mean.append(numpy.mean(trace))
            print('DEBUG: mean', mean[-1])
            if mean[-1] > mean[-2]:
                if sign == 'p':
                    pitaya_eom.asg1.offset -= 0.1
                    if last == 'gp':
                        sign = 'm'
                    else:
                        sign = 'p'
                    last = 'gp'
                if sign == 'm':
                    pitaya_eom.asg1.offset += 0.1
                    if last == 'gm':
                        sign = 'p'
                    else:
                        sign = 'm'
                    last = 'gm'
            elif mean[-1] < mean[-2]:
                if sign == 'p':
                    pitaya_eom.asg1.offset += 0.1
                    if last == 'lp':
                        sign = 'm'
                    else:
                        sign = 'p'
                    last = 'lp'
                if sign == 'm':
                    pitaya_eom.asg1.offset -= 0.1
                    if last == 'lm':
                        sign = 'p'
                    else:
                        sign = 'm'
                    last = 'lm'
            print('DEBUG sign', sign)
            print('DEBUG last', last)
        '''
        pitaya_eom.pid0.output_direct = "out2"
        pitaya_eom.asg1.output_direct = "off"
    # ------------------------------------------
    def turn_off_time_multiplexing_signals(self):
        aom = self.devices['aoms']['instrument']
        if self.eom:
            eom = self.devices['amplitude_eom']['instrument']
        gate = self.pyrpl_obj_aom.rp.asg1
        sample_hold = self.time_multiplexing['instrument']

        aom_value = self.devices['aoms']['levels']['lock']/self.devices['aoms']['amplification_gain'] - self.devices['aoms']['instrument_offset']
        if self.eom:
            eom_value = self.devices['amplitude_eom']['levels']['lock']/self.devices['amplitude_eom']['amplification_gain'] - self.devices['amplitude_eom']['instrument_offset']
        sample_hold_value = 1

        aom.setup(waveform='dc', trigger_source="immediately", amplitude=1, offset=aom_value)
        if self.eom:
            eom.setup(waveform='dc', trigger_source="immediately", amplitude=1, offset=eom_value)
        gate.setup(waveform='dc', trigger_source="immediately", amplitude=1, offset=sample_hold_value)
        sample_hold.setup(waveform='dc', trigger_source="immediately", amplitude=1, offset=sample_hold_value)

        aom.output_direct = "out1"
        if self.eom:
            eom.output_direct = "out1"
        gate.output_direct = "out2"
        sample_hold.output_direct = "out2"

        lecroy = self.state_measurement.hd_controller.acquisition_system.scope
        lecroy.setup_trigger(trigger_source='C4', trigger_level=1.36)
    # ------------------------------------------
    def _time_multiplexing_signals(self, max_delay, aom_levels, eom_levels, frequency, aom_duty_cycles, eom_duty_cycles, gate_flag = True):
        '''
        Generate the 3 level sample hold for the AOM and 2 level for the EOM, with the delay optimization between them.
        :return:
        '''
        #acq = self.state_measurement.hd_controller.acquisition_system
        #acq.scope.set_save_directory('Scissor QKD\\Test - delete')
        #acquisition_channels = {"aom": 2, "eom": 3}
        #channel_names = list(acquisition_channels.keys())
        #self._setup_acquisition_calibration(channels=acquisition_channels)

        aom = self.devices['aoms']['instrument'] # AOMs controlling input state
        if self.eom:
            eom = self.devices['amplitude_eom']['instrument']
        sample_hold = self.time_multiplexing['instrument'] # OPO probe and pid gate signal
        sspd_aom = self.devices['sspd_aom']['instrument'] # AOMs that protect SSPD
        aom_scope = self.pyrpl_obj_aom.rp.scope
        gate = self.pyrpl_obj_aom.rp.asg1 # Gate for input state AOMs

        n_points = 2 ** 14
        scope_decimation = int(numpy.ceil(125e6/(frequency*n_points)))
        Ts = scope_decimation / 125e6

        real_aom_levels = aom_levels
        print('AOM levels:', real_aom_levels)
        real_eom_levels = eom_levels
        aom_levels = numpy.array(aom_levels)/self.devices['aoms']['amplification_gain'] - self.devices['aoms']['instrument_offset']
        eom_levels = numpy.array(eom_levels)/self.devices['amplitude_eom']['amplification_gain'] - self.devices['amplitude_eom']['instrument_offset']
        gate_levels = [1, -1]
        gate_duty_cycles = [0.8, 0.2]
        sample_hold_levels = [1, -1]
        sample_hold_duty_cycles = [0.6, 0.4]
        sspd_aom_levels = [-1, 1, -1]
        sspd_aom_duty_cycles_begin = [0.7, 0.2]
        sspd_aom_duty_cycles = [*sspd_aom_duty_cycles_begin, 1-numpy.sum(sspd_aom_duty_cycles_begin)] #Had to do it like this, because for some dumb reason [0.7, 0.2, 0.1] add up to 0.9999999999999999

        if gate_flag == False:
            gate_levels = [1, 1]

        aom.setup(trigger_source="ext_negative_edge", frequency = self.time_multiplexing['frequency'], amplitude=1, offset=0)
        aom.data = self._n_level_step_function(frequency, aom_levels, aom_duty_cycles, n_points, Ts)
        #aom.setup(waveform='dc', trigger_source="immediately", amplitude=1, offset=aom_levels[1])
        #eom.setup(trigger_source="ext_negative_edge", frequency=self.time_multiplexing['frequency'], amplitude=1, offset=0)
        #eom.data = self._n_level_step_function(frequency, eom_levels, eom_duty_cycles, n_points, Ts)
        if self.eom:
            eom.setup(waveform='dc', trigger_source="immediately", amplitude=1, offset=eom_levels[1])
        gate.setup(trigger_source="ext_negative_edge", frequency=self.time_multiplexing['frequency'], amplitude=1, offset=0)
        gate.data = self._n_level_step_function(frequency, gate_levels, gate_duty_cycles, n_points, Ts)
        sample_hold.setup(trigger_source="ext_negative_edge", frequency = self.time_multiplexing['frequency'], amplitude=1, offset=0)
        sample_hold.data = self._n_level_step_function(frequency, sample_hold_levels, sample_hold_duty_cycles, n_points, Ts)
        sspd_aom.setup(trigger_source="ext_negative_edge", frequency = self.time_multiplexing['frequency'], amplitude=1, offset=0)
        sspd_aom.data = self._n_level_step_function(frequency, sspd_aom_levels, sspd_aom_duty_cycles, n_points, Ts)

        aom.output_direct = "out1"
        if self.eom:
            eom.output_direct = "out1"
        gate.output_direct = "out2"
        sample_hold.output_direct = "out2"
        sspd_aom.output_direct = 'out1'

        lecroy = self.state_measurement.hd_controller.acquisition_system.scope
        lecroy.setup_trigger(trigger_source='C4', trigger_level=1.36)#0.5*real_aom_levels[0])

        print("If sample-hold doesn't work, check if FG sending the synch signal is turned on.")
        #plt.figure()
        #plt.plot(numpy.arange(len(asg0.data)), asg0.data, 'm')
        #plt.title('asg 0')
        #plt.show()

        # time.sleep(0.1)
        # Optimize the generation of the two waveforms by requiring that their (random) time delay between be lower
        # than a certain target value
        # visualize and acquire the waveforms on the scope
        #scope.input1 = "asg0"
        #scope.input2 = "asg1"
        #scope.duration = 3/frequency

        #Find delay by sending signals multiple times
        """
        delay = 2*max_delay
        print("Preparing the time-multiplexing signals for state calibration")
        eom_levels_array = numpy.array(eom_levels)/2.5 + 1
        begin = time.time()
        while abs(delay) > max_delay or delay == 0:
            if self.calibration_stopped:
                self.calibration_stopped = False
                return
            #print("delay: %.2f ms > %.2f ms" % (delay * 1e3, max_delay * 1e3))
            eom.setup(trigger_source="immediately", frequency = self.time_multiplexing['frequency'], amplitude=1, offset=0)
            eom.data = self._n_level_step_function(frequency, eom_levels, eom_duty_cycles, n_points, Ts)
            # acquire
            x = numpy.linspace(0, Ts * n_points, n_points)
            #traces = scope.curve()
            #scope.continuous()
            #trace_aom = traces[0]
            #trace_eom = traces[1]

            for channel_name in channel_names:
                acq.filenames[channel_name] = channel_name + "_" + "sample-hold"
            acq.acquire()

            _trace_aom = acq.scope.traces[channel_names[0]]
            _trace_eom = acq.scope.traces[channel_names[1]]
            lecroy_Ts = _trace_aom[0]['horiz_interval']
            trace_aom = _trace_aom[2][0]
            trace_eom = _trace_eom[2][0]

            lecroy_time = []
            for element in numpy.arange(len(trace_aom)):
                lecroy_time.append(element*lecroy_Ts)
            print('eom levels', real_eom_levels[0], real_eom_levels[1], real_eom_levels[2])

            #corr_obj = correlator(signal_1=numpy.gradient(trace_aom), signal_2=-numpy.gradient(trace_eom), sampling_period=Ts)
            #delay = abs(corr_obj.getDelay(absolute_correlation=False))

            #Find the delay between the last samples of the aom and eom (either high or medium) levels
            #for j in range(3):
            #    print('j', j)
            aom_index = numpy.where(numpy.isclose(trace_aom, real_aom_levels[1], atol=5e-4))[0][0]
            eom_index = numpy.where(numpy.isclose(trace_eom, real_eom_levels[1], atol=5e-4))[0][0]
            if trace_aom[aom_index] != 0 or trace_eom[eom_index] != 0:
                if aom_index != 0 and eom_index != 0:
                    delay = (eom_index - aom_index) * lecroy_Ts
                    #print('AOM-EOM delay:', delay)
        print('AOM-EOM delay found:', delay)
        '''
        plt.figure()
        plt.plot(lecroy_time, trace_aom, 'm')
        plt.plot(lecroy_time, trace_eom, 'c')
        plt.axvline(x=aom_index*lecroy_Ts)
        plt.axvline(x=eom_index*lecroy_Ts)
        plt.title('AOM and EOM')
        plt.show()
        '''
        # Delay between AOM and Sample Hold
        sh_delay = 2*max_delay
        count = 0
        while abs(sh_delay) > max_delay or sh_delay == 0:
            sample_hold.setup(trigger_source="immediately", frequency = self.time_multiplexing['frequency'], amplitude=1, offset=0)
            sample_hold.data = self._n_level_step_function(frequency, sample_hold_levels, sample_hold_duty_cycle, n_points, Ts)
            sample_hold.output_direct = "out2"
            traces = aom_scope.curve()
            aom_scope.continuous()
            trace_aom = traces[0]
            trace_sh = traces[1]

            aom_index = numpy.where(numpy.isclose(trace_aom, aom_levels[-1], atol=5e-4))[0][0]
            sh_index = numpy.where(numpy.isclose(trace_sh, sample_hold_levels[-1], atol=5e-4))[0][0]
            if trace_aom[aom_index] != 0 or trace_eom[sh_index] != 0:
                if aom_index != 0 and sh_index != 0:
                    sh_delay = (aom_index - sh_index) * Ts
                    #print('AOM-Sample hold delay:', sh_delay)

            pitaya_time = []
            for element in numpy.arange(len(trace_aom)):
                pitaya_time.append(element*Ts)

            count += 1

            time.sleep(abs(max_delay - sh_delay))
        print('AOM-Sample hold delay found:', sh_delay)
        '''
        plt.figure()
        plt.plot(pitaya_time, trace_aom, 'm')
        plt.plot(pitaya_time, trace_sh, 'c')
        plt.title('AOM and Sample hold')
        plt.axvline(x=aom_index * Ts)
        plt.axvline(x=sh_index * Ts)
        plt.show()
        '''
        
        #scope.input1 = "out1"
        #scope.input2 = "out2"
        print("AOM-EOM delay: %f ms < %f ms" % (delay * 1e3, max_delay * 1e3))
        print("AOM- Sample hold delay: %f ms < %f ms" % (sh_delay * 1e3, max_delay * 1e3))
        end = time.time()
        print("Time spent for finding the correct delay: %d min %d s " %(int((end-begin)/60), numpy.ceil((end-begin)%60)))
        time.sleep(1)
        """
    # -----------------------------------------
    def _extract_quadrature_measurements(self, hd_output, time_multiplexing_signal, Ts, dead_time=0.75e-3, measurement_time=-1,
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
        _positive_peak_indices, properties = signal.find_peaks(gradient, height=0.01 * numpy.max(gradient))
        _positive_peak_heights = properties["peak_heights"]
        good_indices = numpy.where(_positive_peak_heights > .9*numpy.max(_positive_peak_heights))
        _positive_peak_indices = _positive_peak_indices[good_indices]
        _positive_peak_heights = _positive_peak_heights[good_indices]

        _negative_peak_indices, properties = signal.find_peaks(-gradient, height=0.05 * numpy.max(-gradient))
        _negative_peak_heights = -properties["peak_heights"]
        remove_negative = []
        remove_positive = []
        print('Positive good indices:', len(_positive_peak_indices))
        print('Negative raw:', len(_negative_peak_indices))
        for i in range(len(_positive_peak_indices)):
            for j in range(len(_negative_peak_indices)):
                time_difference = abs(_positive_peak_indices[i] - _negative_peak_indices[j]) * Ts
                #if len(_positive_peak_indices) > i+1:
                #    time_difference_positive = abs(_positive_peak_indices[i+1] - _positive_peak_indices[i]) * Ts
                if time_difference < 0.15/self.time_multiplexing['frequency']:
                    remove_negative.append(j)
                #if time_difference_positive < 0.15 / self.time_multiplexing['frequency']:
                #    remove_positive.append(i)
                #    print('Positive remove:', i)
        if len(remove_negative) != 0:
            negative_peak_indices = numpy.delete(_negative_peak_indices, numpy.array(remove_negative))
            negative_peak_heights = numpy.delete(_negative_peak_heights, numpy.array(remove_negative))
        else:
            negative_peak_indices = _negative_peak_indices
            negative_peak_heights = _negative_peak_heights
        positive_peak_indices = _positive_peak_indices
        positive_peak_heights = _positive_peak_heights
        #positive_peak_indices = numpy.delete(_positive_peak_indices, numpy.array(remove_positive))
        #positive_peak_heights = numpy.delete(_positive_peak_heights, numpy.array(remove_positive))
        good_indices = numpy.where(negative_peak_heights < .01*numpy.min(negative_peak_heights))
        negative_peak_indices = negative_peak_indices[good_indices]
        negative_peak_heights = negative_peak_heights[good_indices]
        print('Final positive:', len(positive_peak_indices))
        print('Final negative:', len(negative_peak_indices))
        #print('Before collecting missing peaks')
        #print('Positive peaks:', positive_peak_indices * Ts)
        #print('Negative peaks:', negative_peak_indices * Ts)

        '''
        missed_peaks = []
        for j in range(len(negative_peak_indices)-1):
            subtraction = (negative_peak_indices[j+1] - negative_peak_indices[j])*Ts
            if subtraction > (self.time_multiplexing['duty_cycles'][0] + 0.1)/self.time_multiplexing['frequency']:
                missed_peaks.append(j+1)
        negative_peak_indices = negative_peak_indices.tolist()
        negative_peak_heights = negative_peak_heights.tolist()
        for j in range(len(missed_peaks)):
            # Not correcting for negative_peak_heights
            print('Missed peaks:', missed_peaks)
            negative_peak_indices.insert(missed_peaks[j], negative_peak_indices[missed_peaks[j]-1] + 0.002/Ts)
            negative_peak_heights.insert(missed_peaks[j], -1e-3)
            for i in range(len(missed_peaks)):
                missed_peaks[i] += 1
        negative_peak_indices = numpy.array(negative_peak_indices)
        negative_peak_heights = numpy.array(negative_peak_heights)
        if len(missed_peaks) > 0:
            print('There were missing peaks')
        '''

        print('Positive peaks:', positive_peak_indices*Ts)
        print('Negative peaks:', negative_peak_indices*Ts)
        #print('Negative heights:', negative_peak_heights)

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
        end_dead_samples = int(.5e-3/Ts)
        for j in range(len(positive_peak_indices[1:-1])):
            if self.calibration_stopped:
                self.calibration_stopped = False
                return
            start, end = (positive_peak_indices[j], positive_peak_indices[j + 1])
            neg_peak_indices =  negative_peak_indices[numpy.where(numpy.logical_and( \
                negative_peak_indices > start, \
                negative_peak_indices < end))]
            if len(neg_peak_indices) == 1:
                vac_peak_time = neg_peak_indices[0]*Ts + self.time_multiplexing['duty_cycles'][1]/self.time_multiplexing['frequency']
                vac_peak_indice = vac_peak_time/Ts
                neg_peak_indices_list = neg_peak_indices.tolist()
                neg_peak_indices_list.append(vac_peak_indice)
                neg_peak_indices = numpy.array(neg_peak_indices_list)
                print('Missing negative peaks were added.')
            #Debug printing
           # print("(start time, end time) = (%f, %f)"%(start*Ts, end*Ts))
           # print("negative peak times: %s"%(numpy.array(neg_peak_indices)*Ts))
            peak1, peak2 = (neg_peak_indices[0], neg_peak_indices[1])
            peak1 = int(peak1)
            peak2 = int(peak2)
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
            #pyplot.axvline(x = peak1*Ts, color = "m")
            #pyplot.axvline(x = start*Ts, color = "blue")
            pyplot.legend(loc="upper right")
        return vac, sig

    # -------------------------------------------
    # def find_low_eom_transmission(self, voltage_range=[2.5, 5], n_points=100):
    #     '''
    #     Scan EOM voltage and return value with minimum state transmission
    #
    #     :return:
    #     '''
    #     asg_aom = self.devices['aoms']['instrument']
    #     asg_eom =  self.devices['amplitude_eom']['instrument']
    #
    #     #Set aom in a locking state
    #     amplification_gain = self.devices['aoms']['amplification_gain']
    #     rp_offset = self.devices['aoms']['instrument_offset']
    #     asg_aom.setup(waveform='dc', \
    #                   offset=self.devices['aoms']['levels']['lock']/self.devices['aoms']['amplification_gain']-self.devices['aoms']['instrument_offset'], \
    #                   trigger_source='immediately')
    #     asg_aom.output_direct = "out1"
    #     #Find minimum transmission
    #     asg_eom.setup(waveform='dc', \
    #                   offset=self.devices['amplitude_eom']['levels']['lock']/self.devices['amplitude_eom']['amplification_gain']-self.devices['amplitude_eom']['instrument_offset'], \
    #                   trigger_source='immediately')
    #     asg_eom.output_direct = "out2"
    #     voltages = numpy.linspace(*voltage_range, n_points)
    #     amplitudes = numpy.zeros((n_points,))
    #     #Scan LO phase
    #     self.state_measurement.hd_controller.phase_controller.scan()
    #     for j in numpy.arange(n_points):
    #         voltage = voltages[j]
    #         asg_eom.offset = voltage/self.devices['amplitude_eom']['amplification_gain']-self.devices['amplitude_eom']['instrument_offset']
    #         amplitudes[j] = self.state_measurement.hd_controller.phase_controller.get_signal_amplitude()
    #     asg_eom.offset = 0
    #     #Find minimum transmission voltage
    #     self.state_measurement.hd_controller.phase_controller.turn_off_scan()
    #     self.amplitude_eom_low = voltages[numpy.argmin(amplitudes)]
    #     asg_aom.offset = 0.1
    #     asg_eom.offset = 0
    #     print('EOM low voltage = %.2f'%self.amplitude_eom_low)
    # -------------------------------------------
    def lock(self, phase):
        phase_controller = self.state_measurement.hd_controller.phase_controller
        # Calibration
        phase_controller.calibrate()
        # Set homodyne detection phase
        phase_controller.set_phase(phase)
        # Lock the phase
        phase_controller.lock()
    # -------------------------------------------
    def measure_quadrature_calibration(self, phase, acquisition_channels):
        '''
        Lock HD to specified phase and acquire data from desired channels

        :param phase: float
        HD phase [degrees]
        :return:
        '''
        attempt = 0
        run = 0
        while attempt == 0:
            run += 1
            print('trial: %d'%run)
            acq = self.state_measurement.hd_controller.acquisition_system
            phase_controller = self.state_measurement.hd_controller.phase_controller
            #if self.phase_calibration == "once": # Phase calibration is only done once for all the measurements. It avoids moments in which iq demodulation phase is 180 deg apart.
            #    if self.calibrated == "no":
            #        # Calibration
            #        phase_controller.calibrate()
            #        self.calibrated = "yes"
            #else:
            #    # Calibration
            #    phase_controller.calibrate()
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
                print("Acquisition didn't work, trying again")
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
    def calibrate_aoms(self, voltage_range=[1e-2, 0.1], n_points = 6, polynomial_fit_order = 2, acquisition_channels={"hd": 1, "time-multiplexing": 2}, sample_hold = True, fit = True):
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
        self.phase_calibration = "once"
        self.calibrated = "no"
        start_time = time.time()
        self.state_measurement.hd_controller.phase_controller.turn_off_scan()
        acq = self.state_measurement.hd_controller.acquisition_system
        aom_high = self.devices['aoms']['levels']['lock']
        aom_low = self.devices['aoms']['levels']['vacuum']
        aom_medium_array = numpy.linspace(voltage_range[0], voltage_range[1], n_points)
        # EOM transmits when there is no voltage and blocks when there is voltage
        eom_high = self.devices['amplitude_eom']['levels']['lock']
        eom_low = self.amplitude_eom_low
        eom_medium = self.devices['amplitude_eom']['levels']['state_generation']
        # Define properties of time multiplexing signals
        aom_duty_cycles = eom_duty_cycles = self.time_multiplexing['duty_cycles']
        frequency = self.time_multiplexing['frequency']
        max_delay = self.time_multiplexing['max_delay']
        #Set up acquisition system
        self._setup_acquisition_calibration(channels=acquisition_channels)
        channel_names = list(acquisition_channels.keys())
        #Calibrate phase controller
        amplification_gain, rp_offset = [self.devices['aoms'][x] for x in ['amplification_gain', 'instrument_offset']]
        self.devices['aoms']['instrument'].waveform = "dc"
        self.devices['aoms']['instrument'].offset = self.aoms_high/amplification_gain - rp_offset
        #self.state_measurement.hd_controller.phase_controller.calibrate()
        self.devices['aoms']['instrument'].offset = 0
        #define array of displacements
        self.displacements = []
        #self.turn_off_time_multiplexing_signals()
        self.state_measurement.hd_controller.phase_controller.calibrate()
        for i in range(n_points):
            if self.calibration_stopped:
                self.calibration_stopped = False
                return
            # Generate the AOM and EOMtime multiplexing signals for calibration
            print('Calibrate AOM: measure %d of %d' %(i, len(aom_medium_array)))
            aom_levels = [aom_high, aom_medium_array[i], aom_low]
            print('Calibrate AOM levels ', aom_levels)
            eom_levels = [eom_high, eom_medium, eom_low]
            phase = 45
            #self.lock(phase)
            self._time_multiplexing_signals(max_delay, aom_levels, eom_levels, frequency, aom_duty_cycles, eom_duty_cycles)
            self._setup_acquisition_calibration(channels=acquisition_channels)
            #Lock the homodyne detector and measure its output, together with the AOM time multiplexing signal
            #phase = 45
            #generalized q quadrature
            self.measure_quadrature_calibration(phase=phase, acquisition_channels=acquisition_channels)
            traces_q = {channel_names[0]: acq.scope.traces[channel_names[0]], \
                        channel_names[1]: acq.scope.traces[channel_names[1]]}
            #self.turn_off_time_multiplexing_signals()
            #generalized p quadrature
            phase = phase + 90
            #self.lock(phase)
            #self._time_multiplexing_signals(max_delay, aom_levels, eom_levels, frequency, aom_duty_cycles, eom_duty_cycles)
            self._setup_acquisition_calibration(channels=acquisition_channels)
            self.measure_quadrature_calibration(phase=phase, acquisition_channels=acquisition_channels)
            traces_p = {channel_names[0]: acq.scope.traces[channel_names[0]], \
                        channel_names[1]: acq.scope.traces[channel_names[1]]}
            #self.turn_off_time_multiplexing_signals()
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

        if fit == False:
            return aom_medium_array, self.displacements

        self.phase_calibration = None
        #Skip the first point (we don't know why the first measurement is always bad!!)
        # Define polynomial for fitting
        fit_params = numpy.polyfit(aom_medium_array, numpy.abs(self.displacements), polynomial_fit_order)
        print(fit_params)
        aom_medium_fit = numpy.linspace(*voltage_range, 1000)
        aom_fitted = numpy.poly1d(fit_params)(aom_medium_fit)
        # Define calibration function
        def calibration_function(amplitude):
            poly_coefficients = fit_params.copy()
            poly_coefficients[0] -= amplitude
            roots = numpy.roots(poly_coefficients)
            print('AOM calibration parameters', roots)
            return numpy.array([r for r in roots if numpy.isreal(r) and numpy.real(r) >= 0])[0]

        self.calibrations["aoms"] = {"function": calibration_function, "parameters": fit_params}
        '''
        order = 2
        coeff = numpy.ones(order)
        def polynomial(x, *coeff):
            f = 0
            for i in range(order):
                f += coeff[i]*x**(2*i)
            return f
        # Fitting
        coeff_guess = coeff
        fit_guess = [0, 0, 10]
        fit_params, fit_covariance = optimize.curve_fit(f=polynomial, xdata=aom_medium_array,
                                                        ydata=numpy.abs(self.displacements), \
                                                        p0=fit_guess)
        coeff = fit_params
        print(fit_params)
        aom_medium_fit = numpy.linspace(*voltage_range, 1000)
        aom_fitted = polynomial(aom_medium_fit, *coeff)

        calibration_function = lambda amplitudes: (numpy.mod(numpy.arccos((amplitudes - offset) / amplitude),
                                                            numpy.pi) - phase) / (numpy.pi / v_pi)
        fit_params = dict(zip(["amplitude", "v_pi", "phase", "offset"], fit_params))
        self.calibrations["amplitude_eom"] = {"function": calibration_function, "parameters": fit_params}
        '''
        stop_time = time.time()
        time_spent = stop_time - start_time
        minutes = time_spent//60
        seconds = round(time_spent%60)
        print('Time for AOM calibration: %d min %d sec'%(minutes, seconds))
        print('DEBUG: params', fit_params)
        return aom_medium_array, self.displacements, aom_medium_fit, aom_fitted

        # --------------------------------------------
    def calibrate_aoms_n_times(self, num_of_iterations=5, polynomial_fit_order=2, voltage_range=[1e-2, 0.1], n_points = 6, acquisition_channels={"hd": 1, "time-multiplexing": 2}):
        '''
        Calibrates the state generation AOMs num_of_iterations times with respect to the induced amplitude variation.
        Consequently, mean and std values are calculated over all datasets.
        Data saved to .csv file
        '''
        start_n_time = time.time()
        aom_medium_arrays = []
        displacements_arrays = []

        # Running AOM calibration n times
        for i in range(num_of_iterations):
            print('Iteration number', i)
            aom_medium_array, displacements = self.calibrate_aoms(voltage_range, n_points, fit = False)
            aom_medium_arrays.append([aom_medium_array]) #Check if rectangular brackets are fine
            displacements_arrays.append([numpy.abs(displacements)])
        # Calculating statistics with data
        displacement_mean = numpy.mean(displacements_arrays, axis=0)
        displacement_mean = numpy.array(displacement_mean)
        displacement_mean = displacement_mean.flatten()
        displacement_mean = numpy.abs(displacement_mean)
        displacement_sdev = numpy.std(displacements_arrays, axis=0)
        displacement_sdev = numpy.array(displacement_sdev)
        displacement_sdev = displacement_sdev.flatten()
        aom_medium_mean = numpy.mean(aom_medium_arrays, axis=0)
        aom_medium_mean = numpy.array(aom_medium_mean)
        aom_medium_mean = aom_medium_mean.flatten()

        # Fitting parameters to the mean values
        fit_params = numpy.polyfit(aom_medium_mean, displacement_mean, polynomial_fit_order)
        aom_medium_fit = numpy.linspace(*voltage_range, 1000)
        aom_fitted = numpy.poly1d(fit_params)(aom_medium_fit)

        # Sacinf mean values and std of data
        df_disps_mean_sdev = pd.DataFrame(
            {"displacement_mean": displacement_mean, "displacement_sdev": displacement_sdev})
        displacement_abs = {'Calibration ' + str(i): displacements_arrays[i][0] for i in range(len(displacements_arrays))}
        df_raw_data = pd.DataFrame(displacement_abs) #pd.DataFrame({"displacements": displacement_abs, "aom_medium_vals": aom_medium_arrays})

        save_csv = self.state_measurement.hd_controller.acquisition_system.host_save_directory
        df_disps_mean_sdev.to_csv(save_csv + "\displacements_mean_sdev.csv", index=True)
        df_raw_data.to_csv(save_csv + "\displacements_raw_data.csv", index=True)

        # Define the function that fits the measured data
        def calibration_function(amplitude):
            poly_coefficients = fit_params.copy()
            poly_coefficients[0] -= amplitude
            roots = numpy.roots(poly_coefficients)
            print('AOM calibration parameters', roots)
            return numpy.array([r for r in roots if numpy.isreal(r) and numpy.real(r) >= 0])[0]

        self.calibrations["aoms"] = {"function": calibration_function,
                                     "parameters": fit_params}
        stop_n_time = time.time()
        time_spent = stop_n_time - start_n_time
        minutes = time_spent // 60
        seconds = round(time_spent % 60)
        print('Time for n times AOM calibration: %d min %d sec' % (minutes, seconds))

        numpy.savetxt(save_csv + "fit_params.txt", fit_params)
        return aom_medium_mean, displacement_mean, aom_medium_fit, aom_fitted, displacement_sdev
        # --------------------------------------------
    def _find_aoms_high(self):
        '''
        Find a value for the high voltage for the AOMs without saturating the detector.
        :return:
        '''
        '''
        # Scan HD
        aom_high = numpy.linspace(1, -1, 11)
        self.state_measurement.hd_controller.phase_controller.scan()
        # Set scope acquisition system
        acquisition_channels = {"hd": 1, "time-multiplexing": 2}
        acq = self.state_measurement.hd_controller.acquisition_system
        self._setup_acquisition_calibration(channels=acquisition_channels)
        channel_names = list(acquisition_channels.keys())
        for name in channel_names:
            acq.filenames[name] = name + "_scanned"
        # Scan AOM high voltages
        for voltage in aom_high:
            print('Pitaya voltage = %.1f' %voltage)
            self.devices['aoms']['instrument'].setup(waveform='dc', offset=voltage, trigger_source='immediately')
            self.devices['aoms']['instrument'].output_direct = 'out1'
            # Acquire trace from Pitaya
            _trace_pitaya = self.pyrpl_obj_calibr.rp.scope.curve()
            self.pyrpl_obj_calibr.rp.scope.continuous()
            trace_pitaya = _trace_pitaya[0]
            # Acquire trace from Lecroy
            traces_lecroy = acq.acquire()
            _hd_scanned = traces_lecroy[channel_names[0]]
            _aom_voltage = traces_lecroy[channel_names[1]]
            meta = _hd_scanned[0]
            hd_scanned = _hd_scanned[2].flatten()
            aom_voltage = _aom_voltage[2].flatten()
            # Find voltage with no saturation
            limit = 0.8
            if all(value < limit for value in hd_scanned) and all(value > -limit for value in hd_scanned):
                rounded_voltage = round(voltage, 1)
                print('Found! Pitaya voltage = %.1f V' % rounded_voltage)
                amplitude_pitaya = (numpy.max(trace_pitaya) - numpy.min(trace_pitaya)) / 2
                amplitude_lecroy = (numpy.max(aom_voltage) - numpy.min(aom_voltage)) / 2
                self.aoms_amplification_gain = amplitude_pitaya / amplitude_lecroy
                self.aoms_high = (rounded_voltage + rp_offset)*self.devices['aoms']['amplification_gain']
        '''
        # Scan HD
        phase_controller = self.state_measurement.hd_controller.phase_controller
        phase_controller.scan()
        # Test different AOM voltages
        asg_aom = self.devices['aoms']['instrument']
        amplification_gain = self.aoms_amplification_gain
        rp_offset = self.devices['aoms']['instrument_offset']
        phase_controller.scope.input1 = 'in1'
        phase_controller.scope.input2 = "in2"
        voltages = numpy.linspace(amplification_gain + rp_offset, 0, 10)
        print('Finding high voltage for AOMs')
        print(voltages)
        for voltage in voltages:
            print('voltage = %.2f V'%voltage)
            asg_aom.waveform = "dc"
            asg_aom.offset = voltage/amplification_gain - 1
            asg_aom.output_direct = 'out1'

            time.sleep(1)
            traces = phase_controller.scope.curve()
            phase_controller.scope.continuous()
            trace_ac = traces[1]
            #Prepare acquisition
            n_points = len(trace_ac)
            Ts = 1 / 125e6
            x = numpy.linspace(0, Ts*(n_points-1), n_points)
            '''
            acquisition_channels = {"hd": 1, "time-multiplexing": 2}
            acq = self.state_measurement.hd_controller.acquisition_system
            self._setup_acquisition_calibration(channels=acquisition_channels)
            channel_names = list(acquisition_channels.keys())
            for name in channel_names:
                acq.filenames[name] = name + "_scanned"
            traces = acq.acquire()
            _hd_scanned = traces[channel_names[0]]
            _aom_voltage = traces[channel_names[1]]
            # Load traces
            # hd_scanned = acq.scope.traces[channel_names[0]]
            meta = _hd_scanned[0]
            hd_scanned = _hd_scanned[2].flatten()
            # aom_voltage = acq.scope.traces[channel_names[1]]
            aom_voltage = _aom_voltage[2].flatten()
            print('Max = %.3f'%numpy.max(hd_scanned))
            print('Min = %.3f'%numpy.min(hd_scanned))
            Ts = meta['horiz_interval']
            points = len(hd_scanned)
            '''

            pyplot.figure()
            pyplot.title('Voltage = %.2f'%voltage)
            pyplot.plot(x, trace_ac, 'm')
            pyplot.show()

            limit = 0.75
            if all(value < limit for value in trace_ac) and all(value > -limit for value in trace_ac):
                rounded_voltage = round(voltage, 2)
                self.aoms_high = rounded_voltage
                print("Highest AOM voltage which doesn't saturate HD = %f V" %rounded_voltage)
                return

    # -------------------------------------------
    def _find_eom_high(self):
        print('Finding EOM high')
        # Scan HD
        phase_controller = self.state_measurement.hd_controller.phase_controller
        phase_controller.scan()
        # Test different EOM voltages
        asg_eom = self.devices['amplitude_eom']['instrument']
        amplification_gain = self.devices['amplitude_eom']['amplification_gain']
        rp_offset = self.devices['amplitude_eom']['instrument_offset']
        phase_controller.scope.input1 = 'in1'
        phase_controller.scope.input2 = "in2"
        voltages = numpy.linspace(0, 2.5, 11)
        amplitudes = []
        for voltage in voltages:
            asg_eom.waveform = "dc"
            asg_eom.offset = voltage / amplification_gain - rp_offset
            asg_eom.output_direct = 'out1'

            traces = phase_controller.scope.curve()
            phase_controller.scope.continuous()
            trace_ac = traces[1]
            #n_points = len(trace_ac)
            #Ts = 1 / 125e6
            #x = numpy.linspace(0, Ts * (n_points - 1), n_points)

            amplitude = (numpy.max(trace_ac) - numpy.min(trace_ac))/2
            amplitudes.append(amplitude)
        max_index = numpy.where(amplitudes == numpy.max(amplitudes))[0][0]
        self.eom_high = voltages[max_index]
        print('EOM high voltage is %.2f V' %self.eom_high)
    # -------------------------------------------
    # def calibrate_phase_eom(self, voltage_range=[-1, 1], n_points = 6, acquisition_channels={"hd": 1, "time-multiplexing": 2}): #TODO
    #     '''
    #      Calibrates the state generation phase EOM with respect to the induced phase rotation.
    #      The aim is to draw a functional dependence of the induced phase rotation on the voltage output from the
    #      RedPitaya that controls the phase EOM.
    #      It iteratively inputs a voltage to the EOM and measures the displacement of the coherent state
    #      from a homodyne detector.
    #      :param voltage_range: iterable of float (size=2)
    #          range of voltages output from the signal generator that feeds into the EOM driver [V]
    #      :return:
    #      '''
    #     acq = self.state_measurement.hd_controller.acquisition_system
    #     aom_high, aom_medium, aom_low = [self.devices['aoms']['levels'][x] for x in ['lock', 'state_generation', 'vacuum']]
    #     #Amplitude EOM
    #     ## EOM transmits when there is no voltage and blocks when there is voltage
    #     amplitude_eom_high, amplitude_eom_medium = [self.devices['amplitude_eom']['levels'][x] for x in ['lock', 'state_generation']]
    #     amplitude_eom_low = self.amplitude_eom_low
    #     #Phase EOM
    #     phase_eom_level_array = numpy.linspace(*voltage_range, n_points)
    #     amplification_gain, rp_offset = [self.devices['phase_eom'][x] for x in ['amplification_gain', 'instrument_offset']]
    #     # Define properties of time multiplexing signals
    #     aom_duty_cycles = eom_duty_cycles = self.time_multiplexing['duty_cycles']
    #     frequency = self.time_multiplexing['frequency']
    #     max_delay = self.time_multiplexing['max_delay']
    #     #Set up acquisition system
    #     self._setup_acquisition_calibration(channels=acquisition_channels)
    #     channel_names = list(acquisition_channels.keys())
    #     #Calibrate phase controller
    #     self.devices['aoms']['instrument'].waveform = "dc"
    #     self.devices['aoms']['instrument'].offset = 1
    #     self.state_measurement.hd_controller.phase_controller.calibrate()
    #     self.state_measurement.hd_controller.set_iq_qfactor()
    #     self.devices['aoms']['instrument'].offset = 0
    #     #define array of displacements
    #     self.displacements = []
    #     for i in range(n_points):
    #         if self.calibration_stopped:
    #             self.calibration_stopped = False
    #             return
    #         # Generate the AOM and EOMtime multiplexing signals for calibration
    #         aom_levels = [aom_high, aom_medium, aom_low]
    #         amplitude_eom_levels = [amplitude_eom_high, amplitude_eom_medium, amplitude_eom_low]
    #         self._time_multiplexing_signals(max_delay, aom_levels, amplitude_eom_levels, frequency, aom_duty_cycles,
    #                                    eom_duty_cycles)
    #         # Send voltage to phase modulator
    #         self.devices['phase_eom']['instrument'].offset = phase_eom_level_array[i]/\
    #             self.devices['phase_eom']['amplification_gain']-self.devices['phase_eom']['instrument_offset']
    #         #self.signal_enabler.channels["C1"].offset =  phase_eom_level_array[i]
    #         #Lock the homodyne detector and measure its output, together with the AOM time multiplexing signal
    #         phase = 45
    #         #generalized q quadrature
    #         self.measure_quadrature_calibration(phase=phase, acquisition_channels=acquisition_channels)
    #         traces_q = {channel_names[0]: acq.scope.traces[channel_names[0]], \
    #                     channel_names[1]: acq.scope.traces[channel_names[1]]}
    #         #generalized p quadrature
    #         self.measure_quadrature_calibration(phase=phase+90, acquisition_channels=acquisition_channels)
    #         traces_p = {channel_names[0]: acq.scope.traces[channel_names[0]], \
    #                     channel_names[1]: acq.scope.traces[channel_names[1]]}
    #         #Extract acquisition time information
    #         Ts = traces_q[channel_names[0]][0]['horiz_interval'] #acquisition sampling period [s]
    #         #Analyze generalized q quadrature
    #         vac, sig = self._extract_quadrature_measurements(hd_output=traces_q[channel_names[0]][2][0, :], \
    #                                                          time_multiplexing_signal=traces_q[channel_names[1]][2][0, :], \
    #                                                          Ts=Ts, dead_time=0.15e-3, plot = False)
    #         vac_std = numpy.std(vac)
    #         q_mean = numpy.mean(sig)/(vac_std / (1/numpy.sqrt(2)))
    #         #Analyze generalized p quadrature
    #         vac, sig = self._extract_quadrature_measurements(hd_output=traces_p[channel_names[0]][2][0, :], \
    #                                                          time_multiplexing_signal=traces_p[channel_names[1]][2][0, :], \
    #                                                          Ts=Ts, dead_time=0.15e-3, plot = False)
    #         vac_std = numpy.std(vac)
    #         p_mean = numpy.mean(sig) / (vac_std / (1/numpy.sqrt(2)))
    #         phase_rad = phase*numpy.pi/180
    #         #Compute and save displacement
    #         self.displacements.append(numpy.exp(1j*phase_rad)/numpy.sqrt(2)*(q_mean+1j*p_mean))
    #     return phase_eom_level_array, self.displacements
    # # -------------------------------------------
    # def scan_calibrate_phase_eom(self, voltage_range = [-10, 10], ramp_frequency = 50,  acquisition_channels={"hd": 1, "ramp": 3}, plot=False):
    #     start_time = time-time()
    #     print('Scanned calibration of the phase EOM')
    #     # Pitayas at DC
    #     aom_high = self.devices['aoms']['levels']['lock']
    #     amplification_gain, rp_offset = [self.devices['aoms'][x] for x in ['amplification_gain', 'instrument_offset']]
    #     self.devices['aoms']['instrument'].waveform = "dc"
    #     self.devices['aoms']['instrument'].offset = aom_high / amplification_gain - rp_offset
    #     amplitude_eom_high = self.devices['amplitude_eom']['levels']['lock']
    #     amplification_gain, rp_offset = [self.devices['amplitude_eom'][x] for x in ['amplification_gain', 'instrument_offset']]
    #     self.devices['amplitude_eom']['instrument'].waveform = "dc"
    #     self.devices['amplitude_eom']['instrument'].offset = amplitude_eom_high / amplification_gain - rp_offset
    #     # Phase EOM scan
    #     self.scan_phase_eom(voltage_range = voltage_range, frequency = ramp_frequency)
    #     # Acquisition
    #     acq = self.state_measurement.hd_controller.acquisition_system
    #     self._setup_acquisition_calibration(channels=acquisition_channels)
    #     channel_names = list(acquisition_channels.keys())
    #     sleep(3)
    #     for name in channel_names:
    #         acq.filenames[name] = name + "_scanned"
    #     attempt = 0
    #     while attempt == 0:
    #         try:
    #             acq.acquire()
    #             attempt = 1
    #         except:
    #             attempt = 0
    #     #Load traces
    #     hd_scanned = acq.scope.traces[channel_names[0]]
    #     ramp = acq.scope.traces[channel_names[1]][2].flatten()
    #     Ts = hd_scanned[0]["horiz_interval"]
    #     hd_scanned = hd_scanned[2].flatten()
    #     time = Ts * numpy.arange(len(hd_scanned))
    #     '''
    #     #Let the user select a single period of the scanned interference on a graph
    #     figure_scan = pyplot.figure(figsize=(13, 8))
    #     axis = figure_scan.add_subplot(111)
    #     axis.set_xlabel("time ($\\mu$s)")
    #     axis.set_ylabel("voltage (V)")
    #     axis.set_title("Phase EOM calibration: \n Mark one period with two clicks (avoid turning points)")
    #     axis.grid()
    #     figure_scan.show()
    #     '''
    #     #Find correspondence between time and voltage input to the phase eom
    #     ##The best way would be to fit a ramp function to the ramp signal and drawing a correspondence between voltage
    #     ##and time. I will for now identify the first half period of the ramp by looking for the first maximum and minimum
    #     min_index = numpy.where(numpy.isclose(ramp, numpy.min(ramp), atol=1e-2))[0][0]
    #     max_index = min_index+numpy.where(numpy.isclose(ramp[min_index:], numpy.max(ramp[min_index:]), atol=1e-2))[0][0]
    #     min_voltage, max_voltage = [ramp[min_index], ramp[max_index]]
    #     hd_scanned = hd_scanned[min_index:max_index+1]
    #     time = numpy.arange(max_index-min_index+1)*Ts
    #     voltages = numpy.linspace(min_voltage, max_voltage, len(time))
    #     if plot:
    #         pyplot.figure()
    #         pyplot.plot(time[::10], voltages[::10])
    #     # Fit with a sine function
    #     def cos(x, amplitude, v_pi, phase, offset):
    #         return offset + amplitude * numpy.cos(numpy.pi/v_pi * x + phase)
    #     amplitude_guess = (numpy.max(hd_scanned) - numpy.min(hd_scanned))/2
    #     #Estimate the frequency of the scanned interference from a periodogram
    #     T_v = abs(voltages[1] - voltages[0])
    #     print("T_v = %f"%T_v)
    #     #frequency_array, PSD = signal.welch(x=hd_scanned-numpy.mean(hd_scanned), fs=1/T_v, nperseg=len(hd_scanned), noverlap=20)
    #     #frequency_guess = frequency_array[numpy.argmax(PSD)] #[1/V]
    #     min_index = numpy.where(numpy.isclose(hd_scanned, numpy.min(hd_scanned), atol=1e-2))[0][0]
    #     max_index = numpy.where(numpy.isclose(hd_scanned, numpy.max(hd_scanned), atol=1e-2))[0][0]
    #     start, stop = numpy.sort([min_index, max_index])
    #     if plot:
    #         pyplot.figure()
    #         pyplot.plot(voltages[start:stop], hd_scanned[start:stop])
    #     frequency_guess_Hz = 1/(2*abs(max_index-min_index)*Ts)
    #     frequency_guess = 1/(2*abs(max_index-min_index)*T_v)
    #     v_pi_guess = 1/(2*frequency_guess)
    #     hd_scanned = hd_scanned[0:int(2/frequency_guess/T_v)]
    #     voltages = voltages[0:int(2/frequency_guess/T_v)]
    #     print("(frequency_guess, v_pi_guess, frequency_guess_Hz): (%f, %f, %f)" % (frequency_guess, v_pi_guess, frequency_guess_Hz))
    #     '''
    #     pyplot.figure()
    #     pyplot.plot(frequency_array, 10*numpy.log10(PSD), linestyle="None", marker=".")
    #     pyplot.grid()
    #     '''
    #     #hd_scanned = hd_scanned[0:int(2/frequency_guess/Ts)]
    #     #time = time[0:int(2/frequency_guess/Ts)]
    #     #time = Ts*numpy.arange(len(hd_scanned))
    #     phase_guess = 0
    #     offset_guess = 0
    #     fit_guess = [amplitude_guess, v_pi_guess, phase_guess, offset_guess]
    #     #bounds = ((amplitude_guess/2, ),(2*amplitude_guess))
    #     fit_params, fit_covariance = optimize.curve_fit(f=cos, xdata=voltages, ydata=hd_scanned, \
    #                                                     p0=fit_guess)
    #     amplitude, v_pi, phase, offset = fit_params
    #     # In[Plot the sinusoidal fit to the scanned HD output
    #     hd_scanned_fitted = cos(voltages, amplitude, v_pi, phase, offset)
    #
    #     distance = self._L2_norm(hd_scanned-hd_scanned_fitted, Ts)/numpy.sqrt(self._L2_norm(hd_scanned, Ts)*self._L2_norm(hd_scanned_fitted, Ts))
    #     if distance > 0.1:
    #         print('distance = %f' % distance)
    #         print('Try again')
    #         return self.scan_calibrate_phase_eom(voltage_range, ramp_frequency,  acquisition_channels, plot)
    #     if plot:
    #         fonts = {"title": {"fontsize": 26, "family": "Times New Roman"}, \
    #                  "axis": {"fontsize": 22, "family": "Times New Roman"}, \
    #                  "legend": {"size": 24, "family": "Times New Roman"}}
    #         fig_raw = pyplot.figure(figsize=(16, 9))
    #         axis_scanned = fig_raw.add_subplot(111)
    #         axis_scanned.plot(time * 1e6, hd_scanned, linestyle="None", marker="o", markersize=8)
    #         axis_scanned.set_xlabel("time ($\\mu$s)", fontdict=fonts["axis"])
    #         axis_scanned.set_ylabel("voltage (V)", fontdict=fonts["axis"])
    #         axis_scanned.set_title("scanned HD output", fontdict=fonts["title"])
    #         axis_scanned.grid(True)
    #         axis_scanned.plot(time * 1e6, hd_scanned_fitted, label="sinusoidal fit", linewidth=3.5)
    #         axis_scanned.legend(loc="upper right", prop=fonts["legend"])
    #     # Stop phase EOM scan
    #     self.devices['phase_eom']['instrument'].output_direct = 'off'
    #     # Prints
    #     print("frequency guess: %f"%frequency_guess)
    #     print('Phase EOM calibration parameters', fit_params)
    #     print('distance = %f'%distance)
    #     # In[Plot the phases corresponding to the locked HD output voltages]
    #     calibration_function = lambda angle_alpha: (angle_alpha - phase)/(numpy.pi/v_pi)
    #     fit_params = dict(zip(["amplitude", "v_pi", "phase", "offset"], fit_params))
    #     self.calibrations["phase_eom"] = {"function": calibration_function, "parameters": fit_params}
    #     stop_time = time.time()
    #     time_spent = stop_time - start_time
    #     minutes = time_spent // 60
    #     seconds = round(time_spent % 60)
    #     print('Time for phase EOM calibration: %d min %d sec' % (minutes, seconds))
    #     return hd_scanned[::10], hd_scanned_fitted[::10], voltages[::10]
    # # -------------------------------------------
    # def _find_phase_eom_amplification_gain(self, frequency=50):
    #     aom_high = self.devices['aoms']['levels']['lock']
    #     amplitude_eom_high = self.devices['amplitude_eom']['levels']['lock']
    #     amplification_gain, rp_offset = [self.devices['aoms'][x] for x in ['amplification_gain', 'instrument_offset']]
    #     self.devices['aoms']['instrument'].setup(waveform='dc', offset=aom_high / amplification_gain - rp_offset,
    #                                         trigger_source='immediately')
    #     amplification_gain, rp_offset = [self.devices['amplitude_eom'][x] for x in ['amplification_gain', 'instrument_offset']]
    #     self.devices['amplitude_eom']['instrument'].setup(waveform='dc', offset=amplitude_eom_high / amplification_gain - rp_offset,
    #                                         trigger_source='immediately')
    #     # Gain and offset calibration
    #     self.devices['phase_eom']['instrument'].setup(waveform='ramp', amplitude=1, \
    #                                      offset=0, frequency=frequency, \
    #                                      trigger_source='immediately')
    #     self.devices['phase_eom']['instrument'].output_direct = 'out2'
    #     traces = self.pyrpl_obj_mod.rp.scope.curve()
    #     self.pyrpl_obj_mod.rp.scope.continuous()
    #     trace_amplified = 10 * traces[0]
    #     trace_phase_eom = traces[1]
    #     amplitude_pitaya_amplified = (numpy.max(trace_amplified) - numpy.min(trace_amplified)) / 2
    #     amplitude_pitaya_phase_eom = (numpy.max(trace_phase_eom) - numpy.min(trace_phase_eom)) / 2
    #     offset_amplified = (numpy.max(trace_amplified) + numpy.min(trace_amplified)) / 2
    #     self.phase_eom_amplification_gain = amplitude_pitaya_amplified / amplitude_pitaya_phase_eom
    #     self.devices['phase_eom']['instrument'].output_direct = 'off'
    #     print('Phase EOM amplification gain = %.1f'%self.phase_eom_amplification_gain)
    # -------------------------------------------
    def _find_aoms_amplification_gain(self):
        # Set scope acquisition system
        acquisition_channels = {"hd": 1, "time-multiplexing": 2}
        acq = self.state_measurement.hd_controller.acquisition_system
        self._setup_acquisition_calibration(channels=acquisition_channels)
        channel_names = list(acquisition_channels.keys())
        for name in channel_names:
            acq.filenames[name] = name + "_aom_calibration"
        pitaya_voltages = [1, -1]
        voltage_pitaya = []
        voltage_lecroy = []
        for voltage in pitaya_voltages:
            print('Pitaya voltage = %.1f' %voltage)
            self.devices['aoms']['instrument'].setup(waveform='dc', offset=voltage, trigger_source='immediately')
            self.devices['aoms']['instrument'].output_direct = 'out1'
            # Acquire trace from Pitaya
            _trace_pitaya = self.pyrpl_obj_aom.rp.scope.curve()
            self.pyrpl_obj_aom.rp.scope.continuous()
            trace_pitaya = _trace_pitaya[0]
            # Acquire trace from Lecroy
            traces_lecroy = acq.acquire()
            _aom_voltage = traces_lecroy[channel_names[1]]
            meta = _aom_voltage[0]
            aom_voltage = _aom_voltage[2].flatten()
            voltage_pitaya.append(numpy.mean(trace_pitaya))
            voltage_lecroy.append(numpy.mean(aom_voltage))
        self.aoms_amplification_gain = (voltage_lecroy[0] - voltage_lecroy[1])/(voltage_pitaya[0] - voltage_pitaya[1])
        print('AOMs amplification gain = %.2f'%self.aoms_amplification_gain)
    # -------------------------------------------
    # def scan_phase_eom(self, frequency=2e2, voltage_range = [-10, 10]):
    #     amplitude_phase_eom_scan = (numpy.max(voltage_range) - numpy.min(voltage_range)) / (2 * self.phase_eom_amplification_gain)
    #     offset_phase_eom_scan = (numpy.max(voltage_range) + numpy.min(voltage_range)) / (2 * self.phase_eom_amplification_gain)
    #     self.devices['phase_eom']['instrument'].setup(waveform='ramp', amplitude=amplitude_phase_eom_scan,
    #                                      offset=offset_phase_eom_scan, frequency=frequency,
    #                                      trigger_source='immediately')
    #     self.devices['phase_eom']['instrument'].output_direct = 'out2'
    # # -------------------------------------------
    # def turn_off_phase_eom_scan(self):
    #     self.devices['phase_eom']['instrument'].output_direct = 'off'
    # -------------------------------------------
    def calibrate_amplitude_eom(self, voltage_range = [5, 0], n_points = 6, acquisition_channels={"hd": 1, "time-multiplexing": 2}):
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
        start_time = time.time()
        self.state_measurement.hd_controller.phase_controller.turn_off_scan()
        acq = self.state_measurement.hd_controller.acquisition_system
        aom_high, aom_medium, aom_low = [self.devices['aoms']['levels'][x] for x in ['lock', 'state_generation', 'vacuum']]
        # EOM transmits when there is no voltage and blocks when there is voltage
        eom_high = self.devices['amplitude_eom']['levels']['lock']
        eom_medium_array = numpy.linspace(voltage_range[0], voltage_range[1], n_points)
        eom_low = self.amplitude_eom_low
        # Define properties of time multiplexing signals
        aom_duty_cycles = eom_duty_cycles = self.time_multiplexing['duty_cycles']
        frequency = self.time_multiplexing['frequency']
        max_delay = self.time_multiplexing['max_delay']
        #Set up acquisition system
        self._setup_acquisition_calibration(channels=acquisition_channels)
        channel_names = list(acquisition_channels.keys())
        #Calibrate phase controller
        amplification_gain, rp_offset = [self.devices['aoms'][x] for x in ['amplification_gain', 'instrument_offset']]
        self.devices['aoms']['instrument'].waveform = "dc"
        self.devices['aoms']['instrument'].offset = self.aoms_high/amplification_gain - rp_offset
        self.state_measurement.hd_controller.phase_controller.calibrate()
        self.state_measurement.hd_controller.phase_controller.set_iq_qfactor()
        self.devices['aoms']['instrument'].offset = 0
        #define array of displacements
        self.displacements = []
        for i in range(n_points):
            if self.calibration_stopped:
                self.calibration_stopped = False
                return
            # Generate the AOM and EOM time multiplexing signals for calibration
            print("Calibrate EOMs")
            aom_levels = [aom_high, aom_medium, aom_low]
            eom_levels = [eom_high, eom_medium_array[i], eom_low]
            self._time_multiplexing_signals(max_delay, aom_levels, eom_levels, frequency, aom_duty_cycles,
                                       eom_duty_cycles)
            #Lock the homodyne detector and measure its output, together with the AOM time multiplexing signal
            phase = 45
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
                                                             Ts=Ts, dead_time=0.5e-3, plot = True)
            vac_std = numpy.std(vac)
            q_mean = numpy.mean(sig)/(vac_std / (1/numpy.sqrt(2)))
            #Analyze generalized p quadrature
            vac, sig = self._extract_quadrature_measurements(hd_output=traces_p[channel_names[0]][2][0, :], \
                                                             time_multiplexing_signal=traces_p[channel_names[1]][2][0, :], \
                                                             Ts=Ts, dead_time=0.5e-3, plot = True)
            vac_std = numpy.std(vac)
            p_mean = numpy.mean(sig) / (vac_std / (1/numpy.sqrt(2)))
            phase_rad = phase*numpy.pi/180
            #Compute and save displacement
            self.displacements.append(numpy.exp(1j*phase_rad)/numpy.sqrt(2)*(q_mean+1j*p_mean))
        # GUI plot
        def cos(x, offset, amplitude, v_pi, phase):
            return offset + amplitude * numpy.cos(numpy.pi/v_pi  * x + phase)

        print('Displacements:', numpy.abs(self.displacements))
        voltages = eom_medium_array
        if numpy.min(voltage_range) < 2.5:
            offset_guess = (numpy.max(numpy.abs(self.displacements)) + numpy.min(numpy.abs(self.displacements))) / 2
        else:
            offset_guess = numpy.max(numpy.abs(self.displacements))
        if numpy.min(voltage_range) < 2.5:
            amplitude_guess = (numpy.max(numpy.abs(self.displacements)) - numpy.min(numpy.abs(self.displacements))) / 2
        else:
            amplitude_guess = (numpy.max(numpy.abs(self.displacements)) - numpy.min(numpy.abs(self.displacements))) / 4
        v_pi_guess = self.amplitude_eom_low
        phase_guess = 0
        fit_guess = [offset_guess, amplitude_guess, v_pi_guess, phase_guess]
        print('Guessed parameters:', fit_guess)
        fit_params, fit_covariance = optimize.curve_fit(f=cos, xdata=voltages, ydata=numpy.abs(self.displacements), p0=fit_guess, maxfev=5000)
        offset, amplitude, v_pi, phase = fit_params
        amplitude_eom_fitted = cos(voltages, offset, amplitude, v_pi, phase)
        '''
        distances = numpy.zeros((len(self.displacements)))
        fit_params = numpy.zeros((4, len(self.displacements)))
        for j in range(len(self.displacements)):
            displacements_temp = numpy.delete(self.displacements[:], j)
            voltages = numpy.delete(eom_medium_array[:], j)
            if numpy.min(voltage_range) < 2.5:
                amplitude_guess = (numpy.max(numpy.abs(displacements_temp))-numpy.min(numpy.abs(displacements_temp)))/2
            else:
                amplitude_guess = (numpy.max(numpy.abs(displacements_temp))-numpy.min(numpy.abs(displacements_temp)))/4
            v_pi_guess = self.amplitude_eom_low
            phase_guess = 0
            if numpy.min(voltage_range) < 2.5:
                offset_guess = (numpy.max(numpy.abs(displacements_temp))+numpy.min(numpy.abs(displacements_temp)))/2
            else:
                offset_guess = numpy.max(numpy.abs(displacements_temp))
            fit_guess = [offset_guess, amplitude_guess, v_pi_guess, phase_guess]
            print("Guessed parameters:", fit_guess)
            fit_params[:, j], fit_covariance = optimize.curve_fit(f=cos, xdata=voltages, ydata=numpy.abs(displacements_temp), \
                                                        p0=fit_guess, maxfev=5000)
            offset, amplitude, v_pi, phase = fit_params[:, j]
            amplitude_eom_fitted = cos(voltages, offset, amplitude, v_pi, phase)
            Vs = eom_medium_array[1] - eom_medium_array[0]
            distances[j] = self._L2_norm(displacements_temp - amplitude_eom_fitted, Vs)/numpy.sqrt(self._L2_norm(displacements_temp, Vs)*self._L2_norm(amplitude_eom_fitted, Vs))
        print('All fitted parameters:', fit_params)
        print('L2 distances:', distances)
        best_fit_index = numpy.argmin(distances)
        print('Best fit index:', best_fit_index)
        fit_params = fit_params[:, best_fit_index]
        '''
        eom_medium_fit = numpy.linspace(*voltage_range, 1000)
        #amplitude_eom_fitted = []
        #for j in range(len(distances)):
        #    _fit_params = fit_params[:, j]
        #    offset, amplitude, v_pi, phase = _fit_params
        #    amplitude_eom_fitted.append(cos(eom_medium_fit, offset, amplitude, v_pi, phase))
        print('Best fitted parameters:', fit_params)
        offset, amplitude, v_pi, phase = fit_params
        amplitude_eom_fitted = cos(eom_medium_fit, offset, amplitude, v_pi, phase)
        '''
        fonts = {"title": {"fontsize": 26, "family": "Times New Roman"}, \
                 "axis": {"fontsize": 22, "family": "Times New Roman"}, \
                 "legend": {"size": 24, "family": "Times New Roman"}}
        fig_raw = pyplot.figure(figsize=(16, 9))
        axis = fig_raw.add_subplot(111)
        axis.plot(eom_medium_array, numpy.abs(self.displacements), linestyle="None", marker="o", markersize=8)
        axis.set_xlabel("Input voltage (V)", fontdict=fonts["axis"])
        axis.set_ylabel("$|\\alpha|$ (SNU)", fontdict=fonts["axis"])
        axis.set_title("Displacement x Voltage", fontdict=fonts["title"])
        axis.grid(True)
        axis.plot(eom_medium_array, amplitude_eom_fitted, label="sinusoidal fit", linewidth=3.5)
        axis.legend(loc="upper right", prop=fonts["legend"])
        '''
        calibration_function = lambda modulus_alpha: (numpy.arccos((modulus_alpha - offset) / amplitude) - phase)/(numpy.pi/v_pi)
        fit_params = dict(zip(["offset", "amplitude", "v_pi", "phase"], fit_params))
        self.calibrations["amplitude_eom"] = {"function": calibration_function, "parameters": fit_params}
        stop_time = time.time()
        time_spent = stop_time - start_time
        minutes = time_spent // 60
        seconds = round(time_spent % 60)
        print('Time for amplitude EOM calibration: %d min %d sec' % (minutes, seconds))
        return eom_medium_array, self.displacements, eom_medium_fit, amplitude_eom_fitted
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
    def tomography_mesaurement_for_calibration(self, phases=15+numpy.array([0, 30, 60, 90, 120, 150]), \
                                               acquisition_channels={"hd": 1, "time-multiplexing": 2}):
        '''
        Performs a homodyne tomography measurement with the time-multiplexing sch

        :param phases:
        :return:
        '''
        phases = numpy.array(phases)
        acq = self.state_measurement.hd_controller.acquisition_system
        aom_high, aom_medium, aom_low = [self.devices['aoms']['levels'][x] for x in ['lock', 'state_generation', 'vacuum']]
        # EOM transmits when there is no voltage and blocks when there is voltage
        amplitude_eom_high = self.devices['amplitude_eom']['levels']['lock']
        amplitude_eom_medium = self.devices['amplitude_eom']['levels']['state_generation']
        amplitude_eom_low = self.amplitude_eom_low
        # Define properties of time multiplexing signals
        aom_duty_cycles = eom_duty_cycles = self.time_multiplexing['duty_cycles']
        frequency = self.time_multiplexing['frequency']
        max_delay = self.time_multiplexing['max_delay']
        #Set up acquisition system
        self._setup_acquisition_calibration(channels=acquisition_channels)
        channel_names = list(acquisition_channels.keys())
        #Calibrate phase controller
        self.devices['aoms']['instrument'].waveform = "dc"
        self.devices['aoms']['instrument'].offset = 1
        self.state_measurement.hd_controller.phase_controller.calibrate()
        self.devices['aoms']['instrument'].offset = 0
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
        frequency, PSD = signal.welch(vac[phases[0]], fs=1 / Ts, nperseg=len(vac[phases[0]]) / 100, noverlap=len(vac[phases[0]]) / 200)
        axis.plot(frequency * 1e-6, 10 * numpy.log10(abs(PSD) / 1e-3), label='vacuum quadrature', marker=".")
        PSDs_raw['vacuum quadrature'] = PSD
        PSDs_raw['frequency'] = frequency

        for phase in phases:
            trace = sig[phase]
            frequency, PSD = signal.welch(trace, fs=1 / Ts, nperseg=len(trace) / 100, noverlap=len(trace) / 200)
            axis.plot(frequency * 1e-6, 10 * numpy.log10(abs(PSD) / 1e-3), label="phase = %.1f$^\\circ$" % phase,
                      marker=".")
            PSDs_raw[phase] = PSD

        axis.legend(loc='upper right')
        #%%

        # Do tomography
        ## Prefilter the data
        b = signal.firwin(501, cutoff=1/(3*Ts), fs=1 / Ts, pass_zero="lowpass") #cutoff used to be 20e6
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

    # -------------------------------------------
    def generate_state(self):
        #Set voltages
        for name in ['aoms', 'amplitude_eom']:#, 'phase_eom']:
            level, amplification_gain, offset = (self.devices[name]['levels']['state_generation'], \
                                                 self.devices[name]['amplification_gain'], \
                                                 self.devices[name]['instrument_offset'])
            self.devices[name]['instrument'].setup(waveform="dc", offset=level/amplification_gain-offset, trigger_source="immediately")
    # -------------------------------------------
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
    # -------------------------------------------
    def _L2_norm(self, f, Ts):
        return numpy.sqrt(Ts * numpy.sum(numpy.abs(f) ** 2))
    # -------------------------------------------
