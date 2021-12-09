"""
This module describes a cavity locking system realized with a RedPitaya pyrpl module.
"""
#%%
import pyrpl
from pyqtgraph.Qt import QtGui
from .Exceptions import ConnectionError, ResonanceNotFoundError
import time
import numpy as np
import threading
#%%
print_separator = "---------------------------------------"
#%%
class CavityLock:
    def __init__(self, redpitaya_config_filename, name="Cavity Lock", redpitaya_name="Cavity Lock Redpitaya", connect=True, show_pyrpl_GUI=True):
        self.name = name
        self.redpitaya_name = redpitaya_name
        self.redpitaya_config_filename = redpitaya_config_filename
        self.pyrpl_obj = None
        self.pyrpl_GUI = None
        self.lockbox = None
        self.high_finesse_lock = None
        if connect:
            self.connect_to_redpitaya(show_pyrpl_GUI=show_pyrpl_GUI)
     
    def connect_to_redpitaya(self, show_pyrpl_GUI=True):
        self.pyrpl_obj = pyrpl.Pyrpl(config=self.redpitaya_config_filename)
        self.lockbox = self.pyrpl_obj.lockbox
        self.scope = self.pyrpl_obj.rp.scope
        if show_pyrpl_GUI:
            self.pyrpl_GUI = QtGui.QApplication.instance()


    def lock(self):
        self.lockbox.lock()

    def lock_high_finesse(self):
        if not self.high_finesse_lock:
            self.high_finesse_lock = HighFinesseCavityLock(self.pyrpl_obj)
        self.high_finesse_lock.scan()
        self.high_finesse_lock.set_iq_phase()
        self.high_finesse_lock.lock()

    def scan(self):
        self.lockbox.sweep()

    def unlock(self):
        self.lockbox.unlock()
#%%
class HighFinesseCavityLock:
    """
    This class describes a custom lock based on RedPitaya pyrpl, used to lock a high-finesse cavity.
    """

    def __init__(self, pyrpl_obj):
        """

        :param pyrpl_obj: pyrpl.Pyrpl object associated to the RedPitaya that performs the cavity lock
        """
        time.sleep(1)
        self.pyrpl_obj = pyrpl_obj
        self.redpitaya = self.pyrpl_obj.rp
        self.error_signal_input = 'in1'  # typically the broadband output of the hodyne detector
        self.auxiliary_input = 'in2'  # typically the DC output of the homodyne detector
        self.error_signal = 'iq0'
        self.control_signal_coarse = 'out1'
        self.control_signal_fine = 'out2'
        # Define some useful variables
        self.scanning_frequency = 60  # Hz
        self.modulation_frequency = 28.5e6
        self.error_signal_amplitude_scanned = 0
        self.error_signal_amplitude = 0
        self.input_signal_max = 0
        self.is_scanning = False
        self.is_locking = False
        time.sleep(0.05)
        # Set up asgs
        self.asg0 =self.redpitaya.asg0
        self.asg1 =self.redpitaya.asg1
        self.setup_ASG0()
        # Set up scope
        self.scope =self.redpitaya.scope
        self.setup_scope()
        # Set up PIDs
        self.pid0 =self.redpitaya.pid0
        self.pid1 =self.redpitaya.pid1
        self.pid2 =self.redpitaya.pid2
        self.setup_PID0()
        # self.setup_PID1()
        # self.setup_PID2()
        # Setup  iq
        self.iq0 =self.redpitaya.iq0
        self.setup_IQ0()

        # Set locks off
        self.unlock()
        self.figures_count = 0

    def setup_PID0(self):
        self.pid0.input = self.error_signal
        self.pid0.output_direct = 'off'
        # self.pid0.inputfilter = [2e3, 2e3, 0, 0]
        # Set the initial proportional value such that the curve fits in a fraction of the scope range
        self.pid0.setpoint = 0
        self.pid0.p = 4.8828e-04
        self.pid0.i = 2.8255e-01
        self.pid0.ival = 0

    def setup_PID1(self):
        self.pid1.input = 'iq0'
        self.pid1.output_direct = 'off'
        self.pid1.inputfilter = [0, 0, 0, 0]
        self.pid1.p = -1.0039e-02
        self.pid1.i = -1e-2
        self.pid1.ival = 0

    def setup_PID2(self):
        self.pid2.input = self.error_signal
        self.pid2.output_direct = 'off'
        self.pid2.inputfilter = [2e3, 2e3, 0, 0]
        self.pid2.p = 1
        self.pid2.i = 0
        self.pid2.ival = 0

    def enable_PID0(self):
        self.pid0.output_direct = self.control_signal_coarse

    def setup_ASG0(self):
        self.asg0.waveform = 'ramp'
        self.asg0.amplitude = 1
        self.asg0.trigger_source = 'immediately'
        self.asg0.offset = 0
        self.asg0.frequency = self.scanning_frequency

    def setup_ASG1(self):
        self.asg1.waveform = 'ramp'
        self.asg1.amplitude = 1
        self.asg1.trigger_source = 'immediately'
        self.asg1.offset = 0
        self.asg1.frequency = self.scanning_frequency

    def setup_IQ0(self):
       # self.iq0.free()
        self.iq0.input = self.error_signal_input
        self.iq0.acbandwidth = 0.8 * self.modulation_frequency
        self.iq0.bandwidth = [2e3, 2e3]
        self.iq0.quadrature_factor = 100
        self.iq0.gain = 0
        self.iq0.amplitude = 0.5
        self.iq0.phase = 0
        self.iq0.output_direct = 'off'
        self.iq0.output_signal = 'quadrature'

    def setup_scope(self):
        self.scope.duration = 0.002
        self.scope.trigger_source = 'ch2_positive_edge'
        self.scope.threshold = 0
        self.scope.trigger_delay = 0
        # self.scope.average = False
        self.scope.input1 = self.error_signal_input
        self.scope.input2 = self.control_signal_coarse

    def get_scope_curve(self, channel=1, fast=True):
        """
        This function gets the scope's rolling curve values for
        the selected channels, excluding the nan values
        """
        if fast:
            self.scope._start_acquisition()
            time.sleep(self.scope.duration)
            trace = self.scope._get_curve()[channel - 1]
        else:
            trace = self.scope.curve()[channel - 1]
            self.scope.continuous()
        """
        if self.figures_count < 10:
            plt.figure()
            plt.plot(trace)
            plt.show()
            self.figures_count += 1
        """
        return trace

    def unlock(self, signal_type='all'):
        self.pid0.output_direct = 'off'
        self.turn_off_scan(signal_type)
        self.is_locking = False

    def turn_off_scan(self, signal_type='all'):
        if signal_type == 'coarse':
            self.asg0.output_direct = 'off'
        elif signal_type == 'fine':
            self.asg1.output_direct = 'off'
        else:
            self.asg0.output_direct = 'off'
            self.asg1.output_direct = 'off'
        self.scope.trigger_source = 'immediately'
        self.is_scanning = False

    def scan(self, scan_type='coarse'):
        # self.unlock(signal_type=signal_type)
        if scan_type == 'coarse':
            self.setup_ASG0()
            self.asg0.output_direct = self.control_signal_coarse
            self.scope.trigger_source = "asg0"
        else:
            self.setup_ASG1()
            self.asg1.output_direct = self.control_signal_fine
            self.scope.trigger_source = "asg1"
        self.is_scanning = True

    def set_iq_phase(self):
        was_not_scanning = not self.is_scanning

        self.scan()
        self.scope.input1 = self.error_signal_input
        self.scope.input2 = self.error_signal

        x_phase = np.linspace(0, 175, 36)
        amplitudes = np.zeros(36)
        for i, k in enumerate(x_phase):
            self.iq0.phase = k
            trace = self.get_scope_curve(channel=2)
            amplitudes[i] = np.max(trace) - np.min(trace)
        getattr(self, self.error_signal).phase = x_phase[np.argmax(amplitudes)]  # set the error signal iq module phase

        if was_not_scanning:
            self.turn_off_scan()

    def flip_phase(self):
        self.iq0.phase = np.mod(self.iq0.phase + 180, 360)

    def get_signal_amplitude(self, signal_name=None, fast_acquisition=True):
        """¨
        This function does the following:
            - scan
            - acquire a trace from the error signal
            - return the amplitude of the error signal
        """
        # previous_scope_input = self.scope.input1
        if signal_name is not None:
            self.scope.input1 = signal_name
        # Get trace
        trace = self.get_scope_curve(channel=1, fast=fast_acquisition)
        # Compute the amplitude
        signal_amplitude = np.max(trace) - np.min(trace)

        return signal_amplitude

    def get_scanned_signal_amplitude(self, signal_name, scan_type='coarse'):

        """¨
        This function does the following:
            - acquire a trace from the error signal
            - return the amplitude of the error signal
        """
        was_not_scanning = not self.is_scanning

        self.scan(scan_type)

        signal_amplitude_scanned = self.get_signal_amplitude(signal_name=signal_name, fast_acquisition=False)
        if was_not_scanning:
            self.turn_off_scan()

        return signal_amplitude_scanned

    def calibrate(self):
        self.set_iq_phase()
        return

    def lock(self, keep_locked=True):
        self.unlock()
        # Search for resonance -coarse stage
        try:
          self.search_resonance(search_type='coarse', relative_threshold=0.5)
        except ResonanceNotFoundError:
            pass
        # Lock
        # coarse
        self.pid0.ival = 0
        self.pid0.input = self.error_signal
        self.pid0.output_direct = self.control_signal_coarse

        # fine
        self.pid1.ival = 0
        self.pid1.input = self.error_signal
        self.pid1.output_direct = self.control_signal_fine
        self.scope.duration = 0.04
        return

    def search_resonance(self, search_type, relative_threshold):
        """
        This function slowly scans the cavity to find its resonance.
        INPUTS
        -----------
            search_type : str
                - 'coarse': uses self.control_signal_coarse to look for resonance (should be used first)
                - 'fine': uses self.control_signal_fine to look for resonance (should be used after 'coarse')
            relative_threshold : float (in [0, 1])
                The search stops if the current signal level is greater or equal than the resonance height times relative_threshold.
        OUTPUTS
        -----------
            None
        """
        """
        figure = plt.figure()
        axis = figure.add_subplot(111)
        line = axis.plot(0, -np.inf)
        values = [-np.inf]
        """
        asg = None
        scope_duration = None
        if search_type == 'coarse':
            asg = self.asg0
            scope_duration = 0.002
            step = 0.000122071
        else:
            asg = self.asg1
            scope_duration = 0.2
            step = 0.001
        self.scope.duration = scope_duration
        output_direct = getattr(self, 'control_signal_' + search_type)
        print('Searching resonance for ' + search_type + ' search')
        # First get the resonance peak height
        if search_type == 'coarse':
            self.resonance_height = self.get_scanned_signal_amplitude(signal_name=self.error_signal_input,
                                                                      scan_type=search_type)
            print('Resonance height: %0.6f' % self.resonance_height)
        else:
            resonance_height = -np.inf
            while resonance_height < 0.9 * self.resonance_height:
                self.scope.trigger_delay -= 0.001
                resonance_height = self.get_scanned_signal_amplitude(signal_name=self.error_signal_input,
                                                                     scan_type=search_type)
                print('Resonance height: %0.6f' % self.resonance_height)
            self.resonance_height = resonance_height
        # Setup scope
        self.scope.input1 = self.error_signal_input
        self.scope.input2 = output_direct
        self.scope.trigger_source = 'immediately'
        # set up the asg
        asg.waveform = 'dc'
        asg.offset = -1
        asg.output_direct = output_direct
        height = -np.inf
        height_temp = None
        while (height <= relative_threshold * self.resonance_height) and (asg.offset <= 0.99):
            asg.offset += step
            height_temp = self.scope.voltage_in1
            if height_temp >= height:
                height = height_temp
            """
            if search_type=='fine':
                values.append(height)
                line.set_y(values)
                plt.draw()
                plt.flush_events()
                plt.pause(0.001)
            """
        if asg.offset > 0.99:
            raise ResonanceNotFoundError('Resonance search for ' + search_type + ' stage failed')
        print("Found resonance height: %0.6f" % height)
        print(print_separator)
        return