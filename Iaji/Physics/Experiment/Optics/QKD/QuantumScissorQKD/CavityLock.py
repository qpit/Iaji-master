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
    def __init__(self, redpitaya_config_filename, name="Cavity Lock", redpitaya_name="Cavity Lock Redpitaya", connect=True, show_pyrpl_GUI=True, lock_type="regular"):
        """

        :param redpitaya_config_filename: str
        :param name: str
        :param redpitaya_name: str
        :param connect: bool
        :param show_pyrpl_GUI: bool
        :param lock_type: str
            Type of lock.
            Valid arguments are:
                - "regular"
                - "high finesse"
        """
        self.name = name
        self.redpitaya_name = redpitaya_name
        self.redpitaya_config_filename = redpitaya_config_filename
        self.lock_type = lock_type
        self.pyrpl_obj = None
        self.pyrpl_GUI = None
        self.lockbox = None
        self.high_finesse_lock = None
        if connect:
            self.connect_to_redpitaya(show_pyrpl_GUI=show_pyrpl_GUI)
            self.high_finesse_lock = HighFinesseCavityLock(self.pyrpl_obj)
            self.lockbox = self.pyrpl_obj.lockbox
            self.scope = self.pyrpl_obj.rp.scope
     
    def connect_to_redpitaya(self, show_pyrpl_GUI=True):
        self.pyrpl_obj = pyrpl.Pyrpl(config=self.redpitaya_config_filename)
        if show_pyrpl_GUI:
            self.pyrpl_GUI = QtGui.QApplication.instance()


    def lock(self):
        if self.lock_type is "high finesse":
            self.high_finesse_lock.scan()
            self.high_finesse_lock.set_demodulation_phase()
            self.high_finesse_lock.lock()
        else:
            self.lockbox.lock()

    def scan(self):
        if self.lock_type == "high finesse":
            print("High finesse lock scan")
            self.high_finesse_lock.scan()
        else:
            print("sweep")
            self.lockbox.sweep()

    def unlock(self):
        if self.lock_type == "high finesse":
            self.high_finesse_lock.unlock()
        else:
            self.lockbox.unlock()

    def set_demodulation_phase(self):
        if self.lock_type == "high finesse":
            self.high_finesse_lock.set_demodulation_phase()
        else:
            self.lockbox.calibrate_all()

    def calibrate(self):
        if self.lock_type == "high finesse":
            self.high_finesse_lock.calibrate()
        else:
            self.lockbox.calibrate_all()

    def flip_phase(self):
        if self.lock_type == "high finesse":
            self.high_finesse_lock.flip_phase()

    def set_lock_type(self, lock_type):
        """

        :param lock_type: str
            Type of lock.
            Valid arguments are:
                - "regular"
                - "high finesse"
        :return:
        """
        self.lock_type = lock_type
#%%
class HighFinesseCavityLock:
    """
    This class describes a custom lock based on RedPitaya pyrpl, used to lock a high-finesse cavity.
    """

    def __init__(self, pyrpl_obj, name="High-finesse Cavity Lock"):
        """

        :param pyrpl_obj: pyrpl.Pyrpl object associated to the RedPitaya that performs the cavity lock
        """
        time.sleep(1)
        self.pyrpl_obj = pyrpl_obj
        self.name = name
        self.redpitaya = self.pyrpl_obj.rp
        #Setup modules and signals
        self.assign_modules()
        self.assign_input_output()
        # Define some useful variables
        self.scanning_frequency = 60  # Hz
        self.modulation_frequency = 28.5e6
        self.error_signal_amplitude_scanned = 0
        self.error_signal_amplitude = 0
        self.input_signal_max = 0
        self.is_scanning = False
        self.is_locking = False
        time.sleep(0.05)
        #Set up scope
        self.scope =self.redpitaya.scope
        self.setup_scope()
        #Set up PIDs
        self.setup_pid_coarse()
        self.setup_pid_fine()
        # Setup  iq
        self.setup_iq()
        #Setup asgs
        self.setup_asg_coarse()
        self.setup_asg_fine()
        # Set locks off
        self.unlock()
        self.figures_count = 0

    def assign_modules(self, asg_coarse="asg0", iq="iq0", pid_coarse="pid0"):
        self.asg_coarse = getattr(self.redpitaya, asg_coarse)
        self.asg_fine = getattr(self.redpitaya, "asg" + str(0 + 1 * (asg_coarse == "asg0")))
        self.iq = getattr(self.redpitaya, iq)
        self.error_signal = iq
        self.pid_coarse = getattr(self.redpitaya, pid_coarse)
        self.pid_fine = getattr(self.redpitaya, "pid"+ str(0+1*(pid_coarse=="pid0")))

    def assign_input_output(self, error_signal_input="in1", control_signal_output_coarse="out1"):
        #Input
        self.error_signal_input = error_signal_input
        self.auxiliary_input = "in" + str(1+1*(error_signal_input=="in1"))
        #Output
        self.control_signal_output_coarse = control_signal_output_coarse
        self.control_signal_output_fine = "out" + str(1+1*(control_signal_output_coarse=="out1"))

    def setup_pid_coarse(self):
        self.pid_coarse.input = self.error_signal
        self.pid_coarse.output_direct = 'off'
        # self.pid_coarse.inputfilter = [2e3, 2e3, 0, 0]
        # Set the initial proportional value such that the curve fits in a fraction of the scope range
        self.pid_coarse.setpoint = 0
        self.pid_coarse.p = 4.8828e-04
        self.pid_coarse.i = 2.8255e-01
        self.pid_coarse.ival = 0

    def setup_pid_fine(self):
        self.pid_fine.input = self.iq.name
        self.pid_fine.output_direct = 'off'
        self.pid_fine.inputfilter = [0, 0, 0, 0]
        self.pid_fine.p = -1.0039e-02
        self.pid_fine.i = -1e-2
        self.pid_fine.ival = 0


    def enable_pid_coarse(self):
        self.pid_coarse.output_direct = self.control_signal_output_coarse

    def enable_pid_fine(self):
        self.pid_fine.output_direct = self.control_signal_output_fine

    def setup_asg_coarse(self):
        self.asg_coarse.waveform = 'ramp'
        self.asg_coarse.amplitude = 1
        self.asg_coarse.trigger_source = 'immediately'
        self.asg_coarse.offset = 0
        self.asg_coarse.frequency = self.scanning_frequency

    def setup_asg_fine(self):
        self.asg_fine.waveform = 'ramp'
        self.asg_fine.amplitude = 1
        self.asg_fine.trigger_source = 'immediately'
        self.asg_fine.offset = 0
        self.asg_fine.frequency = self.scanning_frequency

    def setup_iq(self):
        self.iq.free()
        self.iq.input = self.error_signal_input
        self.iq.acbandwidth = 0.8 * self.modulation_frequency
        self.iq.bandwidth = [2e4, 2e4]
        self.iq.quadrature_factor = 100
        self.iq.gain = 0
        self.iq.amplitude = 0.5
        self.iq.phase = 0
        self.iq.output_direct = 'off'
        self.iq.output_signal = 'quadrature'

    def setup_scope(self):
        self.scope.duration = 0.002
        self.scope.trigger_source = 'ch2_positive_edge'
        self.scope.threshold = 0
        self.scope.trigger_delay = 0
        # self.scope.average = False
        self.scope.input1 = self.error_signal_input
        self.scope.input2 = self.control_signal_output_coarse

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
        self.pid_coarse.output_direct = 'off'
        self.turn_off_scan(signal_type)
        self.is_locking = False

    def turn_off_scan(self, signal_type='all'):
        if signal_type == 'coarse':
            self.asg_coarse.output_direct = 'off'
        elif signal_type == 'fine':
            self.asg_fine.output_direct = 'off'
        else:
            self.asg_coarse.output_direct = 'off'
            self.asg_fine.output_direct = 'off'
        self.scope.trigger_source = 'immediately'
        self.is_scanning = False

    def scan(self, scan_type='coarse'):
        # self.unlock(signal_type=signal_type)
        if scan_type == 'coarse':
            self.setup_asg_coarse()
            self.asg_coarse.output_direct = self.control_signal_output_coarse
            self.scope.trigger_source = self.asg_coarse.name
        else:
            self.setup_asg_fine()
            self.asg_fine.output_direct = self.control_signal_output_fine
            self.scope.trigger_source = self.asg_fine.name
        self.is_scanning = True

    def set_demodulation_phase(self):
        was_not_scanning = not self.is_scanning

        self.scan()
        self.scope.input1 = self.error_signal_input
        self.scope.input2 = self.error_signal

        x_phase = np.linspace(0, 175, 36)
        amplitudes = np.zeros(36)
        for i, k in enumerate(x_phase):
            self.iq.phase = k
            trace = self.get_scope_curve(channel=2)
            amplitudes[i] = np.max(trace) - np.min(trace)
        self.iq.phase = x_phase[np.argmax(amplitudes)]  # set the error signal iq module phase

        if was_not_scanning:
            self.turn_off_scan()

    def flip_phase(self):
        self.iq.phase = np.mod(self.iq.phase + 180, 360)

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
        self.set_demodulation_phase()
        return

    def lock(self):
        self.unlock()
        # Search for resonance -coarse stage
        self.search_resonance(search_type='coarse', relative_threshold=0.5)
        # Lock
        # coarse
        self.pid_coarse.p = 5e-4
        self.pid_coarse.i = 0.5
        self.pid_coarse.ival = 0
        self.enable_pid_coarse()
        time.sleep(1)
        # fine
        self.pid_fine.ival = 0
        self.pid_fine.p = 0
        self.pid_fine.i = 0
        self.enable_pid_fine()
        time.sleep(0.5)
        self.pid_fine.p = -0.5
        time.sleep(0.5)
        self.pid_coarse.p = 0
        time.sleep(0.5)
        self.pid_fine.i = -1
        time.sleep(0.5)
        self.pid_coarse.i = 0
        self.scope.duration = 0.04
        return

    def search_resonance(self, search_type, relative_threshold):
        """
        This function slowly scans the cavity to find its resonance.
        INPUTS
        -----------
            search_type : str
                - 'coarse': uses self.control_signal_output_coarse to look for resonance (should be used first)
                - 'fine': uses self.control_signal_output_fine to look for resonance (should be used after 'coarse')
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
            asg = self.asg_coarse
            scope_duration = 0.002
            step = 0.000122071
        else:
            asg = self.asg_fine
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