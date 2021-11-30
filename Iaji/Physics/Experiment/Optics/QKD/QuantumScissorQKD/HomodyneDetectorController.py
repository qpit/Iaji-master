"""
This class describes a homodyne detector used in the experiment.
It consists of:
    - A phase controller, associated to a RedPitaya that controls the interferometer's phase
"""
#%%
import pyrpl
from pyqtgraph.Qt import QtGui
from .Exceptions import ConnectionError, ResonanceNotFoundError
import time
import numpy as np
import threading
#%%
class HomodyneDetectorController:
    def __init__(self, redpitaya_config_filename, name="Homodyne Detector", redpitaya_name="Homodyne Detector Lock Redpitaya", connect=True, show_pyrpl_GUI=True):
        self.redpitaya_name = redpitaya_name
        self.redpitaya_config_filename = redpitaya_config_filename
        self.pyrpl_obj = None
        self.pyrpl_GUI = None
        if connect:
            self.connect_to_redpitaya(show_pyrpl_GUI=show_pyrpl_GUI)
            self.phase_controller = PhaseController(self.pyrpl_obj)

    def connect_to_redpitaya(self, show_pyrpl_GUI=True):
        self.pyrpl_obj = pyrpl.Pyrpl(config=self.redpitaya_config_filename)
        if show_pyrpl_GUI:
            self.pyrpl_GUI = QtGui.QApplication.instance()
        return

    def scan(self):
        self.phase_controller.scan()

    def lock(self):
        self.phase_controller.lock()

    def unlock(self):
        self.phase_controller.unlock()

    def calibrate_lock(self):
        self.phase_controller.calibrate()


#%%
class PhaseController:
    def __init__(self, pyrpl_obj, enable_modulation_output=False):
        self.redpitaya = pyrpl_obj.rp
        self.modulation_output_on = enable_modulation_output
        self.assign_modules()
        self.assign_input()
        # Define some useful variables
        self.scanning_frequency = 5  # Hz
        self.modulation_frequency = 25e6  # [Hz]
        self.phase_LO = 0
        self.error_signal_amplitude_scanned = 0
        self.error_signal_amplitude = 0
        self.auxiliary_input_amplitude_scanned = 0
        self.is_scanning = False
        self.is_locking = False
        time.sleep(0.05)
        # Set up asgs
        self.setup_asg_control()
        # Set up scope
        self.scope = self.redpitaya.scope
        self.setup_scope()
        # Set up PIDs
        try:
            self.setup_pid_DC()
        except OverflowError:
            print('Warning in HD_quantum_scissors_QKD_lockbox.__init__: could not setup pid_DC because the scanned signal amplitude is invalid.')
        self.setup_pid_AC()
        self.setup_pid_control()
        # Setup  iq
        self.iq = self.redpitaya.iq
        self.setup_iq()
        # Set locks off
        self.unlock()

    def assign_modules(self, asg_control="asg_control", iq="iq", pid_DC="pid_DC", pid_AC="pid_AC"):
        self.asg_control = getattr(self.redpitaya, asg_control)
        asg_modulation = asg_control[:-1] + str(np.mod(int(asg_control[-1])+1, 2))
        self.asg_modulation = getattr(self.redpitaya, asg_modulation)
        self.control_signal_output = asg_control
        self.iq = getattr(self.redpitaya, iq)
        self.AC_error_signal = iq
        self.pid_DC = getattr(self.redpitaya, pid_DC)
        self.DC_error_signal = pid_DC
        self.pid_AC = getattr(self.redpitaya, pid_AC)
        self.error_signal = "iq2"


    def assign_input(self, error_signal_input="in2"):
        self.error_signal_input = error_signal_input
        self.auxiliary_input = error_signal_input[:-1] + str(np.mod(int(error_signal_input[-1])+1, 2))


    def setup_pid_DC(self):
        self.pid_DC.input = self.error_signal_input
        self.pid_DC.output_direct = 'off'
        self.pid_DC.inputfilter = [2e3, 2e3, 0, 0]
        # Set the initial proportional value such that the curve fits in a fraction of the scope range

        self.pid_DC.p = 1 / 5 * (2 / self.get_scanned_signal_amplitude(signal_name=self.error_signal_input))
        self.pid_DC.i = 0
        self.pid_DC.ival = 0
        self.pid_DC_p_initial = self.pid_DC.p

    def setup_pid_AC(self):
        self.pid_AC.input = 'iq'
        self.pid_AC.output_direct = 'off'
        self.pid_AC.inputfilter = [0, 0, 0, 0]
        self.pid_AC.p = 1
        self.pid_AC.i = 0
        self.pid_AC.ival = 0

    def setup_pid_control(self):
        self.pid_control.input = self.error_signal
        self.pid_control.output_direct = 'off'
        self.pid_control.inputfilter = [2e3, 2e3, 0, 0]
        self.pid_control.p = 1
        self.pid_control.i = 0
        self.pid_control.ival = 0

    def enable_pid_control(self):
        self.pid_control.output_direct = "on"

    def setup_asg_control(self):
        self.asg_control.waveform = 'ramp'
        self.asg_control.amplitude = 0.6
        self.asg_control.trigger_source = 'immediately'
        self.asg_control.offset = 0
        self.asg_control.frequency = self.scanning_frequency

    def setup_iq(self):
        self.iq.input = self.error_signal_input
        self.iq.acbandwidth = 0.8 * self.modulation_frequency
        self.iq.bandwidth = [2e3, 2e3]
        self.iq.quadrature_factor = 20
        self.iq.gain = 0
        self.iq.amplitude = 0.5
        self.iq.phase = 0
        self.iq.output_direct = "off"
        if self.modulation_output_on:
            self.iq.output_direct = self.modulation_signal_output
        self.iq.output_signal = 'quadrature'

    def setup_scope(self):
        self.scope.duration = 4 / self.asg_control.frequency / 50
        self.scope.trigger_source = 'asg_control'
        self.scope.threshold = 0
        # self.scope.average = False
        self.scope.input1 = 'in1'
        self.scope.input2 = 'in2'

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
        """
        plt.figure()
        plt.plot(trace)
        plt.show()
        """
        return trace

    def unlock(self):
        self.pid_control.output_direct = 'off'
        self.asg_control.output_direct = 'off'
        self.asg1.output_direct = 'off'
        self.is_scanning = False
        self.is_locking = False

    def turn_off_scan(self):
        self.asg_control.output_direct = 'off'
        self.is_scanning = False

    def scan(self):
        self.unlock()
        self.asg_control.output_direct = self.control_signal_output
        self.is_scanning = True

    def set_iq_phase(self):
        was_not_scanning = not self.is_scanning

        self.scan()
        self.scope.input1 = self.DC_error_signal
        self.scope.input2 = 'iq'

        x_phase = np.linspace(0, 175, 36)
        amplitudes = np.zeros(36)
        for i, k in enumerate(x_phase):
            self.iq.phase = k
            trace = self.get_scope_curve(channel=2)
            amplitudes[i] = np.max(trace) - np.min(trace)
        self.iq.phase = x_phase[np.argmax(amplitudes)]

        if was_not_scanning:
            self.turn_off_scan()

    def set_iq_qfactor(self):
        was_not_scanning = not self.is_scanning

        self.scan()
        self.scope.input1 = self.DC_error_signal
        self.scope.input2 = 'iq'

        amplitude_pid_DC = self.get_signal_amplitude(signal_name=self.DC_error_signal)
        amplitude_iq = self.get_signal_amplitude(signal_name='iq')
        self.iq.quadrature_factor *= amplitude_pid_DC / amplitude_iq

        if was_not_scanning:
            self.turn_off_scan()

    def remove_offset_pid_DC(self):

        was_scanning = self.is_scanning

        self.scope.input1 = self.error_signal_input
        self.scope.input2 = self.DC_error_signal

        if was_scanning:
            self.turn_off_scan()
        self.external_signal_generator_input_state.turn_off()
        time.sleep(0.2)
        trace = self.get_scope_curve(channel=1)
        self.pid_DC.setpoint = np.mean(trace)

        self.external_signal_generator_input_state.turn_on()

        if was_scanning:
            self.scan()

    def flip_iq_phase(self):
        self.iq.phase = np.mod(self.iq.phase + 180, 360)

    def get_signal_amplitude(self, signal_name=None):
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
        trace = self.get_scope_curve(channel=1)
        # Compute the amplitude
        signal_amplitude = np.max(trace) - np.min(trace)

        return signal_amplitude

    def get_scanned_signal_amplitude(self, signal_name):

        """¨
        This function does the following:
            - acquire a trace from the error signal
            - return the amplitude of the error signal
        """
        was_not_scanning = not self.is_scanning

        self.scan()

        signal_amplitude_scanned = self.get_signal_amplitude(signal_name=signal_name)
        if was_not_scanning:
            self.turn_off_scan()

        return signal_amplitude_scanned

    def lock(self, keep_locked=True):
        self.unlock()
        # Get the amplitude of the scanned error signal

        self.auxiliary_input_amplitude_scanned = self.get_scanned_signal_amplitude(signal_name=self.auxiliary_input)
        time.sleep(0.5)  # this pause prevents detecting a false ringing at the beginning of the control algorithm
        print("Amplitude of the scanned auxiliary input signal: %0.3f" % self.auxiliary_input_amplitude_scanned)

        self.is_locking = True
        self.PID_manual_autotune(keep_locked)

    def PID_manual_autotune(self, keep_locked=True):
        """
        This function attempts to replicate a manual (human-made) autotune of the homodyne detection PI control.
        The procedure is the following:
            1) Start from P=I=ival=0
            2) Increase P until the lock rings
            3) Decrease P until the lock does not ring anymore
            4) Increase I until the lock rings
            5) Decrease I until the lock does not ring anymore
        """

        # Phase 1): Initialize PID parameters
        self.pid_control.p = 0
        self.pid_control.i = 0
        self.pid_control.ival = 0
        initial_p = 0.5e-1
        initial_i = 0.5e-1

        # Set up PID connections
        self.pid_control.input = self.error_signal
        self.pid_control.output_direct = self.control_signal_output

        # Set up scope connections
        self.scope.input1 = self.error_signal
        self.scope.input2 = self.error_signal_input

        # Phase 2)
        # --------------
        print("Phase 2): increase P until ringing" + print_separator)
        self.pid_control.p = initial_p
        while not self.is_lock_ringing():
            self.pid_control.p *= 2
            print('P = %0.4f' % self.pid_control.p)
        print("Phase 2): lock is ringing" + print_separator)
        # ---------------
        # Phase 3)
        # ------------------
        while self.is_lock_ringing():
            self.pid_control.p *= 0.8
            print('P = %0.4f' % self.pid_control.p)
        self.pid_control.p *= 0.9 ** 3  # reduce a bit further for safety
        # ----------------
        # Phase 4)
        # ----------------
        print("Phase 4): increase I until ringing" + print_separator)
        self.pid_control.i = initial_i
        while not self.is_lock_ringing():
            self.pid_control.i *= 2
            print('I = %0.4f' % self.pid_control.i)
        print("Phase 4): lock is ringing" + print_separator)
        # --------------
        # Phase 5)
        # --------------
        while self.is_lock_ringing():
            self.pid_control.i *= 0.8
            print('P = %0.4f' % self.pid_control.i)
            # -------------
        self.pid_control.i *= 0.9 ** 2  # reduce a bit further for safety

        print("Locked" + print_separator)

        if not keep_locked:
            self.unlock()

    def is_lock_ringing(self, relative_threshold=0.1):
        """
        This scope checks whether the homodyne detection lock is ringing by
        checking if the current amplitude of the error signal is above a certain
        fraction of the error signal´s scanned amplitude.

        INPUTS
        ----------
            relative_threshold : float (>0)
                fraction of the scanned error signal amplitude, above which ringing is detected

        OUTPUTS
        ----------
        True if error_signal_amplitude >= relative_threshold * self.error_signal_amplitude_scanned
        """

        if not self.is_locking:
            print("Warning in homodyne detection lock: I am checking for lock ringing, but I am not locking now.")
            return False

        return self.get_signal_amplitude(
            signal_name=self.auxiliary_input) >= relative_threshold * self.auxiliary_input_amplitude_scanned

    def set_LO_phase(self, x):
        theta = x * np.pi / 180
        self.phase_LO = theta
        self.pid_AC.p = self.pid_DC_p_initial * np.cos(theta)
        self.pid_DC.p = self.pid_DC_p_initial * np.sin(theta)

    def measure_quadrature(self, phase=None):
        """
        This function measures a single quadrature of the signal field.

        INPUTS
        ---------
            phase : float
                Phase of the local oscillator [degrees]
        """
        if phase is None:
            phase = self.phase_LO * 180 / np.pi
        # Lock to the specified phase
        self.set_LO_phase(phase)
        self.remove_offset_pid_DC()
        time.sleep(0.2)
        self.lock(keep_locked=True)
        # self.lock(keep_locked=True) #sometimes it needs to hits of the "lock" button
        # Acquire the quadrature
        time.sleep(2)
        self.external_oscilloscope.saveTrace(filename=str(phase) + "deg", channels="C1")
        self.external_oscilloscope.display_continuous()
        # Acquire the vacuum quadrature trace
        self.external_signal_generator_input_state.turn_off()
        time.sleep(1)
        self.external_oscilloscope.saveTrace(filename="vacuum", channels="C1")  # vacuum quadrature measurement
        self.external_oscilloscope.display_continuous()
        self.external_signal_generator_input_state.turn_on()
        time.sleep(0.2)

    def quantum_state_tomography(self, phases=[0, 30, 60, 90, 120, 150]):

        """
        This function perform quadrature measurements on a single quadrature of the signal field,
        for different phases of the local oscillator, aimed at tomographyc state reconstruction
        based on homodyne detection. The assumptions on the RedPitaya configuration are the same as
        in the function self.measure_quadrature.
        """
        for phase in phases:
            self.measure_quadrature(phase)
        return

    def save_oscilloscope_trace(self):
        """
        This functions saves a trace from the external oscilloscope.
        """
        self.external_oscilloscope.saveTrace()
        self.external_oscilloscope.display_continuous()

    def turn_off_signal_generator_input_state(self):
        """
        This function turns OFF the signal generator output related to the AOM drivers of the
        input coherent state
        """
        self.external_signal_generator_input_state.turn_off()

    def calibrate(self):
        self.remove_offset_pid_DC()
        time.sleep(0.2)
        self.setup_pid_DC()
        time.sleep(0.2)
        self.setup_pid_AC()
        time.sleep(0.2)
        self.set_iq_phase()
        time.sleep(0.2)
        self.set_iq_qfactor()
        time.sleep(0.2)
        self.setup_pid_control()
