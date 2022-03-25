#%%
import time
import numpy as np
import threading
import pyrpl
from pyqtgraph.Qt import QtGui

print_separator = "-------------------------------------------------------------------"
#%%
class PhaseController:
    """
    This class describes a phase controller for a Mach-Zender interferometer, physically implemented with a redpitaya.
    It controls the phase difference between the two interfering signals (local oscillator and proper signal).
    It can:
    - scan the phase
    - lock the phase to an arbitrary value

    Locking the interferometer to an arbitrary phase is done by providing to the PID controller
    a convex combination of the carrier-frequency error signal
    and the modulation sideband-frequency error signal, demodulated at the sideband frequency, so that the total error signal
    has the form:

        e(t) \propto cos(theta)cos(phi) + sin(theta)sin(phi) \propto cos(theta - phi)

    where theta is the interferometer phase and phi is the demodulation phase. By tuning the coefficients of the
    convex combination one can lock to an arbitrary theta, by locking e(t) at 0 value.
    """
    def __init__(self, redpitaya_config_filename, name="Phase Controller", enable_modulation_output=False, pid_autotune=True, \
                 show_pyrpl_GUI=True):
        """
        :param redpitaya_config_filename: str
            object representing the pyrpl interface for the controlling RedPitaya
        :param name: str
            name of the phase controller
        :param modulation_output_enabled: bool
            If True, the controlling RedPitaya outputs a phase modulation signal
        :param pid_autotune:
            If True, the locking system uses an automatic software procedure to tune the PID parameters.
        :param show_pyrpl_GUI: bool
        """
        self.redpitaya_config_filename = redpitaya_config_filename
        self.connect_to_redpitaya(show_pyrpl_GUI=show_pyrpl_GUI)
        self.redpitaya = self.pyrpl_obj.rp
        self.name = name
        self.modulation_output_enabled = enable_modulation_output
        self.pid_autotune = pid_autotune
        # Define some useful variables
        self.scanning_frequency = 5  # Hz
        self.modulation_frequency = 25e6  # [Hz]
        self.phase = 0
        self.error_signal_amplitude_scanned = 0
        self.error_signal_amplitude = 0
        self.auxiliary_input_amplitude_scanned = 0
        self.is_scanning = False
        self.is_locking = False
        #Define and setup modules
        self.assign_modules()
        self.assign_input_output()
        # Set up asgs
        self.setup_asg_control()
        # Set up scope
        self.scope = self.redpitaya.scope
        self.setup_scope()
        # Set up PIDs
        try:
            self.setup_pid_DC()
        except OverflowError:
            print('Warning: phase controller could not set up pid_DC because the scanned error signal has invalid amplitude.')
        self.setup_pid_AC()
        self.setup_pid_control()
        # Setup  iq
        self.setup_iq()
        # Set locks off
        self.unlock()

    def connect_to_redpitaya(self, show_pyrpl_GUI=True):
        self.pyrpl_obj = pyrpl.Pyrpl(config=self.redpitaya_config_filename)
        if show_pyrpl_GUI:
            self.pyrpl_GUI = QtGui.QApplication.instance()

    def assign_modules(self, asg_control="asg0", iq="iq0", pid_DC="pid0", pid_AC="pid1"):
        self.asg_control = getattr(self.redpitaya, asg_control) #asg module used to scan the phase
        self.iq = getattr(self.redpitaya, iq) #iq module used to generate the AC error signal
        self.AC_error_signal = iq #name of the output signal from the iq module
        self.pid_DC = getattr(self.redpitaya, pid_DC)  #PID module used to generate the DC error signal
        self.DC_error_signal = pid_DC#name of the output signal from the PID DC module
        self.pid_AC = getattr(self.redpitaya, pid_AC) #PID module used to generate the scaled AC error signal
        self.error_signal = "iq2" #name of the signal that contains the phase error signal
        self.pid_control = self.redpitaya.pid2 #PID module that control the phase error

    def assign_input_output(self, error_signal_input="in2", control_signal_output="out1"):
        #Input
        self.error_signal_input = error_signal_input #signal used to generate the phase error signal
        self.auxiliary_input = "in" + str(1+1*(error_signal_input=="in1")) #complementary signal, typically used for monitoring
        #Output
        self.control_signal_output = control_signal_output #name of the analog output that carries the control signal
        self.modulation_signal_output = "out" + str(1+1*(control_signal_output=="out1")) #name of the other analog output, carrying the phase modulation signal


    def setup_pid_DC(self):
        self.pid_DC.free()
        self.pid_DC.input = self.error_signal_input
        self.pid_DC.output_direct = 'off'
        self.pid_DC.inputfilter = [2e3, 2e3, 0, 0]
        # Set the initial proportional value such that the curve fits in a fraction of the scope range
        self.pid_DC.p = 1 / 5 * (2 / self.get_scanned_signal_amplitude(signal_name=self.error_signal_input))
        self.pid_DC.i = 0
        self.pid_DC.ival = 0
        self.pid_DC_p_initial = self.pid_DC.p

    def setup_pid_AC(self):
        self.pid_AC.free()
        self.pid_AC.input = self.iq.name
        self.pid_AC.output_direct = 'off'
        self.pid_AC.inputfilter = [0, 0, 0, 0]
        self.pid_AC.p = 1
        self.pid_AC.i = 0
        self.pid_AC.ival = 0

    def setup_pid_control(self):
        self.pid_control.free()
        self.pid_control.input = self.error_signal
        self.pid_control.output_direct = 'off'
        self.pid_control.inputfilter = [2e3, 2e3, 0, 0]
        self.pid_control.p = 1
        self.pid_control.i = 0
        self.pid_control.ival = 0

    def enable_pid_control(self):
        if self.is_scanning:
            self.turn_off_scan()
        self.pid_control.output_direct = self.control_signal_output
        self.is_locking = True

    def setup_asg_control(self):
        self.asg_control.waveform = 'ramp'
        self.asg_control.amplitude = 0.6
        self.asg_control.trigger_source = 'immediately'
        self.asg_control.offset = 0
        self.asg_control.frequency = self.scanning_frequency

    def setup_iq(self):
        self.iq.input = self.error_signal_input
        self.iq.acbandwidth = 0.8 * self.modulation_frequency
        self.iq.frequency = self.modulation_frequency
        self.iq.bandwidth = [2e3, 2e3]
        self.iq.quadrature_factor = 20
        self.iq.gain = 0
        self.iq.amplitude = 0.5
        self.iq.phase = 0
        self.iq.output_direct = "off"
        if self.modulation_output_enabled:
            self.iq.output_direct = self.modulation_signal_output
        self.iq.output_signal = 'quadrature'

    def setup_scope(self):
        self.scope.duration = 4 / self.scanning_frequency / 50
        self.scope.trigger_source = self.asg_control.name
        self.scope.threshold = 0
        self.scope.input1 = self.error_signal_input
        self.scope.input2 = self.auxiliary_input

    def set_modulation_frequency(self, modulation_frequency):
        self.modulation_frequency = modulation_frequency
        self.setup_iq()

    def set_scanning_frequency(self, scanning_frequency):
        self.scanning_frequency = scanning_frequency
        self.setup_asg_control()
        self.setup_scope()

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
        plt.figure()
        plt.plot(trace)
        plt.show()
        """
        return trace

    def unlock(self):
        self.pid_control.output_direct = 'off'
        self.asg_control.output_direct = 'off'
        self.is_scanning = False
        self.is_locking = False

    def turn_off_scan(self):
        self.asg_control.output_direct = 'off'
        self.is_scanning = False

    def scan(self):
        self.unlock()
        self.asg_control.output_direct = self.control_signal_output
        self.is_scanning = True
        self.is_locking = False

    def set_demodulation_phase(self):
        was_not_scanning = not self.is_scanning

        self.scan()
        self.scope.input1 = self.DC_error_signal
        self.scope.input2 = self.iq.name

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
        self.scope.input2 = self.iq.name

        amplitude_pid_DC = self.get_signal_amplitude(signal_name=self.DC_error_signal)
        amplitude_iq = self.get_signal_amplitude(signal_name=self.iq.name)
        self.iq.quadrature_factor *= amplitude_pid_DC / amplitude_iq

        if was_not_scanning:
            self.turn_off_scan()

    def remove_offset_pid_DC(self):
        """
        1. scan the error signal input
        2. get the mean value
        3. set the setpoint of pid_DC to the mean value
        """
        was_scanning = self.is_scanning
        was_locking = self.is_locking
        self.scope.input1 = self.error_signal_input
        self.scope.input2 = self.DC_error_signal
        self.scan()
        trace = self.get_scope_curve(channel=1)
        self.pid_DC.setpoint = (np.max(trace) + np.min(trace))/2
        if not was_scanning:
            self.turn_off_scan()
        if was_locking:
            self.lock()

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
        was_locking = self.is_locking

        self.scan()
        signal_amplitude_scanned = self.get_signal_amplitude(signal_name=signal_name)

        if was_not_scanning:
            self.turn_off_scan()
        if was_locking:
            self.lock()
        return signal_amplitude_scanned

    def lock(self, keep_locked=True):
        self.unlock()
        if self.pid_autotune:
            # Get the amplitude of the scanned error signal
            self.error_signal_amplitude_scanned = self.get_scanned_signal_amplitude(signal_name=self.error_signal)
            print("Amplitude of the error signal: %0.3f" % self.error_signal_amplitude_scanned)
            self.is_locking = True
            self.remove_offset_pid_DC()
            self.PID_manual_autotune(keep_locked)
        else:
            self.enable_pid_control()

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
        initial_p = 0.05e-1
        initial_i = 0.05e-1
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
         #   print('P = %0.4f' % self.pid_control.p)
        print("Phase 2): lock is ringing" + print_separator)
        # ---------------
        # Phase 3)
        # ------------------
        while self.is_lock_ringing():
            self.pid_control.p *= 0.8
          #  print('P = %0.4f' % self.pid_control.p)
        self.pid_control.p *= 0.9 ** 4# reduce a bit further for safety
        # ----------------
        # Phase 4)
        # ----------------
        print("Phase 4): increase I until ringing" + print_separator)
        self.pid_control.i = initial_i
        while not self.is_lock_ringing():
            self.pid_control.i *= 2
         #   print('I = %0.4f' % self.pid_control.i)
        print("Phase 4): lock is ringing" + print_separator)
        # --------------
        # Phase 5)
        # --------------
        while self.is_lock_ringing():
            self.pid_control.i *= 0.8
          #  print('P = %0.4f' % self.pid_control.i)
            # -------------
        self.pid_control.i *= 0.9 ** 4 # reduce a bit further for safety
        print("Locked" + print_separator)
        if not keep_locked:
            self.unlock()

    def is_lock_ringing(self, relative_threshold=0.3):
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
       # print("Error signal amplitude during ringing check: %0.3f"%self.get_signal_amplitude(signal_name=self.error_signal))
        return self.get_signal_amplitude(signal_name=self.error_signal) >= relative_threshold * self.error_signal_amplitude_scanned

    def set_phase(self, phase):
        phase_rad = phase * np.pi / 180
        self.phase = phase_rad
        #See Iyad's PhD thesis (Methods) for reference
        V_DC = self.get_scanned_signal_amplitude(signal_name=self.error_signal_input)
        P_DC = -np.sin(phase_rad) / V_DC
        V_AC = self.get_scanned_signal_amplitude(signal_name=self.AC_error_signal)
        P_AC = np.cos(phase_rad) / V_AC
        self.pid_AC.p = P_AC
        self.pid_DC.p = P_DC

    def calibrate(self):
        #self.remove_offset_pid_DC()
        #time.sleep(0.2)
        #self.setup_pid_DC()
        #time.sleep(0.2)
        #self.setup_pid_AC()
        #time.sleep(0.2)
        self.set_demodulation_phase()
        #time.sleep(0.2)
        #self.setup_pid_control()
