#%%
import time

import numpy
import numpy as np
import matplotlib.pyplot as plt
import threading
import pyrpl
import time
from scipy.fft import fft, fftfreq
from scipy import optimize
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
    def __init__(self, redpitaya_config_filename, frequency="freq", name="Phase Controller", enable_modulation_output=False, pid_autotune=True, \
                 show_pyrpl_GUI=True):
        """
        :param redpitaya_config_filename: str
            object representing the pyrpl interface for the controlling RedPitaya
        :param frequency: str
            modulation frequency for locking
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
        self.scanning_frequency = 15  # Hz
        if self.name == "Relay Phase Controller":
            self.scanning_frequency = 60
        #Setting modulation frequencies
        self.calibration_frequency = 120e3
        self.measurement_frequency = 50e3
        if frequency == "calibration_frequency":
            self.calibration_frequency_on = True
            self.measurement_frequency_on = False
            self.modulation_frequency = self.calibration_frequency
        elif frequency == "measurement_frequency":
            print('Measurement frequency')
            self.calibration_frequency_on = False
            self.measurement_frequency_on = True
            self.modulation_frequency = self.measurement_frequency
        print(name, 'Modulation frequency:', self.modulation_frequency)
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
        # Setup scope
        self.scope.input1 = self.AC_error_signal
        self.scope.input2 = self.error_signal_input
        self.ival_jump = None
        if self.name == "Relay Phase Controller":
            self.ival_jump = 'yes'
    '''
    @property
    def modulation_output_enabled(self):
        return self._modulation_output_enabled
    @modulation_output_enabled.setter
    def modulation_output_enabled(self):
        if self._modulation_output_enabled == True:
            self.iq.output_direct = self.modulation_signal_output
        else:
            self.iq.output_direct = "off"
    @modulation_output_enabled.deleter
    def modulation_output_enabled(self):
        del self._modulation_output_enabled
    '''
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
        self.error_signal = "pid01sum" #name of the signal that contains the phase error signal
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

    def setup_pid_AC(self):
        self.pid_AC.free()
        self.pid_AC.input = self.iq.name
        self.pid_AC.output_direct = 'off'
        self.pid_AC.inputfilter = [2e5, 2e5, 0, 0]
        self.pid_AC.p = 1
        self.pid_AC.i = 0
        self.pid_AC.ival = 0

    def setup_pid_control(self):
        self.pid_control.free()
        self.pid_control.input = self.error_signal
        self.pid_control.output_direct = 'off'
        self.pid_control.inputfilter = [2e5, 2e5, 0, 0]
        self.pid_control.p = 0.1
        self.pid_control.i = 16
        self.pid_control.ival = 0
        if self.name == "Relay Phase Controller":
            # These parameters work well when sample hold is off
            #self.pid_control.p = 0.054
            #self.pid_control.i = 20

            # These parameters work "better" when sample hold is on
            self.pid_control.p = 0.075
            self.pid_control.i = 45
            self.pid_control.inputfilter = [6e2, 2.5e3, 7.8e4, 0]

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
        if self.name == "Relay Phase Controller":
            self.asg_control.amplitude = 0.25
            self.asg_control.offset = -0.75

    def setup_iq(self):
        """
        Send modulation signal

        Positive bandwitdhs are low-pass filters and negativa are low-pass
        :return:
        """
        self.iq.input = self.error_signal_input
        #if self.modulation_frequency > 10**5:
        #    self.iq.acbandwidth = 0.8 * self.modulation_frequency
        #else:
        self.iq.acbandwidth = 0.3 * self.modulation_frequency
        self.iq.frequency = self.modulation_frequency
        self.iq.gain = 0
        self.iq.bandwidth = [1e3, 1e3]
        self.iq.amplitude = 0.09
        if self.name == "Relay Phase Controller":
            self.iq.acbandwidth = 4.8e3
            self.iq.bandwidth = [2.4e3, 9.7e3]
            self.iq.amplitude = 1
        if self.name == "Dr. Jacoby Phase Controller":
            self.iq.amplitude = 0.05
        #if self.modulation_frequency > 10**5:
        #    self.iq.quadrature_factor = 20
        #    self.iq.amplitude = 0.05
        #else:
        self.iq.quadrature_factor = 10
        self.iq.phase = 0
        if self.modulation_output_enabled:
            self.iq.output_direct = self.modulation_signal_output
        else:
            self.iq.output_direct = "off"
        self.iq.output_signal = 'quadrature'

    def setup_scope(self):
        self.scope.duration = 24 / self.scanning_frequency / 50
        self.scope.trigger_source = self.asg_control.name
        self.scope.threshold = 0
        self.scope.input1 = self.error_signal_input
        self.scope.input2 = self.auxiliary_input

    def set_modulation_frequency(self, modulation_frequency):
        self.modulation_frequency = modulation_frequency
        self.iq.acbandwidth = 0.8 * self.modulation_frequency
        self.iq.frequency = self.modulation_frequency

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
        """
        time.sleep(2)
        trace = self.get_scope_curve(channel=1)
        Ts = self.scope.decimation/125e6
        time_s = Ts*numpy.array(numpy.arange(len(trace)))

        plt.figure()
        plt.plot(time_s, trace)
        plt.title('HD scan')
        plt.xlabel('Time')
        plt.ylabel('HD signal')
        plt.show()
        """

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
        plot = True
        was_not_scanning = not self.is_scanning

        self.scan()
        self.scope.input1 = self.DC_error_signal
        self.scope.input2 = self.iq.name

        amplitude_pid_DC = self.get_signal_amplitude(signal_name=self.DC_error_signal)
        #print('Amplitude pid DC:', amplitude_pid_DC)
        amplitude_iq = self.get_signal_amplitude(signal_name=self.iq.name)
        #print('Amplitude iq before calibration:', amplitude_iq)
        if amplitude_pid_DC == 0:
            print('DEBUG: Amplitude pid DC is zero')
            self.phase = 45*np.pi/180
            amplitude_pid_DC = self.get_signal_amplitude(signal_name=self.DC_error_signal)
        if amplitude_iq == 0:
            print('DEBUG: Amplitude iq is zero')
            self.iq.quadrature_factor = amplitude_pid_DC
        if amplitude_iq != 0:
            self.iq.quadrature_factor *= amplitude_pid_DC / amplitude_iq

        amplitude_iq = self.get_signal_amplitude(signal_name=self.iq.name)
        #print('Quadrature factor:', self.iq.quadrature_factor)
        #print('Amplitude iq after calibration:', amplitude_iq)

        self.scope.input1 = self.AC_error_signal
        self.scope.input2 = self.DC_error_signal

        if was_not_scanning:
            self.turn_off_scan()

    def remove_offset_pid_DC(self):
        """
        1. scan the error signal input
        2. get the mean value
        3. set the setpoint of pid_DC to the mean value
        """
        plot = False
        was_scanning = self.is_scanning
        was_locking = self.is_locking
        self.scope.input1 = self.error_signal_input
        self.scope.input2 = self.DC_error_signal
        self.scan()
        _trace = self.get_scope_curve(channel=1)
        # There is a jump in the signal that I want to avoid
        max_index = int(len(_trace)/2)
        skip = 50
        trace = _trace[0:max_index]
        trace = trace[::skip]
        self.pid_DC.setpoint = (np.max(trace) + np.min(trace))/2
        if plot:
            plt.figure()
            plt.plot(range(len(trace)), trace)
            plt.axhline(y = np.max(trace), color = 'r', linestyle = 'dashed')
            plt.axhline(y = np.min(trace), color = 'r', linestyle = 'dashed')
            plt.title('pid DC')
            plt.show()
        if not was_scanning:
            self.turn_off_scan()
        if was_locking:
            self.lock()

    def flip_iq_phase(self):
        self.iq.phase = np.mod(self.iq.phase + 180, 360)

    def get_signal_mean(self, signal_name=None):
        """
        This function gets the mean value of the desired signal.
        """
        if signal_name is not None:
            self.scope.input1 = signal_name
        # Get trace
        trace = self.get_scope_curve(channel=1)
        # Compute the amplitude
        signal_mean = np.mean(trace)

        return signal_mean

    def get_signal_amplitude(self, signal_name=None, plot = False):
        """¨
        This function does the following:
            - scan
            - acquire a trace from the error signal
            - return the amplitude of the error signal
        """
        # previous_scope_input = self.scope.input1
        plot = False
        if signal_name is not None:
            self.scope.input1 = signal_name
        # Get trace
        _trace = self.get_scope_curve(channel=1)
        # There is a jump in the signal that I want to avoid
        max_index = int(len(_trace) / 2)
        skip = 50
        trace = _trace[0:max_index]
        trace = trace[::5]
        # Compute the amplitude
        signal_amplitude = np.max(trace) - np.min(trace)

        if plot == True:
            plt.figure()
            plt.plot(numpy.linspace(1, len(trace), len(trace)), trace)
            plt.axhline(y=np.max(trace), color='r', linestyle='dashed')
            plt.axhline(y=np.min(trace), color='r', linestyle='dashed')
            plt.title(signal_name)
            plt.show()

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

    def get_voltage_frequency(self):
        if self.ival_jump == None:
            pass
        else:
            #Fixed values for ival jump

            #Calculate the voltage which makes the signal advance one period

            plot = False

            self.scan()
            self.scope.input1 = self.error_signal_input

            _trace = self.get_scope_curve(channel=1)
            trace = _trace[0:int(len(_trace)/3)]

            Ts = self.scope.decimation/125e6
            time = Ts*numpy.array(numpy.arange(len(trace)))

            offset = (numpy.max(trace) + numpy.min(trace)) / 2
            amplitude = (numpy.max(trace) - numpy.min(trace)) / 2

            def cos(x, freq, phase):
                return offset + amplitude * numpy.cos(freq * x + phase)

            #offset_guess = (numpy.max(trace) - numpy.min(trace))/2
            #amplitude_guess = (numpy.max(trace) + numpy.min(trace))/2
            freq_guess = 2500
            phase_guess = 0
            fit_guess = [freq_guess, phase_guess]
            fit_params, fit_covar = optimize.curve_fit(f=cos, xdata = time, ydata=trace, p0=fit_guess)
            freq, phase = fit_params
            fitted_scan = cos(time, freq, phase)

            if plot:
                plt.figure()
                plt.plot(time, trace)
                plt.plot(time, fitted_scan)
                plt.axhline(y=np.max(trace), color='r', linestyle='dashed')
                plt.axhline(y=np.min(trace), color='r', linestyle='dashed')
                plt.xlabel('Time')
                plt.ylabel('Data')
                plt.show()

            period = 1/freq
            self.ival_jump = period
            voltage_period = period*self.asg_control.amplitude/((1/self.asg_control.frequency)/2)
            #print('Voltage period:', voltage_period)
            #print('Guessed parameters:', fit_guess)
            #print('Frequency, Phase:', fit_params)

            self.pid_control.ival_max_bound = 0.9
            self.pid_control.ival_min_bound = - self.pid_control.ival_max_bound
            self.pid_control.ival_max_jump_dst = 2*abs(voltage_period)
            self.pid_control.ival_min_jump_dst = - self.pid_control.ival_max_jump_dst


    def get_ringing_frequency(self, signal_name):
        """
        This function gets the frequency of the ringing
        """
        if signal_name is not None:
            self.scope.input1 = signal_name
        trace = self.get_scope_curve(channel=1)

        Ts = self.scope.decimation/125e6
        time = Ts*numpy.array(numpy.arange(len(trace)))

        plt.figure()
        plt.plot(time, trace)
        plt.title('Scope data')
        plt.xlabel('Time')
        plt.ylabel('Data')
        plt.show()

        fourier_transform = fft(trace)
        fourier_frequencies = fftfreq(len(trace), Ts)
        '''
        fft_freq = [element for element in fourier_frequencies if element > 0]
        indexes = [numpy.where(fourier_frequencies == element for element in fourier_frequencies if element > 0)[0][0]]
        print('DEBUG: indexes', indexes)
        fft_data = [fourier_transform[index] for index in indexes]

        print('DEBUG:', numpy.abs(fourier_frequencies))
        '''
        plt.figure()
        plt.plot(fourier_frequencies, numpy.abs(fourier_transform))
        plt.title('Fourier transform')
        plt.xlabel('Frequency')
        plt.ylabel('Fourier transform')
        plt.show()

        max_index = numpy.where(numpy.abs(fft_data) == numpy.max(numpy.abs(fft_data)))[0][0]
        print('DEBUG: max index', max_index)
        if max_index == 0:
            fft_data.remove(0)
        max_index = numpy.where(numpy.abs(fft_data) == numpy.max(numpy.abs(fft_data)))[0][0]
        ringing_frequency = fft_freq[max_index]
        print('Ringing frequency is', ringing_frequency)
        return ringing_frequency

    def get_phase_difference(self, signal_1, signal_2):
        """
        Fits a sine wave to both signals and measure the phase difference
        between them.
        """
        if signal_1 is not None:
            self.scope.input1 = signal_1
        if signal_2 is not None:
            self.scope.input2 = signal_2

    def lock(self, keep_locked=True):
        self.unlock()
        if self.phase == None:
            self.set_phase(0)
        phase = self.phase * 180/numpy.pi
        #print('Phase:', phase)
        if self.pid_autotune:
            #Get the amplitude of the scanned error signal
            self.error_signal_amplitude_scanned = self.get_scanned_signal_amplitude(signal_name=self.error_signal)
            print("Amplitude of the error signal: %0.3f" % self.error_signal_amplitude_scanned)
            self.is_locking = True
            print('Manual autotune')
            self.PID_manual_autotune(keep_locked)
            #signal_mean = self.get_signal_mean(signal_name=self.error_signal_input)
            #self.unlock()
            #scan_mean = self.get_scan_mean(signal_name=self.error_signal_input)
            #signal_amplitude = self.get_scanned_signal_amplitude(signal_name=self.error_signal_input)
            #angle = np.arcsin(2*(signal_mean - scan_mean)/signal_amplitude)
            #print("Locking angle = %.1f"%(angle*180/np.pi))
        else:
            self.enable_pid_control()

    def PID_manual_autotune(self, keep_locked=True):
        """
        This function attempts to replicate a manual (human-made) autotune of the homodyne detection PI control.
        The procedure is the following:
            1) Start from P=I=ival=0
            2) Increase P until the lock rings and then decrease it until it doesn't ring anymore
            3) Repeat the same procedure for I
            4) Again repeat the procedure for P
        """
        """
        #TODO: First find a good value for I with P = 0, then find a good value for P, and then a good value for I again
        """

        # Phase 1): Initialize PID parameters
        start_time = time.time()
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
        print("Phase 2): adjust P" + print_separator)
        self.pid_control.p = initial_p
        self.pid_control.i = initial_i
        while not self.is_lock_ringing():
            self.pid_control.p *= 2
         #   print('P = %0.4f' % self.pid_control.p)
        print("Lock is ringing" + print_separator)
        while self.is_lock_ringing():
            #print('DEBUG: ringing')
            self.pid_control.p *= 0.8
            time.sleep(0.2)
            #self.get_ringing_frequency(self.error_signal)
          #  print('P = %0.4f' % self.pid_control.p)
        self.pid_control.p *= 0.9 ** 4# reduce a bit further for safety
        print("P = %.4f"%self.pid_control.p)
        print('Time elapsed: %.2f s' %(time.time() - start_time))
        # ----------------
        # Phase 3)
        # ----------------
        print("Phase 3): adjust I" + print_separator)
        self.pid_control.i = initial_i
        while not self.is_lock_ringing():
            self.pid_control.i *= 2
         #   print('I = %0.4f' % self.pid_control.i)
        print("Lock is ringing" + print_separator)
        while self.is_lock_ringing():
            self.pid_control.i *= 0.8
            time.sleep(0.2)
          #  print('P = %0.4f' % self.pid_control.i)
            # -------------
        self.pid_control.i *= 0.9 ** 4 # reduce a bit further for safety
        print("I = %.4f"%self.pid_control.i)
        print('Time elapsed: %.2f s' %(time.time() - start_time))
        # --------------
        # Phase 4)
        # --------------
        print("Phase 4): adjust P" + print_separator)
        while not self.is_lock_ringing():
            self.pid_control.p *= 2
        #   print('P = %0.4f' % self.pid_control.p)
        while self.is_lock_ringing():
            # print('DEBUG: ringing')
            self.pid_control.p *= 0.8
            time.sleep(0.2)
            # self.get_ringing_frequency(self.error_signal)
        #  print('P = %0.4f' % self.pid_control.p)
        self.pid_control.p *= 0.9 ** 4  # reduce a bit further for safety
        print("P = %.4f" % self.pid_control.p)
        print("Locked" + print_separator)
        print('Time elapsed: %.2f s' %(time.time() - start_time))
        if not keep_locked:
            self.unlock()
        '''
        time.sleep(5)
        while self.is_lock_ringing():
            print('Lock is ringing, try again')
            print('1. Adjust P')
            while not self.is_lock_ringing():
                self.pid_control.p *= 2
                time.sleep(1)
            while self.is_lock_ringing():
                self.pid_control.p *= 0.8
            self.pid_control.p *= 0.9 ** 4  # reduce a bit further for safety
            print("P = %.4f" % self.pid_control.p)
            print('2. Adjust I')
            while not self.is_lock_ringing():
                self.pid_control.i *= 2
                time.sleep(1)
            while self.is_lock_ringing():
                self.pid_control.i *= 0.8
            self.pid_control.i *= 0.9 ** 4  # reduce a bit further for safety
            print("I = %.4f" % self.pid_control.i)
            print('3. Final adjustment of P')
            while not self.is_lock_ringing():
                self.pid_control.p *= 2
                time.sleep(1)
            while self.is_lock_ringing():
                self.pid_control.p *= 0.8
            self.pid_control.p *= 0.9 ** 4  # reduce a bit further for safety
            print("P = %.4f" % self.pid_control.p)
            time.sleep(5)
        '''

    def is_lock_ringing(self, relative_threshold=0.2):
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
        phase = float(phase)
        phase_rad = phase * np.pi / 180
        self.phase = phase_rad
        #See Iyad's PhD thesis (Methods) for reference
        V_DC = self.get_scanned_signal_amplitude(signal_name=self.error_signal_input)
        P_DC = -np.sin(phase_rad) / V_DC
        V_AC = 0
        while V_AC == 0: # In case it catches the moment in which iq is turned off
            V_AC = self.get_scanned_signal_amplitude(signal_name=self.AC_error_signal)
            if V_AC == 0:
                print('Amplitude of AC error signal is zero')
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
        self.get_voltage_frequency()
        self.setup_pid_DC()
        self.set_demodulation_phase()
        #time.sleep(0.2)
        #self.setup_pid_control()
        self.remove_offset_pid_DC()
        self.set_iq_qfactor()
        self.scope.input1 = self.AC_error_signal
        self.scope.input2 = self.DC_error_signal

    def find_transfer_function(self):
        #TODO
        # Assure HD is locked and calibrated
        if self.is_locking:
            pass
        else:
            self.calibrate()
            self.set_iq_qfactor()
            self.lock()

        # Parameters of the sine wave
        frequency_initial = 1e2
        frequency_final = 1e4
        points = 1e2
        frequency_range = numpy.linspace(frequency_initial, frequency_final, points)
        amplitude = 0.1
        asg = self.redpitaya.asg1
        reflection = []
        transmission = []
        for i in frequency_range:
            asg.setup(waveform='sin', amplitude=amplitude, frequency = frequency_range[i], offset=0)
            asg.output_direct = self.control_signal_output
        #Set the scope
        self.scope.input1 = "asg1"
        self.scope.input2 = self.error_signal_input
        trace_R = self.get_scope_curve(channel=1)
        trace_T = self.get_scope_curve(channel=2)
        #Extract ampl R and T
