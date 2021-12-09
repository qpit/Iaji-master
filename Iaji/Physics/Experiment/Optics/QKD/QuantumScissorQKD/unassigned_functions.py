"""
This file contains potentially useful bu unassigned function.
"""


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
