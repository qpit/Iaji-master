"""
This module defines a state measurement system for quantum scissor QKD, using homodyne detection
as mesaurement.
"""
# In[imports]
import numpy
import time
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD import HomodyneDetectionController
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator import QuantumStateFock
from Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.QuantumStateTomography import QuadratureTomographer

# In[]
class StateMeasurementController:
    """
    This class describes a state measurement controller. It manages the measurement of a single-mode state of light
    through homodyne detection. It contains:
        - A controller for homodyne detection
        - A numerical quantum state in the Fock basis
        - A quantum state tomographer based on homodyne measurements
    It can perform:
        - Homodyne state tomography measurements
        - Tomographic state reconstruction based on homodyne measurement
        - Displacement measurement #TODO
    """
    #------------------------------------------------------------------
    def __init__(self, hd_controller: HomodyneDetectionController, name="State Measurement Controller"):
        '''
        :param hd_controller: Iaji HomodyneDetectionController
        :param name: str
        '''
        self.hd_controller = hd_controller
        self.name = name
        self.tomographer = None
        self.quantum_state = None
        self.displacement = None
    #------------------------------------------------------------------
    def tomography_measurement(self, phases):
        '''
        :param phases: iterable of float'
            phase angles [deg]
        '''
        phases = numpy.atleast_1d(phases)
        self.phases = phases #[deg]
        self.quadratures = dict(zip(phases, [None for j in range(len(phases))]))
        #Calibration
        self.hd_controller.phase_controller.calibrate()
        self.hd_controller.phase_controller.remove_offset_pid_DC()
        #Extract AC channel name
        channel_names = list(self.hd_controller.acquisition_system.scope.channels.keys())
        channel_ac = [c for c in channel_names if "AC" in c][0]
        for phase in phases:
            traces = self.hd_controller.measure_quadrature(phase)
            #Only store the AC output of the homodyne detector
            self.quadratures[phase] = traces[channel_ac]
        return self.quadratures
    # ------------------------------------------------------------------
    def scanned_measurement(self):
        '''
        :return:
        '''
        #Scan the phase
        self.hd_controller.phase_controller.scan()
        #Acquire
        traces = self.hd_controller.acquisition_system.acquire()
        #Save the AC homodyne detector channel
        channel_names = list(self.hd_controller.acquisition_system.scope.channels.keys())
        channel_ac = [c for c in channel_names if "AC" in c][0]
        self.quadrature_scan = traces[channel_ac]
        self.hd_controller.phase_controller.turn_off_scan()
        return self.quadrature_scan
    #------------------------------------------------------------------

