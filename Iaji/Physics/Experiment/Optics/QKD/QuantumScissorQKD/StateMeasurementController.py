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
from Iaji.InstrumentsControl.SigilentSignalGenerator import SigilentSignalGenerator

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
        self.signal_enabler = None
    #------------------------------------------------------------------
    def tomography_measurement(self, phases):
        '''
        Quadrature measurement for quantum state tomography
        :param phases: iterable of float
            phase angles [deg]
        :param signal_controller: Iaji SigilentSignalGenerator
            signal generator blocking and transmitting the signal mode. If specified, vacuum quadrature measurements
            are taken after every signal quadrature measurement.
        '''
        phases = numpy.atleast_1d(phases)
        self.phases = phases #[deg]
        self.quadratures = dict(zip(phases, [None for j in range(len(phases))]))
        self.vacuum_quadratures = dict(zip(phases, [None for j in range(len(phases))]))
        #Calibration
        self.hd_controller.phase_controller.calibrate()
        #Extract AC channel name
        channel_names = list(self.hd_controller.acquisition_system.scope.channels.keys())
        channel_ac = [c for c in channel_names if "AC" in c][0]
        channel_dc = [c for c in channel_names if "DC" in c][0]
        for phase in phases:
            self.hd_controller.phase_controller.remove_offset_pid_DC()
            traces = self.hd_controller.measure_quadrature(phase)
            #Only store the AC output of the homodyne detector
            self.quadratures[phase] = traces[channel_ac]
            if self.signal_enabler is not None:
                # Block signal and disable AC channel
                self.signal_enabler.enable(False)
                self.hd_controller.acquisition_system.scope.channels[channel_dc].enable(False)
                traces =  self.hd_controller.acquisition_system.acquire(filenames=["_vacuum_AC_"+str(phase)])
                # Unblock signal and enable AC channel
                self.hd_controller.acquisition_system.scope.channels[channel_dc].enable(True)
                self.signal_enabler.enable(True)
                self.vacuum_quadratures[phase] = traces[channel_ac]
        return self.quadratures, self.vacuum_quadratures
    # ------------------------------------------------------------------
    def scanned_measurement(self):
        '''
        :return:
        '''
        #Scan the phase
        self.hd_controller.phase_controller.scan()
        # Save the AC homodyne detector channel
        channel_names = list(self.hd_controller.acquisition_system.scope.channels.keys())
        channel_ac = [c for c in channel_names if "AC" in c][0]
        #Acquire
        traces = self.hd_controller.acquisition_system.acquire(filenames=["quadrature_scanned_DC", "quadrature_scanned_AC"])
        self.quadrature_scanned = traces[channel_ac]
        self.hd_controller.phase_controller.turn_off_scan()
        #If a signal mode controller is present, measure the vacuum quadrature right away
        self.vacuum_quadrature_scanned = None
        if self.signal_enabler is not None:
            self.signal_enabler.enable(False)
            traces = self.hd_controller.acquisition_system.acquire(filenames=["vacuum_scanned_DC", "vacuum_scanned_AC"])
            self.vacuum_quarature_scan = traces[channel_ac]
            self.signal_enabler.enable(True)
        return self.quadrature_scanned, self.vacuum_quadrature_scanned
    #------------------------------------------------------------------

