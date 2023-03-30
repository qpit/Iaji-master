"""
This module describes a parametric gain locking system realized with a RedPitaya pyrpl module. It consists of:

    -The pyrpl object of the RedPitaya controlling the parametric gain
"""
# %%
import pyrpl
from pyqtgraph.Qt import QtGui
from .Exceptions import ConnectionError
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.PhaseController import PhaseController
# %%
class GainLock:
    def __init__(self, phase_controller, name="Parametric Gain Lock"):
        self.name = name
        self.phase_controller = phase_controller

    def scan(self):
        self.phase_controller.scan()

    def lock(self):
        self.phase_controller.lock()

    def unlock(self):
        self.phase_controller.unlock()

    def calibrate_lock(self):
        self.phase_controller.calibrate()
