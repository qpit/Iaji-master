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
    def __init__(self, redpitaya_config_filename, name="Parametric Gain Lock", redpitaya_name="Gain Lock Redpitaya", connect=True, show_pyrpl_GUI=True, enable_modulation_output=False):
        self.name = name
        self.redpitaya_name = redpitaya_name
        self.redpitaya_config_filename = redpitaya_config_filename
        self.name = name
        self.redpitaya_name = redpitaya_name
        self.redpitaya_config_filename = redpitaya_config_filename
        self.pyrpl_obj = None
        self.pyrpl_GUI = None
        if connect:
            self.connect_to_redpitaya(show_pyrpl_GUI=show_pyrpl_GUI)
            self.phase_controller = PhaseController(self.pyrpl_obj, modulation_output_enabled=enable_modulation_output)

    def connect_to_redpitaya(self, show_pyrpl_GUI=True):
        self.pyrpl_obj = pyrpl.Pyrpl(config=self.redpitaya_config_filename)
        if show_pyrpl_GUI:
            self.pyrpl_GUI = QtGui.QApplication.instance()


    def scan(self):
        self.phase_controller.scan()

    def lock(self):
        self.phase_controller.lock()

    def unlock(self):
        self.phase_controller.unlock()

    def calibrate_lock(self):
        self.phase_controller.calibrate()
