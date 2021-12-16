"""
This class describes a homodyne detector used in the experiment.
It consists of:
    - A phase controller, associated to a RedPitaya that controls the interferometer's phase
"""
#%%
import pyrpl
from pyqtgraph.Qt import QtGui
from .Exceptions import ConnectionError, ResonanceNotFoundError
from .PhaseController import PhaseController
#%%
print_separator = "---------------------------------------------"
#%%
class HomodyneDetectorController:
    def __init__(self, redpitaya_config_filename, name="Homodyne Detector", redpitaya_name="Homodyne Detector Lock Redpitaya", connect=True, show_pyrpl_GUI=True, enable_modulation_output=False):
        self.redpitaya_name = redpitaya_name
        self.redpitaya_config_filename = redpitaya_config_filename
        self.name = name
        self.pyrpl_obj = None
        self.pyrpl_GUI = None
        if connect:
            self.connect_to_redpitaya(show_pyrpl_GUI=show_pyrpl_GUI)
            self.phase_controller = PhaseController(self.pyrpl_obj, modulation_output_enabled=enable_modulation_output)

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




