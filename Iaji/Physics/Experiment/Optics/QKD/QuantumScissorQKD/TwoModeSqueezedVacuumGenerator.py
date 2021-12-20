"""
This class describes the that generates the single photon.
It consists in:

- pump controller
- OPO controller
"""
#%%
from .PumpController import PumpController
from .OPOController import OPOController
#%%
class TwoModeSqueezedVacuumGenerator:
    def __init__(self, pump_controller, OPO_controller, name="Two-mode Squeezed Vacuum Generator"):
        self.pump_controller = pump_controller
        self.OPO_controller = OPO_controller
        self.name = name
    #SHG control methods
    def scan_SHG_cavity(self):
        self.pump_controller.SHG_cavity_lock.scan()

    def lock_SHG_cavity(self):
        self.pump_controller.SHG_cavity_lock.lock()

    def unlock_SHG_cavity(self):
        self.pump_controller.SHG_cavity_lock.unlock()
    #OPO control methods
    #Cavity
    def scan_OPO_cavity(self):
        self.OPO_controller.cavity_lock.scan()

    def lock_OPO_cavity(self):
        self.OPO_controller.cavity_lock.lock()

    def unlock_OPO_cavity(self):
        self.OPO_controller.cavity_lock.unlock()
    #Parametric gain
    def scan_OPO_gain(self):
        self.OPO_controller.gain_lock.scan()

    def lock_OPO_gain(self):
        self.OPO_controller.gain_lock.lock()

    def unlock_OPO_gain(self):
        self.gain_lock.unlock()

    def calibrate_OPO_gain(self):
        self.OPO_controller.calibrate_gain()
