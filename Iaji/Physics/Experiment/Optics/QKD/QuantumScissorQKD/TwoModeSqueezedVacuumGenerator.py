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
    def __init__(self, SHG_cavity_redpitaya_config_filename, OPO_cavity_redpitaya_config_filename, OPO_gain_redpitaya_config_filename, name="Two-mode Squeezed Vacuum Generator",\
                 SHG_cavity_redpitaya_name="Second Harmonic Generator Cavity Lock Redpitaya", \
                 OPO_cavity_redpitaya_name="OPO Cavity Lock Redpitaya", OPO_gain_redpitaya_name="OPO Gain Lock Redpitaya",show_pyrpl_GUI=True):

        self.SHG_cavity_redpitaya_config_filename, self.SHG_cavity_redpitaya_name = (SHG_cavity_redpitaya_config_filename, SHG_cavity_redpitaya_name)
        self.OPO_cavity_redpitaya_config_filename, self.OPO_cavity_redpitaya_name = (OPO_cavity_redpitaya_config_filename, OPO_cavity_redpitaya_name)
        self.OPO_gain_redpitaya_config_filename, self.OPO_gain_redpitaya_name = (OPO_gain_redpitaya_config_filename, OPO_gain_redpitaya_name)
        self.name = name

        self.pump_controller = PumpController(redpitaya_config_filename=self.SHG_cavity_redpitaya_config_filename, redpitaya_name=self.SHG_cavity_redpitaya_name)
        self.OPO_controller = OPOController(cavity_redpitaya_config_filename=self.OPO_cavity_redpitaya_config_filename, gain_redpitaya_config_filename=self.OPO_gain_redpitaya_config_filename, \
                                            cavity_redpitaya_name=self.OPO_cavity_redpitaya_name, gain_redpitaya_name=self.OPO_gain_redpitaya_name)
    #SHG control methods
    def scan_SHG_cavity(self):
        self.pump_controller.cavity_lock.scan()

    def lock_SHG_cavity(self):
        self.pump_controller.cavity_lock.lock()

    def unlock_SHG_cavity(self):
        self.pump_controller.cavity_lock.unlock()
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
