"""
This class describes the system controlling the type II OPO that generates the single photon.
It consists in:

- OPO cavity lock
- OPO gain lock
"""
#%%
from .CavityLock import CavityLock
from .GainLock import GainLock
#%%
class OPOController:
    def __init__(self, cavity_redpitaya_config_filename, gain_redpitaya_config_filename, name="OPO Controller",\
                 cavity_redpitaya_name="OPO Cavity Lock Redpitaya", gain_redpitaya_name="OPO Gain Lock Redpitaya",show_pyrpl_GUI=True):
        self.cavity_redpitaya_config_filename, self.cavity_redpitaya_name = (cavity_redpitaya_config_filename, cavity_redpitaya_name)
        self.gain_redpitaya_config_filename, self.gain_redpitaya_name = (gain_redpitaya_config_filename, gain_redpitaya_name)
        self.name = name
        self.cavity_lock = CavityLock(redpitaya_config_filename=self.cavity_redpitaya_config_filename,
                                      name="OPO Cavity Lock", redpitaya_name=self.cavity_redpitaya_name,
                                      show_pyrpl_GUI=show_pyrpl_GUI)
        self.gain_lock = GainLock(redpitaya_config_filename=self.gain_redpitaya_config_filename,
                                      name="OPO Gain Lock", redpitaya_name=self.gain_redpitaya_name,
                                      show_pyrpl_GUI=show_pyrpl_GUI)

    def scan_cavity(self):
        self.cavity_lock.scan()

    def lock_cavity(self):
        self.cavity_lock.lock()

    def unlock_cavity(self):
        self.cavity_lock.unlock()

    def scan_gain(self):
        self.gain_lock.scan()

    def lock_gain(self):
        self.gain_lock.lock()

    def unlock_gain(self):
        self.gain_lock.unlock()
