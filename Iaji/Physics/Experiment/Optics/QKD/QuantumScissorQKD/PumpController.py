"""
This module describes the system for controlling the pump field that pumps the type II OPO generating the single photons
for NLA.
It consists of:

    - The second harmonic generation cavity lock
"""
from .CavityLock import CavityLock
#%%
class PumpController:
    def __init__(self, redpitaya_config_filename, name="Pump Controller", redpitaya_name="Second Harmonic Generator Cavity Lock RedPitaya", show_pyrpl_GUI=True):
        self.redpitaya_config_filename = redpitaya_config_filename
        self.redpitaya_name = redpitaya_name
        self.name = name
        self.SHG_cavity_lock = CavityLock(redpitaya_config_filename=self.redpitaya_config_filename, name="Second Harmonic Generator Cavity Lock", redpitaya_name=self.redpitaya_name, show_pyrpl_GUI=True)

    def scan(self):
        self.SHG_cavity_lock.scan()

    def lock(self):
        self.SHG_cavity_lock.lock()

    def unlock(self):
        self.SHG_cavity_lock.unlock()


