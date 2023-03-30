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
    def __init__(self, cavity_lock, gain_lock, name="OPO Controller"):
        self.cavity_lock = cavity_lock
        self.gain_lock = gain_lock
        self.name = name

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

    def calibrate_gain(self):
        self.gain_lock.calibrate_lock()
