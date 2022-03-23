"""
This module describes the system for controlling the pump field that pumps the type II OPO generating the single photons
for NLA.
It consists of:

    - The second harmonic generation cavity lock
"""
from .CavityLock import CavityLock
#%%
class PumpController:
    def __init__(self, shg_cavity_lock, name="Pump Controller"):
        self.shg_cavity_lock = shg_cavity_lock
        self.name = name

    def scan(self):
        self.shg_cavity_lock.scan()

    def lock(self):
        self.shg_cavity_lock.lock()

    def unlock(self):
        self.shg_cavity_lock.unlock()


