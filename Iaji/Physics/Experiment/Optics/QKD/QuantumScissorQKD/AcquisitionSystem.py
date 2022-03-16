"""
This module describes the acquisition system used for acquiring traces during the experiment.
It consists of:

    - An oscilloscope module
"""
#%%
import os
import shutil
from Iaji.InstrumentsControl.LecroyOscilloscope import LecroyOscilloscope
from qopt import lecroy as LecroyReader
import numpy as np
import threading
#%%
class AcquisitionSystem:
    def __init__(self, scope_IP_address, channel_names=["C1", "C2", "C3", "C4"], name="Acquisition System", save_directory=os.getcwd()):
        self.name = name
        self.scope = LecroyOscilloscope(IP_address=scope_IP_address,channel_names=channel_names, connect=True)
        self.traces = dict(zip(channel_names, [None for j in range(len(channel_names))]))
        self.save_directory = save_directory

    def set_save_directory(self, save_directory):
        self.save_directory = save_directory


