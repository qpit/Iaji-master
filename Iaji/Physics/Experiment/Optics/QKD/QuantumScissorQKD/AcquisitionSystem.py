"""
This module describes the acquisition system used for acquiring traces during the experiment.
It consists of:

    - An oscilloscope module
"""
#%%
import os
import shutil
from Iaji.InstrumentsControl.LecroyOscilloscope import LecroyOscilloscope
#%%
class AcquisitionSystem:
    def __init__(self, scope_IP_address, channel_names, name="Acquisition System"):
        self.name = name
        self.scope = LecroyOscilloscope(IP_address=scope_IP_address,channel_names=channel_names, connect=True)

    def acquire(self, channel_names): #TODO
        self.scope.start_acquisition(store_all_traces=False)
        for channel_name in channel_names:
            channel_names
            pass


