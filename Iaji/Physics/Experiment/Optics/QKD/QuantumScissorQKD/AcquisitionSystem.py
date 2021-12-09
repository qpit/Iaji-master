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
    def __init__(self, scope_IP_address, channel_names, name="Acquisition System", save_directory=os.getcwd()):
        self.name = name
        self.scope = LecroyOscilloscope(IP_address=scope_IP_address,channel_names=channel_names, connect=True)
        self.save_directory = save_directory

    def acquire(self, channel_names, save_directory=None):
        #Enable the input channels in the scope
        for name in channel_names:
            if not self.scope.channels[name].is_enabled():
                self.scope.channels[name].enable(True)
        #Start acquisition
        self.scope.start_acquisition(store_all_traces=False)
        #Store traces locally in the oscilloscope
        for channel_name in channel_names:
            self.scope.store_trace(channel_name=channel_name)
        #Transfer the selected traces to the host save directory
        if save_directory is None:
            save_directory = self.save_directory



    def set_save_directory(self, save_directory):
        self.save_directory = save_directory


