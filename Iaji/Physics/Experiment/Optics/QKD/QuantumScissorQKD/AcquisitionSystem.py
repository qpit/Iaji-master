"""
This module describes the acquisition system used for acquiring traces during the experiment.
It consists of:

    - An oscilloscope module
"""
# In[acquisition system]
class AcquisitionSystem:
    def __init__(self, scope, name="Acquisition System"):
        self.name = name
        self.scope = scope
        self.host_save_directory = "."
        self.filenames = dict(zip(list(self.scope.channels.keys()), \
                                  ["trace.trc" for j in range(len(list(self.scope.channels.keys())))]))
    # --------------------------------    
    def set_host_save_directory(self, save_directory):
        self.host_save_directory = save_directory
    # --------------------------------
