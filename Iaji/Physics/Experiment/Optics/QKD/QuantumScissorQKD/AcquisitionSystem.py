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
        channel_names = channel_names = list(self.scope.channels.keys())
        self.filenames = self.filenames = dict(zip(channel_names, ["%s_trace"%channel_name for channel_name in channel_names]))
    # --------------------------------    
    def set_host_save_directory(self, save_directory):
        self.host_save_directory = save_directory
    # --------------------------------
    def acquire(self, filenames=None):
        active_channels = [c for c in list(self.scope.channels.keys()) if self.scope.channels[c].is_enabled()]
        if filenames is None:
            filenames = [self.filenames[channel_name] for channel_name in active_channels]
        else:
            assert len(filenames) == len(active_channels), \
            "%d file names were specified, but there are %d active channels"%(len(filenames), len(active_channels))
        self.scope.acquire(channel_names=active_channels, filenames=filenames,
                           save_directory=self.host_save_directory, \
                           load_traces=True, \
                           adapt_vertical_axis=False, adapt_horizontal_axis=False)
        return self.scope.traces
    # --------------------------------
    def set_channel_names(self, channel_indices, channel_names):
        self.scope.set_channel_names(channel_indices, channel_names)
        self.filenames = dict(zip(list(self.scope.channels.keys()), list(self.filenames.values())))
