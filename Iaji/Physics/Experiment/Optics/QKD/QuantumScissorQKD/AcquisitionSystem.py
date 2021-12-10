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
#%%
class AcquisitionSystem:
    def __init__(self, scope_IP_address, channel_names=["C1", "C2", "C3", "C4"], name="Acquisition System", save_directory=os.getcwd()):
        self.name = name
        self.scope = LecroyOscilloscope(IP_address=scope_IP_address,channel_names=channel_names, connect=True)
        self.traces = dict(zip(channel_names, [None for j in range(len(channel_names))]))
        self.save_directory = save_directory

    def acquire(self, channel_names, filenames=None, save_directory=None, \
                      load_traces=False, \
                      adapt_vertical_axis=False, adapt_horizontal_axis=False):
        """

        :param channel_names: list of str
            the names of the channels from which traces are saved.
        :param save_directory: str
            the destination directory in the host, where the saved traces will be transfered.
        :param filenames: list of str (same size as channel_names)
            the file names associated to the saved traces.
        :return:
        """
        #Enable the input channels in the scope
        adapt_display = "vertical" * (adapt_vertical_axis and not adapt_horizontal_axis) +\
                        "horizontal" * (adapt_horizontal_axis and not adapt_vertical_axis) +\
                        "all" * (adapt_horizontal_axis and adapt_vertical_axis) +\
                        "" * (not(adapt_horizontal_axis or adapt_vertical_axis))
        for channel_name in channel_names:
            if not self.scope.channels[channel_name].is_enabled():
                self.scope.channels[channel_name].enable(True)
                if adapt_display is not "":
                    self.scope.setup_automatic(channel_name=channel_name, setup_type=adapt_display)
        #Start acquisition
        self.scope.start_acquisition(store_all_traces=False)
        #Store traces locally in the oscilloscope
        for channel_name in channel_names:
            self.scope.store_trace(channel_name=channel_name)
        #Transfer the selected traces to the host save directory
        if save_directory is None:
            save_directory = self.save_directory
        #If the destination save directory does not exist, create it
        if not os.path.isdir(save_directory):
            try:
                os.mkdir(save_directory)
            except:
                raise FileNotFoundError
        #See what is there in the scope's save directory
        scope_save_directory = self.scope.host_drive+"\\"+self.scope.save_directory
        scope_filenames = os.listdir(scope_save_directory)
        scope_filenames_latest = []
        #Select the most recent files for the selected traces
        #These will be the files to be sent
        for j in range(len(channel_names)):
            channel_name = channel_names[j]
            channel_number = self.scope.channels[channel_name].number
            #Load all the file names that relate to the current channel
            scope_filenames_temp = [f for f in scope_filenames if "C"+str(channel_number) in f]
            #Find the most recent filename
            scope_filename_latest = scope_filenames_temp[np.argmax(scope_filenames_temp)]
            scope_filenames_latest.append(scope_filename_latest)
        #Transfer the corresponding trace to the host save directory
        if filenames is None:
            filenames = scope_filenames_latest
        for j in range(len(channel_names)):
            channel_name = channel_names[j]
            if ".trc" not in filenames[j]:
                filenames[j] += ".trc"
            shutil.copy(src=scope_save_directory+"\\"+scope_filenames_latest[j], dst=save_directory+"\\"+filenames[j])
            #Load the traces if requested
            if load_traces:
                self.traces[channel_name] = LecroyReader.read(save_directory+"\\"+filenames[j])

    def set_save_directory(self, save_directory):
        self.save_directory = save_directory


