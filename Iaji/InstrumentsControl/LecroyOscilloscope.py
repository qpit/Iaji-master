"""
This module describes a Lecroy Oscilloscope, remotely controlled with vx1ii protocol.
For a list of commands, refer to http://cdn.teledynelecroy.com/files/manuals/maui-remote-control-and-automation-manual.pdf

NOTES:

    - So far, I haven't found a way to automatically set the sampling rate, so that has to be set manually.
    -
"""
#%%
import os
import vxi11
from .Exceptions import ConnectionError, InvalidParameterError
import shutil
import numpy as np
from qopt import lecroy as LecroyReader
#%%
print_separator = '\n-------------------------------------------'
#%%
class LecroyOscilloscope:
    def __init__(self, IP_address, channel_names=["C1", "C2", "C3", "C4"], name="Lecroy Oscilloscope", connect=True):
        self.name = name
        self.IP_address = IP_address
        self.host_drive = "L:"
        self.channels = {}
        if connect:
            try:
                self.connect()
                self.local_drive = self.get_local_drive_name()
                self.save_directory = self.get_save_directory()
                self.channels = dict(zip(channel_names, [LecroyOscilloscpeChannel(instrument=self.instrument, channel_number=j+1, name=channel_names[j]) for j in range(len(channel_names))]))
                self.traces = dict(zip(channel_names, [None for j in range(len(channel_names))]))
            except: 
                print('WARNING: it was not possible to connect to the oscilloscope '+self.name)
                return

        
    def connect(self):
        try:
            self.instrument = vxi11.Instrument(self.IP_address)
        except:
            raise ConnectionError

    def setup_automatic(self, channel_name, setup_type, keep_trigger_settings=True):
        """
        Uses the automatic setup function of the oscilloscope to set up the horizontal
        or vertical axis of the current channel to fit the input data.

        :param channel_name: str
        :param setup_type: str
            Type of setup.
            Valid arguments are:
                - "horizontal": automatically sets the horizontal axis, and resets the previous vertical axis
                - "vertical": automatically sets the vertical axis, and resets the previous horizontal axis
                - "all": automatically sets the horizontal and the vertical axis.
        :param keep_trigger_settings: bool
            If true, it resets the previous trigger settings after automatic setup

        :return:
        """
        channel = self.channels[channel_name]
        trigger_setup = self.get_trigger_setup()
        horizontal_setup = self.get_horizontal_setup()
        vertical_setup = channel.get_vertical_setup()

        self.instrument.write("C" + str(channel.number) + ":ASET")
        if setup_type == "horizontal":
            channel.setup_vertical(voltage_range=vertical_setup["full range"],
                                offset=vertical_setup["offset"])
        elif setup_type == "vertical":
            self.setup_horizontal(duration=horizontal_setup["duration"])
        else:
            raise InvalidParameterError("The setup type is not valid. See documentation for valid arguments.")
        if keep_trigger_settings:
            self.setup_trigger(trigger_type=trigger_setup["trigger type"], \
                               trigger_source=trigger_setup["trigger source"], \
                               trigger_level=trigger_setup["trigger level"], \
                               hold_type=trigger_setup["hold type"])
    #----------------------------------------------
    #Timebase

    def setup_horizontal(self, duration):
        """
        INPUTS
        ------------
            sampling_rate : float (>0)
                sampling rate of the acquisition [Hz]
            duration : float (>0, divisible by 10)
                time duration of the acquisition [s]
            trigger_source : str
                source signal for triggering the acquisition
        """
        self.set_duration(duration)

    def get_horizontal_setup(self):
        setup = {}
        setup["duration"] = float(self.instrument.ask("TDIV?").split("TDIV")[1].replace("S", ""))*10
        return setup

    def set_duration(self, duration):
        """
        :param duration: float (>0)
            acquisition duration [s]
        :return:
        """
        secons_per_division = duration/10
        self.instrument.write('TDIV '+str(secons_per_division)) #set the time duration [s]
        self.instrument.ask('*OPC?')

    #-------------------------------------------------------------------------------------------
    #Trigger

    def setup_trigger(self, trigger_type='EDGE', trigger_source='LINE', trigger_level=0, hold_type="off"):
        """

        :param trigger_type: str
        :param trigger_source: str
        :param trigger_level: float
            trigger level [V]
        :param hold_type: str

        :return:
        """
        self.instrument.write('TRSE '+trigger_type+',SR,' + trigger_source+',HT,'+hold_type)  # set trigger type and source
        if trigger_source != "LINE":
           self.instrument.write(trigger_source + ':TRIG_LEVEL '+str(trigger_level))  # set trigger level
        self.instrument.ask('*OPC?')

    def get_trigger_setup(self):
        trigger_setup = {}
        trigger_setup["trigger type"], _, trigger_setup["trigger source"], _, trigger_setup["hold type"]\
        = self.instrument.ask("TRIG_SELECT?").split(" ")[1].split(",")
        trigger_setup["trigger level"] = float(self.instrument.ask("TRIG_LEVEL?").split("TRLV")[1].replace("V", ""))
        return trigger_setup



    #----------------------------------------------------------------------------------------------
    #Acquisition

    def display_continuous(self):
        self.instrument.write('TRMD NORM')
        self.instrument.ask('*OPC?')

    def enable_sequenced_mode(self, n_segments, n_samples_per_segment):
        command_string = 'SEQ ON,'+str(int(n_segments))+','+str(int(n_samples_per_segment))
        self.instrument.write(command_string)
        self.instrument.ask('*OPC?')
        return command_string
    '''
    def save_trace(self, directory=None, channels='C1 to C4', filename=None, keep_displaying=True): #Selecting the directory or the channels is not working
        """
        Save current displayed traces to the specified filename.      
        INPUTS
        ------------
            channels : str
                The channel numbers in a readable format for the oscilloscope. e,g, 'C1' or 'C1 to C4'.
        """
        self.stop()
        if directory is not None:
            self.set_save_directory(directory)
        self.instrument.write('ARM')
        self.instrument.write('WAIT')
        self.instrument.write('STO '+channels+', FILE')
        self.instrument.write('WAIT')
        self.instrument.ask('*OPC?')
        if keep_displaying:
            self.display_continuous()
    '''

    def start_acquisition(self, store_all_traces=True, keep_displaying=True):
        self.stop()
        self.instrument.write('ARM')
        self.wait()
        self.instrument.ask('*OPC?')
        if store_all_traces:
            for channel_name in list(self.channels.keys()):
                self.store_trace(channel_name)
        self.instrument.ask('*OPC?')
        if keep_displaying:
            self.display_continuous()


    def store_trace(self, channel_name):
        """
        Saves the last acquired trace in a file in the scope's current save directory

        :return:
        """
        channel_number = self.channels[channel_name].number
        command_string = 'STO ' + "C"+str(channel_number) + ', FILE'
        self.instrument.write(command_string)
        return command_string

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
            if not self.channels[channel_name].is_enabled():
                self.channels[channel_name].enable(True)
                if adapt_display != "":
                    self.setup_automatic(channel_name=channel_name, setup_type=adapt_display)
        #Start acquisition
        self.start_acquisition(store_all_traces=False)
        #Store traces locally in the oscilloscope
        for channel_name in channel_names:
            self.store_trace(channel_name=channel_name)
        #Transfer the selected traces to the host save directory
        if save_directory is None:
            save_directory = self.save_directory
        #If the destination save directory does not exist, create it
        if not os.path.isdir(save_directory):
            try:
                os.mkdir(save_directory)
            except:
                raise FileNotFoundError("Save directory on the host does not exist")
        #See what is there in the scope's save directory
        scope_save_directory = self.host_drive+"\\"+self.save_directory
        scope_filenames = os.listdir(scope_save_directory)
        print(scope_filenames)
        scope_filenames_latest = []
        #Select the most recent files for the selected traces
        #These will be the files to be sent
        for j in range(len(channel_names)):
            channel_name = channel_names[j]
            channel_number = self.channels[channel_name].number
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
        self.display_continuous()


    #-----------------------------------------------------------------------------------------------------------
    #File management
        
    def create_directory(self, directory, force_create=False):
        """
        Creates a new directory in the scope's hard drive.
        
        INPUTS
        --------------
            directory : str
                Path of the target directory, relative to the scope's hard drive path 'self.local_drive'.
                Separations between directories must be represented as '\\'.
        
        OUTPUTS
        --------------
            The command being sent by the host computer.
        """
        command_string = "DIR DISK, HDD, ACTION, CREATE, "+self.local_drive+"\\"+directory
        if not os.path.isdir(self.host_drive + "\\" + directory):
            self.instrument.write(command_string)
        else:
            if force_create:
                self.delete_directory(directory)
                self.create_directory(directory, force_create=False)
        return command_string

    def delete_directory(self, directory):
        """
        Deletes a directory in the scope's hard drive.

        INPUTS
        --------------
            directory : str
                Path of the target directory, relative to the scope's hard drive path 'self.local_drive'.
                Separations between directories must be represented as '\\'.

        OUTPUTS
        --------------
            The command being sent by the host computer.
        """
        command_string = "DIR DISK, HDD, ACTION, DELETE, " + self.local_drive + "\\" + directory
        if os.path.isdir(self.host_drive + "\\" + directory):
            self.instrument.write(command_string)
            self.instrument.ask('*OPC?')
        return command_string

    def set_save_directory(self, directory, force_create=False): #Switching the oscilloscope to the save directory is not working
        """
        Sets the directory where traces are saved. It does the following:
            1. Check, from the host computer, if the target directory exists
            2. If it does not, create a new one; else do nothing
            3. Set the save directory
        INPUTS
        ----------
            directory : str
                Path of the target directory, relative to the scope's hard drive path 'self.local_drive'.
                Separations between directories must be represented as '\\'.

        OUTPUTS
        -----------
        """
        self.create_directory(directory, force_create)
        command_string = "DIR DISK,HDD,ACTION,SWITCH," + self.local_drive + "\\" + directory
        self.instrument.write(command_string)
        self.instrument.ask('*OPC?')
        self.save_directory = directory

    def get_save_directory(self):
        """
        Gets the current save directory.

        :return:
        """
        return  "\\".join(self.instrument.ask("DIRECTORY? DISK,HDD").split('\r\n')[1].split("Directory of ")[1].split("\\")[1:])


    def get_local_drive_name(self):
        """
        Gets the name of the scope local drive

        :return:
        """
        return (self.instrument.ask("DIRECTORY? DISK,HDD").split('\r\n')[1].split("Directory of ")[1].split("\\")[0])

    '''
    def set_filename(self, filename, channels, force_create=False): #NOT WORKING
        """
        Sets the default file name to be given to the next files, saved in the current save directory.
        INPUTS
        ----------
            filename : str
                default file name to be given to the next saved files. It must not contain the file extension.
            channels : str
                channels to which the default file name is given
        OUTPUTS
        -----------
        """
        command_string = "FLNM TYPE,"+channels+" FILE" + filename
        self.instrument.write(command_string)
        self.instrument.write("WAIT")
        self.instrument.ask('*OPC?')
        self.last_filenames = {channels: filename}
    '''

    def delete_file(self, filename, relative_to_save_directory=False):
        """
        Deletes a file from the scope's system.

        :param filename: str
            Absolute path of the file
        :param relative_to_save_directory: bool
            If True, filename is considered a path relative to the current save directory
        :return:
        """
        if relative_to_save_directory:
            filename = self.local_drive+"\\"+self.save_directory+"\\"+filename
        command_string = "DELF DISK,HDD,FILE,'"+filename+"'"
        self.instrument.write(command_string)
        self.instrument.ask('*OPC?')
        return command_string

    def stop(self):
        """
        Stops the current acquisition
        """
        self.instrument.write('STOP')
        self.instrument.ask('*OPC?')

    def wait(self):
        """
        Stops the current acquisition
        """
        self.instrument.write("WAIT")



class LecroyOscilloscpeChannel:
    """
    This class describes a single channel of a Lecroy oscilloscope
    """
    def __init__(self, instrument, channel_number, name="no name"):
        self.instrument = instrument
        self.number = int(channel_number)
        self.name = name
    #----------------------------------------------------------------------------------------------------
    #Vertical

    def setup_vertical(self, voltage_range, offset):
        """
        Sets the voltage range and the vertical offset for the input channel.

        :param channel: str
        :param voltage_range: float
            full range of the voltages displayed by the scope for the input channel.
        :param offset: float
            vertical offset of the scope for the input channel.
        :return:
            the list of command strings being sent to the scope
        """
        volt_per_division = voltage_range/10
        command_string = []
        command_string.append("C"+str(self.number)+":VOLT_DIV "+str(volt_per_division))
        command_string.append("C"+str(self.number)+":OFST "+str(offset))
        for c in command_string:
            self.instrument.write(c)
        self.instrument.ask('*OPC?')
        return command_string

    def get_vertical_setup(self):
        setup = {}
        setup["full range"] = float(self.instrument.ask("C"+str(self.number)+":VOLT_DIV?").split("VDIV")[1].replace("V", ""))*10
        setup["offset"] = float(self.instrument.ask("C"+str(self.number)+":OFST?").split("OFST")[1].replace("V", ""))
        return setup

    def set_vertical_range(self, voltage_range):
        """
        Sets the voltage range for the input channel.

        :param channel: str
        :param voltage_range: float
            full range of the voltages displayed by the scope for the input channel.
        :return:
        """
        volt_per_division = voltage_range / 10
        command_string = "C"+str(self.number)+":VOLT_DIV "+str(volt_per_division)
        self.instrument.write(command_string)
        self.instrument.ask('*OPC?')
        return command_string

    def set_vertical_offset(self, offset):
        """
        Sets the vertical offset for the input channel.

        :param channel: str
        :param offset: float
            vertical offset of the scope for the input channel.
        :return:
        """
        command_string = "C"+str(self.number) + ":OFST " + str(offset)
        self.instrument.write(command_string)
        self.instrument.ask('*OPC?')
        return command_string

    # --------------------------------------------------------------------------------------------------------------
    # Other

    def enable(self, enabled):
        """
        Enables or disables the input channels.

        :param channel: str
        :param enable: bool
            If true, the input "C"+str(self.number)s are displayed.
        :return:
        """
        mode = "ON" * (enabled is True) + "OFF" * (enabled is False)
        command_string = "C"+str(self.number) + ":TRACE " + mode
        self.instrument.write(command_string)
        self.instrument.ask('*OPC?')
        return command_string

    def is_enabled(self):
        reply = self.instrument.ask("C"+str(self.number) + ":TRACE?").split("TRA")[1].replace(" ", "")
        if reply == "ON":
            return True
        else:
            return False

    def set_coupling(self, coupling):
        """
        Sets the coupling type of the channel input.

        :param channel: str
        :param coupling: str
            Coupling type.
            Valid arguments:
                - "A1M": (AC, 1 MOhm input impedance)
                - "D1M": (DC, 1 MOhm input impedance)
                - "D50": (DC, 50 Ohm input impedance)
                - "GND": grounded input
        :return:
            the command being sent to the scope
        """
        command_string = "C"+str(self.number) + ":CPL " + coupling
        self.instrument.write(command_string)
        self.instrument.ask('*OPC?')
        return command_string

    def get_coupling(self):
        """
        Gets the coupling type of the input channel

        :param channel: str
        :return:
        """
        return self.instrument.ask("C"+str(self.number) + ":COUPLING?").split("CPL")[1].replace(" ", "")




