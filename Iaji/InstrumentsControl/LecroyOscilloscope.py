"""
This module describes a Lecroy Oscilloscope, remotely controlled with vx1ii protocol.
For a list of commands, refer to http://cdn.teledynelecroy.com/files/manuals/maui-remote-control-and-automation-manual.pdf
"""
#%%
import vxi11
from .Exceptions import ConnectionError
#%%
print_separator = '\n-------------------------------------------'
#%%
class LecroyOscilloscope:
    def __init__(self, IP_address, sampling_rate=None, duration=None, name="no name", trigger_source="LINE", connect=True):
        self.name = name
        self.IP_address = IP_address
        if connect:
            try:
                self.connect()
                if sampling_rate and duration:
                    self.setup_horizontal(sampling_rate, duration, trigger_source)
            except ConnectionError: 
                print("WARNING: it was not possible to connect to the oscilloscope "+self.name)
                return
        
    def connect(self):
        try:
            self.instrument = vxi11.Instrument(self.IP_address)
        except:
            raise ConnectionError

    def setup_horizontal(self, sampling_rate, duration, trigger_source="LINE"):
        """
        for a list of commands and parameters    

        for a list of commands and parameters.  

        INPUTS
        ------------
            sampling_rate : float (>0)
                sampling rate of the acquisition [Hz]
            duration : float (>0, divisible by 10)
                time duration of the acquisition [ms]
            trigger_source : str
                source signal for triggering the acquisition
        """
        self.instrument.write("TDIV "+str(int(duration/10))+"ms") #set the time duration [ms]
        #self.instrument.write("MSIZ "+str(int(sampling_rate*duration))) #set number of samples per acquisition
        self.instrument.write("TRSE EDGE, SR, "+trigger_source) #set trigger type and source
        self.instrument.write(trigger_source+":TRLV 0V") #set trigger level
        self.instrument.write('WAIT')
        self.instrument.ask('*OPC?')      
        
        
    def setup_vertical(self, channel, volt_per_division):
        return
    
    def display_continuous(self):
        self.instrument.write("TRMD NORM")
    

    def saveTrace(self, filename=None, channels='C1 to C2'):
        """
        Save current displayed traces to the specified filename.      
        INPUTS
        ------------
            filename : str
                The filename after the channel name and before the file index number, WITHOUT EXTENSION.
            channels : str
                The channel numbers in a readable format for the oscilloscope. e,g, "C1" or "C1 to C4".
        """
        if filename is not None:

            self.instrument.write("FILE, "+"""+filename+""")    

        self.instrument.write("STOP")    
        self.instrument.write('ARM')
        self.instrument.write('WAIT')
        self.instrument.write("STO "+channels+", FILE")
        self.instrument.write('WAIT')
        self.instrument.ask('*OPC?')        