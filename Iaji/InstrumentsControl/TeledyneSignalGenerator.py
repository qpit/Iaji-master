"""
This module describes a Teledyne Signal generator, remotely controlled with vx1ii protocol
"""
#%%
import vxi11
from .Exceptions import ConnectionError
#%%
print_separator = '\n-------------------------------------------'
#%%
class TeledyneSignalGeneratorChannel:
    def __init__(self, IP_address, name="no name", channel=1, waveform=None, frequency=0, amplitude=0, offset=0, connect=True):
        self.name = name
        self.IP_address = IP_address
        self.channel = "C"+str(int(channel))
        if connect:
            try:
                self.connect()
                if waveform and frequency and amplitude and offset:
                    self.setup(waveform, frequency, amplitude, offset)
            except ConnectionError: 
                print("WARNING: it was not possible to connect to the signal generator "+self.name)
    
    def connect(self):
        try:
            self.instrument = vxi11.Instrument(self.IP_address)
            self.connected = True
        except:
            raise ConnectionError
    
    def setup(self, waveform, frequency, amplitude, offset):
        """
        INPUTS
        ------------
            amplitude : float (>0)
                Peak-to-peak voltage amplitude of the waveform [Vpp]
        """
        if not self.connected:
            return
        setup_string = self.channel \
                       + ":BSWV "\
                       + "WVTP," + waveform + "," \
                       + "FRQ," + str(frequency) + "HZ," \
                       + "AMP," + str(amplitude) + "V," \
                       + "OFST," + str(offset) + "V"
        self.instrument.write(setup_string)
        return setup_string
    
    def set_waveform(self, waveform):
        setup_string = self.channel \
                       + ":BSWV "\
                       + "WVTP," + waveform 
        return setup_string
    
    def set_frequency(self, frequency):
        setup_string = self.channel \
                       + ":BSWV "\
                      + "FRQ," + str(frequency) + "HZ"
        return setup_string

    def set_amplitdue(self, amplitude):
        setup_string = self.channel \
                       + ":BSWV "\
                       + "AMP," + str(amplitude) + "V" 
        return setup_string
    
    def set_offset(self, offset):
        setup_string = self.channel \
                       + ":BSWV "\
                       + "OFST," + str(offset) + "V" 
        return setup_string
                   
        
    def turn_off(self):
        if not self.connected:
            return
        self.instrument.write(self.channel+":OUTP off")


        print(self.channel+":OUTP off")

        
    def turn_on(self):
        if not self.connected:
            return
        self.instrument.write(self.channel+":OUTP on")
    
    def setAmplitude(self, amplitude):
        """
        Set the peak-to-peak amplitude of the input state function generator waveform to 'amplitude' [Vpp]
        """
        if not self.connected:
            return
        self.instrument.write(self.channel+":AMP, "+str(amplitude)+"V")
        
