"""
This module describes a Sigilent signal generator, remotely controlled with vx1ii protocol.
For a list of commands, refer to https://siglentna.com/USA_website_2014/Documents/Program_Material/SDG_ProgrammingGuide_PG_E03B.pdf
"""
#%%
import vxi11
from .Exceptions import ConnectionError, InvalidParameterError
#%%
print_separator = '\n-------------------------------------------'
#%%
class SigilentSignalGenerator:
    def __init__(self, IP_address, name="no name", channel_names=["C1", "C2"]):
        self.IP_address = IP_address
        self.name = name
        try:
            self.connect()
        except ConnectionError: 
            print("WARNING: it was not possible to connect to the signal generator "+self.name)
        self.channels = [SigilentSignalGeneratorChannel(instrument=self.instrument, name=channel_names[j], channel=j+1) for j in range(2)]
        
    def connect(self):
        try:
            self.instrument = vxi11.Instrument(self.IP_address)
            self.connected = True
        except:
            raise ConnectionError
#%%
class SigilentSignalGeneratorChannel:
    def __init__(self, instrument, name="no name", channel=1):
        self.instrument = instrument
        self.name = name
        self.channel = "C"+str(int(channel))
    
            
    def set_parameter(self, parameter_name, parameter_value):
        if parameter_name == "waveform":
            self.set_waveform(parameter_value)
        elif parameter_name == "frequency":
            self.set_frequency(parameter_value)
        elif parameter_name == "amplitude":
            self.set_amplitude(parameter_value)
        elif parameter_name == "offset":
            self.set_offset(parameter_value)
        elif parameter_name == "duty cycle":
            self.set_duty_cycle(parameter_value)
        else:
            error_message = "Trying to set invalid parameter to signal generator '"+self.name+"'." \
                            +"\n Parameter with name "+parameter_name+" was not recognized."
            raise InvalidParameterError(error_message)
    
    def set_waveform(self, waveform):
        setup_string = self.channel \
                       + ":BSWV "\
                       + "WVTP," + waveform 
        self.instrument.write(setup_string)
        self.waveform = waveform
        return setup_string
    
    def set_frequency(self, frequency):
        setup_string = self.channel \
                       + ":BSWV "\
                      + "FRQ," + str(frequency)
        self.instrument.write(setup_string)
        self.frequency = frequency
        return setup_string

    def set_amplitude(self, amplitude):
        setup_string = self.channel \
                       + ":BSWV "\
                       + "AMP," + str(amplitude)
        self.instrument.write(setup_string)
        self.amplitude = amplitude
        return setup_string
    
    def set_offset(self, offset):
        setup_string = self.channel \
                       + ":BSWV "\
                       + "OFST," + str(offset)
        self.instrument.write(setup_string)
        self.offset = offset
        return setup_string
    
    def set_duty_cycle(self, duty_cycle):
        if self.waveform != "SQUARE":
            print("WARNING: did not set duty cycle to signal generator '"+self.name+"' because its waveform is not 'SQUARE'. It is "+self.waveform)
            return            
        setup_string = self.channel \
                       + ":BSWV "\
                       + "OFST," + str(duty_cycle) 
        self.instrument.write(setup_string)
        self.duty_cycle = duty_cycle
        return setup_string
                   
        
    def turn_off(self):
        self.instrument.write(self.channel+":OUTP off")
        print(self.channel+":OUTP off")
        self.output = "off"

        
    def turn_on(self):
        self.instrument.write(self.channel+":OUTP on")
        self.output = "on"
        
    
        
