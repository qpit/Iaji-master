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
        self.channels = dict(zip(channel_names, [SigilentSignalGeneratorChannel(instrument=self.instrument, name=channel_names[j], channel_number=j+1) for j in range(2)]))
        
    def connect(self):
        try:
            self.instrument = vxi11.Instrument(self.IP_address)
            self.connected = True
        except:
            raise ConnectionError
#%%
class SigilentSignalGeneratorChannel:
    def __init__(self, instrument, channel_number, name="no name"):
        self.instrument = instrument
        self.name = name
        self.number = int(channel_number)
    
            
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
        command_string = "C"+str(self.number) \
                       + ":BSWV "\
                       + "WVTP," + waveform 
        self.instrument.write(command_string)
        self.waveform = waveform
        return command_string
    
    def set_frequency(self, frequency):
        command_string = "C"+str(self.number) \
                       + ":BSWV "\
                      + "FRQ," + str(frequency)
        self.instrument.write(command_string)
        self.frequency = frequency
        return command_string

    def set_amplitude(self, amplitude):
        command_string = "C"+str(self.number) \
                       + ":BSWV "\
                       + "AMP," + str(amplitude)
        self.instrument.write(command_string)
        self.amplitude = amplitude
        return command_string
    
    def set_offset(self, offset):
        command_string = "C"+str(self.number) \
                       + ":BSWV "\
                       + "OFST," + str(offset)
        self.instrument.write(command_string)
        self.offset = offset
        return command_string
    
    def set_duty_cycle(self, duty_cycle):
        if self.waveform != "SQUARE":
            print("WARNING: did not set duty cycle to signal generator '"+self.name+"' because its waveform is not 'SQUARE'. It is "+self.waveform)
            return            
        command_string = "C"+str(self.number) \
                       + ":BSWV "\
                       + "OFST," + str(duty_cycle) 
        self.instrument.write(command_string)
        self.duty_cycle = duty_cycle
        return command_string
                   
        
    def turn_off(self):
        self.instrument.write("C"+str(self.number)+":OUTP off")
        print("C"+str(self.number)+":OUTP off")
        self.output = "off"

        
    def turn_on(self):
        self.instrument.write("C"+str(self.number)+":OUTP on")
        self.output = "on"
        
    
        
