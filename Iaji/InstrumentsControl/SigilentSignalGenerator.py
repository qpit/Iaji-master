"""
This module describes a Sigilent signal generator, remotely controlled with vx1ii protocol.
For a list of commands, refer to https://siglentna.com/USA_website_2014/Documents/Program_Material/SDG_ProgrammingGuide_PG_E03B.pdf
"""
#%%
import vxi11, pyvisa
from .Exceptions import ConnectionError, InvalidParameterError
import signalslot
#%%
print_separator = '\n-------------------------------------------'
# In[Global variables]
PROTOCOLS = ["visa", "vxi"]
WAVEFORM_TYPES = ["SINE", "SQUARE", "RAMP", "PULSE", "NOISE", "ARB", "DC", "PRBS"]
# In[Signal generator]
class SigilentSignalGenerator:
    # -------------------------------------------
    def __init__(self, address, name="no name", channel_names=["C1", "C2"], protocol="visa"):
        assert protocol in PROTOCOLS, \
        "Invalid communication protocol. Valid arguments are %s"%PROTOCOLS
        self.address = address
        self.name = name
        self.communication_protocol = protocol
        try:
            self.connect()
        except ConnectionError: 
            print("WARNING: it was not possible to connect to the signal generator "+self.name)
        self.channels = dict(zip(channel_names, [SigilentSignalGeneratorChannel(instrument=self.instrument, name=channel_names[j], channel_number=j+1) for j in range(2)]))
    # -------------------------------------------
    def connect(self):
        try:
            if self.communication_protocol == "vxi":
                self.instrument = vxi11.Instrument(self.address)
            else:
                self.instrument = pyvisa.ResourceManager().open_resource(self.address)
            self.connected = True
        except:
            raise ConnectionError
    # -------------------------------------------
    def enable(self, enabled):
        '''
        Turn ON or OFF all channels
        :param enabled: bool
            if True, all channels are turned ON.
        :return:
        '''
        for channel_name in list(self.channels.keys()):
            self.channels[channel_name].enable(enabled)
    # -------------------------------------------
    def lock_phase(self, locked=True):
        state = "ON"*locked + "OFF"*(not locked)
        command_string = "PCOUP,%s"%state
        self.instrument.write(command_string)
        channel_names = list(self.channels.keys())
        if locked:
            self.channels[channel_names[0]].phase_changed.connect(
                self.channels[channel_names[1]].set_phase)
            self.channels[channel_names[1]].phase_changed.connect(
                self.channels[channel_names[0]].set_phase)
        else:
            self.channels[channel_names[0]].phase_changed.disconnect()
            self.channels[channel_names[1]].phase_changed.disconnect()
        self.phase_locked = locked
        return command_string
    # -------------------------------------------
# In[Signal generator channel]
class SigilentSignalGeneratorChannel:
    # -------------------------------------------
    def __init__(self, instrument, channel_number, name="no name"):
        self.instrument = instrument
        self.name = name
        self.number = int(channel_number)
        self.phase_changed = signalslot.Signal()
    # -------------------------------------------
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
    # -------------------------------------------
    def set_waveform(self, waveform):
        command_string = "C"+str(self.number) \
                       + ":BSWV "\
                       + "WVTP," + waveform
        self.instrument.write(command_string)
        self.waveform = waveform
        return command_string
    # -------------------------------------------
    def set_frequency(self, frequency):
        command_string = "C"+str(self.number) \
                       + ":BSWV "\
                      + "FRQ," + str(frequency)
        self.instrument.write(command_string)
        self.frequency = frequency
        return command_string
    # -------------------------------------------
    def set_phase(self, phase, **kwargs):
        """
        :param phase: float
            phase [degrees]
        :return:
        """
        command_string = "C"+str(self.number) \
                       + ":BSWV "\
                      + "PHSE," + str(phase)
        self.instrument.write(command_string)
        self.phase = phase
        self.phase_changed.emit(phase)
        return command_string
    # -------------------------------------------
    def set_amplitude(self, amplitude):
        command_string = "C"+str(self.number) \
                       + ":BSWV "\
                       + "AMP," + str(amplitude)
        self.instrument.write(command_string)
        self.amplitude = amplitude
        return command_string
    # -------------------------------------------
    def set_offset(self, offset):
        command_string = "C"+str(self.number) \
                       + ":BSWV "\
                       + "OFST," + str(offset)
        self.instrument.write(command_string)
        self.offset = offset
        return command_string
    # -------------------------------------------
    def set_duty_cycle(self, duty_cycle):
        if self.waveform != "SQUARE":
            print("WARNING: did not set duty cycle to signal generator '"+self.name+"' because its waveform is not 'SQUARE'. It is "+self.waveform)
            return            
        command_string = "C"+str(self.number) \
                       + ":BSWV "\
                       + "DUTY," + str(duty_cycle)
        self.instrument.write(command_string)
        self.duty_cycle = duty_cycle
        return command_string
    # -------------------------------------------
    def set_low_level(self, level):
        if self.waveform != "SQUARE":
            print("WARNING: did not set low level to signal generator '"+self.name+"' because its waveform is not 'SQUARE'. It is "+self.waveform)
            return
        command_string = "C" + str(self.number) \
                         + ":BSWV " \
                         + "LLEV," + str(level)
        self.instrument.write(command_string)
        self.low_level = level
        return command_string
    # -------------------------------------------
    def set_high_level(self, level):
        if self.waveform != "SQUARE":
            print("WARNING: did not set high level to signal generator '"+self.name+"' because its waveform is not 'SQUARE'. It is "+self.waveform)
            return
        command_string = "C" + str(self.number) \
                         + ":BSWV " \
                         + "HLEV," + str(level)
        self.instrument.write(command_string)
        self.high_level = level
        return command_string
    # -------------------------------------------
    def get_parameter(self, parameter_type):
        parameter_types = ["waveform", "frequency", "phase", "amplitude", "offset", "low level", "high level", "duty cycle"]
        assert parameter_type in parameter_types,\
        "invalid parameter type. Accepted arguments are %s"%parameter_types
        #answer = self.instrument.write("C%s:BSWV?"%self.number)
        answer = self.instrument.query("C%s:BSWV?" % self.number)
        if parameter_type == "waveform":
            value = answer.split("WVTP")[1].split(",")[1]
        elif parameter_type == "frequency":
            value = answer.split("FRQ")[1].split(",")[1].split("HZ")[0]
        elif parameter_type == "phase":
            value = answer.split("PHSE")[1].split(",")[1]
        elif  parameter_type == "amplitude":
            value = answer.split("AMP")[1].split(",")[1].split("V")[0]
        elif  parameter_type == "offset":
            value = answer.split("OFST")[1].split(",")[1].split("V")[0]
        elif parameter_type == "low level":
            value = answer.split("LLEV")[1].split(",")[1].split("V")[0]
        elif parameter_type == "high level":
            value = answer.split("HLEV")[1].split(",")[1].split("V")[0]
        elif parameter_type == "duty cycle":
            value = answer.split("DUTY")[1].split(",")[1]
        else:
            return
        if parameter_type != "waveform":
            if value != "":
                value = float(value)
            else:
                return
        setattr(self, parameter_type, value)
        return value
    # -------------------------------------------
    '''
    def turn_off(self):
        self.instrument.write("C"+str(self.number)+":OUTP off")
        print("C"+str(self.number)+":OUTP off")
        self.output = "off"

        
    def turn_on(self):
        self.instrument.write("C"+str(self.number)+":OUTP on")
        self.output = "on"
    '''
    # -------------------------------------------
    def enable(self, enabled):
        """
        Same as self.turn_on() and self.turn_off(), depending on the input argument.
        :param enabled: bool
            If true, the channel is enabled.
        :return:
        """
        state = "on"*(enabled==True) + "off"*(enabled==False)
        self.instrument.write("C"+str(self.number)+":OUTP "+state)
        self.output = state
        
    

