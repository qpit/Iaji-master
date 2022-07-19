# -*- coding: utf-8 -*-
"""
This script tests the StateGenerator module.
It also generates a 3-level step signal to be used for calibrating the input
state.
"""

from PyQt5.QtWidgets import QApplication
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.PhaseController import PhaseController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.AcquisitionSystem import AcquisitionSystem
from Iaji.InstrumentsControl.LecroyOscilloscope import LecroyOscilloscope as Scope
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.HomodyneDetectionController import HomodyneDetectionController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateMeasurementController import StateMeasurementController
from Iaji.InstrumentsControl.SigilentSignalGenerator import SigilentSignalGenerator

from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateGenerator import StateGenerator
import sys
import time, threading, numpy
from scipy import signal
#%%
#----------------------------------------------------------------------------------------------------------

#Test application
#State measurement controller
#phase_controller = PhaseController(redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\HD_Tx_lock",\
                                #enable_modulation_output=True, pid_autotune=False)
#acquisition_system = AcquisitionSystem(Scope(IP_address="10.54.10.222"))
#hd = HomodyneDetectionController(phase_controller, acquisition_system)
state_measurement = StateMeasurementController(None)
#Signal generator
#signal_generator = SigilentSignalGenerator(address="USB0::0xF4ED::0xEE3A::NDG2XCA4160177::INSTR", protocol="visa")
#State generator
rp_config = "O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\channel_losses"
state_generator = StateGenerator(redpitaya_config_filename=rp_config, \
                                 signal_enabler=None, state_measurement=state_measurement)
#%%
#Define the function that generates the 3-level step signal
def n_level_step_function(frequency, levels, duty_cycles, n_points, Ts):
    assert len(levels) == len(duty_cycles), \
    'The number of levels must be equal to the number of duty cycles'
    assert numpy.sum(duty_cycles) == 1, \
    'The duty cycles must add up to 100%%'
    n_levels = len(levels)
    period = 1/frequency
    n_period = int(numpy.floor(period/Ts))
    function_period = []
    for j in range(n_levels):
        function_period = numpy.concatenate((function_period, levels[j]*numpy.ones((int(duty_cycles[j]*n_period),)), \
                                         ))
    #Determine the number of periods contained in the specified number of points
    n_periods = numpy.floor(n_points/n_period)
    n_leftover = n_points%n_period
    function = numpy.repeat(function_period, n_periods)
    function = numpy.concatenate((function, function_period[:n_leftover]))
    return function

asg0 = state_generator.pyrpl_obj.rp.asg0
asg1 = state_generator.pyrpl_obj.rp.asg1
scope = state_generator.pyrpl_obj.rp.scope
scope.decimation = 128
Ts = scope.decimation/125e6

amplification_gain = 2.5
amplification_gain = 2.5
rp_offset = 1
levels_3 = numpy.array([5, 0.5, 0]) / amplification_gain - rp_offset
levels_2 = numpy.array([0, 5]) / amplification_gain - rp_offset
n_points = 2**14
frequency = int(1/(n_points*Ts))
asg0.data = n_level_step_function(frequency, levels_3, [0.6, 0.2, 0.2], n_points, Ts)
#time.sleep(0.1)
#Optimize the generation of the two waveforms by requiring that their (random) time delay between be lower
#than a certain target value
#visualize and acquire the waveforms on the scope
scope.input1 = "asg0"
scope.input2 = "asg1"
max_delay = 2e-4
delay = 4e-4
while delay > max_delay:
    print("delay: %.2f ms > %.2f ms" % (delay*1e3, max_delay*1e3))
    asg1.setup(trigger_source="immediately")
    asg1.data = n_level_step_function(frequency, levels_2, [0.6, 0.4], n_points, Ts)
    #acquire
    traces = scope.curve()
    scope.continuous()
    trace_3 = traces[0]
    trace_2 = traces[1]
    #Compute the gradients
    gradient_3 = numpy.gradient(trace_3)
    gradient_2 = numpy.gradient(trace_2)
    #Find the first positive peak the 3-level step function
    peak_3 = signal.find_peaks(-gradient_3)[0][0]
    #Find the first negative peak the 2-level step function
    peak_2 =  signal.find_peaks(gradient_2)[0][0]
    delay = abs(Ts*(peak_3-peak_2))

scope.input1 = "out1"
scope.input2 = "out2"



