from PyQt5.QtWidgets import QApplication
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.PhaseController import PhaseController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.AcquisitionSystem import AcquisitionSystem
from Iaji.InstrumentsControl.LecroyOscilloscope import LecroyOscilloscope as Scope
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.HomodyneDetectionController import HomodyneDetectionController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateMeasurementController import StateMeasurementController
from Iaji.InstrumentsControl.SigilentSignalGenerator import SigilentSignalGenerator

from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateGenerator import StateGenerator
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateChecking import StateChecking
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.StateCheckingWidget import StateCheckingWidget
import os
import sys
from matplotlib import pyplot
#%%
#----------------------------------------------------------------------------------------------------------
###
# Connect to scope as
# Ch1: HD signal
# Ch2: AOM
###

MainScope = True
OPO2 = False

local_config_files_folder = "C:\\Users\\qpitlab\\Desktop\\Scissor QKD data\\Config-files\\"
network_config_files_folder = "O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\"
folder = local_config_files_folder

#Test application
#State measurement controller
'''
In PhaseController, set frequency to either "calibration_frequency" for modulation done with amplitude EOM or
"measurement_frequency" for modulation done with PG OPO piezo. Values are defined in PhaseController module.
'''

Dr_Jacoby_phase_controller = PhaseController(redpitaya_config_filename=os.path.join(folder, "HD_Dr_Jacoby"),\
                                frequency="calibration_frequency", name="Dr. Jacoby Phase Controller", enable_modulation_output=False, pid_autotune=False)
Blue_Velvet_phase_controller = PhaseController(redpitaya_config_filename=os.path.join(folder, "HD_Blue_Velvet"),\
                                        frequency="measurement_frequency", name="Blue Velvet Phase Controller", enable_modulation_output=True, pid_autotune=False)
if OPO2:
    pass

#Relay interference phase controller
relay_phase_controller = PhaseController(redpitaya_config_filename=os.path.join(local_config_files_folder, "relay_phase_lock"),\
                                frequency="calibration_frequency", name="Relay Phase Controller", enable_modulation_output=True, pid_autotune=False) # Change PhaseController if name is changed

if MainScope:
#Main scope
    acquisition_system = AcquisitionSystem(Scope(IP_address="192.168.1.63"))
    print('Main scope')
#Test scope
else:
    acquisition_system = AcquisitionSystem(Scope(IP_address="10.54.11.44"))
    print('Test scope')

Dr_Jacoby_hd = HomodyneDetectionController(Dr_Jacoby_phase_controller, acquisition_system)
Blue_Velvet_hd = HomodyneDetectionController(Blue_Velvet_phase_controller, acquisition_system)
relay_hd = HomodyneDetectionController(relay_phase_controller, acquisition_system)

Dr_Jacoby_state_measurement = StateMeasurementController(Dr_Jacoby_hd)
Blue_Velvet_state_measurement = StateMeasurementController(Blue_Velvet_hd)
relay_state_measurement = StateMeasurementController(relay_hd)

#Signal generator
#signal_generator = SigilentSignalGenerator(address="USB0::0xF4ED::0xEE3A::NDG2XCA4160177::INSTR", protocol="visa")
#State generator
print('State generator')
state_generator = StateGenerator(eom_redpitaya_config_filename=os.path.join(local_config_files_folder, "eom_pitaya"), \
                                aom_redpitaya_config_filename=os.path.join(local_config_files_folder, "aom_pitaya"), \
                                sample_hold_redpitaya_config_filename=os.path.join(local_config_files_folder, "sample_hold"), \
                                state_measurement=Dr_Jacoby_state_measurement)
#State checking
print('Checking')
state_checking = StateChecking(state_measurement=Blue_Velvet_state_measurement, state_generator=state_generator)

#State generator widget
app = QApplication(sys.argv)
widget = StateCheckingWidget(state_generator=state_generator, state_checking=state_checking, relay_lock=relay_state_measurement)
print("Connected to " + "Main"*(MainScope) + "Secundary"*(not MainScope) + " scope.")
print("Not c"*(not OPO2) + "C"*(OPO2) + "onnected to OPO2.")
widget.show()
pyplot.close('all')
app.exec()