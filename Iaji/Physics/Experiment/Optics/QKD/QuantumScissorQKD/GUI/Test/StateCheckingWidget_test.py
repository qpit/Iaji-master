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
MainScope = True
OPO2 = False

local_config_files_folder = "C:\\Users\\qpitlab\\Desktop\\Scissor QKD data\\Config-files\\"
network_config_files_folder = "O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\"

#Test application
#State measurement controller

Dr_Jacoby_phase_controller = PhaseController(redpitaya_config_filename=os.path.join(local_config_files_folder, "HD_Dr_Jacoby"),\
                                frequency="calibration_frequency", name="Dr Jacoby Phase Controller", enable_modulation_output=False, pid_autotune=True)
Blue_Velvet_phase_controller = PhaseController(redpitaya_config_filename=os.path.join(local_config_files_folder, "HD_Blue_Velvet"),\
                                        frequency="measurement_frequency", name="Blue Velvet Phase Controller", enable_modulation_output=True, pid_autotune=True)

'''
Alice_phase_controller = PhaseController(redpitaya_config_filename=os.path.join(local_config_files_folder, "HD_Rx1_lock"),\
                                        frequency=25e6, name="Silencio HD on Blue Velvet pitaya", enable_modulation_output=True, pid_autotune=True)
Bob_phase_controller = PhaseController(redpitaya_config_filename=os.path.join(local_config_files_folder, "HD_Tx_lock"),\
                                frequency=80e3, name="Blue Velvet HD on Silencio pitaya", enable_modulation_output=True, pid_autotune=True)
'''

#Relay interference phase controller
relay_phase_controller = PhaseController(redpitaya_config_filename=os.path.join(local_config_files_folder, "relay_phase_lock"),\
                                frequency="calibration_frequency", name="Relay Phase Controller", enable_modulation_output=True, pid_autotune=True)

if MainScope:
#Main scope
    acquisition_system = AcquisitionSystem(Scope(IP_address="10.54.10.222"))
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
signal_generator = SigilentSignalGenerator(address="USB0::0xF4ED::0xEE3A::NDG2XCA4160177::INSTR", protocol="visa")
#State generator
print('State generator')
state_generator = StateGenerator(modulation_redpitaya_config_filename=os.path.join(local_config_files_folder, "input_state_modulation"), \
                                calibration_redpitaya_config_filename=os.path.join(local_config_files_folder, "channel_losses"), \
                                 signal_enabler=signal_generator, state_measurement=Dr_Jacoby_state_measurement)
#State checking
state_checking = StateChecking(state_measurement=Blue_Velvet_state_measurement, state_generator=state_generator)

#State generator widget
app = QApplication(sys.argv)
widget = StateCheckingWidget(state_generator=state_generator, state_checking=state_checking, relay_lock=relay_state_measurement)
print("Connected to " + "Main"*(MainScope) + "Secundary"*(not MainScope) + " scope.")
print("Not c"*(not OPO2) + "C"*(OPO2) + "onnected to OPO2.")
widget.show()
pyplot.close('all')
app.exec()