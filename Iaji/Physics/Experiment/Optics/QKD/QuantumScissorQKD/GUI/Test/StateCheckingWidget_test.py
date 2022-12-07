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
Bob_HD = True
Bob_detector = "Blue velvet" #"Blue velvet" for Rx1, "Dr Jacoby" for Rx2
OPO2 = False

modulation_frequency = 25e6

local_config_files_folder = "C:\\Users\\qpitlab\\Desktop\\Scissor QKD data\\Config-files\\"
network_config_files_folder = "O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\"

#Test application
#State measurement controller

print('Alice phase controller')
Alice_phase_controller = PhaseController(redpitaya_config_filename=os.path.join(local_config_files_folder, "HD_Tx_lock"),\
                                frequency=modulation_frequency, name="Alice Phase Controller", enable_modulation_output=True, pid_autotune=True)
if Bob_HD:
    print('Bob phase controller')
    print('Rx' + '1'*(Bob_detector == "Blue velvet") + '2'*(Bob_detector == "Dr Jacoby") + ' - ' + Bob_detector)
    if Bob_detector == "Blue velvet":
        Bob_phase_controller = PhaseController(redpitaya_config_filename=os.path.join(local_config_files_folder, "HD_Rx1_lock"),\
                                        frequency=50e3, name="Blue Velvet Phase Controller", enable_modulation_output=True, pid_autotune=True)
    elif Bob_detector == "Dr Jacoby":
        Bob_phase_controller = PhaseController(redpitaya_config_filename=os.path.join(local_config_files_folder, "HD_Rx2_lock"),\
                                        frequency=50e3, name = "Dr Jacoby Phase Controller", enable_modulation_output=True, pid_autotune=True)
    else:
        print("Bob wasn't assigned a valid detector")
else:
    Bob_phase_controller = None
if OPO2:
    pass
'''
Alice_phase_controller = PhaseController(redpitaya_config_filename=os.path.join(local_config_files_folder, "HD_Rx1_lock"),\
                                        frequency=25e6, name="Silencio HD on Blue Velvet pitaya", enable_modulation_output=True, pid_autotune=True)
Bob_phase_controller = PhaseController(redpitaya_config_filename=os.path.join(local_config_files_folder, "HD_Tx_lock"),\
                                frequency=80e3, name="Blue Velvet HD on Silencio pitaya", enable_modulation_output=True, pid_autotune=True)
'''

#Relay interference phase controller
relay_phase_controller = PhaseController(redpitaya_config_filename=os.path.join(local_config_files_folder, "input_state_single_photon_interference"),\
                                frequency=modulation_frequency, name="Relay Phase Controller", enable_modulation_output=False, pid_autotune=True)

if MainScope:
#Main scope
    acquisition_system = AcquisitionSystem(Scope(IP_address="10.54.10.222"))
    print('Main scope')
#Test scope
else:
    acquisition_system = AcquisitionSystem(Scope(IP_address="10.54.11.44"))
    print('Test scope')

Alice_hd = HomodyneDetectionController(Alice_phase_controller, acquisition_system)
if Bob_HD:
    Bob_hd = HomodyneDetectionController(Bob_phase_controller, acquisition_system)
else:
    Bob_hd = None
relay_hd = HomodyneDetectionController(relay_phase_controller, acquisition_system)
Alice_state_measurement = StateMeasurementController(Alice_hd)
if Bob_HD:
    Bob_state_measurement = StateMeasurementController(Bob_hd)
else:
    Bob_state_measurement = None
relay_state_measurement = StateMeasurementController(relay_hd)

#Signal generator
signal_generator = SigilentSignalGenerator(address="USB0::0xF4ED::0xEE3A::NDG2XCA4160177::INSTR", protocol="visa")
#State generator
print('State generator')
state_generator = StateGenerator(modulation_redpitaya_config_filename=os.path.join(local_config_files_folder, "input_state_modulation"), \
                                calibration_redpitaya_config_filename=os.path.join(local_config_files_folder, "channel_losses"), \
                                 signal_enabler=signal_generator, state_measurement=Alice_state_measurement)
#State checking
if Bob_HD:
    state_checking = StateChecking(state_measurement=Bob_state_measurement, state_generator=state_generator)
else:
    state_checking = None
#State generator widget
app = QApplication(sys.argv)
widget = StateCheckingWidget(state_generator=state_generator, state_checking=state_checking, relay_lock=relay_state_measurement)
print("Connected to " + "Main"*(MainScope) + "Secundary"*(not MainScope) + " scope.")
print("Not c"*(not Bob_HD) + "C"*(Bob_HD) + "onnected to Bob's HD.")
print("Not c"*(not OPO2) + "C"*(OPO2) + "onnected to OPO2.")
widget.show()
pyplot.close('all')
app.exec()