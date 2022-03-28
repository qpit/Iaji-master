from PyQt5.QtWidgets import QApplication
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.PhaseController import PhaseController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.AcquisitionSystem import AcquisitionSystem
from Iaji.InstrumentsControl.LecroyOscilloscope import LecroyOscilloscope as Scope
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.HomodyneDetectionController import HomodyneDetectionController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateMeasurementController import StateMeasurementController
from Iaji.InstrumentsControl.SigilentSignalGenerator import SigilentSignalGenerator

from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateGenerator import StateGenerator
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.StateGeneratorWidget import StateGeneratorWidget
import sys
#%%
#----------------------------------------------------------------------------------------------------------

#Test application
#State measurement controller
phase_controller = PhaseController(redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\HD_Tx_lock",\
                                enable_modulation_output=True, pid_autotune=False)
acquisition_system = AcquisitionSystem(Scope(IP_address="10.54.11.189"))
hd = HomodyneDetectionController(phase_controller, acquisition_system)
state_measurement = StateMeasurementController(hd)
#Signal generator
signal_generator = SigilentSignalGenerator(IP_address="10.54.11.240")
#State generator
state_generator = StateGenerator(redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\input_state_modulation", \
                                 signal_generator=signal_generator, state_measurement=state_measurement)
#State generator widget
app = QApplication(sys.argv)
widget = StateGeneratorWidget(state_generator=state_generator)
widget.show()
app.exec()