from PyQt5.QtWidgets import QApplication
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.PhaseController import PhaseController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.AcquisitionSystem import AcquisitionSystem
from Iaji.InstrumentsControl.LecroyOscilloscope import LecroyOscilloscope as Scope
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.HomodyneDetectionController import HomodyneDetectionController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.StateMeasurementController import StateMeasurementController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.StateMeasurementControllerWidget import StateMeasurementControllerWidget
import sys
#%%
#----------------------------------------------------------------------------------------------------------

#Test application
phase_controller = PhaseController(redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\HD_Tx_lock",\
                                enable_modulation_output=True, pid_autotune=False)
acquisition_system = AcquisitionSystem(Scope(IP_address="10.54.11.187"))
hd = HomodyneDetectionController(phase_controller, acquisition_system)
state_measurement = StateMeasurementController(hd)
app = QApplication(sys.argv)
widget = StateMeasurementControllerWidget(state_measurement)
widget.show()
app.exec()