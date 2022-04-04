from PyQt5.QtWidgets import QApplication
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.PhaseController import PhaseController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.AcquisitionSystem import AcquisitionSystem
from Iaji.InstrumentsControl.LecroyOscilloscope import LecroyOscilloscope as Scope
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.HomodyneDetectionController import HomodyneDetectionController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.HomodyneDetectionControllerWidget import HomodyneDetectionControllerWidget
import sys
#%%
#----------------------------------------------------------------------------------------------------------

#Test application
phase_controller = PhaseController(redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\input_state_single_photon_interference",\
                                enable_modulation_output=True, pid_autotune=False)
phase_controller.assign_input_output(error_signal_input="in1", control_signal_output="out1")
acquisition_system = AcquisitionSystem(Scope(IP_address="10.54.11.187"))
hd = HomodyneDetectionController(phase_controller, acquisition_system)
app = QApplication(sys.argv)
widget = HomodyneDetectionControllerWidget(hd)
widget.show()
app.exec()