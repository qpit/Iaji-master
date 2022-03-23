from PyQt5.QtWidgets import QApplication
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.PhaseController import PhaseController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI import PhaseControllerWidget
import sys
#%%
#----------------------------------------------------------------------------------------------------------

#Test application
phase_controller = PhaseController(redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\HD_Tx_lock",\
                                enable_modulation_output=True, pid_autotune=False)
app = QApplication(sys.argv)
widget = PhaseControllerWidget.PhaseControllerWidget(phase_controller)
widget.show()
app.exec()