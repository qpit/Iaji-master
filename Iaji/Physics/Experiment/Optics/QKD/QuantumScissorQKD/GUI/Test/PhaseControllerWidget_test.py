from PyQt5.QtWidgets import QApplication
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.PhaseController import PhaseController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI import PhaseControllerWidget
import sys
#%%
#----------------------------------------------------------------------------------------------------------

#Test application
phase_controller = PhaseController(redpitaya_config_filename="C:\\Users\\Qpitlab\\Documents\\FPGAs\\RP-F09009_displacement_AC-DC_lock\\RP-F09009_displacement_AC-DC_lock.yml",\
                                frequency="measurement_frequency", name="AC/DC lock", enable_modulation_output=True, pid_autotune=False)
app = QApplication(sys.argv)
widget = PhaseControllerWidget.PhaseControllerWidget(phase_controller)
widget.show()
app.exec(