from PyQt5.QtWidgets import QApplication
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.HomodyneDetectorController import HomodyneDetectorController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.HomodyneDetectorControllerWidget import HomodyneDetectorControllerWidget
import sys
#%%
#----------------------------------------------------------------------------------------------------------

#Test application
hd = HomodyneDetectorController(redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\HD_Tx_lock", name="HD Controller Test")
app = QApplication(sys.argv)
widget = HomodyneDetectorControllerWidget(hd)
widget.show()
app.exec()