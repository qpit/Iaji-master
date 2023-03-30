from PyQt5.QtWidgets import QApplication
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.PumpController import PumpController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.PumpControllerWIdget import PumpControllerWidget
import sys
#%%
#----------------------------------------------------------------------------------------------------------

#Test application
pump_controller = PumpController(redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\00_SHG_RP5", name="Pump Controller Test")
app = QApplication(sys.argv)
widget = PumpControllerWidget(pump_controller)
widget.show()
app.exec()