from PyQt5.QtWidgets import QApplication
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.OPOController import OPOController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.OPOControllerWIdget import OPOControllerWidget
import sys
#%%
#----------------------------------------------------------------------------------------------------------

#Test application
OPO_controller = OPOController(cavity_redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\02_OPO2_RP3", \
                               gain_redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\OPO_gain_lock_ACDC_lock", name="OPO Controller Test")
app = QApplication(sys.argv)
widget = OPOControllerWidget(OPO_controller)
widget.show()
app.exec()