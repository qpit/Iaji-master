from PyQt5.QtWidgets import QApplication
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.PumpController import PumpController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.OPOController import OPOController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.TwoModeSqueezedVacuumGenerator import TwoModeSqueezedVacuumGenerator
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.TwoModeSqueezedVacuumGeneratorWidget import TwoModeSqueezedVacuumGeneratorWidget
import sys
#%%
#----------------------------------------------------------------------------------------------------------

#Test application
pump_controller = PumpController(redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\00_SHG_RP5")
OPO_controller = OPOController(cavity_redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\02_OPO2_RP3", \
                               gain_redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\OPO_gain_lock_ACDC_lock")
TMSV_generator = TwoModeSqueezedVacuumGenerator(pump_controller=pump_controller, OPO_controller=OPO_controller, name="Two-mode Squeezed Vacuum Generator Test")
app = QApplication(sys.argv)
widget = TwoModeSqueezedVacuumGeneratorWidget(TMSV_generator=TMSV_generator)
widget.setMinimumSize(50, 50)
widget.setGeometry(0, 0, 100, 100)
widget.show()
app.exec()