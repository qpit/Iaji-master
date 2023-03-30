from PyQt5.QtWidgets import QApplication
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD import \
    CavityLock, \
    GainLock, \
    PhaseController, \
    OPOController,\
    PumpController, \
    TwoModeSqueezedVacuumGenerator
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI import \
    TwoModeSqueezedVacuumGeneratorWidget
import sys
#%%
#----------------------------------------------------------------------------------------------------------

#Test application
#Pump controller
shg_cavity_lock = CavityLock.CavityLock(redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\00_SHG_RP5", \
                                        name="SHG Cavity Lock")
pump_controller = PumpController.PumpController(shg_cavity_lock)
#OPO controller
gain_phase_controller = PhaseController.PhaseController(redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\OPO_gain_lock_ACDC_lock")
gain_lock = GainLock.GainLock(phase_controller=gain_phase_controller)
OPO_cavity_lock = CavityLock.CavityLock(redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\02_OPO2_RP3", name="OPO Cavity Lock")
OPO_controller = OPOController.OPOController(cavity_lock=OPO_cavity_lock, gain_lock=gain_lock)
#Two-mode squeezed vacuum generator
TMSV_generator = TwoModeSqueezedVacuumGenerator.TwoModeSqueezedVacuumGenerator(pump_controller=pump_controller, OPO_controller=OPO_controller, name="Two-mode Squeezed Vacuum Generator Test")
app = QApplication(sys.argv)
widget = TwoModeSqueezedVacuumGeneratorWidget.TwoModeSqueezedVacuumGeneratorWidget(TMSV_generator=TMSV_generator)
widget.setMinimumSize(50, 50)
widget.setGeometry(0, 0, 100, 100)
widget.show()
app.exec()