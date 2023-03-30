from PyQt5.QtWidgets import QApplication
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GainLock import GainLock
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.GainLockWidget import GainLockWidget
import sys
#%%
#----------------------------------------------------------------------------------------------------------

#Test application
gain_lock = GainLock(redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\OPO_gain_lock_ACDC_lock", name="Gain Lock Test")
gain_lock.phase_controller.assign_input_output(error_signal_input="in2", control_signal_output="out2")
gain_lock.phase_controller.assign_modules(asg_control="asg0", iq="iq0")
app = QApplication(sys.argv)
widget = GainLockWidget(gain_lock)
widget.show()
app.exec()