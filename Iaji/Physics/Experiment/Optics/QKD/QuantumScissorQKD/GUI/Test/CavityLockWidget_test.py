from PyQt5.QtWidgets import QApplication
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.CavityLock import CavityLock
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.CavityLockWidget import CavityLockWidget
#%%
#----------------------------------------------------------------------------------------------------------

#Test application
cavity_lock = CavityLock(redpitaya_config_filename="O:\\LIST-QPIT\\Catlab\\Quantum-Scissors-QKD\\Software\\RedPitaya\\Pyrpl\\Config-files\\FC2_QKD-lunfa")
app = QApplication(sys.argv)
widget = CavityLockWidget(cavity_lock)
widget.show()
app.exec()