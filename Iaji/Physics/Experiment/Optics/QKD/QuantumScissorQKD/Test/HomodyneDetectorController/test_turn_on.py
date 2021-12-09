"""
This file tests the HomodyneDetectorController module.
It turns two homodyne detector controllers on.
"""

from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.HomodyneDetectorController import \
    HomodyneDetectorController as HD

#%%
hd_Tx = HD(
    redpitaya_config_filename="O:\LIST-QPIT\Catlab\Quantum-Scissors-QKD\Software\RedPitaya\Pyrpl\Config-files\HD_Tx_lock",\
    enable_modulation_output = True)

hd_Rx1 = HD(
    redpitaya_config_filename="O:\LIST-QPIT\Catlab\Quantum-Scissors-QKD\Software\RedPitaya\Pyrpl\Config-files\HD_Rx1_lock",\
    enable_modulation_output = False)





