'''
Test script for the AcquisitionSystem module
'''
# In[imports]
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.AcquisitionSystem import AcquisitionSystem
from Iaji.InstrumentsControl.LecroyOscilloscope import LecroyOscilloscope
# In[]
scope = LecroyOscilloscope(IP_address = "10.54.10.222")
acq = AcquisitionSystem(scope)