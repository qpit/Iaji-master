# In[imports]
from Iaji.InstrumentsControl.SigilentSignalGenerator import SigilentSignalGenerator as SG

# In[]
sg = SG(address="USB0::0xF4ED::0xEE3A::NDG2XCA4160177::INSTR", name="Coherent State AOMs")

import time
states = [10, 5, 0]
periods = [60e-3, 20e-3, 20e-3]
state_count = 0
count = 0
while(count < 50):
    state = states[state_count]
    sg.channels["C1"].set_offset(state)
    time.sleep(periods[state_count])
    state_count = (state_count + 1)%3
    if state_count == 2:
        count += 1