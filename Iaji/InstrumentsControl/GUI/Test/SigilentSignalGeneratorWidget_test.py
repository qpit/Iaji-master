from PyQt5.QtWidgets import QApplication
from Iaji.InstrumentsControl.SigilentSignalGenerator import SigilentSignalGenerator as SG
from Iaji.InstrumentsControl.GUI.SigilentSignalGeneratorWidget import SigilentSignalGeneratorWidget
import sys
#%%
#----------------------------------------------------------------------------------------------------------

#Test application
sg = SG(address="USB0::0xF4ED::0xEE3A::NDG2XCA4160177::INSTR", name="Coherent State AOMs", channel_names=["MOD", "GATE"])
app = QApplication(sys.argv)
widget = SigilentSignalGeneratorWidget(signal_generator=sg)
widget.show()
app.exec()


