from PyQt5.QtWidgets import QApplication
from Iaji.InstrumentsControl.LecroyOscilloscope import LecroyOscilloscope as Scope
from Iaji.InstrumentsControl.GUI.LecroyOscilloscopeWidget import LecroyOscilloscopeWidget as ScopeWidget
import sys
#%%
#----------------------------------------------------------------------------------------------------------

#Test application
scope = Scope(IP_address="10.54.11.187")
app = QApplication(sys.argv)
widget = ScopeWidget(scope)
widget.show()
scope.set_channel_names([2, 3], ["test 2", "test 3"])
app.exec()


