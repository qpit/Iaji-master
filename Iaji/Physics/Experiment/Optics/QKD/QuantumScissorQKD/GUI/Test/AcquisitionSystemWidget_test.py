#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:50:44 2022

@author: jiedz
"""
from PyQt5.QtWidgets import QApplication
from Iaji.InstrumentsControl.LecroyOscilloscope import LecroyOscilloscope as Scope
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.AcquisitionSystem import \
    AcquisitionSystem
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.AcquisitionSystemWidget \
    import AcquisitionSystemWidget
import sys
#%%
#----------------------------------------------------------------------------------------------------------

#Test application
scope = Scope(IP_address="10.54.11.187")
acquisition_system = AcquisitionSystem(scope)
app = QApplication(sys.argv)
widget = AcquisitionSystemWidget(acquisition_system)
widget.show()
sys.exit(app.exec_())
