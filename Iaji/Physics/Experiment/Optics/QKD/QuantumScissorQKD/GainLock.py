"""
This module describes a parametric gain locking system realized with a RedPitaya pyrpl module. It consists of:

    -The pyrpl object of the RedPitaya controlling the parametric gain
"""
# %%
import pyrpl
from pyqtgraph.Qt import QtGui
from .Exceptions import ConnectionError


# %%
class GainLock:
    def __init__(self, redpitaya_config_filename, name="Parametric Gain Lock", redpitaya_name="Gain Lock Redpitaya", connect=True, show_pyrpl_GUI=True):
        self.name = name
        self.redpitaya_name = redpitaya_name
        self.redpitaya_config_filename = redpitaya_config_filename
        self.name = name
        self.redpitaya_name = redpitaya_name
        self.redpitaya_config_filename = redpitaya_config_filename
        self.pyrpl_obj = None
        self.pyrpl_GUI = None
        if connect:
            self.connect_to_redpitaya(show_pyrpl_GUI=show_pyrpl_GUI)

    def connect_to_redpitaya(self, show_pyrpl_GUI=True):
        self.pyrpl_obj = pyrpl.Pyrpl(config=self.redpitaya_config_filename)
        if show_pyrpl_GUI:
            self.pyrpl_GUI = QtGui.QApplication.instance()


    def scan(self): #TODO
        return

    def lock(self): #TODO
        return

    def unlock(self): #TODO
        return