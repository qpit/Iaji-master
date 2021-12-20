"""
This module defines the GUI of the HomodyneDetectorController module.
"""
#%%
from PyQt5.QtCore import Qt, QRect
from PyQt5.Qt import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDateTimeEdit,
    QDial,
    QDoubleSpinBox,
    QFontComboBox,
    QLabel,
    QLCDNumber,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QBoxLayout,
    QGridLayout,
    QWidget,
)

from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.HomodyneDetectorController import HomodyneDetectorController
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.PhaseControllerWidget import PhaseControllerWidget
from Iaji.Physics.Experiment.Optics.QKD.QuantumScissorQKD.GUI.WidgetStyles import HomodyneDetectorControllerStyle
import numpy as np

class HomodyneDetectorControllerWidget(QWidget):
    """
    This class describes the widget of a homodyne detector controller.
    Documentation to be completed.
    """
    def __init__(self, homodyne_detector_controller, name="Homodyne Detector Controller Widget"):
        super().__init__()
        self.hd_controller = homodyne_detector_controller
        self.setWindowTitle(name)
        self.layout = QVBoxLayout()
        # Define widget title label
        self.name_label = QLabel()
        self.name_label.setText(self.hd_controller.name)
        self.layout.addWidget(self.name_label, Qt.AlignCenter)
        #Add phase controller widget
        self.phase_controller_widget = PhaseControllerWidget(self.hd_controller.phase_controller)
        self.layout_phase_controller = QVBoxLayout()
        self.layout_phase_controller.addWidget(self.phase_controller_widget)
        self.layout.addLayout(self.layout_phase_controller)

        self.style_sheets = HomodyneDetectorControllerStyle().style_sheets
        self.set_style(theme="dark")
        self.setLayout(self.layout)


    def set_style(self, theme): #TODO
        """
        This function sets the visual appearance of the widget

        :param theme: str
            Visual theme.
            Accepted values are:
                - "light" - not implemented yet
                - "dark"
        :return:
        """
        self.setStyleSheet(self.style_sheets["main"][theme])
        self.name_label.setStyleSheet(self.style_sheets["label"][theme])


