"""
This module defines the styles of different widgets belonging to this pckage
"""

class WidgetStyle:
    def __init__(self):
        self.widget_types = ["main", "button", "label", "slider", "tabs", "radiobutton", "doublespinbox", \
                             "linedit", "checkbox"]
        self.theme_types = ["dark", "light"]
        self.style_sheets = {}
        for widget_type in self.widget_types:
            self.style_sheets[widget_type] = dict(zip(self.theme_types, [{} for j in range(len(self.theme_types))]))
        #Set default dark theme
        self.style_sheets["main"]["dark"] = "background-color: #082032; color: #EEEEEE;"
        self.style_sheets["label"]["dark"] = """
                                            QLabel
                                            {
                                            background-color : #082032; 
                                            color: #EEEEEE;
                                            border-color: #EEEEEE; 
                                            border: 2px;
                                            font-family: Times New Roman ;
                                            font-size: 18pt;
                                            max-width : 400px;
                                            max-height :  50px;
                                            }
                                            """
        self.style_sheets["button"]["dark"] = """
                                              QPushButton
                                              {
                                              background-color: #2C394B; 
                                              color: #EEEEEE; 
                                              border: 1.5px solid #C4C4C3;
                                              border-color: #EEEEEE;
                                              border-top-left-radius: 4px;
                                              border-top-right-radius: 4px;
                                              border-bottom-left-radius: 4px;
                                              border-bottom-right-radius: 4px;
                                              max-width : 200px;
                                              max-height :  30px;
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              }
                                              QPushButton:pressed
                                              {
                                              background-color: #EEEEEE; 
                                              color: #2C394B;
                                              }
                                              QPushButton:hover
                                              {
                                              background-color: #EEEEEE; 
                                              color: #2C394B;
                                              }
                                              """
        self.style_sheets["tabs"]["dark"] ="""
                                            QTabBar::tab 
                                            {
                                            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                                        stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,
                                                                        stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);
                                            border: 2px solid #C4C4C3;
                                            border-bottom-color: #C2C7CB; /* same as the pane color */
                                            border-top-left-radius: 4px;
                                            border-top-right-radius: 4px;
                                            min-width: 8ex;
                                            width : 250px;
                                            height: 30px;
                                            padding: 2px;
                                            color: black;
                                            font-family: Times New roman;
                                            font-size: 13pt;
                                                }
                                            """
        self.style_sheets["slider"]["dark"] = """
                                              QSlider
                                              {
                                              background-color: #2C394B;
                                              color: #EEEEEE;
                                              border: 1px solid #C4C4C3;
                                              border-color: #EEEEEE;
                                              max-width : 200px;
                                              max-height :  20px;
                                              }
                                              QSlider::handle
                                              {
                                              color : #EEEEEE;
                                              background-color : #EEEEEE;
                                              width : 18px;
                                              height: 35px;
                                              border-radius : 4px;
                                              border: 1px solid #C4C4C3;
                                              border-color: #2C394B; 
                                              }
                                              QSlider::groove
                                              {
                                              background-color: #2C394B;
                                              color: #EEEEEE;
                                              }
                                              """
        self.style_sheets["checkbox"]["dark"] = """
                                              QCheckBox
                                              {
                                              background-color: #2C394B; 
                                              color: #EEEEEE; 
                                              border-color: #EEEEEE;
                                              border: 1.5px solid #C4C4C3;
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              max-width : 200px;
                                              max-height :  20px;
                                              }
                                              """
        self.style_sheets["linedit"]["dark"] = """
                                                 QLineEdit
                                                 {
                                                 background-color : #2C394B;
                                                 border-color:#EEEEEE;
                                                 font-family: Times New Roman;
                                                 font-size: 12pt;
                                                 }
                                                 """
        self.style_sheets["doublespinbox"]["dark"] = """
                                                    QDoubleSpinBox
                                                    {
                                                    background-color : #2C394B;
                                                     border-color:#EEEEEE;
                                                     font-family: Times New Roman;
                                                     font-size: 12pt;
                                                    }
                                                    """

class PhaseControllerWidgetStyle(WidgetStyle):
    def __init__(self):
        super().__init__()
        '''
        self.style_sheets["main"]["dark"] = "background-color: #37474; color:white;"
        self.style_sheets["button"]["dark"] = """
                                              QPushButton
                                              {
                                              background-color: #37474F; 
                                              color: #EEEEEE; 
                                              border-color: #EEEEEE;
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              }
                                              """
        self.style_sheets["label"]["dark"] =  """
                                            QLabel
                                            {
                                            background-color : #37474F; 
                                            color: #EEEEEE;
                                            border-color: #EEEEEE; 
                                            font-family: Times New Roman ;
                                            font-size: 15pt;
                                            }
                                            """
        self.style_sheets["slider"]["dark"] = """
                                              QSlider
                                              {
                                              background-color: #37474F;
                                              color: #EEEEEE;
                                              border-color: #EEEEEE
                                              }
                                              """
        self.style_sheets["tabs"]["dark"] = """
                                            QTabWidget
                                            {
                                            background-color: #37474F;
                                            color: #EEEEEE;
                                            border-color: #EEEEEE
                                            font-family: Times New Roman;
                                            font-size: 12pt
                                            }
                                            """
        self.style_sheets["checkbox"]["dark"] = """
                                              QCheckBox
                                              {
                                              background-color: #37474F; 
                                              color: #EEEEEE; 
                                              border-color: #EEEEEE;
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              }
                                              """
        self.style_sheets["linedit"]["dark"] = """
                                                 QLineEdit
                                                 {
                                                 border-color:#EEEEEE;
                                                 font-family: Times New Roman;
                                                 font-size: 12pt
                                                 }
                                                 """
        '''

class HomodyneDetectionControllerStyle(WidgetStyle):
    def __init__(self):
        super().__init__()

class StateMeasurementControllerStyle(WidgetStyle):
    def __init__(self):
        super().__init__()

class CavityLockWidgetStyle(WidgetStyle):
    def __init__(self):
        super().__init__()
        '''
        self.style_sheets["main"]["dark"] = "background-color: #512da8; color:white;"
        self.style_sheets["button"]["dark"] = """
                                              QPushButton
                                              {
                                              background-color: #512da8; 
                                              color: #EEEEEE; 
                                              border-color: #EEEEEE;
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              }
                                              """
        self.style_sheets["label"]["dark"] =  """
                                            QLabel
                                            {
                                            background-color : #512da8; 
                                            color: #EEEEEE;
                                            border-color: #EEEEEE; 
                                            font-family: Times New Roman ;
                                            font-size: 15pt;
                                            }
                                            """
        self.style_sheets["slider"]["dark"] = """
                                              QSlider
                                              {
                                              background-color: #512da8;
                                              color: #EEEEEE;
                                              border-color: #EEEEEE
                                              }
                                              """
        self.style_sheets["tabs"]["dark"] = """
                                            QTabWidget
                                            {
                                            background-color: #512da8;
                                            color: #EEEEEE;
                                            border-color: #EEEEEE
                                            font-family: Times New Roman;
                                            font-size: 12pt
                                            }
                                            """
        self.style_sheets["radiobutton"]["dark"] = """
                                              QRadioButton
                                              {
                                              background-color: #512da8; 
                                              color: #EEEEEE; 
                                              border-color: #EEEEEE;
                                              font-family: Times New Roman;
                                              font-size: 15pt;
                                              }
                                              """
        '''

class GainLockWidgetStyle(WidgetStyle):
    def __init__(self):
        super().__init__()
        '''
        self.style_sheets["main"]["dark"] = "background-color: #5d4037; color:white;"
        self.style_sheets["button"]["dark"] = """
                                                      QPushButton
                                                      {
                                                      background-color: #5d4037; 
                                                      color: #EEEEEE; 
                                                      border-color: #EEEEEE;
                                                      font-family: Times New Roman;
                                                      font-size: 13pt;
                                                      }
                                                      """
        self.style_sheets["label"]["dark"] = """
                                                    QLabel
                                                    {
                                                    background-color : #5d4037; 
                                                    color: #EEEEEE;
                                                    border-color: #EEEEEE; 
                                                    font-family: Times New Roman ;
                                                    font-size: 15pt;
                                                    }
                                                    """
        self.style_sheets["slider"]["dark"] = """
                                                      QSlider
                                                      {
                                                      background-color: #5d4037;
                                                      color: #EEEEEE;
                                                      border-color: #EEEEEE
                                                      }
                                                      """
        self.style_sheets["tabs"]["dark"] = """
                                                    QTabWidget
                                                    {
                                                    background-color: #5d4037;
                                                    color: #EEEEEE;
                                                    border-color: #EEEEEE
                                                    font-family: Times New Roman;
                                                    font-size: 12pt
                                                    }
                                                    """
        self.style_sheets["checkbox"]["dark"] = """
                                                      QCheckBox
                                                      {
                                                      background-color: #5d4037; 
                                                      color: #EEEEEE; 
                                                      border-color: #EEEEEE;
                                                      font-family: Times New Roman;
                                                      font-size: 13pt;
                                                      }
                                                      """
        self.style_sheets["linedit"]["dark"] = """
                                                         QLineEdit
                                                         {
                                                         border-color:#EEEEEE;
                                                         font-family: Times New Roman;
                                                         font-size: 12pt
                                                         }
                                                         """
        '''

class PIDControlWidgetStyle(WidgetStyle):
    def __init__(self):
        super().__init__()
        '''
        self.style_sheets["main"]["dark"] = "background-color: #37474; color:white;"
        self.style_sheets["button"]["dark"] = """
                                              QPushButton
                                              {
                                              background-color: #1A237E; 
                                              color: #EEEEEE; 
                                              border-color: #EEEEEE;
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              }
                                              """
        self.style_sheets["label"]["dark"] =  """
                                            QLabel
                                            {
                                            background-color : #1A237E; 
                                            color: #EEEEEE;
                                            border-color: #EEEEEE; 
                                            font-family: Times New Roman ;
                                            font-size: 15pt;
                                            }
                                            """
        self.style_sheets["slider"]["dark"] = """
                                              QSlider
                                              {
                                              background-color: #1A237E;
                                              color: #EEEEEE;
                                              border-color: #EEEEEE
                                              }
                                              """
        self.style_sheets["doublespinbox"]["dark"] = """
                                                      QDoubleSpinbox
                                                      {
                                                      background-color: #1A237E;
                                                      color: #EEEEEE;
                                                      border-color: #EEEEEE
                                                      }
                                                      """
        self.style_sheets["tabs"]["dark"] = """
                                            QTabWidget
                                            {
                                            background-color: #1A237E;
                                            color: #EEEEEE;
                                            border-color: #EEEEEE
                                            font-family: Times New Roman;
                                            font-size: 12pt
                                            }
                                            """
        '''
class AcquisitionSystemWidgetStyle(WidgetStyle):
    def __init__(self):
        super().__init__()
        '''
        self.style_sheets["main"]["dark"] = "background-color: #6d597a; color:white;"
        self.style_sheets["button"]["dark"] = """
                                              QPushButton
                                              {
                                              background-color: #6d597a; 
                                              color: #EEEEEE; 
                                              border-color: #EEEEEE;
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              max-width : 400px;
                                              max-height : 50px;
                                              }
                                              """
        self.style_sheets["label"]["dark"] =  """
                                            QLabel
                                            {
                                            background-color : #6d597a; 
                                            color: #EEEEEE;
                                            border-color: #EEEEEE; 
                                            font-family: Times New Roman ;
                                            font-size: 15pt;
                                            }
                                            """
        self.style_sheets["slider"]["dark"] = """
                                              QSlider
                                              {
                                              background-color: #6d597a;
                                              color: #EEEEEE;
                                              border-color: #EEEEEE
                                              }
                                              """
        self.style_sheets["doublespinbox"]["dark"] = """
                                                      QDoubleSpinbox
                                                      {
                                                      background-color: #6d597a;
                                                      color: #EEEEEE;
                                                      border-color: #EEEEEE
                                                      }
                                                      """
        self.style_sheets["tabs"]["dark"] = """
                                            QTabWidget
                                            {
                                            background-color: #6d597a;
                                            color: #EEEEEE;
                                            border-color: #EEEEEE
                                            font-family: Times New Roman;
                                            font-size: 12pt
                                            }
                                            """
        self.style_sheets["linedit"]["dark"] = """
                                                 QLineEdit
                                                 {
                                                 border-color:#EEEEEE;
                                                 font-family: Times New Roman;
                                                 font-size: 12pt
                                                 }
                                                 """
        self.style_sheets["checkbox"]["dark"] = """
                                               QCheckBox
                                               {
                                               background-color: #6d597a; 
                                               color: #EEEEEE; 
                                               border-color: #EEEEEE;
                                               font-family: Times New Roman;
                                               font-size: 13pt;
                                               }
                                               """
        '''





