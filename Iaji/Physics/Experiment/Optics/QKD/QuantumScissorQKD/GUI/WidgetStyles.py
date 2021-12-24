"""
This module defines the styles of different widgets belonging to this pckage
"""

class WidgetStyle:
    def __init__(self):
        self.widget_types = ["main", "button", "label", "slider", "tabs", "radiobutton", "doublespinbox"]
        self.theme_types = ["dark", "light"]
        self.style_sheets = {}
        for widget_type in self.widget_types:
            self.style_sheets[widget_type] = dict(zip(self.theme_types, [{} for j in range(len(self.theme_types))]))

class PhaseControllerWidgetStyle(WidgetStyle):
    def __init__(self):
        super().__init__()
        self.style_sheets["main"]["dark"] = "background-color: #37474; color:white;"
        self.style_sheets["button"]["dark"] = """
                                              QPushButton
                                              {
                                              background-color: #37474F; 
                                              color: 'white'; 
                                              border-color: 'white';
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              }
                                              """
        self.style_sheets["label"]["dark"] =  """
                                            QLabel
                                            {
                                            background-color : #37474F; 
                                            color: 'white';
                                            border-color: 'white'; 
                                            font-family: Times New Roman ;
                                            font-size: 15pt;
                                            }
                                            """
        self.style_sheets["slider"]["dark"] = """
                                              QSlider
                                              {
                                              background-color: #37474F;
                                              color: 'white';
                                              border-color: 'white'
                                              }
                                              """
        self.style_sheets["tabs"]["dark"] = """
                                            QTabWidget
                                            {
                                            background-color: #37474F;
                                            color: 'white';
                                            border-color: 'white'
                                            font-family: Times New Roman;
                                            font-size: 12pt
                                            }
                                            """

class HomodyneDetectorControllerStyle(WidgetStyle):
    def __init__(self):
        super().__init__()
        self.style_sheets["main"]["dark"] = "background-color: #1c1c1c; color:white;"
        self.style_sheets["label"]["dark"] = """
                                            QLabel
                                            {
                                            background-color : #1c1c1c; 
                                            color: 'white';
                                            border-color: 'white'; 
                                            font-family: Times New Roman ;
                                            font-size: 18pt;
                                            }
                                            """
class CavityLockWidgetStyle(WidgetStyle):
    def __init__(self):
        super().__init__()
    def __init__(self):
        super().__init__()
        self.style_sheets["main"]["dark"] = "background-color: #512da8; color:white;"
        self.style_sheets["button"]["dark"] = """
                                              QPushButton
                                              {
                                              background-color: #512da8; 
                                              color: 'white'; 
                                              border-color: 'white';
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              }
                                              """
        self.style_sheets["label"]["dark"] =  """
                                            QLabel
                                            {
                                            background-color : #512da8; 
                                            color: 'white';
                                            border-color: 'white'; 
                                            font-family: Times New Roman ;
                                            font-size: 15pt;
                                            }
                                            """
        self.style_sheets["slider"]["dark"] = """
                                              QSlider
                                              {
                                              background-color: #512da8;
                                              color: 'white';
                                              border-color: 'white'
                                              }
                                              """
        self.style_sheets["tabs"]["dark"] = """
                                            QTabWidget
                                            {
                                            background-color: #512da8;
                                            color: 'white';
                                            border-color: 'white'
                                            font-family: Times New Roman;
                                            font-size: 12pt
                                            }
                                            """
        self.style_sheets["radiobutton"]["dark"] = """
                                              QRadioButton
                                              {
                                              background-color: #512da8; 
                                              color: 'white'; 
                                              border-color: 'white';
                                              font-family: Times New Roman;
                                              font-size: 15pt;
                                              }
                                              """

class GainLockWidgetStyle(WidgetStyle):
    def __init__(self):
        super().__init__()
    def __init__(self):
        super().__init__()
        self.style_sheets["main"]["dark"] = "background-color: #5d4037; color:white;"
        self.style_sheets["button"]["dark"] = """
                                              QPushButton
                                              {
                                              background-color: #5d4037; 
                                              color: 'white'; 
                                              border-color: 'white';
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              }
                                              """
        self.style_sheets["label"]["dark"] =  """
                                            QLabel
                                            {
                                            background-color : #5d4037; 
                                            color: 'white';
                                            border-color: 'white'; 
                                            font-family: Times New Roman ;
                                            font-size: 15pt;
                                            }
                                            """
        self.style_sheets["slider"]["dark"] = """
                                              QSlider
                                              {
                                              background-color: #5d4037;
                                              color: 'white';
                                              border-color: 'white'
                                              }
                                              """
        self.style_sheets["tabs"]["dark"] = """
                                            QTabWidget
                                            {
                                            background-color: #5d4037;
                                            color: 'white';
                                            border-color: 'white'
                                            font-family: Times New Roman;
                                            font-size: 12pt
                                            }
                                            """

class PIDControlWidgetStyle(WidgetStyle):
    def __init__(self):
        super().__init__()
        self.style_sheets["main"]["dark"] = "background-color: #37474; color:white;"
        self.style_sheets["button"]["dark"] = """
                                              QPushButton
                                              {
                                              background-color: #1A237E; 
                                              color: 'white'; 
                                              border-color: 'white';
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              }
                                              """
        self.style_sheets["label"]["dark"] =  """
                                            QLabel
                                            {
                                            background-color : #1A237E; 
                                            color: 'white';
                                            border-color: 'white'; 
                                            font-family: Times New Roman ;
                                            font-size: 15pt;
                                            }
                                            """
        self.style_sheets["slider"]["dark"] = """
                                              QSlider
                                              {
                                              background-color: #1A237E;
                                              color: 'white';
                                              border-color: 'white'
                                              }
                                              """
        self.style_sheets["doublespinbox"]["dark"] = """
                                                      QDoubleSpinbox
                                                      {
                                                      background-color: #1A237E;
                                                      color: 'white';
                                                      border-color: 'white'
                                                      }
                                                      """
        self.style_sheets["tabs"]["dark"] = """
                                            QTabWidget
                                            {
                                            background-color: #1A237E;
                                            color: 'white';
                                            border-color: 'white'
                                            font-family: Times New Roman;
                                            font-size: 12pt
                                            }
                                            """





