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

class LecroyOscilloscopeWidgetStyle(WidgetStyle):
    def __init__(self):
        super().__init__()
        background_color = "#E91E63"
        self.style_sheets["main"]["dark"] = "background-color: #E91E63; color:white;"
        self.style_sheets["button"]["dark"] = """
                                              QPushButton
                                              {
                                              background-color: #E91E63; 
                                              color: 'white'; 
                                              border-color: 'white';
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              }
                                              """
        self.style_sheets["label"]["dark"] =  """
                                            QLabel
                                            {
                                            background-color : #E91E63; 
                                            color: 'white';
                                            border-color: 'white'; 
                                            font-family: Times New Roman ;
                                            font-size: 15pt;
                                            }
                                            """
        self.style_sheets["slider"]["dark"] = """
                                              QSlider
                                              {
                                              background-color: #E91E63;
                                              color: 'white';
                                              border-color: 'white'
                                              }
                                              """
        self.style_sheets["tabs"]["dark"] = """
                                            QTabWidget
                                            {
                                            background-color: #E91E63;
                                            color: 'white';
                                            border-color: 'white'
                                            font-family: Times New Roman;
                                            font-size: 12pt
                                            }
                                            """
class SigilentSignalGeneratorWidgetStyle(WidgetStyle):
    def __init__(self):
        super().__init__()
        background_color = "#29b6f6"
        self.style_sheets["main"]["dark"] = "background-color: %s; color:white;"
        self.style_sheets["button"]["dark"] = """
                                              QPushButton
                                              {
                                              background-color: %s; 
                                              color: 'white'; 
                                              border-color: 'white';
                                              font-family: Times New Roman;
                                              font-size: 13pt;
                                              }
                                              """%(background_color)
        self.style_sheets["label"]["dark"] =  """
                                            QLabel
                                            {
                                            background-color : %s; 
                                            color: 'white';
                                            border-color: 'white'; 
                                            font-family: Times New Roman ;
                                            font-size: 15pt;
                                            }
                                            """%(background_color)
        self.style_sheets["slider"]["dark"] = """
                                              QSlider
                                              {
                                              background-color: %s;
                                              color: 'white';
                                              border-color: 'white'
                                              }
                                              """%(background_color)
        self.style_sheets["tabs"]["dark"] = """
                                            QTabWidget
                                            {
                                            background-color: %s;
                                            color: 'white';
                                            border-color: 'white'
                                            font-family: Times New Roman;
                                            font-size: 12pt
                                            }
                                            """%(background_color)
