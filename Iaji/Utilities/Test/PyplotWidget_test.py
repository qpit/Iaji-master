"""
This script tests the PyplotWidget module
"""
# In[imports]
from Iaji.Utilities.GUI import PyplotWidget
from PyQt5.QtWidgets import QApplication
import sys
from matplotlib import pyplot
import numpy
# In[]
app = QApplication(sys.argv)
figure = pyplot.figure(figsize=(9, 8))
widget = PyplotWidget()
axis = figure.add_subplot(1, 1, 1)
widget.show()
axis.set_xlabel("ciao", fontdict={"size": 20, "family":"Times New Roman"})
x = numpy.linspace(-1, 1, 100)
y = numpy.arccos(x)
axis.plot(x, y)
app.exec()



