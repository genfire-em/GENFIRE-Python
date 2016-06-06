from PyQt4 import QtCore, QtGui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import PhaseRetrieval_GUI_ui
import numpy as np

class PhaseRetrieval_GUI(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(QtGui.QMainWindow,self).__init__()
        self.ui = PhaseRetrieval_GUI_ui.Ui_ProjectionCalculator()
        self.ui.setupUi(self)
        self.figure = plt.figure(1)
        self.canvas = FigureCanvas(self.figure)
        self.navigationToolbar = NavigationToolbar(self.canvas, self)
        self.ui.verticalLayout_figure.addWidget(self.navigationToolbar)
        self.ui.verticalLayout_figure.addWidget(self.canvas)

        self.diffraction_pattern = np.load('diffraction_pattern.npy')
        myFig = self.figure.add_subplot(111)
        myFig.imshow(np.log(np.abs(self.diffraction_pattern)))
        self.canvas.draw()
if __name__ == "__main__":
    import sys

    # Startup the application
    app = QtGui.QApplication(sys.argv)
    # app.setStyle('plastique')
    app.setStyle('mac')

    # Create the GUI
    PhaseRetrieval_GUI = PhaseRetrieval_GUI()

    # Render the GUI
    PhaseRetrieval_GUI.show()

    sys.exit(app.exec_())