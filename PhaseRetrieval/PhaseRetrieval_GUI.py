from PyQt4 import QtCore, QtGui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import PhaseRetrieval_GUI_ui
from PhaseRetrieval import *
import numpy as np
import os
import sys

class PhaseRetrieval_GUI(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(QtGui.QMainWindow,self).__init__()
        self.ui = PhaseRetrieval_GUI_ui.Ui_ProjectionCalculator()
        self.ui.setupUi(self)
        self.ReconstructionParameters = ReconstructionParameters()

        self.ui.lineEdit_diffpatFile.textEdited.connect(self.ReconstructionParameters.setDiffPatFile)

        self.ui.btn_selectDiffpat
        self.figure = plt.figure(1)
        self.canvas = FigureCanvas(self.figure)
        self.navigationToolbar = NavigationToolbar(self.canvas, self)
        self.ui.verticalLayout_figure.addWidget(self.navigationToolbar)
        self.ui.verticalLayout_figure.addWidget(self.canvas)

        # self.diffraction_pattern = np.load('diffraction_pattern.npy')
        # myFig = self.figure.add_subplot(111)
        # myFig.imshow(np.log(np.abs(self.diffraction_pattern)))
        # self.canvas.draw()

    def selectDiffPatFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Diffraction Pattern",filter="MATLAB files (*.mat);;TIFF images (*.tif *.tiff);;MRC (*.mrc);;All Files (*)")
        if filename:
            self.ReconstructionParameters.setDiffPatFile(filename)
            print ("Diffraction Pattern Filename:", self.ReconstructionParameters._diffpatFile)
            self.ui.lineEdit_BGfilename.setText(QtCore.QString(filename))


class ReconstructionParameters(object):
    def __init__(self):
        self._diffpatFile = ""
        self._bgFile = ""
        self._bgScale = ""

    def setDiffPatFile(self, filename):
        if filename:
            self._diffpatFile = os.path.join(os.getcwd(),unicode(filename.toUtf8(), encoding='UTF-8'))
            # print ("filename set to {}".format(self._diffpatFile))

    def setBGFile(self, filename):
        if filename:
            self._bgFile = os.path.join(os.getcwd(),unicode(filename.toUtf8(), encoding='UTF-8'))



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