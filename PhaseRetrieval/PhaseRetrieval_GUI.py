from PyQt4 import QtCore, QtGui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import PhaseRetrieval_GUI_ui
from PhaseRetrieval import *
import numpy as np
import os
import sys
import GENFIRE_io


class PhaseRetrieval_GUI(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(QtGui.QMainWindow,self).__init__()
        self.ui = PhaseRetrieval_GUI_ui.Ui_ProjectionCalculator()
        self.ui.setupUi(self)
        self.CurrentReconstructionParameters = ReconstructionParameters()

        self.ui.lineEdit_diffpatFile.textEdited.connect(self.CurrentReconstructionParameters.setDiffPatFile)
        self.ui.lineEdit_BGfilename.textEdited.connect(self.CurrentReconstructionParameters.setBGFile)
        self.ui.lineEdit_diffpatFile.textChanged.connect(self.loadDiffractionPattern)

        self.ui.btn_selectDiffpat.clicked.connect(self.selectDiffPatFile)
        self.ui.btn_BGfilename.clicked.connect(self.selectBGFile)


        self.figure = plt.figure(1)
        self.canvas = FigureCanvas(self.figure)
        self.navigationToolbar = NavigationToolbar(self.canvas, self)
        self.ui.verticalLayout_figure.addWidget(self.navigationToolbar)
        self.ui.verticalLayout_figure.addWidget(self.canvas)


    def updateDisplay(self,image):
        myFig = self.figure.add_subplot(111)
        myFig.imshow(np.log(np.abs(self.CurrentReconstructionParameters._DiffractionPattern.data)))
        # myFig.imshow(np.log(np.abs(np.load('diffraction_pattern.npy'))))
        self.canvas.draw()

    def selectDiffPatFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Diffraction Pattern",filter="MATLAB files (*.mat);;TIFF images (*.tif* *.tiff);;MRC (*.mrc);;All Files (*)")
        if filename:
            self.CurrentReconstructionParameters.setDiffPatFile(filename)
            print ("Diffraction Pattern Filename:", self.CurrentReconstructionParameters._diffpatFile)
            self.ui.lineEdit_diffpatFile.setText(QtCore.QString(filename))
            self.loadDiffractionPattern(self.CurrentReconstructionParameters._diffpatFile)

    def selectBGFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Background Pattern",filter="MATLAB files (*.mat);;TIFF images (*.tif *.tiff);;MRC (*.mrc);;All Files (*)")
        if filename:
            self.CurrentReconstructionParameters.setBGFile(filename)
            print ("BG Filename:", self.CurrentReconstructionParameters._bgFile)
            self.ui.lineEdit_BGfilename.setText(QtCore.QString(filename))

    def loadDiffractionPattern(self, str=None):
        self.CurrentReconstructionParameters._DiffractionPattern = DiffractionPattern(loadImage(self.CurrentReconstructionParameters._diffpatFile))
        self.updateDisplay(self.CurrentReconstructionParameters._DiffractionPattern)

    def loadBGImage(self):
        self.CurrentReconstructionParameters._BGImage = loadImage(self.CurrentReconstructionParameters._bgFile)

def loadImage(filename):
        """
        * loadImage *

        Wrapper function for loading in loadImage of arbitrary (supported) extension

        Author: Alan (AJ) Pryor, Jr.
        Jianwei (John) Miao Coherent Imaging Group
        University of California, Los Angeles
        Copyright 2015-2016. All rights reserved.

        :param filename: Filename of images to load
        :return: NumPy array containing projections
        """

        filename, file_extension = os.path.splitext(filename)
        if file_extension == ".mat":
            print ("GENFIRE: reading projections from MATLAB file.\n")
            return GENFIRE_io.readMAT(filename + file_extension)
        elif (file_extension == ".tif") | (file_extension == ".tiff") | (file_extension == ".TIF") | (file_extension == ".TIFF"):
            print ("GENFIRE: reading projections from .TIFF file.\n")
            return GENFIRE_io.readTIFF(filename + file_extension)
        elif file_extension == ".mrc":
            print ("GENFIRE: reading projections from .mrc file.\n")
            return GENFIRE_io.readMRC(filename + file_extension)
        else:
            raise Exception('GENFIRE: File format %s not supported.', file_extension)

class ReconstructionParameters(object):
    def __init__(self):
        self._diffpatFile = ""
        self._DiffractionPattern = None

        self._bgFile = ""
        self.__BGImage = None
        self._bgScale = ""

    def setDiffPatFile(self, filename):
        if filename:
            self._diffpatFile = os.path.join(os.getcwd(),unicode(filename.toUtf8(), encoding='UTF-8'))

    def setBGFile(self, filename):
        if filename:
            self._bgFile = os.path.join(os.getcwd(),unicode(filename.toUtf8(), encoding='UTF-8'))
            print ("filename set to {}".format(self._bgFile))

    def setBGscale(self, value):
        if value:
            self._bgScale = value


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