from __future__ import division
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

SLIDER_SCALE = 100
class PhaseRetrieval_GUI(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(QtGui.QMainWindow,self).__init__()
        self.ui = PhaseRetrieval_GUI_ui.Ui_ProjectionCalculator()
        self.ui.setupUi(self)
        self.resize(1200,600)

        self.parameters = ReconstructionParameters()

        self.ui.lineEdit_diffpatFile.textEdited.connect(self.parameters.setDiffPatFile)
        self.ui.lineEdit_BGfilename.textEdited.connect(self.parameters.setBGFile)
        self.ui.lineEdit_diffpatFile.textChanged.connect(self.loadDiffractionPattern)
        self.ui.lineEdit_contrastHigh.textChanged.connect(self.setHighContrast_Qstring)
        self.ui.lineEdit_contrastLow.textChanged.connect(self.setLowContrast_Qstring)
        self.ui.lineEdit_BGscale.textChanged.connect(self.setBGscale_Qstring)
        self.ui.lineEdit_satThresh.textChanged.connect(self.setSatThresh_Qstring)

        # self.ui.slder_contrastHigh.valueChanged.connect(self.setHighContrast_slider)
        # self.ui.slder_contrastLow.valueChanged.connect(self.setLowContrast_slider)
        self.ui.slder_contrastHigh.valueChanged.connect(self.setLineEditText_contrastHigh_fromSlider)
        self.ui.slder_contrastLow.valueChanged.connect(self.setLineEditText_contrastLow_fromSlider)
        self.ui.slder_BGscale.valueChanged.connect(self.setLineEditText_BGscale_fromSlider)
        self.ui.slder_satThresh.valueChanged.connect(self.setLineEditText_satThresh_fromSlider)

        self.setupSliders()

        self.ui.btn_selectDiffpat.clicked.connect(self.selectDiffPatFile)
        self.ui.btn_BGfilename.clicked.connect(self.selectBGFile)


        self.figure = plt.figure(1)
        plt.imshow(np.random.rand(50,50))
        self.handle = self.figure.add_subplot(111)
        self.handle.hold(False)
        self.current_image_handle = plt.gci()
        self.canvas = FigureCanvas(self.figure)
        self.navigationToolbar = NavigationToolbar(self.canvas, self)
        self.ui.verticalLayout_figure.addWidget(self.navigationToolbar)
        self.ui.verticalLayout_figure.addWidget(self.canvas)

        self._lowContrastSetting = 0
        self._highContrastSetting = 1
        self._saturated_threshold = 1e30 #default to large

    def setLineEditText_contrastHigh_fromSlider(self, value):
        self.ui.lineEdit_contrastHigh.setText(str(value / SLIDER_SCALE))

    def setLineEditText_contrastLow_fromSlider(self, value):
        self.ui.lineEdit_contrastLow.setText(str(value / SLIDER_SCALE))

    def setLineEditText_satThresh_fromSlider(self, value):
        self.ui.lineEdit_satThresh.setText(str(value))

    def setLineEditText_BGscale_fromSlider(self, value):
        self.ui.lineEdit_BGscale.setText(str(value))


    def setupSliders(self):
        if self.parameters._DiffractionPattern is None:
            minVal = minVal_log = 0
            maxVal = maxVal_log = 1

        else:
            # minVal = np.max((0,np.amin(np.log(np.abs(self.parameters._DiffractionPattern.data)))))
            # maxVal = np.min((self._saturated_threshold,np.amax(np.log(np.abs(self.parameters._DiffractionPattern.data)))))
            minVal = np.max((0,np.amin(np.abs(self.parameters._DiffractionPattern.data))))
            maxVal = np.min((self._saturated_threshold,np.amax(np.abs(self.parameters._DiffractionPattern.data))))
            minVal_log = np.log(minVal)
            maxVal_log = np.log(maxVal)
        self.ui.slder_contrastHigh.setValue(maxVal_log * SLIDER_SCALE)
        self.ui.slder_contrastHigh.setMinimum(minVal_log * SLIDER_SCALE)
        self.ui.slder_contrastHigh.setMaximum(maxVal_log * SLIDER_SCALE)

        self.ui.slder_contrastLow.setValue(minVal_log * SLIDER_SCALE)
        self.ui.slder_contrastLow.setMinimum(minVal_log * SLIDER_SCALE)
        self.ui.slder_contrastLow.setMaximum(maxVal_log * SLIDER_SCALE)

        self.ui.slder_satThresh.setValue(minVal)
        self.ui.slder_satThresh.setMinimum(minVal)
        self.ui.slder_satThresh.setMaximum(maxVal)

        self.ui.slder_BGscale.setValue(100)
        self.ui.slder_BGscale.setMinimum(0)
        self.ui.slder_BGscale.setMaximum(1000)

    def updateDisplay(self,image=None): ##need to fix
        self.handle.imshow(np.log(self.parameters._DiffractionPattern.data),clim=[self._lowContrastSetting, self._highContrastSetting])
        self.canvas.draw()

    def setHighContrast(self, value):
        try:
            self._highContrastSetting = max(value, self._lowContrastSetting)
            self.updateDisplay()
        except AttributeError:
            pass

    def setLowContrast(self, value):
        try:
            self._lowContrastSetting = min(value, self._highContrastSetting)
            self.updateDisplay()
        except AttributeError:
            pass

    def setHighContrast_Qstring(self, value):
        self.setHighContrast(value.toFloat()[0])

    def setLowContrast_Qstring(self, value):
        print
        self.setLowContrast(value.toFloat()[0])

    def setBGscale_Qstring(self, value):
        self.setBGscale(value.toFloat()[0])

    def setSatThresh_Qstring(self, value):
        self.setSatThresh(value.toFloat()[0])

    def setHighContrast_slider(self, value):
        self.setHighContrast(value / SLIDER_SCALE)
        print ("setHighContrast_slider value = " , value / SLIDER_SCALE)

    def setLowContrast_slider(self, value):
        self.setLowContrast(value / SLIDER_SCALE)
        print ("setLowContrast_slider value = " , value / SLIDER_SCALE)

    def setBGscale(self, value):
        self.parameters._bgScale = value
        self.subtractBG()
        # self.applyBGandThreshold()

    def setSatThresh(self, value):
        self.parameters._satThresh = value
        self.thresholdSaturated()
        # self.applyBGandThreshold()

    def selectDiffPatFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Diffraction Pattern",filter="MATLAB files (*.mat);;TIFF images (*.tif* *.tiff);;MRC (*.mrc);;All Files (*)")
        if filename:
            self.parameters.setDiffPatFile(filename)
            print ("Diffraction Pattern Filename:", self.parameters._diffpatFile)
            self.ui.lineEdit_diffpatFile.setText(QtCore.QString(filename))
            self.loadDiffractionPattern(self.parameters._diffpatFile)

    def selectBGFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Background Pattern",filter="MATLAB files (*.mat);;TIFF images (*.tif *.tiff);;MRC (*.mrc);;All Files (*)")
        if filename:
            self.parameters.setBGFile(filename)
            print ("BG Filename:", self.parameters._bgFile)
            self.ui.lineEdit_BGfilename.setText(QtCore.QString(filename))

    def loadDiffractionPattern(self, str=None):
        self.parameters._DiffractionPattern = DiffractionPattern(np.abs(loadImage(self.parameters._diffpatFile)))
        self._lowContrastSetting = np.min(np.log(self.parameters._DiffractionPattern.data))
        self._highContrastSetting = np.max(np.log(self.parameters._DiffractionPattern.data))
        self.updateDisplay(self.parameters._DiffractionPattern)
        self.parameters._originalDiffractionPattern = np.copy(self.parameters._DiffractionPattern.data)
        if self.parameters._BGImage is None: #use array of ones if no BG has been provided
            self.parameters._BGImage = np.ones_like(self.parameters._DiffractionPattern.data) * np.mean(self.parameters._DiffractionPattern.data)
        self.setupSliders()

    def loadBGImage(self):
        self.parameters._BGImage = loadImage(self.parameters._bgFile)
    
    def subtractBG(self):
        print ("BG level {}".format(self.parameters._bgScale))
        if self.parameters._BGImage is not None:
            print "TEST"
            self.parameters._DiffractionPattern.data = self.parameters._originalDiffractionPattern - \
                self.parameters._BGImage * self.parameters._bgScale / 100 #subtract background (divide by 100 because it is a percentage)
            self.parameters._DiffractionPattern.data[self.parameters._DiffractionPattern.data < 0 ] = 0
            self.updateDisplay()

    def thresholdSaturated(self):
        if self.parameters._DiffractionPattern is not None:
            self.parameters._DiffractionPattern.data = np.copy(self.parameters._originalDiffractionPattern)
            self.parameters._DiffractionPattern.data[self.parameters._DiffractionPattern.data > \
                self.parameters._satThresh] = -1
            # self.parameters._DiffractionPattern.data  -= \
            #     self.parameters._BGImage * self.parameters._bgScale / 100 #subtract background (divide by 100 because it is a percentage)
            # self.parameters._DiffractionPattern.data[self.parameters._DiffractionPattern.data < 0 ] = 0
            self.updateDisplay()
    #
    # def applyBGandThreshold(self):
    #     if self.parameters._DiffractionPattern is not None and self.parameters._BGImage is not None:
    #         self.parameters._DiffractionPattern.data = np.copy(self.parameters._originalDiffractionPattern)
    #         self.parameters._DiffractionPattern.data[self.parameters._DiffractionPattern.data > \
    #             self.parameters._satThresh] = -1
    #         self.parameters._DiffractionPattern.data = self.parameters._DiffractionPattern.data - \
    #                                                    (self.parameters._BGImage * self.parameters._bgScale / 100) #subtract background (divide by 100 because it is a percentage)
    #         self.parameters._DiffractionPattern.data[self.parameters._DiffractionPattern.data < 0 ] = 0
    #         self.updateDisplay()

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
        self._originalDiffractionPattern = None #saved copy to avoid problem of losing thresholded data

        self._bgFile = ""
        self._BGImage = None
        self._bgScale = 0
        self._satThresh = 1e30

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