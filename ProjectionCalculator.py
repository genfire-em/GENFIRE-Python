from PyQt4 import QtCore, QtGui
import os
import misc
import pyfftw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import ProjectionCalculator_ui
import scipy.io as io

class ProjectionCalculator(QtGui.QMainWindow): #QDialog?
    def __init__(self, parent=None):
        super(ProjectionCalculator, self).__init__()
        self.ui = ProjectionCalculator_ui.Ui_ProjectionCalculator()
        self.ui.setupUi(self)
        self.calculationParameters = ProjectionCalculationParameters()
        self.interpolator = None
        self.ui.lineEdit_outputFilename.setText(QtCore.QString(os.getcwd()))
        self.ui.lineEdit_modelFile.setText(QtCore.QString(os.getcwd()))
        self.ui.lineEdit_angleFile.setText(QtCore.QString(os.getcwd()))
        self.ui.lineEdit_modelFile.textEdited.connect(self.setModelFilename_fromLineEdit)
        self.ui.lineEdit_angleFile.textEdited.connect(self.setAngleFilename_fromLineEdit)

        ## Push Buttons
        self.ui.btn_selectAngleFile.clicked.connect(self.selectAngleFile)
        self.ui.btn_selectModel.clicked.connect(self.selectModelFile)
        self.ui.btn_outputDirectory.clicked.connect(self.selectOutputDirectory)
        self.ui.btn_refresh.clicked.connect(self.refreshModel)

        ## Sliders
        self.ui.verticalSlider_phi.setValue(0)
        self.ui.verticalSlider_phi.setMinimum(0)
        self.ui.verticalSlider_phi.setMaximum(3600)
        # self.ui.verticalSlider_phi.setSingleStep(1)
        self.ui.verticalSlider_phi.valueChanged.connect(self.setPhiLineEditValue)
        # self.ui.verticalSlider_phi.sliderReleased.connect(self.setPhiLineEditValue)

        self.ui.verticalSlider_theta.setValue(0)
        self.ui.verticalSlider_theta.setMinimum(0)
        self.ui.verticalSlider_theta.setMaximum(3600)
        # self.ui.verticalSlider_theta.setSingleStep(1)
        self.ui.verticalSlider_theta.valueChanged.connect(self.setThetaLineEditValue)

        self.ui.verticalSlider_psi.setValue(0)
        self.ui.verticalSlider_psi.setMinimum(0)
        self.ui.verticalSlider_psi.setMaximum(3600)
        # self.ui.verticalSlider_psi.setSingleStep(1)
        self.ui.verticalSlider_psi.valueChanged.connect(self.setPsiLineEditValue)

        self.ui.checkBox_displayFigure.setEnabled(False)
        # self.ui.checkBox_displayProjections.toggled.connect(self.toggleDisplayProjections)
        # self.ui.checkBox_saveProjections.toggled.connect(self.toggleSaveProjections)

        self.ui.lineEdit_outputFilename.textEdited.connect(self.setOutputFilename)
        self.ui.lineEdit_phi.setText(QtCore.QString('0'))
        self.ui.lineEdit_theta.setText(QtCore.QString('0'))
        self.ui.lineEdit_psi.setText(QtCore.QString('0'))
        self.ui.lineEdit_phi.editingFinished.connect(self.setPhiSliderValue)
        self.ui.lineEdit_theta.editingFinished.connect(self.setThetaSliderValue)
        self.ui.lineEdit_psi.editingFinished.connect(self.setPsiSliderValue)

        # self.ui.lineEdit_outputFilename.textEdited
        self.ui.lineEdit_outputFilename.setText(QtCore.QString(os.getcwd()))

        self.ui.btn_go.clicked.connect(self.calculateProjections)

        self.ui.checkBox_displayFigure.toggled.connect(self.toggleDisplayFigure)

        self.figure = plt.figure(1)
        self.canvas = FigureCanvas(self.figure)
        self.navigationToolbar = NavigationToolbar(self.canvas, self)
        self.ui.verticalLayout_figure.addWidget(self.navigationToolbar)
        self.ui.verticalLayout_figure.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)
        self.ax.hold(False)


    def calculateProjections(self):
        import scipy.io as io
        self.calculationParameters.modelFilename = unicode(self.calculationParameters.modelFilename.toUtf8(), encoding='UTF-8')
        if not self.calculationParameters.modelLoadedFlag:

            self.GENFIRE_load(self.calculationParameters.modelFilename)
            self.oversamplingRatio = 3
            self.dims = np.shape(self.model)
            paddedDim = self.dims[0] * self.oversamplingRatio
            padding = int((paddedDim-self.dims[0])/2)
            self.model = np.pad(self.model,((padding,padding),(padding,padding),(padding,padding)),'constant')
            self.model = pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(pyfftw.interfaces.numpy_fft.ifftshift((self.model))))
            self.interpolator = misc.getProjectionInterpolator(self.model)
        self.ncOut = np.shape(self.model)[0]//2


        if self.calculationParameters.angleFileProvided:
            self.calculationParameters.angleFilename = unicode(self.calculationParameters.angleFilename.toUtf8(), encoding='UTF-8')
            self.calculationParameters.outputBaseFilename = unicode(self.calculationParameters.outputBaseFilename.toUtf8(), encoding='UTF-8')
            angles = np.loadtxt(self.calculationParameters.angleFilename)
            phi = angles[:, 0]
            theta = angles[:, 1]
            psi = angles[:, 2]
            filename = self.calculationParameters.outputBaseFilename +'.npy'
            projections = np.zeros((self.dims[0],self.dims[1],np.size(phi)),dtype=float)
            if self.interpolator is None:
                self.interpolator = misc.getProjectionInterpolator(self.model)
            for projNum in range(0,np.size(phi)):
                # pj = misc.calculateProjection_interp(self.model, phi[projNum], theta[projNum], psi[projNum])[self.ncOut-self.dims[0]/2:self.ncOut+self.dims[0]/2, self.ncOut-self.dims[1]/2:self.ncOut+self.dims[1]/2]
                # projections[:, :, projNum] = misc.calculateProjection_interp(self.model, phi[projNum], theta[projNum], psi[projNum])[self.ncOut-self.dims[0]/2:self.ncOut+self.dims[0]/2, self.ncOut-self.dims[1]/2:self.ncOut+self.dims[1]/2]
                pj = misc.calculateProjection_interp_fromInterpolator(self.interpolator, self.calculationParameters.phi,self.calculationParameters.theta,self.calculationParameters.psi, np.shape(self.model))
                # projections[:, :, projNum] = pj
                projections[:, :, projNum] = pj[self.ncOut-self.dims[0]/2:self.ncOut+self.dims[0]/2, self.ncOut-self.dims[1]/2:self.ncOut+self.dims[1]/2]

            np.save(filename,projections)
        else:
            # pj = misc.calculateProjection_interp(self.model, self.calculationParameters.phi,self.calculationParameters.theta,self.calculationParameters.psi)[self.ncOut-self.dims[0]/2:self.ncOut+self.dims[0]/2, self.ncOut-self.dims[1]/2:self.ncOut+self.dims[1]/2]
            if self.interpolator is None:
                self.interpolator = misc.getProjectionInterpolator(self.model)
            pj = misc.calculateProjection_interp_fromInterpolator(self.interpolator, self.calculationParameters.phi,self.calculationParameters.theta,self.calculationParameters.psi, np.shape(self.model))
            pj = pj[self.ncOut-self.dims[0]/2:self.ncOut+self.dims[0]/2, self.ncOut-self.dims[1]/2:self.ncOut+self.dims[1]/2]
            filename = self.calculationParameters.outputBaseFilename +'TEMPprojectionName.mat'
            outDict = {'pj': pj}
            io.savemat(filename,outDict)
            self.showProjection(pj)


    def refreshModel(self):
        self.ui.checkBox_displayFigure.click()
        self.ui.checkBox_displayFigure.click()

    def toggleDisplayFigure(self):
        if not self.calculationParameters.displayProjectionsFlag:
           self.calculationParameters.displayProjectionsFlag = True
           if self.calculationParameters.modelFilenameProvided:
                import scipy.io as io
                modelFilename = unicode(self.calculationParameters.modelFilename.toUtf8(), encoding='UTF-8')
                self.GENFIRE_load(modelFilename)
                self.oversamplingRatio = 3
                print np.shape(self.model)
                self.dims = np.shape(self.model)
                self.paddedDim = self.dims[0] * self.oversamplingRatio
                padding = int((self.paddedDim-self.dims[0])/2)
                self.model = np.pad(self.model,((padding,padding),(padding,padding),(padding,padding)),'constant')
                self.model = pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(pyfftw.interfaces.numpy_fft.ifftshift((self.model))))
                # global ncOut
                self.ncOut = np.shape(self.model)[0]//2
                if self.interpolator is None:
                    self.interpolator = misc.getProjectionInterpolator(self.model)
                # pj = misc.calculateProjection_interp(self.model, self.calculationParameters.phi,self.calculationParameters.theta,self.calculationParameters.psi)[self.ncOut-self.dims[0]/2:self.ncOut+self.dims[0]/2, self.ncOut-self.dims[1]/2:self.ncOut+self.dims[1]/2]
                pj = misc.calculateProjection_interp_fromInterpolator(self.interpolator, self.calculationParameters.phi,self.calculationParameters.theta,self.calculationParameters.psi, np.shape(self.model))
                pj = pj[self.ncOut-self.dims[0]/2:self.ncOut+self.dims[0]/2, self.ncOut-self.dims[1]/2:self.ncOut+self.dims[1]/2]
                self.showProjection(pj)
                self.calculationParameters.modelLoadedFlag = True
        else:
            self.calculationParameters.displayProjectionsFlag = False
            self.clearFigure()


    def updateFigure(self):
        # pj = misc.calculateProjection_interp(self.model, self.calculationParameters.phi,self.calculationParameters.theta,self.calculationParameters.psi)[self.ncOut-self.dims[0]/2:self.ncOut+self.dims[0]/2, self.ncOut-self.dims[1]/2:self.ncOut+self.dims[1]/2]
        if self.interpolator is None:
            self.interpolator = misc.getProjectionInterpolator(self.model)
        pj = misc.calculateProjection_interp_fromInterpolator(self.interpolator, self.calculationParameters.phi,self.calculationParameters.theta,self.calculationParameters.psi, np.shape(self.model))
        pj = pj[self.ncOut-self.dims[0]/2:self.ncOut+self.dims[0]/2, self.ncOut-self.dims[1]/2:self.ncOut+self.dims[1]/2]
        self.ax.imshow(pj)
        self.canvas.draw()


    def showProjection(self, pj):
        self.ax.imshow(pj)
        self.canvas.draw()

    def clearFigure(self):
        self.figure.clf()
        self.canvas.draw()

    def setPhiSliderValue(self):
        value = self.ui.lineEdit_phi.text().toFloat()[0]
        value = int(value * 10)
        self.ui.verticalSlider_phi.setValue(value)

    def setThetaSliderValue(self):
        value = self.ui.lineEdit_theta.text().toFloat()[0]
        value = int(value * 10)
        self.ui.verticalSlider_theta.setValue(value)

    def setPsiSliderValue(self):
        value = self.ui.lineEdit_psi.text().toFloat()[0]
        value = int(value * 10)
        self.ui.verticalSlider_psi.setValue(value)


    def setPhiLineEditValue(self, value):
        self.ui.lineEdit_phi.setText(QtCore.QString.number(float(value)/10))
        self.calculationParameters.phi = float(value)/10
        if self.calculationParameters.displayProjectionsFlag:
            self.updateFigure()

    def setThetaLineEditValue(self, value):
        self.ui.lineEdit_theta.setText(QtCore.QString.number(float(value)/10))
        self.calculationParameters.theta = float(value)/10
        if self.calculationParameters.displayProjectionsFlag:
            self.updateFigure()

    def setPsiLineEditValue(self, value):
        self.ui.lineEdit_psi.setText(QtCore.QString.number(float(value)/10))
        self.calculationParameters.psi = float(value)/10
        if self.calculationParameters.displayProjectionsFlag:
            self.updateFigure()

    def setNumberOfProjections(self, number):
        self.calculationParameters.numberOfProjections = number.toInt()[0]

    def setPhiStart(self, phiStart):
        self.calculationParameters.phiStart = phiStart.toFloat()[0]

    def setThetaStart(self, thetaStart):
        self.calculationParameters.thetaStart = thetaStart.toFloat()[0]

    def setPsiStart(self, psiStart):
        self.calculationParameters.psiStart = psiStart.toFloat()[0]

    def setPhiStep(self, phiStep):
        self.calculationParameters.phiStep = phiStep.toFloat()[0]

    def setThetaStep(self, thetaStep):
        self.calculationParameters.thetaStep = thetaStep.toFloat()[0]

    def setPsiStep(self, psiStep):
        self.calculationParameters.psiStep = psiStep.toFloat()[0]

    def selectOutputDirectory(self):
        dirname = QtGui.QFileDialog.getExistingDirectory(QtGui.QFileDialog(), "Select File Containing Angles",options=QtGui.QFileDialog.ShowDirsOnly)
        if dirname:
            # self.calculationParameters.outputBaseFilename = dirname
            # self.calculationParameters.outputBaseFilename =  unicode(dirname.toUtf8(), encoding='UTF-8')
            self.calculationParameters.outputBaseFilename =  dirname
            self.ui.lineEdit_outputFilename.setText(QtCore.QString(dirname))

    def selectAngleFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Angles",filter="txt files (*.txt);;All Files (*)")
        if filename:
            self.calculationParameters.angleFilename = filename
            self.calculationParameters.angleFileProvided = True
            self.ui.lineEdit_angleFile.setText(QtCore.QString(filename))
            # print "angle File, ", self.calculationParameters.angleFilename

    def setAngleFilename_fromLineEdit(self):
        filename = self.ui.lineEdit_angleFile.text()
        if os.path.isfile(unicode(filename.toUtf8(),encoding='UTF-8')):
            self.calculationParameters.angleFilename = filename
            self.calculationParameters.angleFileProvided = True
            self.ui.lineEdit_angleFile.setText(QtCore.QString(filename))

    def setModelFilename_fromLineEdit(self):
        filename = self.ui.lineEdit_modelFile.text()
        # print unicode(filename.toUtf8(),encoding='UTF-8')
        # print os.path.isfile(unicode(filename.toUtf8(),encoding='UTF-8'))
        if os.path.isfile(unicode(filename.toUtf8(),encoding='UTF-8')):
            # self.modelLoadedFlag = False
            self.GENFIRE_load(unicode(filename.toUtf8(),encoding='UTF-8'))
            # print "going"
            # self.setModelFilename(filename)
            self.calculationParameters.modelFilename = filename
            # self.ui.lineEdit_modelFile.setText(QtCore.QString(filename))
            self.calculationParameters.modelFilenameProvided = True
            self.ui.checkBox_displayFigure.setEnabled(True)

    def setModelFilename(self, filename):
        print "test"
        self.calculationParameters.modelFilename = filename
        self.ui.lineEdit_modelFile.setText(QtCore.QString(filename))
        self.calculationParameters.modelFilenameProvided = True
        self.ui.checkBox_displayFigure.setEnabled(True)

    def selectModelFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Model",filter="MATLAB files (*.mat);;TIFF images (*.tif *.tiff);;MRC (*.mrc);;All Files (*)")
        print "myfilename",  filename

        if os.path.isfile(unicode(filename.toUtf8(),encoding='UTF-8')):
            # print "it is a file"
            self.setModelFilename(filename)
            # self.GENFIRE_load(filename)
            self.GENFIRE_load(unicode(filename.toUtf8(),encoding='UTF-8'))
            # self.modelLoadedFlag = False
            # self.model = self.loadModel(filename)
            # self.calculationParameters.modelFilename = filename
            # self.ui.lineEdit_modelFile.setText(QtCore.QString(filename))
            # self.calculationParameters.modelFilenameProvided = True
            # self.ui.checkBox_displayFigure.setEnabled(True)

    #
    # def setModelFilename(self, filename):
    #     if filename:
    #         self.calculationParameters.modelFilename = filename
    #         self.calculationParameters.modelNameProvided = True

    def setAngleFilename(self, filename):
        if filename:
            self.calculationParameters.angleFilename = filename

    def toggleDisplayProjections(self):
        if self.calculationParameters.displayProjectionsFlag:
            self.calculationParameters.displayProjectionsFlag = False
        else:
            self.calculationParameters.displayProjectionsFlag = True

    def toggleSaveProjections(self):
        print GENFIRE_ProjectionCalculator.calculationParameters.modelFilename
        print GENFIRE_ProjectionCalculator.calculationParameters.angleFilename
        print GENFIRE_ProjectionCalculator.calculationParameters.outputBaseFilename
        print GENFIRE_ProjectionCalculator.calculationParameters.outputFilesFlag
        print  GENFIRE_ProjectionCalculator.calculationParameters.displayProjectionsFlag
        print GENFIRE_ProjectionCalculator.calculationParameters.phiStart
        print GENFIRE_ProjectionCalculator.calculationParameters.thetaStart
        print  GENFIRE_ProjectionCalculator.calculationParameters.psiStart
        print GENFIRE_ProjectionCalculator.calculationParameters.phiStep
        print GENFIRE_ProjectionCalculator.calculationParameters.thetaStep
        print GENFIRE_ProjectionCalculator.calculationParameters.psiStep
        print GENFIRE_ProjectionCalculator.calculationParameters.numberOfProjections


        if self.calculationParameters.outputFilesFlag:
            self.calculationParameters.outputFilesFlag = False
        else:
            self.calculationParameters.outputFilesFlag = True

    def setOutputFilename(self, filename):
        if filename:
            self.calculationParameters.outputBaseFilename = filename


    def GENFIRE_load(self, filename):
        print "GENFIRE load file", filename
        print type(filename)
        self.model = io.loadmat(filename)
        self.model = self.model['model']##CHANGE
        print np.shape(self.model)



class ProjectionCalculationParameters:
    def __init__(self):
        self.modelFilename = QtCore.QString('')
        self.angleFilename = QtCore.QString('')
        self.outputBaseFilename = QtCore.QString('')
        self.outputFilesFlag = False
        self.displayProjectionsFlag = False
        self.angleFileProvided = False
        self.modelFilenameProvided = False
        self.modelLoadedFlag = False
        self.phi = 0
        self.theta = 0
        self.psi = 0

if __name__ == "__main__":
    import sys

    # Startup the application
    app = QtGui.QApplication(sys.argv)
    # app.setStyle('plastique')
    app.setStyle('mac')

    # Create the GUI
    GENFIRE_ProjectionCalculator = ProjectionCalculator()

    # Render the GUI
    GENFIRE_ProjectionCalculator.show()

    sys.exit(app.exec_())

