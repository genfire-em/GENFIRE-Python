from PyQt4 import QtCore, QtGui
import os
import GENFIRE
import pyfftw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import ProjectionCalculator_MainWindow
import CalculateProjectionSeries_Dialog

def toString(string):
    if isinstance(string,QtCore.QString):
        string = unicode(string.toUtf8(),encoding='UTF-8')
    return string

class ProjectionCalculator(QtGui.QMainWindow): #QDialog?
    def __init__(self, parent=None):
        super(ProjectionCalculator, self).__init__()
        self.ui = ProjectionCalculator_MainWindow.Ui_ProjectionCalculator()
        self.ui.setupUi(self)
        self.calculationParameters = ProjectionCalculationParameters()
        self.interpolator = None
        # self.ui.lineEdit_outputFilename.setText(QtCore.QString(os.getcwd()))
        self.ui.lineEdit_modelFile.setText(QtCore.QString(os.getcwd()))
        # self.ui.lineEdit_angleFile.setText(QtCore.QString(os.getcwd()))
        self.ui.lineEdit_modelFile.textEdited.connect(self.setModelFilename_fromLineEdit)
        # self.ui.lineEdit_angleFile.textEdited.connect(self.setAngleFilename_fromLineEdit)

        ## Push Buttons
        #
        self.ui.btn_selectModel.clicked.connect(self.selectModelFile)
        # self.ui.btn_outputDirectory.clicked.connect(self.selectOutputDirectory)
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

        # self.ui.lineEdit_outputFilename.textEdited.connect(self.setOutputFilename)
        self.ui.lineEdit_phi.setText(QtCore.QString('0'))
        self.ui.lineEdit_theta.setText(QtCore.QString('0'))
        self.ui.lineEdit_psi.setText(QtCore.QString('0'))
        self.ui.lineEdit_phi.editingFinished.connect(self.setPhiSliderValue)
        self.ui.lineEdit_theta.editingFinished.connect(self.setThetaSliderValue)
        self.ui.lineEdit_psi.editingFinished.connect(self.setPsiSliderValue)

        # self.ui.lineEdit_outputFilename.textEdited
        # self.ui.lineEdit_outputFilename.setText(QtCore.QString(os.getcwd()))

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
        self.calculateProjections_Dialog = CalculateProjectionSeries_popup(self.calculationParameters)
        result = self.calculateProjections_Dialog.exec_()
        if result == QtGui.QDialog.Accepted:
            import scipy.io as io
            # self.calculateProjections_Dialog.calculationParameters.modelFilename = unicode(self.calculateProjections_Dialog.calculationParameters.modelFilename.toUtf8(), encoding='UTF-8')
            self.calculateProjections_Dialog.calculationParameters.modelFilename = toString(self.calculateProjections_Dialog.calculationParameters.modelFilename)
            if not self.calculationParameters.modelLoadedFlag:

                self.GENFIRE_load(self.calculateProjections_Dialog.calculationParameters.modelFilename)
                self.oversamplingRatio = self.calculationParameters.oversamplingRatio
                self.dims = np.shape(self.model)
                paddedDim = self.dims[0] * self.oversamplingRatio
                padding = int((paddedDim-self.dims[0])/2)
                self.model = np.pad(self.model,((padding,padding),(padding,padding),(padding,padding)),'constant')
                self.model = pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(pyfftw.interfaces.numpy_fft.ifftshift((self.model))))
                self.interpolator = GENFIRE.misc.getProjectionInterpolator(self.model)
            self.ncOut = np.shape(self.model)[0]//2


            if self.calculateProjections_Dialog.calculationParameters.angleFileProvided:
                # self.calculateProjections_Dialog.calculationParameters.angleFilename = unicode(self.calculateProjections_Dialog.calculationParameters.angleFilename.toUtf8(), encoding='UTF-8')
                # self.calculateProjections_Dialog.calculationParameters.outputFilename = unicode(self.calculateProjections_Dialog.calculationParameters.outputFilename.toUtf8(), encoding='UTF-8')
                self.calculateProjections_Dialog.calculationParameters.angleFilename = toString(self.calculateProjections_Dialog.calculationParameters.angleFilename)
                self.calculateProjections_Dialog.calculationParameters.outputFilename = toString(self.calculateProjections_Dialog.calculationParameters.outputFilename)
                angles = np.loadtxt(self.calculateProjections_Dialog.calculationParameters.angleFilename)
                phi = angles[:, 0]
                theta = angles[:, 1]
                psi = angles[:, 2]
                filename = self.calculateProjections_Dialog.calculationParameters.outputFilename +'.npy'
                projections = np.zeros((self.dims[0],self.dims[1],np.size(phi)),dtype=float)
                if self.interpolator is None:
                    self.interpolator = GENFIRE.misc.getProjectionInterpolator(self.model)
                for projNum in range(0,np.size(phi)):
                    pj = GENFIRE.misc.calculateProjection_interp_fromInterpolator(self.interpolator, phi[projNum], theta[projNum], psi[projNum], np.shape(self.model))
                    projections[:, :, projNum] = pj[self.ncOut-self.dims[0]/2:self.ncOut+self.dims[0]/2, self.ncOut-self.dims[1]/2:self.ncOut+self.dims[1]/2]

                np.save(filename,projections)
            else:
                if self.interpolator is None:
                    self.interpolator = GENFIRE.misc.getProjectionInterpolator(self.model)
                phi = self.calculateProjections_Dialog.calculationParameters.phi
                psi = self.calculateProjections_Dialog.calculationParameters.psi
                theta = np.arange(self.calculateProjections_Dialog.calculationParameters.thetaStart, \
                                  self.calculateProjections_Dialog.calculationParameters.thetaStop, \
                                  self.calculateProjections_Dialog.calculationParameters.thetaStep)
                projections = np.zeros((self.dims[0],self.dims[1],np.size(theta)),dtype=float)
                for i, current_theta in enumerate(theta):

                    pj = GENFIRE.misc.calculateProjection_interp_fromInterpolator(self.interpolator, phi, current_theta, psi, np.shape(self.model))
                    projections[:, :, i] = pj[self.ncOut-self.dims[0]/2:self.ncOut+self.dims[0]/2, self.ncOut-self.dims[1]/2:self.ncOut+self.dims[1]/2]
                filename = self.calculateProjections_Dialog.calculationParameters.outputFilename
                # if isinstance(filename,QtCore.QString):
                #     filename = (unicode(self.calculateProjections_Dialog.calculationParameters.outputFilename.toUtf8(),encoding='UTF-8'))
                filename = toString(filename)
                GENFIRE.fileio.saveData(filename,projections)
                # self.showProjection(pj)
            print("Finished calculating {}.".format(filename))


    def refreshModel(self):
        self.ui.checkBox_displayFigure.click()
        self.ui.checkBox_displayFigure.click()

    def toggleDisplayFigure(self):
        if not self.calculationParameters.displayProjectionsFlag:
           self.calculationParameters.displayProjectionsFlag = True
           if self.calculationParameters.modelFilenameProvided:
                import scipy.io as io
                # modelFilename = unicode(self.calculationParameters.modelFilename.toUtf8(), encoding='UTF-8')
                modelFilename = toString(self.calculationParameters.modelFilename)
                self.GENFIRE_load(modelFilename)
                self.oversamplingRatio = 3
                self.dims = np.shape(self.model)
                self.paddedDim = self.dims[0] * self.oversamplingRatio
                padding = int((self.paddedDim-self.dims[0])/2)
                self.model = np.pad(self.model,((padding,padding),(padding,padding),(padding,padding)),'constant')
                self.model = pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(pyfftw.interfaces.numpy_fft.ifftshift((self.model))))
                self.ncOut = np.shape(self.model)[0]//2
                if self.interpolator is None:
                    self.interpolator = GENFIRE.misc.getProjectionInterpolator(self.model)
                pj = GENFIRE.misc.calculateProjection_interp_fromInterpolator(self.interpolator, self.calculationParameters.phi,self.calculationParameters.theta,self.calculationParameters.psi, np.shape(self.model))
                pj = pj[self.ncOut-self.dims[0]/2:self.ncOut+self.dims[0]/2, self.ncOut-self.dims[1]/2:self.ncOut+self.dims[1]/2]
                self.showProjection(pj)
                self.calculationParameters.modelLoadedFlag = True
        else:
            self.calculationParameters.displayProjectionsFlag = False
            self.clearFigure()


    def updateFigure(self):
        # pj = misc.calculateProjection_interp(self.model, self.calculationParameters.phi,self.calculationParameters.theta,self.calculationParameters.psi)[self.ncOut-self.dims[0]/2:self.ncOut+self.dims[0]/2, self.ncOut-self.dims[1]/2:self.ncOut+self.dims[1]/2]
        if self.interpolator is None:
            self.interpolator = GENFIRE.misc.getProjectionInterpolator(self.model)
        pj = GENFIRE.misc.calculateProjection_interp_fromInterpolator(self.interpolator, self.calculationParameters.phi,self.calculationParameters.theta,self.calculationParameters.psi, np.shape(self.model))
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
            # self.calculationParameters.outputFilename = dirname
            # self.calculationParameters.outputFilename =  unicode(dirname.toUtf8(), encoding='UTF-8')
            self.calculationParameters.outputFilename =  dirname
            self.ui.lineEdit_outputFilename.setText(QtCore.QString(dirname))




    def setModelFilename_fromLineEdit(self):
        filename = self.ui.lineEdit_modelFile.text()
        if os.path.isfile(toString(filename)):
            self.GENFIRE_load(toString(filename))
            self.calculationParameters.modelFilename = filename
            self.calculationParameters.modelFilenameProvided = True
            self.ui.checkBox_displayFigure.setEnabled(True)

    def setModelFilename(self, filename):
        self.calculationParameters.modelFilename = filename
        self.ui.lineEdit_modelFile.setText(QtCore.QString(filename))
        self.calculationParameters.modelFilenameProvided = True
        self.ui.checkBox_displayFigure.setEnabled(True)

    def selectModelFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Model",filter="MATLAB files (*.mat);;TIFF images (*.tif *.tiff);;MRC (*.mrc);;All Files (*)")

        if os.path.isfile(toString(filename)):
            self.setModelFilename(filename)
            self.GENFIRE_load(toString(filename))

    def setAngleFilename(self, filename):
        if filename:
            self.calculationParameters.angleFilename = filename

    def toggleDisplayProjections(self):
        if self.calculationParameters.displayProjectionsFlag:
            self.calculationParameters.displayProjectionsFlag = False
        else:
            self.calculationParameters.displayProjectionsFlag = True

    def toggleSaveProjections(self):
        print  GENFIRE_ProjectionCalculator.calculationParameters.modelFilename
        print  GENFIRE_ProjectionCalculator.calculationParameters.angleFilename
        print  GENFIRE_ProjectionCalculator.calculationParameters.outputFilename
        print  GENFIRE_ProjectionCalculator.calculationParameters.outputFilesFlag
        print  GENFIRE_ProjectionCalculator.calculationParameters.displayProjectionsFlag
        print  GENFIRE_ProjectionCalculator.calculationParameters.phiStart
        print  GENFIRE_ProjectionCalculator.calculationParameters.thetaStart
        print  GENFIRE_ProjectionCalculator.calculationParameters.psiStart
        print  GENFIRE_ProjectionCalculator.calculationParameters.phiStep
        print  GENFIRE_ProjectionCalculator.calculationParameters.thetaStep
        print  GENFIRE_ProjectionCalculator.calculationParameters.psiStep
        print  GENFIRE_ProjectionCalculator.calculationParameters.numberOfProjections


        if self.calculationParameters.outputFilesFlag:
            self.calculationParameters.outputFilesFlag = False
        else:
            self.calculationParameters.outputFilesFlag = True

    def setOutputFilename(self, filename):
        if filename:
            self.calculationParameters.outputFilename = filename


    def GENFIRE_load(self, filename):
        self.model = GENFIRE.fileio.loadVolume(filename)

class ProjectionCalculationParameters:
    oversamplingRatio = 3
    def __init__(self):
        self.modelFilename = QtCore.QString('')
        self.angleFilename = QtCore.QString('')
        self.outputFilename = QtCore.QString('')
        self.outputFilesFlag = False
        self.displayProjectionsFlag = False
        self.angleFileProvided = False
        self.modelFilenameProvided = False
        self.modelLoadedFlag = False
        self.phi = 0
        self.theta = 0
        self.thetaStart = 0
        self.thetaStep = 1
        self.thetaStop = 1
        self.psi = 0

class CalculateProjectionSeries_popup(QtGui.QDialog):
    def __init__(self, calculation_parameters = ProjectionCalculationParameters()):
        super(CalculateProjectionSeries_popup, self).__init__()
        self.calculationParameters = calculation_parameters


        self.ui = CalculateProjectionSeries_Dialog.Ui_CalculateProjectionSeries_Dialog()
        self.ui.setupUi(self)
        import os
        self.ui.lineEdit_outputFilename.setText(os.path.abspath(os.getcwd()))
        self.ui.lineEdit_phi.setText(QtCore.QString(str(self.calculationParameters.phi)))
        self.ui.lineEdit_psi.setText(QtCore.QString(str(self.calculationParameters.psi)))
        self.ui.lineEdit_thetaStart.setText(QtCore.QString(str(self.calculationParameters.thetaStart)))
        self.ui.lineEdit_thetaStep.setText(QtCore.QString(str(self.calculationParameters.thetaStep)))
        self.ui.lineEdit_thetaStop.setText(QtCore.QString(str(self.calculationParameters.thetaStop)))
        self.ui.lineEdit_angleFile.textEdited.connect(self.setAngleFilename_fromLineEdit)
        self.ui.btn_selectAngleFile.clicked.connect(self.selectAngleFile)
        self.ui.lineEdit_phi.textEdited.connect(self.setPhi)
        self.ui.lineEdit_psi.textEdited.connect(self.setPsi)
        self.ui.lineEdit_thetaStart.textEdited.connect(self.setThetaStart)
        self.ui.lineEdit_thetaStep.textEdited.connect(self.setThetaStep)
        self.ui.lineEdit_thetaStop.textEdited.connect(self.setThetaStop)
        self.ui.lineEdit_outputFilename.textEdited.connect(self.setOutputFilename)

    def setAngleFilename_fromLineEdit(self):
        filename = self.ui.lineEdit_angleFile.text()
        if os.path.isfile(toString(filename)):
            self.calculationParameters.angleFilename = filename
            self.calculationParameters.angleFileProvided = True
            self.ui.lineEdit_angleFile.setText(QtCore.QString(filename))
            self.disableAngleWidgets()

    def selectAngleFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Angles",filter="txt files (*.txt);;All Files (*)")
        if filename:
            self.calculationParameters.angleFilename = filename
            self.calculationParameters.angleFileProvided = True
            self.ui.lineEdit_angleFile.setText(QtCore.QString(filename))
            self.disableAngleWidgets()

    def disableAngleWidgets(self):
        print('disabling')
        self.ui.lineEdit_thetaStart.setEnabled(True)
        self.ui.lineEdit_thetaStep.setDisabled(True)
        self.ui.lineEdit_thetaStop.setDisabled(True)
        self.ui.lineEdit_phi.setDisabled(True)
        self.ui.lineEdit_psi.setDisabled(True)

        self.ui.lineEdit_thetaStart.setStyleSheet("background-color: gray")
        self.ui.lineEdit_thetaStep.setStyleSheet("background-color: gray")
        self.ui.lineEdit_thetaStop.setStyleSheet("background-color: gray")
        self.ui.lineEdit_phi.setStyleSheet("background-color: gray")
        self.ui.lineEdit_psi.setStyleSheet("background-color: gray")

    def setPhi(self, angle):
        self.calculationParameters.phi = angle.toFloat()[0]

    def setPsi(self, angle):
        self.calculationParameters.psi = angle.toFloat()[0]

    def setThetaStart(self, value):
        self.calculationParameters.thetaStart = value.toFloat()[0]

    def setThetaStep(self, value):
        self.calculationParameters.thetaStep = value.toFloat()[0]

    def setThetaStop(self, value):
        self.calculationParameters.thetaStop = value.toFloat()[0]

    def setOutputFilename(self, filename):
        if filename:
            self.calculationParameters.outputFilename = filename

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

