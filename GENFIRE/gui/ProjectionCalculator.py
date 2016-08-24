from __future__ import print_function

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
from GENFIRE.gui.utility import toString

class ProjectionCalculator(QtGui.QMainWindow):
    model_loading_signal = QtCore.pyqtSignal()
    update_filenames_signal = QtCore.pyqtSignal()
    emit_message_signal = QtCore.pyqtSignal(str)
    def __init__(self, parent=None):
        super(ProjectionCalculator, self).__init__()
        self.ui = ProjectionCalculator_MainWindow.Ui_ProjectionCalculator()
        self.ui.setupUi(self)
        self.parent = parent
        self.calculationParameters = ProjectionCalculationParameters()

        self.ui.lineEdit_modelFile.setText(QtCore.QString(os.getcwd()))
        self.ui.lineEdit_modelFile.editingFinished.connect(self.setModelFilename_fromLineEdit)

        ## Push Buttons
        self.ui.btn_selectModel.clicked.connect(self.selectModelFile)

        ## Sliders
        self.ui.verticalSlider_phi.setValue(0)
        self.ui.verticalSlider_phi.setMinimum(0)
        self.ui.verticalSlider_phi.setMaximum(3600)
        self.ui.verticalSlider_phi.valueChanged.connect(self.setPhiLineEditValue)

        self.ui.verticalSlider_theta.setValue(0)
        self.ui.verticalSlider_theta.setMinimum(0)
        self.ui.verticalSlider_theta.setMaximum(3600)
        self.ui.verticalSlider_theta.valueChanged.connect(self.setThetaLineEditValue)

        self.ui.verticalSlider_psi.setValue(0)
        self.ui.verticalSlider_psi.setMinimum(0)
        self.ui.verticalSlider_psi.setMaximum(3600)
        self.ui.verticalSlider_psi.valueChanged.connect(self.setPsiLineEditValue)

        self.ui.lineEdit_phi.setText(QtCore.QString('0'))
        self.ui.lineEdit_theta.setText(QtCore.QString('0'))
        self.ui.lineEdit_psi.setText(QtCore.QString('0'))
        self.ui.lineEdit_phi.editingFinished.connect(self.setPhiSliderValue)
        self.ui.lineEdit_theta.editingFinished.connect(self.setThetaSliderValue)
        self.ui.lineEdit_psi.editingFinished.connect(self.setPsiSliderValue)

        self.calculateProjections_Dialog = CalculateProjectionSeries_popup(self.calculationParameters)
        self.ui.btn_go.clicked.connect(self.showDialog)
        self.calculateProjections_Dialog.ui.buttonBox.accepted.connect(self.readyToCalculateProjections)
        self.ui.btn_go.setEnabled(False)

        self.figure = plt.figure(3)
        self.figure.clf() # clear figure in case it was rendered somewhere else previously
        self.canvas = FigureCanvas(self.figure)
        self.navigationToolbar = NavigationToolbar(self.canvas, self)
        self.ui.verticalLayout_figure.addWidget(self.navigationToolbar)
        self.ui.verticalLayout_figure.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)
        self.ax.hold(False)

        self.ui.btn_clearModel.clicked.connect(self.clearModel)

    def showDialog(self):
        self.calculateProjections_Dialog.show()

    def readyToCalculateProjections(self):
        self.emit_message_signal.emit("GENFIRE: Calculating projections...")
        self.calculation_thread = ProjectionCalculator_thread(self)
        self.calculation_thread.finished.connect(self.readyToClose)
        self.calculation_thread.start()
        self.parent.raise_()

    def readyToClose(self):
        self.update_filenames_signal.emit()
        self.close()

    def calculateProjections(self):
        if self.calculateProjections_Dialog.status:
            self.calculationParameters.modelFilename = toString(self.calculationParameters.modelFilename)
            if not self.calculationParameters.modelLoadedFlag:

                self.calculationParameters.model = GENFIRE.fileio.readVolume(self.calculationParameters.modelFilename)
                self.calculationParameters.oversamplingRatio = self.calculationParameters.oversamplingRatio
                self.calculationParameters.dims = np.shape(self.calculationParameters.model)
                paddedDim = self.calculationParameters.dims[0] * self.calculationParameters.oversamplingRatio
                padding = int((paddedDim-self.calculationParameters.dims[0])/2)
                self.calculationParameters.model = np.pad(self.calculationParameters.model,((padding,padding),(padding,padding),(padding,padding)),'constant')
                self.calculationParameters.model = pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(pyfftw.interfaces.numpy_fft.ifftshift((self.calculationParameters.model))))
                self.calculationParameters.interpolator = GENFIRE.utility.getProjectionInterpolator(self.calculationParameters.model)
            self.calculationParameters.ncOut = np.shape(self.calculationParameters.model)[0]//2


            if self.calculationParameters.angleFileProvided:
                self.calculationParameters.angleFilename = toString(self.calculationParameters.angleFilename)
                self.calculationParameters.outputFilename = toString(self.calculationParameters.outputFilename)
                angles = np.loadtxt(toString(self.calculationParameters.angleFilename))
                phi = angles[:, 0]
                theta = angles[:, 1]
                psi = angles[:, 2]
                filename = self.calculationParameters.outputFilename +'.npy'
                projections = np.zeros((self.calculationParameters.dims[0],self.calculationParameters.dims[1],np.size(phi)),dtype=float)
                if self.calculationParameters.interpolator is None:
                    self.calculationParameters.interpolator = GENFIRE.utility.getProjectionInterpolator(self.calculationParameters.model)
                for projNum in range(0,np.size(phi)):
                    pj = GENFIRE.utility.calculateProjection_interp_fromInterpolator(self.calculationParameters.interpolator, phi[projNum], theta[projNum], psi[projNum], np.shape(self.calculationParameters.model))
                    projections[:, :, projNum] = pj[self.calculationParameters.ncOut-self.calculationParameters.dims[0]/2:self.calculationParameters.ncOut+self.calculationParameters.dims[0]/2, self.calculationParameters.ncOut-self.calculationParameters.dims[1]/2:self.calculationParameters.ncOut+self.calculationParameters.dims[1]/2]
                # projections[ projections<0 ] = 0
                GENFIRE.fileio.writeVolume(filename, projections)
            else:
                if self.calculationParameters.interpolator is None:
                    self.calculationParameters.interpolator = GENFIRE.utility.getProjectionInterpolator(self.calculationParameters.model)
                phi = self.calculationParameters.phi
                psi = self.calculationParameters.psi
                theta = np.arange(self.calculationParameters.thetaStart, \
                                  self.calculationParameters.thetaStop + 1e-10, \
                                  self.calculationParameters.thetaStep)
                projections = np.zeros((self.calculationParameters.dims[0],self.calculationParameters.dims[1],np.size(theta)),dtype=float)
                for i, current_theta in enumerate(theta):

                    pj = GENFIRE.utility.calculateProjection_interp_fromInterpolator(self.calculationParameters.interpolator, phi, current_theta, psi, np.shape(self.calculationParameters.model))
                    projections[:, :, i] = pj[self.calculationParameters.ncOut-self.calculationParameters.dims[0]/2:self.calculationParameters.ncOut+self.calculationParameters.dims[0]/2, self.calculationParameters.ncOut-self.calculationParameters.dims[1]/2:self.calculationParameters.ncOut+self.calculationParameters.dims[1]/2]
                filename = self.calculationParameters.outputFilename
                filename = toString(filename)
                # projections[ projections<0 ] = 0
                GENFIRE.fileio.writeVolume(filename, projections)
                if self.calculationParameters.writeAnglesFlag:
                    output_filename_base, ext  = os.path.splitext(toString(self.calculationParameters.outputFilename))
                    output_angle_filename      = output_filename_base + "_euler_angles.txt"
                    if os.path.isfile(output_angle_filename):
                        print("WARNING! Filename {} already exists and will be overwritten.".format(output_angle_filename))
                    self.calculationParameters.outputAngleFilename = output_angle_filename
                    num_projections = np.size(theta)
                    phi = np.repeat(phi, num_projections)
                    psi = np.repeat(psi, num_projections)
                    with open(output_angle_filename,'w') as fid:
                        for euler in zip(phi, theta, psi):
                            string = str(euler[0]) + ' ' + str(euler[1]) + ' ' + str(euler[2]) + '\n'
                            fid.write(string)
                    print("Successfully calculated {}.".format(output_angle_filename))


            print("Successfully calculated {}.".format(filename))

    def clearModel(self):
        self.calculationParameters.model         = None
        self.calculationParameters.interpolator  = None
        self.calculationParameters.modelFilename = None
        self.clearFigure()
        self.ui.btn_go.setEnabled(False)

    def displayFigure(self):
        if self.calculationParameters.model is not None:
            pj = GENFIRE.utility.calculateProjection_interp_fromInterpolator(self.calculationParameters.interpolator, self.calculationParameters.phi, self.calculationParameters.theta, self.calculationParameters.psi, np.shape(self.calculationParameters.model))
            pj = pj[self.calculationParameters.ncOut-self.calculationParameters.dims[0]/2:self.calculationParameters.ncOut+self.calculationParameters.dims[0]/2, self.calculationParameters.ncOut-self.calculationParameters.dims[1]/2:self.calculationParameters.ncOut+self.calculationParameters.dims[1]/2]
            pj[ pj<0 ] = 0
            self.showProjection(pj)
        else:
            self.clearFigure()

    def updateFigure(self):
        if self.calculationParameters.interpolator is None:
            self.calculationParameters.interpolator = GENFIRE.utility.getProjectionInterpolator(self.calculationParameters.model)
        pj = GENFIRE.utility.calculateProjection_interp_fromInterpolator(self.calculationParameters.interpolator, self.calculationParameters.phi, self.calculationParameters.theta, self.calculationParameters.psi, np.shape(self.calculationParameters.model))
        pj = pj[self.calculationParameters.ncOut-self.calculationParameters.dims[0]/2:self.calculationParameters.ncOut+self.calculationParameters.dims[0]/2, self.calculationParameters.ncOut-self.calculationParameters.dims[1]/2:self.calculationParameters.ncOut+self.calculationParameters.dims[1]/2]
        pj[ pj<0 ] = 0
        self.ax.imshow(pj)
        self.canvas.draw()


    def showProjection(self, pj):
        self.ax.imshow(pj)
        self.canvas.draw()

    def clearFigure(self):
        self.figure.clf()
        self.canvas.draw()
        self.ax = self.figure.add_subplot(111)
        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)
        self.ax.hold(False)

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
        self.updateFigure()

    def setThetaLineEditValue(self, value):
        self.ui.lineEdit_theta.setText(QtCore.QString.number(float(value)/10))
        self.calculationParameters.theta = float(value)/10
        self.updateFigure()

    def setPsiLineEditValue(self, value):
        self.ui.lineEdit_psi.setText(QtCore.QString.number(float(value)/10))
        self.calculationParameters.psi = float(value)/10
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
            self.calculationParameters.outputFilename =  dirname
            # self.ui.lineEdit_outputFilename.setText(QtCore.QString(dirname))

    def setModelFilename_fromLineEdit(self):
        filename = self.ui.lineEdit_modelFile.text()
        if os.path.isfile(toString(filename)) and filename != self.calculationParameters.modelFilename:
            self.setModelFilename(toString(filename))

    def setModelFilename(self, filename):
        self.calculationParameters.modelFilename = filename
        self.ui.lineEdit_modelFile.setText(QtCore.QString(filename))
        self.loadModel(filename)
        self.displayFigure()

    def selectModelFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Model",filter="Volume files (*.mrc *.mat *.npy);;MATLAB files (*.mat);;TIFF images (*.tif *.tiff);;MRC (*.mrc);;numpy (*.npy);;All Files (*)")
        if os.path.isfile(toString(filename)):
            self.ui.btn_go.setEnabled(True)
            self.setModelFilename(filename)

    def loadModel(self, filename):
        from threading import Thread
        t = Thread(target=lambda:print("Loading Model..."))
        t.start()
        self.ui.btn_go.setEnabled(True)
        self.calculationParameters.model = GENFIRE.fileio.readVolume(toString(filename))
        self.calculationParameters.dims = np.shape(self.calculationParameters.model)
        self.calculationParameters.paddedDim = self.calculationParameters.dims[0] * self.calculationParameters.oversamplingRatio
        padding = int((self.calculationParameters.paddedDim-self.calculationParameters.dims[0])/2)
        self.calculationParameters.model = np.pad(self.calculationParameters.model,((padding,padding),(padding,padding),(padding,padding)),'constant')
        self.calculationParameters.model = pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(pyfftw.interfaces.numpy_fft.ifftshift((self.calculationParameters.model))))
        self.calculationParameters.ncOut = np.shape(self.calculationParameters.model)[0]//2
        self.calculationParameters.interpolator = GENFIRE.utility.getProjectionInterpolator(self.calculationParameters.model)
        t.join()

    def setAngleFilename(self, filename):
        if filename:
            self.calculationParameters.angleFilename = filename

    def toggleSaveProjections(self):
        print  (GENFIRE_ProjectionCalculator.calculationParameters.modelFilename)
        print  (GENFIRE_ProjectionCalculator.calculationParameters.angleFilename)
        print  (GENFIRE_ProjectionCalculator.calculationParameters.outputFilename)
        print  (GENFIRE_ProjectionCalculator.calculationParameters.outputFilesFlag)
        print  (GENFIRE_ProjectionCalculator.calculationParameters.phiStart)
        print  (GENFIRE_ProjectionCalculator.calculationParameters.thetaStart)
        print  (GENFIRE_ProjectionCalculator.calculationParameters.psiStart)
        print  (GENFIRE_ProjectionCalculator.calculationParameters.phiStep)
        print  (GENFIRE_ProjectionCalculator.calculationParameters.thetaStep)
        print  (GENFIRE_ProjectionCalculator.calculationParameters.psiStep)
        print  (GENFIRE_ProjectionCalculator.calculationParameters.numberOfProjections)


        if self.calculationParameters.outputFilesFlag:
            self.calculationParameters.outputFilesFlag = False
        else:
            self.calculationParameters.outputFilesFlag = True

    def setOutputFilename(self, filename):
        if filename:
            self.calculationParameters.outputFilename = filename


class ProjectionCalculationParameters:
    oversamplingRatio = 3
    def __init__(self):
        self.modelFilename              = QtCore.QString('')
        self.angleFilename              = QtCore.QString('')
        self.outputFilename             = QtCore.QString('')
        self.outputAngleFilename        = QtCore.QString('')
        self.outputFilesFlag            = False
        self.angleFileProvided          = False
        self.modelFilenameProvided      = False
        self.modelLoadedFlag            = False
        self.writeAnglesFlag            = True
        self.interpolator               = None
        self.model                      = None
        self.ncOut                      = None
        self.phi                        = 0.0
        self.theta                      = 0.0
        self.thetaStart                 = 0.0
        self.thetaStep                  = 3.0
        self.thetaStop                  = 180.0
        self.psi                        = 0.0

class CalculateProjectionSeries_popup(QtGui.QDialog):
    def __init__(self, calculation_parameters = ProjectionCalculationParameters()):
        super(CalculateProjectionSeries_popup, self).__init__()
        self.calculationParameters = calculation_parameters


        self.ui = CalculateProjectionSeries_Dialog.Ui_CalculateProjectionSeries_Dialog()
        self.ui.setupUi(self)
        import os
        from functools import partial
        self.calculationParameters.outputFilename = os.path.abspath(os.getcwd() + '/projections.mrc')
        self.ui.lineEdit_outputFilename.setText(self.calculationParameters.outputFilename)
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
        self.ui.checkBox_saveAngles.setChecked(True)
        self.ui.checkBox_saveAngles.toggled.connect(self.toggleSaveAngles)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).setText("Calculate Projections")
        self.ui.buttonBox.accepted.connect(partial(self.setStatus, 1))

        self.status = 0

    def setStatus(self,status):
        self.status = status
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
    def toggleSaveAngles(self):
        if self.ui.checkBox_saveAngles.isChecked():
            self.calculationParameters.writeAnglesFlag = True
        else:
            self.calculationParameters.writeAnglesFlag = False
    def disableAngleWidgets(self):
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
        self.ui.checkBox_saveAngles.setChecked(False)
        self.ui.checkBox_saveAngles.setEnabled(False)


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

class ProjectionCalculator_thread(QtCore.QThread):
    def __init__(self, parent):
        super(ProjectionCalculator_thread, self).__init__()
        self.parent = parent
    def run(self):
        self.parent.calculateProjections()
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

