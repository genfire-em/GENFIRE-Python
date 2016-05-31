from PyQt4 import QtCore, QtGui
import GENFIRE_GUI
import GENFIRE
import ProjectionCalculator
import GENFIRE_main
import os
import sys

class GenfireMainWindow(QtGui.QMainWindow): #Subclasses QMainWindow
    def __init__(self):

        ## Superclass constructor
        super(GenfireMainWindow,self).__init__()

        ## Initialize UI
        self.ui = GENFIRE_GUI.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('test.png'))

        ## Initialize Reconstruction Parameters
        self.GENFIRE_ReconstructionParameters = ReconstructionParameters()

        ## Initialize file paths in text boxes
        self.ui.lineEdit_results.setText(QtCore.QString(os.path.join(os.getcwd(),'results.mrc')))

        self.ui.lineEdit_support.setText(QtCore.QString(os.getcwd()))

        self.ui.lineEdit_pj.setText(QtCore.QString(os.getcwd()))

        self.ui.lineEdit_io.setText(QtCore.QString(os.getcwd()))

        self.ui.lineEdit_angle.setText(QtCore.QString(os.getcwd()))


        ## Push Buttons -- connect each to their main function and check if the reconstruction parameters are good each
        ## time a parameter is changed
        self.ui.btn_projections.clicked.connect(self.selectProjectionFile)
        self.ui.btn_projections.clicked.connect(self.GENFIRE_ReconstructionParameters.checkParameters)

        self.ui.btn_angles.clicked.connect(self.selectAngleFile)
        self.ui.btn_angles.clicked.connect(self.GENFIRE_ReconstructionParameters.checkParameters)

        self.ui.btn_support.clicked.connect(self.selectSupportFile)
        self.ui.btn_support.clicked.connect(self.GENFIRE_ReconstructionParameters.checkParameters)

        self.ui.btn_io.clicked.connect(self.selectInitialObjectFile)
        self.ui.btn_io.clicked.connect(self.GENFIRE_ReconstructionParameters.checkParameters)

        self.ui.btn_reconstruct.clicked.connect(self.startReconstruction)
        self.ui.btn_reconstruct.clicked.connect(self.GENFIRE_ReconstructionParameters.checkParameters)
        self.ui.btn_reconstruct.setStyleSheet("background-color: red")

        ## Line Edits -- connect each to their main function and check if the reconstruction parameters are good each
        ## time a parameter is changed
        self.ui.lineEdit_pj.textChanged.connect(self.GENFIRE_ReconstructionParameters.setProjectionFilename)
        self.ui.lineEdit_pj.textChanged.connect(self.GENFIRE_ReconstructionParameters.checkParameters)

        self.ui.lineEdit_angle.textChanged.connect(self.GENFIRE_ReconstructionParameters.setAngleFilename)
        self.ui.lineEdit_angle.textChanged.connect(self.GENFIRE_ReconstructionParameters.checkParameters)

        self.ui.lineEdit_support.textChanged.connect(self.GENFIRE_ReconstructionParameters.setSupportFilename)
        self.ui.lineEdit_support.textChanged.connect(self.GENFIRE_ReconstructionParameters.checkParameters)

        self.ui.lineEdit_io.textChanged.connect(self.GENFIRE_ReconstructionParameters.setInitialObjectFilename)
        self.ui.lineEdit_io.textChanged.connect(self.GENFIRE_ReconstructionParameters.checkParameters)

        self.ui.lineEdit_results.textChanged.connect(self.GENFIRE_ReconstructionParameters.setResultsFilename)
        self.ui.lineEdit_results.textChanged.connect(self.GENFIRE_ReconstructionParameters.checkParameters)

        self.ui.lineEdit_numIterations.setText(QtCore.QString("50"))
        self.ui.lineEdit_numIterations.textChanged.connect(self.GENFIRE_ReconstructionParameters.setNumberOfIterations)
        self.ui.lineEdit_numIterations.textChanged.connect(self.GENFIRE_ReconstructionParameters.checkParameters)

        self.ui.lineEdit_oversamplingRatio.setText(QtCore.QString("3"))
        self.ui.lineEdit_oversamplingRatio.textChanged.connect(self.GENFIRE_ReconstructionParameters.setOversamplingRatio)
        self.ui.lineEdit_oversamplingRatio.textChanged.connect(self.GENFIRE_ReconstructionParameters.checkParameters)

        self.ui.lineEdit_interpolationCutoffDistance.setText(QtCore.QString("0.7"))
        self.ui.lineEdit_interpolationCutoffDistance.textChanged.connect(self.GENFIRE_ReconstructionParameters.setInterpolationCutoffDistance)
        self.ui.lineEdit_interpolationCutoffDistance.textChanged.connect(self.GENFIRE_ReconstructionParameters.checkParameters)


        self.ui.lineEdit_displayFrequency.setDisabled(True)
        self.ui.lineEdit_displayFrequency.setStyleSheet("background-color: gray")
        self.ui.lineEdit_displayFrequency.textChanged.connect(self.ShoutDisplayFrequency)

        self.ui.lineEdit_io.setDisabled(True)
        self.ui.lineEdit_io.setStyleSheet("background-color: gray")


        ## Radio Buttons -- default is resolution extension suppression
        self.ui.radioButton_on.setChecked(True)
        self.ui.radioButton_on.toggled.connect(self.selectResolutionExtensionSuppressionState)

        self.ui.radioButton_off.toggled.connect(self.selectResolutionExtensionSuppressionState)

        self.ui.radioButton_extension.toggled.connect(self.selectResolutionExtensionSuppressionState)


        ## Check Boxes

        self.ui.checkBox_displayFigure.toggled.connect(self.GENFIRE_ReconstructionParameters.toggleDisplayFigure)
        self.ui.checkBox_displayFigure.toggled.connect(self.enableDisplayFrequencyChange)

        self.ui.checkBox_log.toggled.connect(self.enableLog)

        self.ui.checkBox_rfree.toggled.connect(self.calculateRfree)

        self.ui.checkBox_provide_io.toggled.connect(self.toggleSelectIO)

        self.ui.checkBox_default_support.toggled.connect(self.toggleUseDefaultSupport)

        self.ui.action_Create_Support.triggered.connect(self.launchProjectionCalculator)

        # ## Message Log
        # sys.stdout = GenfireLog(self.ui.log, sys.stdout )
        # sys.stderr = GenfireLog(self.ui.log, sys.stderr, QtGui.QColor(255,0,0))
        # # bla = GenfireLog(self.ui.log,"test")
        #
        # # sys.stdout = GenfireLog()
        #
        # # sys.stdout.write.connect(self.messageWritten)

    def calculateRfree(self):
        if self.ui.checkBox_rfree.isEnabled() == True:
            self.GENFIRE_ReconstructionParameters.calculateRfree = True
        else:
            self.GENFIRE_ReconstructionParameters.calculateRfree = False

    def launchProjectionCalculator(self):
        print "launching"
        self.GENFIRE_ProjectionCalculator = ProjectionCalculator.ProjectionCalculator()
        self.GENFIRE_ProjectionCalculator.show()

    def enableLog(self):
        sys.stdout = GenfireLogger(self.ui.log, sys.stdout )
        sys.stderr = GenfireLogger(self.ui.log, sys.stderr, QtGui.QColor(255,0,0))

    def toggleSelectIO(self):
        if self.ui.lineEdit_io.isEnabled():
             self.ui.lineEdit_io.setStyleSheet("background-color: gray")
             self.ui.lineEdit_io.setEnabled(False)
             self.GENFIRE_ReconstructionParameters._initialObjectFilename=""
        else:
             self.ui.lineEdit_io.setStyleSheet("background-color: white")
             self.ui.lineEdit_io.setEnabled(True)

    def toggleUseDefaultSupport(self):
        if self.ui.checkBox_default_support.isEnabled():
            self.GENFIRE_ReconstructionParameters._useDefaultSupport = True
        else:
            self.GENFIRE_ReconstructionParameters._useDefaultSupport = False

    def enableDisplayFrequencyChange(self):
        global displayFrequency
        displayFrequency = 5
        self.ui.lineEdit_displayFrequency.setEnabled(True)
        self.ui.lineEdit_displayFrequency.setText(QtCore.QString("5"))
        self.ui.lineEdit_displayFrequency.setStyleSheet("background-color: white")

    def ShoutDisplayFrequency(self, frequency):
        self.GENFIRE_ReconstructionParameters.displayFigure.displayFrequency = frequency.toInt()[0]
        print type(self.GENFIRE_ReconstructionParameters.displayFigure.displayFrequency)
        print "updating displayFrequency", self.GENFIRE_ReconstructionParameters.displayFigure.displayFrequency

    def messageWritten(self, message):
        print message
    def __del__(self):
        sys.stdout = sys.__stdout__ # Restore output stream to default upon exiting the GUI
    #Functions for selecting input files using QFileDialog
    def selectProjectionFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Projections",filter="MATLAB files (*.mat);;TIFF images (*.tif *.tiff);;MRC (*.mrc);;All Files (*)")

        if filename:
            self.GENFIRE_ReconstructionParameters.setProjectionFilename(filename)
            print ("Projection Filename:", self.GENFIRE_ReconstructionParameters.getProjectionFilename())
            self.ui.lineEdit_pj.setText(QtCore.QString(filename))

    def selectAngleFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Support",filter="MATLAB files (*.mat);;text files (*.txt *.tiff);;MRC (*.mrc);;All Files (*)")
        if filename:
            self.GENFIRE_ReconstructionParameters.setAngleFilename(filename)
            print ("Angle Filename:", self.GENFIRE_ReconstructionParameters.getAngleFilename())
            self.ui.lineEdit_angle.setText(QtCore.QString(filename))

    def selectSupportFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Support",filter="MATLAB files (*.mat);;TIFF images (*.tif *.tiff);;MRC (*.mrc);;All Files (*)")
        if filename:
            self.GENFIRE_ReconstructionParameters.setSupportFilename(filename)
            print ("Support Filename:", self.GENFIRE_ReconstructionParameters.getSupportFilename())
            self.ui.lineEdit_support.setText(QtCore.QString(filename))

    def selectInitialObjectFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Initial Object",filter="MATLAB files (*.mat);;TIFF images (*.tif *.tiff);;MRC (*.mrc);;All Files (*)")
        if filename:
            self.GENFIRE_ReconstructionParameters.setInitialObjectFilename(filename)
            print ("Initial Object Filename:", self.GENFIRE_ReconstructionParameters.getProjectionFilename())
            self.ui.lineEdit_io.setText(QtCore.QString(filename))

    #Define constraint enforcement mode
    def selectResolutionExtensionSuppressionState(self):
        if self.ui.radioButton_on.isChecked():
            self.GENFIRE_ReconstructionParameters.setResolutionExtensionSuppressionState(1)
        elif self.ui.radioButton_extension.isChecked():
            self.GENFIRE_ReconstructionParameters.setResolutionExtensionSuppressionState(3)
        else:
            self.GENFIRE_ReconstructionParameters.setResolutionExtensionSuppressionState(2)

    #Function to run GENFIRE reconstruction once all parameters are accounted for
    def startReconstruction(self):
        print('GENFIRE: Launching GENFIRE Reconstruction')

        if self.GENFIRE_ReconstructionParameters.CheckIfInitialObjectIsDefined():
            GENFIRE_main.GENFIRE_main(filename_projections=self.GENFIRE_ReconstructionParameters.getProjectionFilename(),
                             filename_angles=self.GENFIRE_ReconstructionParameters.getAngleFilename(),
                             filename_support=self.GENFIRE_ReconstructionParameters.getSupportFilename(),
                             filename_results=self.GENFIRE_ReconstructionParameters.getResultsFilename(),
                             numIterations=self.GENFIRE_ReconstructionParameters.getNumberOfIterations(),
                             oversamplingRatio=self.GENFIRE_ReconstructionParameters.getOversamplingRatio(),
                             interpolationCutoffDistance=self.GENFIRE_ReconstructionParameters.getInterpolationCutoffDistance(),
                             resolutionExtensionSuppressionState=self.GENFIRE_ReconstructionParameters.getResolutionExtensionSuppressionState(),
                             displayFigure=self.GENFIRE_ReconstructionParameters.displayFigure,
                             calculateRFree=self.GENFIRE_ReconstructionParameters.calculateRfree,
                             filename_initialObject=self.GENFIRE_ReconstructionParameters.getInitialObjectFilename())
        else:
            GENFIRE_main.GENFIRE_main(filename_projections=self.GENFIRE_ReconstructionParameters.getProjectionFilename(),
                             filename_angles=self.GENFIRE_ReconstructionParameters.getAngleFilename(),
                             filename_support=self.GENFIRE_ReconstructionParameters.getSupportFilename(),
                             filename_results=self.GENFIRE_ReconstructionParameters.getResultsFilename(),
                             numIterations=self.GENFIRE_ReconstructionParameters.getNumberOfIterations(),
                             oversamplingRatio=self.GENFIRE_ReconstructionParameters.getOversamplingRatio(),
                             interpolationCutoffDistance=self.GENFIRE_ReconstructionParameters.getInterpolationCutoffDistance(),
                             resolutionExtensionSuppressionState=self.GENFIRE_ReconstructionParameters.getResolutionExtensionSuppressionState(),
                             displayFigure=self.GENFIRE_ReconstructionParameters.displayFigure,
                             calculateRFree=self.GENFIRE_ReconstructionParameters.calculateRfree)


# Define ReconstructionParameters class to hold reconstruction parameters
class ReconstructionParameters():
    def __init__(self):
        self._projectionFilename = ""
        self._angleFilename = ""
        self._supportFilename = ""
        self._resolutionExtensionSuppressionState = 1 #1 for resolution extension/suppression, 2 for off, 3 for just extension
        self._numIterations = 50
        self.displayFigure = GENFIRE.DisplayFigure()
        self._supportedFiletypes = ['.tif', '.mrc', '.mat']
        self._supportedAngleFiletypes = ['.txt', '.mat']
        self._oversamplingRatio = 3
        self._interpolationCutoffDistance = 0.7
        self._isInitialObjectDefined = False
        self._resultsFilename = os.path.join(os.getcwd(),'results.mrc')
        self._useDefaultSupport = True
        self.calculateRfree = False

    def checkParameters(self): #verify file extensions are supported
        parametersAreGood = 1

        projection_extension = os.path.splitext(self._projectionFilename)
        if projection_extension[1] not in self._supportedFiletypes:
            parametersAreGood = 0

        angle_extension = os.path.splitext(self._angleFilename)
        if angle_extension[1] not in self._supportedAngleFiletypes:
            parametersAreGood = 0

        if self._supportFilename != "": #empty support filename is okay, as this will trigger generation of a default support
            support_extension = os.path.splitext(self._supportFilename)
            if support_extension[1] not in self._supportedFiletypes:
                parametersAreGood = 0

        if not self.getResultsFilename():
            parametersAreGood = 0

        if parametersAreGood: #update "go" button if we are ready
            GF_window.ui.btn_reconstruct.setStyleSheet("background-color: GREEN")
            GF_window.ui.btn_reconstruct.setText("Launch Reconstruction!")

    # Define setters/getters
    def setProjectionFilename(self, projectionFilename):
        if projectionFilename:
            self._projectionFilename = os.path.join(os.getcwd(),unicode(projectionFilename.toUtf8(), encoding='UTF-8'))

    def getProjectionFilename(self):
        return self._projectionFilename

    def setAngleFilename(self, angleFilename):
        if angleFilename:
            self._angleFilename = os.path.join(os.getcwd(),unicode(angleFilename.toUtf8(), encoding='UTF-8'))

    def getAngleFilename(self):
        return self._angleFilename

    def setSupportFilename(self, supportFilename):
        if supportFilename:
            self._supportFilename = os.path.join(os.getcwd(),unicode(supportFilename.toUtf8(), encoding='UTF-8'))

    def getSupportFilename(self):
        return self._supportFilename

    def setResultsFilename(self, resultsFilename):
        if resultsFilename:
            self._resultsFilename = os.path.join(os.getcwd(),unicode(resultsFilename.toUtf8(), encoding='UTF-8'))

    def getResultsFilename(self):
        return self._resultsFilename


    def setInitialObjectFilename(self, initialObjectFilename):
        self._initialObjectFilename = os.path.join(os.getcwd(),unicode(initialObjectFilename.toUtf8(), encoding='UTF-8'))
        self._isInitialObjectDefined = True

    def getInitialObjectFilename(self):
        if self.CheckIfInitialObjectIsDefined():
            return self._initialObjectFilename
        else:
            pass

    def CheckIfInitialObjectIsDefined(self):
        return self._isInitialObjectDefined

    def setResolutionExtensionSuppressionState(self, state):
        self._resolutionExtensionSuppressionState = state

    def getResolutionExtensionSuppressionState(self):
        return self._resolutionExtensionSuppressionState

    def setNumberOfIterations(self,numIterations):
        numIterations = numIterations.toInt()
        if numIterations[1]:
            numIterations = numIterations[0]
            if numIterations > 0:
                self._numIterations = numIterations

    def getNumberOfIterations(self):
        return self._numIterations

    def toggleDisplayFigure(self): # whether or not to display figure during reconstruction
        if self.displayFigure.DisplayFigureON:
            self.displayFigure.DisplayFigureON = False
        else:
            self.displayFigure.DisplayFigureON = True

        if self.displayFigure.DisplayErrorFigureON:
            self.displayFigure.DisplayErrorFigureON = False
        else:
            self.displayFigure.DisplayErrorFigureON = True

    def getDisplayFigure(self):
        return self.displayFigure

    def setOversamplingRatio(self, oversamplingRatio):
        self._oversamplingRatio = oversamplingRatio.toInt()[0]

    def getOversamplingRatio(self):
        return self._oversamplingRatio

    def setInterpolationCutoffDistance(self, interpolationCutoffDistance):
        self._interpolationCutoffDistance = interpolationCutoffDistance.toFloat()[0]

    def getInterpolationCutoffDistance(self):
        return self._interpolationCutoffDistance

class GenfireLogger:
    def __init__(self, textEdit, output=None, textColor=None):
        self.textEdit = textEdit
        self.output = None
        self.textColor = textColor

    def write(self, message):
        if self.textColor:
            whichColor = self.textEdit.texttextColor()
            self.textEdit.setTexttextColor(self.textColor)
        self.textEdit.moveCursor(QtGui.QTextCursor.End)
        self.textEdit.insertPlainText(message)
        if self.textColor:
            self.textEdit.setTexttextColor(whichColor)
        if self.output:
            self.output.write(message)




if __name__ == "__main__":

    # Startup the application
    app = QtGui.QApplication(sys.argv)

    # app.setStyle('plastique')
    app.setStyle('mac')
    # app.setStyle('windows')
    # app.setStyle('cleanlooks')
    # app.setStyle('motif')
    # app.setStyle('GTK')


    # Create the GUI
    GF_window = GenfireMainWindow()

    # Render GUI
    GF_window.show()

    # Safely close and exit
    sys.exit(app.exec_())

