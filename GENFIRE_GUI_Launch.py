from PyQt4 import QtCore, QtGui
import matplotlib
matplotlib.use("Qt4Agg")
import GENFIRE_GUI
import ProjectionCalculator
import GENFIRE_main
import os
import sys
from GENFIRE import ReconstructionParameters


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
        self.ui.btn_projections.clicked.connect(self.checkParameters)

        self.ui.btn_angles.clicked.connect(self.selectAngleFile)
        self.ui.btn_angles.clicked.connect(self.checkParameters)

        self.ui.btn_support.clicked.connect(self.selectSupportFile)
        self.ui.btn_support.clicked.connect(self.checkParameters)

        self.ui.btn_io.clicked.connect(self.selectInitialObjectFile)
        self.ui.btn_io.clicked.connect(self.checkParameters)

        self.ui.btn_reconstruct.clicked.connect(self.startReconstruction)
        self.ui.btn_reconstruct.clicked.connect(self.checkParameters)
        self.ui.btn_reconstruct.setStyleSheet("background-color: rgb(221,0,0)")

        ## Line Edits -- connect each to their main function and check if the reconstruction parameters are good each
        ## time a parameter is changed
        self.ui.lineEdit_pj.textChanged.connect(self.GENFIRE_ReconstructionParameters.setProjectionFilename)
        self.ui.lineEdit_pj.textChanged.connect(self.checkParameters)

        self.ui.lineEdit_angle.textChanged.connect(self.GENFIRE_ReconstructionParameters.setAngleFilename)
        self.ui.lineEdit_angle.textChanged.connect(self.checkParameters)

        self.ui.lineEdit_support.textChanged.connect(self.GENFIRE_ReconstructionParameters.setSupportFilename)
        self.ui.lineEdit_support.textChanged.connect(self.checkParameters)

        self.ui.lineEdit_io.textChanged.connect(self.GENFIRE_ReconstructionParameters.setInitialObjectFilename)
        self.ui.lineEdit_io.textChanged.connect(self.checkParameters)

        self.ui.lineEdit_results.textChanged.connect(self.GENFIRE_ReconstructionParameters.setResultsFilename)
        self.ui.lineEdit_results.textChanged.connect(self.checkParameters)

        self.ui.lineEdit_numIterations.setText(QtCore.QString("50"))
        self.ui.lineEdit_numIterations.textChanged.connect(self.GENFIRE_ReconstructionParameters.setNumberOfIterations)
        self.ui.lineEdit_numIterations.textChanged.connect(self.checkParameters)

        self.ui.lineEdit_oversamplingRatio.setText(QtCore.QString("3"))
        self.ui.lineEdit_oversamplingRatio.textChanged.connect(self.GENFIRE_ReconstructionParameters.setOversamplingRatio)
        self.ui.lineEdit_oversamplingRatio.textChanged.connect(self.checkParameters)

        self.ui.lineEdit_interpolationCutoffDistance.setText(QtCore.QString("0.7"))
        self.ui.lineEdit_interpolationCutoffDistance.textChanged.connect(self.GENFIRE_ReconstructionParameters.setInterpolationCutoffDistance)
        self.ui.lineEdit_interpolationCutoffDistance.textChanged.connect(self.checkParameters)


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

        # self.logger = GenfireLogger(notepad=self.ui.log) # create a threaded logger to redirect stdout to the GUI

        # self.logger_thread = QtCore.QThread()
        # self.logger.moveToThread(self.logger_thread)
        # self.logger_thread.start()



        # self.logger_thread = QtCore.QThread()
        # self.logger.moveToThread(self.logger_thread)
        # self.logger_thread.start()

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
        pass
        # sys.stdout = GenfireLogger(self.ui.log, sys.stdout )
        # sys.stderr = GenfireLogger(self.ui.log, sys.stderr, QtGui.QColor(255,0,0))

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
            self.GENFIRE_ReconstructionParameters.useDefaultSupport = True
        else:
            self.GENFIRE_ReconstructionParameters.useDefaultSupport = False

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

    def checkParameters(self):
        parametersAreGood = self.GENFIRE_ReconstructionParameters.checkParameters()
        if parametersAreGood: #update "go" button if we are ready
            GF_window.ui.btn_reconstruct.setStyleSheet("background-color: GREEN")
            GF_window.ui.btn_reconstruct.setText("Launch Reconstruction!")
    #Function to run GENFIRE reconstruction once all parameters are accounted for
    def startReconstruction(self):
        print('GENFIRE: Launching GENFIRE Reconstruction')
        GENFIRE_main.GENFIRE_main(self.GENFIRE_ReconstructionParameters)
#
    @QtCore.pyqtSlot(str)
    def receive_msg(self, msg):
        self.ui.log.append(msg)
# class GenfireLogger:
#     def __init__(self, textEdit, output=None, textColor=None):
#         from Queue import Queue
#         self.textEdit = textEdit
#         self.output = None
#         self.textColor = textColor
#
#     def write(self, message):
#         if self.textColor:
#             whichColor = self.textEdit.texttextColor()
#             self.textEdit.setTexttextColor(self.textColor)
#         self.textEdit.moveCursor(QtGui.QTextCursor.End)
#         self.textEdit.insertPlainText(message)
#         if self.textColor:
#             self.textEdit.setTexttextColor(whichColor)
#         if self.output:
#             self.output.write(message)


class GenfireListener(QtCore.QObject):
    message_pending = QtCore.pyqtSignal(str)
    def __init__(self, msg_queue):
        super(GenfireListener, self).__init__()
        self.msg_queue = msg_queue

    def run(self):
        while True:
            # print("waiting for message")
            msg = self.msg_queue.get() #get next message, blocks if nothing to get
            self.message_pending.emit(msg)

class GenfireWriter(object):
    def __init__(self, msg_queue):
        self.msg_queue = msg_queue

    def write(self, message):
        self.msg_queue.put(message)

class GenfireLogger(QtCore.QObject):
    def __init__(self):
        from Queue import Queue
        import sys
        from threading import Thread
        super(GenfireLogger, self).__init__()
        self.msg_queue = Queue()
        self.listener  = GenfireListener(msg_queue=self.msg_queue)
        self.writer    = GenfireWriter(msg_queue=self.msg_queue)
        sys.stdout     = self.writer


        self.listener_thread = QtCore.QThread()
        self.listener.moveToThread(self.listener_thread)
        self.listener_thread.started.connect(self.listener.run)
        self.listener_thread.start()

    def __del__(self):
        sys.stdout = sys.__stdout__









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
    GF_logger = GenfireLogger()

    # Render GUI
    GF_window.show()

    GF_logger.listener.message_pending[str].connect(GF_window.receive_msg)

    # Safely close and exit
    sys.exit(app.exec_())

