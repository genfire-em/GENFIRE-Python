"""
* genfire.gui.launch *

This is the script for launching the GENFIRE GUI


Author: Alan (AJ) Pryor, Jr.
Jianwei (John) Miao Coherent Imaging Group
University of California, Los Angeles
Copyright 2015-2016. All rights reserved.
"""


from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib
matplotlib.use("Qt5Agg")
from genfire.gui import ProjectionCalculator, VolumeSlicer, GENFIRE_MainWindow
import os
import sys
from genfire.reconstruct import ReconstructionParameters
if sys.version_info >= (3,0):
	from genfire.gui import GENFIRE_qrc_py3
else:
	from genfire.gui import GENFIRE_qrc
from genfire.utility import *
from genfire.gui.utility import toString, toQString, toInt, toFloat

class GenfireMainWindow(QtWidgets.QMainWindow):
    stop_threads = QtCore.pyqtSignal()
    def closeEvent(self, QCloseEvent):
        self.stop_threads.emit()
        GF_logger.listener_thread.wait()
        GF_error_logger.listener_thread.wait()
        import matplotlib.pyplot as plt; plt.close("all")
        QCloseEvent.accept()
    def __init__(self):

        ## Superclass constructor
        super(GenfireMainWindow,self).__init__()

        ## Initialize UI
        self.ui = GENFIRE_MainWindow.Ui_GENFIRE_MainWindow()
        self.ui.setupUi(self)

        ## Initialize Reconstruction Parameters
        self.GENFIRE_ReconstructionParameters = ReconstructionParameters()

        ## Initialize file paths in text boxes
        self.ui.lineEdit_results.setText(toQString(os.path.join(os.getcwd(),'results.mrc')))

        self.ui.lineEdit_support.setText(toQString(os.getcwd()))

        self.ui.lineEdit_pj.setText(toQString(os.getcwd()))

        self.ui.lineEdit_io.setText(toQString(os.getcwd()))

        self.ui.lineEdit_angle.setText(toQString(os.getcwd()))


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

        self.ui.btn_reconstruct.setEnabled(False)
        self.ui.btn_reconstruct.clicked.connect(self.startReconstruction)
        self.ui.btn_reconstruct.clicked.connect(self.checkParameters)
        self.ui.btn_reconstruct.setStyleSheet("background-color: rgb(221,0,0)")

        self.ui.btn_displayResults.clicked.connect(self.displayResults)

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

        self.ui.lineEdit_numIterations.setText(toQString("100"))
        self.ui.lineEdit_numIterations.textChanged.connect(self.setNumberOfIterations)
        self.ui.lineEdit_numIterations.textChanged.connect(self.checkParameters)

        self.ui.lineEdit_oversamplingRatio.setText(toQString("3"))
        self.ui.lineEdit_oversamplingRatio.textChanged.connect(self.GENFIRE_ReconstructionParameters.setOversamplingRatio)
        self.ui.lineEdit_oversamplingRatio.textChanged.connect(self.checkParameters)

        self.ui.lineEdit_interpolationCutoffDistance.setText(toQString("0.7"))
        self.ui.lineEdit_interpolationCutoffDistance.textChanged.connect(self.GENFIRE_ReconstructionParameters.setInterpolationCutoffDistance)
        self.ui.lineEdit_interpolationCutoffDistance.textChanged.connect(self.checkParameters)

        self.ui.lineEdit_io.setDisabled(True)
        self.ui.lineEdit_io.setStyleSheet("background-color: gray")

        ## Radio Buttons -- default is resolution extension suppression
        self.ui.radioButton_on.setChecked(True)
        self.ui.radioButton_on.toggled.connect(self.selectResolutionExtensionSuppressionState)

        self.ui.radioButton_off.toggled.connect(self.selectResolutionExtensionSuppressionState)

        self.ui.radioButton_extension.toggled.connect(self.selectResolutionExtensionSuppressionState)

        self.ui.radioButton_FFT.setChecked(True)
        self.ui.radioButton_DFT.toggled.connect(self.setGriddingMethod)
        self.ui.radioButton_FFT.toggled.connect(self.setGriddingMethod)

        ## Check Boxes
        self.ui.checkBox_rfree.setChecked(True)
        self.ui.checkBox_rfree.toggled.connect(self.calculateRfree)

        self.ui.checkBox_provide_io.toggled.connect(self.toggleSelectIO)

        self.ui.checkBox_default_support.toggled.connect(self.toggleUseDefaultSupport)
        self.ui.checkBox_default_support.setChecked(True)

        self.ui.checkBox_support_constraint.setChecked(True)
        self.ui.checkBox_positivity_constraint.setChecked(True)
        self.ui.checkBox_support_constraint.toggled.connect(self.toggleSupportConstraint)
        self.ui.checkBox_positivity_constraint.toggled.connect(self.togglePositivityConstraint)
        self.ui.checkBox_multiGridding.toggled.connect(self.toggleMultiGridding)
        self.ui.checkBox_resCircle.toggled.connect(self.toggleResCircle)

        self.ui.action_Create_Support.triggered.connect(self.launchProjectionCalculator)

        self.ui.action_Volume_Slicer.triggered.connect(self.launchVolumeSlicer)


    def setNumberOfIterations(self, value):
        number_of_iterations = toInt(value)
        self.GENFIRE_ReconstructionParameters.setNumberOfIterations(number_of_iterations)

    def calculateRfree(self):
        if self.ui.checkBox_rfree.isChecked() == True:
            self.GENFIRE_ReconstructionParameters.calculateRfree = True
        else:
            self.GENFIRE_ReconstructionParameters.calculateRfree = False

    def launchProjectionCalculator(self):
        from functools import partial
        self.GENFIRE_ProjectionCalculator = ProjectionCalculator.ProjectionCalculator(self)
        self.GENFIRE_ProjectionCalculator.model_loading_signal.connect(partial(self.receive_msg, "Loading Model..."))
        self.GENFIRE_ProjectionCalculator.emit_message_signal.connect(self.receive_msg)
        self.GENFIRE_ProjectionCalculator.update_filenames_signal.connect(self.updateFilenames)
        self.GENFIRE_ProjectionCalculator.show()

    def setGriddingMethod(self):
        if self.ui.radioButton_DFT.isChecked():
            self.GENFIRE_ReconstructionParameters.griddingMethod = "DFT"
        else:
            self.GENFIRE_ReconstructionParameters.griddingMethod = "FFT"

    def updateFilenames(self):
        import os
        base, file = os.path.split(toString(self.GENFIRE_ProjectionCalculator.calculationParameters.outputFilename))

        self.ui.lineEdit_angle.setText(toQString(str(self.GENFIRE_ProjectionCalculator.calculationParameters.outputAngleFilename)))
        self.ui.lineEdit_angle.textChanged.emit(self.ui.lineEdit_angle.text())
        self.ui.lineEdit_pj.setText(toQString(str(self.GENFIRE_ProjectionCalculator.calculationParameters.outputFilename)))
        self.ui.lineEdit_pj.textChanged.emit(self.ui.lineEdit_pj.text())
        self.ui.lineEdit_results.setText(toQString(str(base + "/results.mrc")))
        self.ui.lineEdit_results.textChanged.emit(self.ui.lineEdit_results.text())

    def launchVolumeSlicer(self):
        import genfire.fileio
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(QtWidgets.QFileDialog(), "Select Volume",filter="Volume files (*.mat *.mrc *.npy);;All Files (*)")
        if filename:
            filename = toString(filename)
            volume = genfire.fileio.readVolume(filename)
            self.VolumeSlicer = VolumeSlicer.VolumeSlicer(volume)
            self.VolumeSlicer.show()

    def toggleResCircle(self):
        if self.ui.checkBox_resCircle.isChecked():
            self.GENFIRE_ReconstructionParameters.enforceResolutionCircle = True
        else:
            self.GENFIRE_ReconstructionParameters.enforceResolutionCircle = False

    def toggleMultiGridding(self):
        if self.ui.checkBox_multiGridding.isChecked():
            self.GENFIRE_ReconstructionParameters.permitMultipleGridding = True
        else:
            self.GENFIRE_ReconstructionParameters.permitMultipleGridding = False

    def toggleSelectIO(self):
        if self.ui.lineEdit_io.isEnabled():
             self.ui.lineEdit_io.setStyleSheet("background-color: gray")
             self.ui.lineEdit_io.setEnabled(False)
             self.GENFIRE_ReconstructionParameters.initialObjectFilename= ""
        else:
             self.ui.lineEdit_io.setStyleSheet("background-color: white")
             self.ui.lineEdit_io.setEnabled(True)

    def toggleSupportConstraint(self):
        if self.ui.checkBox_support_constraint.isChecked():
            self.GENFIRE_ReconstructionParameters.constraint_support = True
        else:
            self.GENFIRE_ReconstructionParameters.constraint_support = False

    def togglePositivityConstraint(self):
        if self.ui.checkBox_positivity_constraint.isChecked():
            self.GENFIRE_ReconstructionParameters.constraint_positivity = True
        else:
            self.GENFIRE_ReconstructionParameters.constraint_positivity = False

    def toggleUseDefaultSupport(self):
        if self.ui.checkBox_default_support.isChecked():
            self.GENFIRE_ReconstructionParameters.useDefaultSupport = True
            self.ui.lineEdit_support.setDisabled(True)
            self.ui.lineEdit_support.setStyleSheet("background-color: gray")
        else:
            self.GENFIRE_ReconstructionParameters.useDefaultSupport = False
            self.ui.lineEdit_support.setDisabled(False)
            self.ui.lineEdit_support.setStyleSheet("background-color: white")

    def messageWritten(self, message):
        print(message)

    def __del__(self):
        import sys
        sys.stdout = sys.__stdout__ # Restore output stream to default upon exiting the GUI
        sys.stderr = sys.__stderr__

    #Functions for selecting input files using QFileDialog
    def selectProjectionFile(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(QtWidgets.QFileDialog(), "Select File Containing Projections",filter="Projection Stacks (*.mrc *.mat *.tif *.npy);; MATLAB files (*.mat);;TIFF images (*.tif *.tiff);;MRC (*.mrc);;All Files (*)")

        if filename:
            self.GENFIRE_ReconstructionParameters.setProjectionFilename(filename)
            self.ui.lineEdit_pj.setText(toQString(filename))

    def selectAngleFile(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(QtWidgets.QFileDialog(), "Select File Containing Support",filter="Euler Angles (*.txt *.mat);; MATLAB files (*.mat);;text files (*.txt);;All Files (*)")
        if filename:
            self.GENFIRE_ReconstructionParameters.setAngleFilename(filename)
            self.ui.lineEdit_angle.setText(toQString(filename))

    def selectSupportFile(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(QtWidgets.QFileDialog(), "Select File Containing Support", filter="Volume Files (*.mrc *.mat *.npy);; MATLAB files (*.mat);;MRC (*.mrc);;All Files (*)")
        if filename:
            self.GENFIRE_ReconstructionParameters.setSupportFilename(filename)
            self.ui.lineEdit_support.setText(toQString(filename))
            self.ui.checkBox_default_support.setChecked(False)

    def selectInitialObjectFile(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(QtWidgets.QFileDialog(), "Select File Containing Initial Object", filter="Volume Files (*.mrc *.mat *.npy);; MATLAB files (*.mat);;MRC (*.mrc);;All Files (*)")
        if filename:
            self.GENFIRE_ReconstructionParameters.setInitialObjectFilename(filename)
            self.ui.lineEdit_io.setText(toQString(filename))

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
            self.ui.btn_reconstruct.setEnabled(True)
            self.ui.btn_reconstruct.setStyleSheet("background-color: GREEN; color:#ffffff;font-size:30px")
            # GF_window.ui.btn_reconstruct.setStyleSheet("color: WHITE")
            self.ui.btn_reconstruct.setText("Launch Reconstruction!")

    #Function to run GENFIRE reconstruction once all parameters are accounted for

    @QtCore.pyqtSlot()
    def startReconstruction(self):
        print('Launching GENFIRE reconstruction')
        # Launch the reconstruction in a separate thread to prevent the GUI blocking while reconstructing
        from threading import Thread
        from functools import partial
        import genfire.main
        t = Thread(target=partial(genfire.main.main, self.GENFIRE_ReconstructionParameters))
        t.start()
    def displayResults(self):
        outputfilename, _ = QtWidgets.QFileDialog.getOpenFileName(QtWidgets.QFileDialog(), "Select Reconstruction",filter="Volume files (*.mrc *.mat *.npy)  ;; MATLAB files (*.mat);;text files (*.txt *.tiff);;MRC (*.mrc);;All Files (*)")
        outputfilename = toString(outputfilename)
        if outputfilename:
            import numpy as np
            import os
            import genfire.fileio
            import matplotlib.pyplot as plt
            plt.close('all')

            initialObject = genfire.fileio.readVolume(outputfilename)
            dims = np.shape(initialObject)
            n_half_x = int(dims[0]/2) #this assumes even-sized arrays
            n_half_y = int(dims[1]/2)
            n_half_z = int(dims[2]/2)
            reconstructionDisplayWindowSize=dims[0] # array should be cubic
            half_window_x = reconstructionDisplayWindowSize//2
            half_window_y = reconstructionDisplayWindowSize//2
            half_window_z = reconstructionDisplayWindowSize//2
            plt.figure()
            plt.subplot(233)
            plt.imshow(np.squeeze(initialObject[n_half_x, n_half_y-half_window_y:n_half_y+half_window_y, n_half_z-half_window_z:n_half_z+half_window_z]))
            plt.title("central YZ slice")

            plt.subplot(232)
            plt.imshow(np.squeeze(initialObject[n_half_x-half_window_x:n_half_x+half_window_x, n_half_y, n_half_z-half_window_z:n_half_z+half_window_z]))
            plt.title("central XZ slice")

            plt.subplot(231)
            plt.title("central XY slice")
            plt.imshow(np.squeeze(initialObject[n_half_x-half_window_x:n_half_x+half_window_x, n_half_y-half_window_y:n_half_y+half_window_y, n_half_z]))

            plt.subplot(236)
            plt.title("YZ projection")
            plt.imshow(np.squeeze(np.sum(initialObject[n_half_x-half_window_x:n_half_x+half_window_x, n_half_y-half_window_y:n_half_y+half_window_y, n_half_z-half_window_z:n_half_z+half_window_z], axis=0)))

            plt.subplot(235)
            plt.title("XZ projection")
            plt.imshow(np.squeeze(np.sum(initialObject[n_half_x-half_window_x:n_half_x+half_window_x, n_half_y-half_window_y:n_half_y+half_window_y, n_half_z-half_window_z:n_half_z+half_window_z], axis=1)))

            plt.subplot(234)
            plt.title("XY projection")
            plt.imshow(np.squeeze(np.sum(initialObject[n_half_x-half_window_x:n_half_x+half_window_x, n_half_y-half_window_y:n_half_y+half_window_y, n_half_z-half_window_z:n_half_z+half_window_z], axis=2)))
            plt.get_current_fig_manager().window.setGeometry(25,25,600, 1200)

            # now load error curves if they exist
            output_base, output_ext = os.path.splitext(outputfilename)
            errK_filename = output_base + '_errK.txt'
            if os.path.isfile(errK_filename):
                errK = np.loadtxt(errK_filename)
                numIterations = np.shape(errK)[0]
                plt.figure()
                plt.plot(range(0,numIterations),errK)

                mngr = plt.get_current_fig_manager()
                mngr.window.setGeometry(700,25,550, 250)
                plt.title("Reciprocal Error vs Iteration Number")
                plt.xlabel("Iteration Num")
                plt.ylabel("Reciprocal Error")
                # plt.setp(plt.gcf(),)
            Rfree_total_filename = output_base + '_Rfree_total.txt'
            Rfree_bybin_filename = output_base + '_Rfree_bybin.txt'
            if os.path.isfile(Rfree_total_filename):
                Rfree_total = np.loadtxt(Rfree_total_filename)
                numIterations = np.shape(Rfree_total)[0]
                plt.figure()
                plt.title("R-free vs Iteration Number")
                mngr = plt.get_current_fig_manager()
                mngr.window.setGeometry(700,350,550, 250)
                plt.plot(range(0,numIterations),Rfree_total)
                plt.title("Mean R-free Value vs Iteration Number")
                plt.xlabel("Iteration Num")
                plt.ylabel('Mean R-free')

            if os.path.isfile(Rfree_bybin_filename):
                Rfree_bybin = np.loadtxt(Rfree_bybin_filename)
                plt.figure()
                mngr = plt.get_current_fig_manager()
                mngr.window.setGeometry(700,650,550, 250)
                X = np.linspace(0,1,np.shape(Rfree_bybin)[0])
                plt.plot(X, Rfree_bybin[:,-1])
                plt.title("Final Rfree Value vs Spatial Frequency")
                plt.xlabel("Spatial Frequency (% of Nyquist)")
                plt.ylabel('Rfree')

            plt.draw()
            plt.pause(1e-30) # slight pause forces redraw

    @QtCore.pyqtSlot(str)
    def receive_msg(self, msg):
        global process_finished
        if not process_finished:
            self.ui.log.moveCursor(QtGui.QTextCursor.End)
            formatted_msg = "<span style=\" font-size:14pt; font-weight:600; color:#000000;\" >" + msg + "</span/>"
            self.ui.log.append(formatted_msg)

    @QtCore.pyqtSlot(str)
    def receive_error_msg(self, msg):
        global process_finished
        if not process_finished:
            self.ui.log.moveCursor(QtGui.QTextCursor.End)
            formatted_msg = "<span style=\" font-size:14pt; font-weight:600; color:#ff0000;\" >" + msg + "</span/>"
            self.ui.log.append(formatted_msg)

    @QtCore.pyqtSlot()
    def stopRunning(self):
        global process_finished
        process_finished = True

class GenfireListener(QtCore.QObject):
    message_pending = QtCore.pyqtSignal(str)
    def __init__(self, msg_queue):
        super(GenfireListener, self).__init__()
        self.msg_queue = msg_queue
        self.process_finished = False

    @QtCore.pyqtSlot()
    def run(self):
        msg = ''
        while not self.process_finished:
            msg = self.msg_queue.get() #get next message, blocks if nothing to get
            if self.process_finished:
                self.message_pending.emit(msg)
                return
            self.message_pending.emit(msg)

    @QtCore.pyqtSlot()
    def stopRunning(self):
        self.process_finished = True

class GenfireWriter(object):
    def __init__(self, msg_queue):
        self.msg_queue = msg_queue

    def write(self, message):
        if message != "\n":
            self.msg_queue.put("GENFIRE: " + message)

class GenfireLogger(QtCore.QObject):
    def __init__(self, msg_queue):
        super(GenfireLogger, self).__init__()
        self.msg_queue = msg_queue
        self.listener  = GenfireListener(msg_queue=self.msg_queue)
        self.listener_thread = QtCore.QThread()
        self.listener.moveToThread(self.listener_thread)
        self.listener_thread.started.connect(self.listener.run)
        self.listener_thread.start()

    @QtCore.pyqtSlot()
    def cleanup_thread(self):
        global process_finished
        self.listener.process_finished = True
        self.msg_queue.put("Safely Exit.") # write a final message to force i/o threads to unblock and see the exit flag
        if self.listener_thread.isRunning():
            self.listener_thread.quit()
            self.listener_thread.wait()

def main():

    # Startup the application
    app = QtWidgets.QApplication(sys.argv)
    # app.setStyle('plastique')
    app.setStyle('mac')

    # Create the GUI
    GF_window  = GenfireMainWindow()

    # Render GUI
    GF_window.show()

    # Redirect standard output to the GUI
    if sys.version_info >= (3,0):
        from multiprocessing import Queue
    else:
        from Queue import Queue
    global process_finished
    global GF_logger
    global GF_error_logger
    process_finished = False # flag to control save exit of the i/o threads for logging

    msg_queue  = Queue()
    GF_logger  = GenfireLogger(msg_queue)
    sys.stdout = GenfireWriter(msg_queue)
    GF_logger.listener.message_pending[str].connect(GF_window.receive_msg)

    err_msg_queue = Queue()
    GF_error_logger  = GenfireLogger(err_msg_queue)
    sys.stderr = GenfireWriter(err_msg_queue)
    GF_error_logger.listener.message_pending[str].connect(GF_window.receive_error_msg)

    GF_window.stop_threads.connect(GF_error_logger.cleanup_thread)
    GF_window.stop_threads.connect(GF_logger.cleanup_thread)

    GF_window.stop_threads.connect(GF_error_logger.listener.stopRunning)
    GF_window.stop_threads.connect(GF_logger.listener.stopRunning)

    # Start event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


