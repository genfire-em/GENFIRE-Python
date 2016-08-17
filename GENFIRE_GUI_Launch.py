from PyQt4 import QtCore, QtGui
import matplotlib
matplotlib.use("Qt4Agg")
import GENFIRE_GUI
import ProjectionCalculator
import volume_slicer
import GENFIRE_main
import os
import sys
from GENFIRE import ReconstructionParameters


class GenfireMainWindow(QtGui.QMainWindow):
    def __init__(self):

        ## Superclass constructor
        super(GenfireMainWindow,self).__init__()

        ## Initialize UI
        self.ui = GENFIRE_GUI.Ui_GENFIRE_MainWindow()
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

        self.ui.lineEdit_numIterations.setText(QtCore.QString("50"))
        self.ui.lineEdit_numIterations.textChanged.connect(self.GENFIRE_ReconstructionParameters.setNumberOfIterations)
        self.ui.lineEdit_numIterations.textChanged.connect(self.checkParameters)

        self.ui.lineEdit_oversamplingRatio.setText(QtCore.QString("3"))
        self.ui.lineEdit_oversamplingRatio.textChanged.connect(self.GENFIRE_ReconstructionParameters.setOversamplingRatio)
        self.ui.lineEdit_oversamplingRatio.textChanged.connect(self.checkParameters)

        self.ui.lineEdit_interpolationCutoffDistance.setText(QtCore.QString("0.7"))
        self.ui.lineEdit_interpolationCutoffDistance.textChanged.connect(self.GENFIRE_ReconstructionParameters.setInterpolationCutoffDistance)
        self.ui.lineEdit_interpolationCutoffDistance.textChanged.connect(self.checkParameters)


        # self.ui.lineEdit_displayFrequency.setDisabled(True)
        # self.ui.lineEdit_displayFrequency.setStyleSheet("background-color: gray")
        # self.ui.lineEdit_displayFrequency.textChanged.connect(self.ShoutDisplayFrequency)

        self.ui.lineEdit_io.setDisabled(True)
        self.ui.lineEdit_io.setStyleSheet("background-color: gray")


        ## Radio Buttons -- default is resolution extension suppression
        self.ui.radioButton_on.setChecked(True)
        self.ui.radioButton_on.toggled.connect(self.selectResolutionExtensionSuppressionState)

        self.ui.radioButton_off.toggled.connect(self.selectResolutionExtensionSuppressionState)

        self.ui.radioButton_extension.toggled.connect(self.selectResolutionExtensionSuppressionState)


        ## Check Boxes

        # self.ui.checkBox_displayFigure.toggled.connect(self.GENFIRE_ReconstructionParameters.toggleDisplayFigure)
        # self.ui.checkBox_displayFigure.toggled.connect(self.enableDisplayFrequencyChange)


        self.ui.checkBox_rfree.toggled.connect(self.calculateRfree)

        self.ui.checkBox_provide_io.toggled.connect(self.toggleSelectIO)

        self.ui.checkBox_default_support.toggled.connect(self.toggleUseDefaultSupport)

        self.ui.action_Create_Support.triggered.connect(self.launchProjectionCalculator)

        self.ui.action_Volume_Slicer.triggered.connect(self.launchVolumeSlicer)

    def calculateRfree(self):
        if self.ui.checkBox_rfree.isEnabled() == True:
            self.GENFIRE_ReconstructionParameters.calculateRfree = True
        else:
            self.GENFIRE_ReconstructionParameters.calculateRfree = False

    def launchProjectionCalculator(self):
        print "launching"
        self.GENFIRE_ProjectionCalculator = ProjectionCalculator.ProjectionCalculator()
        self.GENFIRE_ProjectionCalculator.show()

    def launchVolumeSlicer(self):
        import os
        import GENFIRE_io
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select Volume",filter="Volume files (*.mat *.mrc);;All Files (*)")
        filename = unicode(filename.toUtf8(), encoding='UTF-8')
        volume = GENFIRE_io.loadVolume(filename)
        self.VolumeSlicer = volume_slicer.VolumeSlicer(volume)
        self.VolumeSlicer.show()

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
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Projections",filter="Projection Stacks (*.mrc *.mat *.tif *.npy);; MATLAB files (*.mat);;TIFF images (*.tif *.tiff);;MRC (*.mrc);;All Files (*)")

        if filename:
            self.GENFIRE_ReconstructionParameters.setProjectionFilename(filename)
            print ("Projection Filename:", self.GENFIRE_ReconstructionParameters.getProjectionFilename())
            self.ui.lineEdit_pj.setText(QtCore.QString(filename))

    def selectAngleFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Support",filter="Euler Angles (*.txt *.mat);; MATLAB files (*.mat);;text files (*.txt);;All Files (*)")
        if filename:
            self.GENFIRE_ReconstructionParameters.setAngleFilename(filename)
            print ("Angle Filename:", self.GENFIRE_ReconstructionParameters.getAngleFilename())
            self.ui.lineEdit_angle.setText(QtCore.QString(filename))

    def selectSupportFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Support", filter="Volume Files (*.mrc *.mat *.npy);; MATLAB files (*.mat);;MRC (*.mrc);;All Files (*)")
        if filename:
            self.GENFIRE_ReconstructionParameters.setSupportFilename(filename)
            print ("Support Filename:", self.GENFIRE_ReconstructionParameters.getSupportFilename())
            self.ui.lineEdit_support.setText(QtCore.QString(filename))

    def selectInitialObjectFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select File Containing Initial Object", filter="Volume Files (*.mrc *.mat *.npy);; MATLAB files (*.mat);;MRC (*.mrc);;All Files (*)")
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
            GF_window.ui.btn_reconstruct.setStyleSheet("background-color: GREEN; color:#ffffff;font-size:30px")
            # GF_window.ui.btn_reconstruct.setStyleSheet("color: WHITE")
            GF_window.ui.btn_reconstruct.setText("Launch Reconstruction!")

    #Function to run GENFIRE reconstruction once all parameters are accounted for

    @QtCore.pyqtSlot()
    def startReconstruction(self):
        print('Launching GENFIRE Reconstruction')
        # Launch the reconstruction in a separate thread to prevent the GUI blocking while reconstructing
        from threading import Thread
        from functools import partial
        t = Thread(target=partial(GENFIRE_main.GENFIRE_main,self.GENFIRE_ReconstructionParameters))
        t.start()

    def displayResults(self):
        outputfilename = QtGui.QFileDialog.getOpenFileName(QtGui.QFileDialog(), "Select Reconstruction",filter="Volume files (*.mrc *.mat *.npy)  ;; MATLAB files (*.mat);;text files (*.txt *.tiff);;MRC (*.mrc);;All Files (*)")
        outputfilename = unicode(outputfilename.toUtf8(), encoding='UTF-8')

        if outputfilename:
            import numpy as np
            import os
            import GENFIRE_io

            initialObject = GENFIRE_io.loadVolume(outputfilename)
            # initialObject = readMRC("outputfilename")
            dims = np.shape(initialObject)
            n_half_x = int(dims[0]/2) #this assumes even-sized arrays
            n_half_y = int(dims[1]/2)
            n_half_z = int(dims[2]/2)
            reconstructionDisplayWindowSize=dims[0] # array should be cubic
            half_window_x = reconstructionDisplayWindowSize//2
            half_window_y = reconstructionDisplayWindowSize//2
            half_window_z = reconstructionDisplayWindowSize//2
            import matplotlib.pyplot as plt
            import numpy as np
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
            Rfree_filename = output_base + '_Rfree.txt'

            if os.path.isfile(Rfree_filename):
                Rfree = np.loadtxt(Rfree_filename)
                numIterations = np.shape(Rfree)[1]
                plt.figure()
                plt.title("R-free vs Iteration Number")
                mngr = plt.get_current_fig_manager()
                mngr.window.setGeometry(700,350,550, 250)
                plt.plot(range(0,numIterations),np.mean(Rfree,axis=0))
                plt.title("Mean R-free Value vs Iteration Number")
                plt.xlabel("Iteration Num")
                plt.ylabel('Mean R-free')


                plt.figure(4)
                mngr = plt.get_current_fig_manager()
                mngr.window.setGeometry(700,650,550, 250)
                plt.hold(False)
                X = np.linspace(0,1,np.shape(Rfree)[0])
                plt.plot(X, Rfree[:,-1])
                plt.hold(False)
                plt.title("Final Rfree Value vs Spatial Frequency")
                plt.xlabel("Spatial Frequency (% of Nyquist)")
                plt.ylabel('Rfree')

            plt.draw()
            plt.pause(1e-30)

#
    @QtCore.pyqtSlot(str)
    def receive_msg(self, msg):
        global process_finished
        if not process_finished:
            self.ui.log.moveCursor(QtGui.QTextCursor.End)
            # self.ui.log.setStyleSheet("color: black")
            formatted_msg = "<span style=\" font-size:11pt; font-weight:600; color:#000000;\" >" + msg + "</span/>"
            # self.ui.log.insertPlainText(msg)

            self.ui.log.append(formatted_msg)


    @QtCore.pyqtSlot(str)
    def receive_error_msg(self, msg):
        global process_finished
        if not process_finished:
            self.ui.log.moveCursor(QtGui.QTextCursor.End)
            formatted_msg = "<span style=\" font-size:11pt; font-weight:600; color:#ff0000;\" >" + msg + "</span/>"
            # self.ui.log.setStyleSheet("color: red")

            self.ui.log.append(formatted_msg)
            # self.ui.log.insertPlainText(msg)

class Launcher(QtCore.QObject):
    def __init__(self, pars):
        super(Launcher,self).__init__()
        self.pars = pars

    @QtCore.pyqtSlot()
    def run(self):
        GENFIRE_main.GENFIRE_main(self.pars)

class GenfireListener(QtCore.QObject):
    message_pending = QtCore.pyqtSignal(str)
    def __init__(self, msg_queue):
        super(GenfireListener, self).__init__()
        self.msg_queue = msg_queue
        self.process_finished = False

    @QtCore.pyqtSlot()
    def run(self):
        while not self.process_finished:
            msg = self.msg_queue.get() #get next message, blocks if nothing to get
            if process_finished:
                self.message_pending.emit(msg)
                return
            # else:
            #     print ("failure")
            self.message_pending.emit(msg)

class GenfireWriter(object):
    def __init__(self, msg_queue):
        self.msg_queue = msg_queue

    def write(self, message):
        # self.msg_queue.put("GENFIRE: " + message)
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
        # QtCore.QCoreApplication.instance().aboutToQuit.connect(self.cleanup_thread)
        self.listener_thread.start()

    @QtCore.pyqtSlot()
    def cleanup_thread(self):
        import sys
        self.listener.process_finished = True
        process_finished = True
        self.msg_queue.put("Safely Exit.") # write a final message to force i/o threads to unblock and see the exit flag
        # self.listener_thread.wait()
        if self.listener_thread.isRunning():
            self.listener_thread.quit()
            self.listener_thread.wait()


if __name__ == "__main__":

    # Startup the application
    app = QtGui.QApplication(sys.argv)
    # app.setStyle('plastique')
    app.setStyle('mac')

    # Create the GUI
    GF_window  = GenfireMainWindow()

    # Render GUI
    GF_window.show()

    # Redirect standard output to the GUI
    from Queue import Queue
    global process_finished
    process_finished = False # flag to control save exit of the i/o threads for logging

    msg_queue  = Queue()
    GF_logger  = GenfireLogger(msg_queue)
    sys.stdout = GenfireWriter(msg_queue)
    GF_logger.listener.message_pending[str].connect(GF_window.receive_msg)
    app.aboutToQuit.connect(GF_logger.cleanup_thread)

    err_msg_queue = Queue()
    GF_error_logger  = GenfireLogger(err_msg_queue)
    sys.stderr = GenfireWriter(err_msg_queue)
    GF_error_logger.listener.message_pending[str].connect(GF_window.receive_error_msg)
    app.aboutToQuit.connect(GF_error_logger.cleanup_thread)

    # app.aboutToQuit.connect(GF_logger.cleanup_threads)

    # Safely close and exit

    sys.exit(app.exec_())

