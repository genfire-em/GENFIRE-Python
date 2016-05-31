from PyQt4 import QtCore, QtGui

import GENFIRE_GUI
import GENFIRE
import GENFIRE_from_GUI_Input as GENFIRE_Reconstruction
import os
class reconstructionParameters():
    def __init__(self):
        self._projectionFilename = ""
        self._angleFilename = ""
        self._supportFilename = ""
        self._supportedFiletypes = ['.tif','.mrc','.mat']

    def checkParameters(self):
        parametersAreGood = 1

        projection_extension = os.path.splitext(self._projectionFilename)
        if projection_extension not in self._supportedFiletypes:
            parametersAreGood = 0

        angle_extension = os.path.splitext(self._angleFilename)
        if angle_extension not in self._supportedFiletypes:
            parametersAreGood = 0

        support_extension = os.path.splitext(self._supportFilename)
        if support_extension not in self._supportedFiletypes:
            parametersAreGood = 0

        if self.




    def setProjectionFilename(self,projectionFilename):
        self._projectionFilename = projectionFilename
        print("projection filename set to", self._projectionFilename )

    def getProjectionFilename(self):
        return self._projectionFilename

    def setAngleFilename(self,angleFilename):
        self._angleFilename = angleFilename

    def getAngleFilename(self):
        return self._angleFilename

    def setSupportFilename(self,supportFilename):
        self._supportFilename = supportFilename

    def getSupportFilename(self):
        return self._supportFilename






def selectProjectionFile():
    filename = QtGui.QFileDialog.getOpenFileName(dialog, "Select Projection File")
    # print("Projection Filename: ",filename)
    GENFIRE_reconstructionParameters.setProjectionFilename(unicode(filename.toUtf8(), encoding='UTF-8'))

def selectAngleFile():
    filename = QtGui.QFileDialog.getOpenFileName(dialog, "Select Angle File")
    # print("Projection Filename: ",filename)
    GENFIRE_reconstructionParameters.setAngleFilename(unicode(filename.toUtf8(), encoding='UTF-8'))

def selectSupportFile():
    filename = QtGui.QFileDialog.getOpenFileName(dialog, "Select Support File")
    # print("Projection Filename: ",filename)
    GENFIRE_reconstructionParameters.setSupportFilename(unicode(filename.toUtf8(), encoding='UTF-8'))


def listPars():
    print("marker", GENFIRE_reconstructionParameters._projectionFilename)
    print(GENFIRE_reconstructionParameters._angleFilename)
    print(GENFIRE_reconstructionParameters._supportFilename)
    GENFIRE_Reconstruction.GENFIRE_from_GUI_Input(GENFIRE_reconstructionParameters.getProjectionFilename()
                                   ,GENFIRE_reconstructionParameters.getAngleFilename()
                                   ,GENFIRE_reconstructionParameters.getSupportFilename())

def myPrint(texttt):
          print(texttt)

def setText(a):
    print "triggered"
    print a

class myCustomButton(QtGui.QPushButton):
    mySignal = QtCore.pyqtSignal('QString')
    def __init__(self,text):
        super(myCustomButton,self).__init__(text)
        print "made the class"
        self.mySignal.connect(setText)
        self.clicked.connect(self.emitMySignal)
    def emitMySignal(self):
        a = raw_input("enter something: ")
        a = QtCore.QString(a)
        self.mySignal.emit(a)

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    dialog = QtGui.QFileDialog()
    window = QtGui.QMainWindow()
    GF = GENFIRE_GUI.Ui_MainWindow()
    GF.setupUi(window)

    GENFIRE_reconstructionParameters = reconstructionParameters()

    #connect buttons to setters for filenames
    GF.btn_projections.clicked.connect(selectProjectionFile)
    GF.btn_angles.clicked.connect(selectAngleFile)
    GF.btn_support.clicked.connect(selectSupportFile)
    GF.btn_reconstruct.clicked.connect(listPars)

    # GF.lineEdit.setText(QtCore.QString("bla"))
    GF.lineEdit.textEdited.connect(myPrint)

    # window.pushBtn3 = QtGui.QPushButton('Bla')
    # window.pushBtn3.setText(QtCore.QString("bla"))
    # window.pushBtn3.move(300,300)
    # window.pushBtn3.resize(300,300)
    # window.pushBtn3.show()
    # GF.pushBtn3 = QtGui.QPushButton('Bla',GF.verticalLayoutWidget)
    GF.pushBtn3 = myCustomButton('Bla')
    # GF.pushBtn3.setText(QtCore.QString("bla"))
    GF.pushBtn3.move(1,1)
    GF.pushBtn3.resize(300,300)
    # GF.pushBtn3.show()
    GF.verticalLayout.addWidget(GF.pushBtn3)
    # GF.pushBtn3.show()

    window.move(25,25)
    window.resize(700,500)
    window.show()
    sys.exit(app.exec_())





