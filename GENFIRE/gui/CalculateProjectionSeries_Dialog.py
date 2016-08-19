# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'calculateprojectionseries_dialog.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_CalculateProjectionSeries_Dialog(object):
    def setupUi(self, CalculateProjectionSeries_Dialog):
        CalculateProjectionSeries_Dialog.setObjectName(_fromUtf8("CalculateProjectionSeries_Dialog"))
        CalculateProjectionSeries_Dialog.resize(764, 255)
        self.buttonBox = QtGui.QDialogButtonBox(CalculateProjectionSeries_Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(6, 219, 164, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.label_7 = QtGui.QLabel(CalculateProjectionSeries_Dialog)
        self.label_7.setGeometry(QtCore.QRect(10, 120, 59, 16))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.lineEdit_phi = QtGui.QLineEdit(CalculateProjectionSeries_Dialog)
        self.lineEdit_phi.setGeometry(QtCore.QRect(50, 120, 113, 21))
        self.lineEdit_phi.setObjectName(_fromUtf8("lineEdit_phi"))
        self.lineEdit_psi = QtGui.QLineEdit(CalculateProjectionSeries_Dialog)
        self.lineEdit_psi.setGeometry(QtCore.QRect(70, 150, 113, 21))
        self.lineEdit_psi.setObjectName(_fromUtf8("lineEdit_psi"))
        self.label_8 = QtGui.QLabel(CalculateProjectionSeries_Dialog)
        self.label_8.setGeometry(QtCore.QRect(10, 160, 59, 16))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.line = QtGui.QFrame(CalculateProjectionSeries_Dialog)
        self.line.setGeometry(QtCore.QRect(92, 46, 500, 16))
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.lineEdit_angleFile = QtGui.QLineEdit(CalculateProjectionSeries_Dialog)
        self.lineEdit_angleFile.setGeometry(QtCore.QRect(92, 14, 500, 21))
        self.lineEdit_angleFile.setMinimumSize(QtCore.QSize(500, 0))
        self.lineEdit_angleFile.setObjectName(_fromUtf8("lineEdit_angleFile"))
        self.lineEdit_outputFilename = QtGui.QLineEdit(CalculateProjectionSeries_Dialog)
        self.lineEdit_outputFilename.setGeometry(QtCore.QRect(110, 200, 125, 21))
        self.lineEdit_outputFilename.setObjectName(_fromUtf8("lineEdit_outputFilename"))
        self.label_3 = QtGui.QLabel(CalculateProjectionSeries_Dialog)
        self.label_3.setGeometry(QtCore.QRect(14, 14, 66, 64))
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.label_2 = QtGui.QLabel(CalculateProjectionSeries_Dialog)
        self.label_2.setGeometry(QtCore.QRect(14, 86, 59, 16))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label = QtGui.QLabel(CalculateProjectionSeries_Dialog)
        self.label.setGeometry(QtCore.QRect(10, 190, 66, 53))
        self.label.setWordWrap(True)
        self.label.setObjectName(_fromUtf8("label"))
        self.label_5 = QtGui.QLabel(CalculateProjectionSeries_Dialog)
        self.label_5.setGeometry(QtCore.QRect(132, 62, 28, 16))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.label_4 = QtGui.QLabel(CalculateProjectionSeries_Dialog)
        self.label_4.setGeometry(QtCore.QRect(168, 62, 29, 16))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.label_6 = QtGui.QLabel(CalculateProjectionSeries_Dialog)
        self.label_6.setGeometry(QtCore.QRect(94, 62, 30, 16))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.lineEdit_thetaStep = QtGui.QLineEdit(CalculateProjectionSeries_Dialog)
        self.lineEdit_thetaStep.setGeometry(QtCore.QRect(229, 90, 125, 21))
        self.lineEdit_thetaStep.setObjectName(_fromUtf8("lineEdit_thetaStep"))
        self.lineEdit_thetaStop = QtGui.QLineEdit(CalculateProjectionSeries_Dialog)
        self.lineEdit_thetaStop.setGeometry(QtCore.QRect(364, 90, 125, 21))
        self.lineEdit_thetaStop.setObjectName(_fromUtf8("lineEdit_thetaStop"))
        self.lineEdit_thetaStart = QtGui.QLineEdit(CalculateProjectionSeries_Dialog)
        self.lineEdit_thetaStart.setGeometry(QtCore.QRect(94, 90, 125, 21))
        self.lineEdit_thetaStart.setObjectName(_fromUtf8("lineEdit_thetaStart"))
        self.btn_outputFilename = QtGui.QPushButton(CalculateProjectionSeries_Dialog)
        self.btn_outputFilename.setGeometry(QtCore.QRect(380, 210, 148, 32))
        self.btn_outputFilename.setObjectName(_fromUtf8("btn_outputFilename"))
        self.btn_selectAngleFile = QtGui.QPushButton(CalculateProjectionSeries_Dialog)
        self.btn_selectAngleFile.setGeometry(QtCore.QRect(603, 13, 87, 32))
        self.btn_selectAngleFile.setObjectName(_fromUtf8("btn_selectAngleFile"))

        self.retranslateUi(CalculateProjectionSeries_Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), CalculateProjectionSeries_Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), CalculateProjectionSeries_Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(CalculateProjectionSeries_Dialog)

    def retranslateUi(self, CalculateProjectionSeries_Dialog):
        CalculateProjectionSeries_Dialog.setWindowTitle(_translate("CalculateProjectionSeries_Dialog", "Dialog", None))
        self.label_7.setText(_translate("CalculateProjectionSeries_Dialog", "Phi", None))
        self.label_8.setText(_translate("CalculateProjectionSeries_Dialog", "Psi", None))
        self.label_3.setText(_translate("CalculateProjectionSeries_Dialog", "Filename Containing Euler Angles", None))
        self.label_2.setText(_translate("CalculateProjectionSeries_Dialog", "Theta (Tilt)", None))
        self.label.setText(_translate("CalculateProjectionSeries_Dialog", "Output Filename", None))
        self.label_5.setText(_translate("CalculateProjectionSeries_Dialog", "Step", None))
        self.label_4.setText(_translate("CalculateProjectionSeries_Dialog", "Stop", None))
        self.label_6.setText(_translate("CalculateProjectionSeries_Dialog", "Start", None))
        self.btn_outputFilename.setText(_translate("CalculateProjectionSeries_Dialog", "Browse", None))
        self.btn_selectAngleFile.setText(_translate("CalculateProjectionSeries_Dialog", "Browse", None))

