# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'correctcenter.ui'
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

class Ui_CorrectCenter(object):
    def setupUi(self, CorrectCenter):
        CorrectCenter.setObjectName(_fromUtf8("CorrectCenter"))
        CorrectCenter.resize(407, 343)
        self.verticalLayoutWidget = QtGui.QWidget(CorrectCenter)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(50, 60, 226, 80))
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label = QtGui.QLabel(self.verticalLayoutWidget)
        self.label.setWordWrap(True)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_2.addWidget(self.label)
        self.lineEdit_searchBoxSize = QtGui.QLineEdit(self.verticalLayoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_searchBoxSize.sizePolicy().hasHeightForWidth())
        self.lineEdit_searchBoxSize.setSizePolicy(sizePolicy)
        self.lineEdit_searchBoxSize.setObjectName(_fromUtf8("lineEdit_searchBoxSize"))
        self.horizontalLayout_2.addWidget(self.lineEdit_searchBoxSize)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.btn_cancel = QtGui.QPushButton(self.verticalLayoutWidget)
        self.btn_cancel.setObjectName(_fromUtf8("btn_cancel"))
        self.horizontalLayout.addWidget(self.btn_cancel)
        self.btn_go = QtGui.QPushButton(self.verticalLayoutWidget)
        self.btn_go.setObjectName(_fromUtf8("btn_go"))
        self.horizontalLayout.addWidget(self.btn_go)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(CorrectCenter)
        QtCore.QMetaObject.connectSlotsByName(CorrectCenter)

    def retranslateUi(self, CorrectCenter):
        CorrectCenter.setWindowTitle(_translate("CorrectCenter", "Dialog", None))
        self.label.setText(_translate("CorrectCenter", "Search Box Size", None))
        self.btn_cancel.setText(_translate("CorrectCenter", "Cancel", None))
        self.btn_go.setText(_translate("CorrectCenter", "Optimize Center", None))

