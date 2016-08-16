# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'volume_slicer.ui'
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

class Ui_VolumeSlicer(object):
    def setupUi(self, VolumeSlicer):
        VolumeSlicer.setObjectName(_fromUtf8("VolumeSlicer"))
        VolumeSlicer.resize(480, 640)
        self.centralwidget = QtGui.QWidget(VolumeSlicer)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.hz_lyt_figures = QtGui.QHBoxLayout()
        self.hz_lyt_figures.setObjectName(_fromUtf8("hz_lyt_figures"))
        self.hz_lyt_fig1 = QtGui.QHBoxLayout()
        self.hz_lyt_fig1.setObjectName(_fromUtf8("hz_lyt_fig1"))
        self.hz_lyt_figures.addLayout(self.hz_lyt_fig1)
        self.hz_lyt_fig2 = QtGui.QHBoxLayout()
        self.hz_lyt_fig2.setObjectName(_fromUtf8("hz_lyt_fig2"))
        self.hz_lyt_figures.addLayout(self.hz_lyt_fig2)
        self.hz_lyt_fig3 = QtGui.QHBoxLayout()
        self.hz_lyt_fig3.setObjectName(_fromUtf8("hz_lyt_fig3"))
        self.hz_lyt_figures.addLayout(self.hz_lyt_fig3)
        self.verticalLayout_2.addLayout(self.hz_lyt_figures)
        self.hz_lyt_tools = QtGui.QHBoxLayout()
        self.hz_lyt_tools.setObjectName(_fromUtf8("hz_lyt_tools"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.hz_lyt_sliders = QtGui.QHBoxLayout()
        self.hz_lyt_sliders.setObjectName(_fromUtf8("hz_lyt_sliders"))
        self.sldr_fig1 = QtGui.QVBoxLayout()
        self.sldr_fig1.setObjectName(_fromUtf8("sldr_fig1"))
        self.hz_lyt_sliders.addLayout(self.sldr_fig1)
        self.sldr_fig2 = QtGui.QVBoxLayout()
        self.sldr_fig2.setObjectName(_fromUtf8("sldr_fig2"))
        self.hz_lyt_sliders.addLayout(self.sldr_fig2)
        self.sldr_fig3 = QtGui.QVBoxLayout()
        self.sldr_fig3.setObjectName(_fromUtf8("sldr_fig3"))
        self.hz_lyt_sliders.addLayout(self.sldr_fig3)
        self.verticalLayout.addLayout(self.hz_lyt_sliders)
        self.hz_lyt_btns = QtGui.QHBoxLayout()
        self.hz_lyt_btns.setObjectName(_fromUtf8("hz_lyt_btns"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.hz_lyt_btns.addLayout(self.horizontalLayout_5)
        self.verticalLayout.addLayout(self.hz_lyt_btns)
        self.hz_lyt_tools.addLayout(self.verticalLayout)
        self.verticalLayout_2.addLayout(self.hz_lyt_tools)
        VolumeSlicer.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(VolumeSlicer)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 480, 22))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        VolumeSlicer.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(VolumeSlicer)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        VolumeSlicer.setStatusBar(self.statusbar)

        self.retranslateUi(VolumeSlicer)
        QtCore.QMetaObject.connectSlotsByName(VolumeSlicer)

    def retranslateUi(self, VolumeSlicer):
        VolumeSlicer.setWindowTitle(_translate("VolumeSlicer", "MainWindow", None))

