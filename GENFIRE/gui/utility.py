"""
* GENFIRE.gui.utility *

The gui utility function module


Author: Alan (AJ) Pryor, Jr.
Jianwei (John) Miao Coherent Imaging Group
University of California, Los Angeles
Copyright 2015-2016. All rights reserved.
"""

from PyQt4 import QtCore, QtGui
import sys
if sys.version_info >=(3,0):
    def toString(string):
        str(string)
else:
    def toString(string):
        if isinstance(string,QtCore.QString):
            string = unicode(string.toUtf8(),encoding='UTF-8')
        return str(string)


import sys
if sys.version_info >=(3,0):
    def toQString(string):
        return str(string)
else:
    import PyQt4.QtCore
    def toQString(string):
        return PyQt4.QtCore.QString(string)