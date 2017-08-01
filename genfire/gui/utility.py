"""
* genfire.gui.utility *

The gui utility function module


Author: Alan (AJ) Pryor, Jr.
Jianwei (John) Miao Coherent Imaging Group
University of California, Los Angeles
Copyright 2015-2016. All rights reserved.
"""

from PyQt5 import QtCore, QtGui
import sys

# strings are very different objects in Python 2 and 3, so a little modification of these
# helper functions is necessary dependent on the version
if sys.version_info >=(3,0):
    def toString(string):
        return str(string)
    def toQString(string):
        return str(string)
    def toFloat(string):
        return float(string)
    def toInt(value):
        return int(value)
else: #python 2
    import PyQt5.QtCore
    def toString(string):
        if isinstance(string,QtCore.QString):
            string = unicode(string.toUtf8(),encoding='UTF-8')
        return string
    def toQString(string):
        return PyQt5.QtCore.QString(str(string))
    def toFloat(value):
        if isinstance(value,QtCore.QString):
            return value.toFloat()[0]
        else:
            return float(value)
    def toInt(value):
        if isinstance(value,QtCore.QString):
            return value.toInt()[0]
        else:
            return int(value)
