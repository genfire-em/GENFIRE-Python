"""
* utility *

The gui utility function module


Author: Alan (AJ) Pryor, Jr.
Jianwei (John) Miao Coherent Imaging Group
University of California, Los Angeles
Copyright 2015-2016. All rights reserved.
"""

from PyQt4 import QtCore, QtGui
def toString(string):
    if isinstance(string,QtCore.QString):
        string = unicode(string.toUtf8(),encoding='UTF-8')
    return str(string)