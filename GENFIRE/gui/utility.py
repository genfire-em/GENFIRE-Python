from PyQt4 import QtCore, QtGui
def toString(string):
    if isinstance(string,QtCore.QString):
        string = unicode(string.toUtf8(),encoding='UTF-8')
    return str(string)