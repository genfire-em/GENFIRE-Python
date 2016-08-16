import volume_slicer_ui
from PyQt4 import QtCore, QtGui

class VolumeSlicer(QtGui.QMainWindow):
    def __init__(self):
        super(VolumeSlicer, self).__init__()
        self.ui = volume_slicer_ui.Ui_VolumeSlicer()