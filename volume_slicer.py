import volume_slicer_ui
from PyQt4 import QtCore, QtGui
import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np

class VolumeSlicer(QtGui.QMainWindow):
    def __init__(self, volume):
        super(VolumeSlicer, self).__init__()
        self.volume = volume
        self.ui = volume_slicer_ui.Ui_VolumeSlicer()
        self.ui.setupUi(self)

        self.fig1 = plt.figure()
        self.fig2 = plt.figure()
        self.fig3 = plt.figure()

        self.canvas1 = FigureCanvas(self.fig1)
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas3 = FigureCanvas(self.fig3)

        self.slice1 = self.fig1.add_subplot(111)
        self.slice2 = self.fig2.add_subplot(111)
        self.slice3 = self.fig3.add_subplot(111)

        self.slice1.hold(False)
        self.slice2.hold(False)
        self.slice3.hold(False)

        self.navigationToolbar1 = NavigationToolbar(self.canvas1, self)
        self.navigationToolbar2 = NavigationToolbar(self.canvas2, self)
        self.navigationToolbar3 = NavigationToolbar(self.canvas3, self)

        self.ui.vt_lyt_fig1.addWidget(self.navigationToolbar1)
        self.ui.vt_lyt_fig1.addWidget(self.canvas1)
        self.ui.vt_lyt_fig2.addWidget(self.navigationToolbar2)
        self.ui.vt_lyt_fig2.addWidget(self.canvas2)
        self.ui.vt_lyt_fig3.addWidget(self.navigationToolbar3)
        self.ui.vt_lyt_fig3.addWidget(self.canvas3)

        dimx, dimy, dimz = np.shape(volume)
        ncx,  ncy,  ncz  = dimx//2., dimy//2, dimz//2

        self.ui.scrlbr_fig1.setValue(ncx)
        self.ui.scrlbr_fig1.setMinimum(0)
        self.ui.scrlbr_fig1.setMaximum(dimx)

        self.ui.scrlbr_fig2.setValue(ncy)
        self.ui.scrlbr_fig2.setMinimum(0)
        self.ui.scrlbr_fig3.setMaximum(dimy)

        self.ui.scrlbr_fig3.setValue(ncz)
        self.ui.scrlbr_fig3.setMinimum(0)
        self.ui.scrlbr_fig3.setMaximum(dimz)

        self.updateSliceX(ncx)
        self.updateSliceY(ncy)
        self.updateSliceZ(ncz)

    def updateSliceX(self, nx):
        self.slice1.imshow(np.squeeze(self.volume[nx, :, :]))
        self.canvas1.draw()

    def updateSliceY(self, ny):
        self.slice2.imshow(np.squeeze(self.volume[:, ny, :]))
        self.canvas2.draw()

    def updateSliceZ(self, nz):
        self.slice3.imshow(np.squeeze(self.volume[:, :, nz]))
        self.canvas3.draw()