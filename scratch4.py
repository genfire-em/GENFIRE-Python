import multiprocessing
import numpy as np
import time
import itertools
from PyQt4 import QtCore, QtGui
import os
import misc
import pyfftw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import ProjectionCalculator_ui
import scipy.io as io
import Queue
import threading
from multiprocessing import Process,Pool,Array
import sys
import struct
# a = np.random.rand(401,401)
# nv = (a>.5).astype(int)
# nv3 = nv
# print np.max(nv)
# nv[0,0] = 500
# print nv3[0,0]
#
# a = np.random.rand(401,401)
# nv = (a>.5)
# nv = nv.astype(float)
# nv3 = nv
# print np.max(nv)
# nv[0,0] = 500
# print nv3[0,0]


a = np.random.rand(401,401)
nv = a
# nv = nv.astype(float)
# nv3 = a
# print nv3[0,0]
# # print np.max(nv)
# nv[0,0] = 500
# a[0,0] = 55
# print nv3[0,0]
# print nv[0,0]
# print np.allclose(nv3,nv).all()

def myReadMRC(filename):
    headerIntNumber = 56
    numberOfBytes_int = 4

    headerCharNumber = 800
    numberOfBytes_char = 1

    image = open(filename,'rb')
    formatter_int = '=56l'
    formatter_char = '=800c'

    dimx, dimy, dimz, data_type = struct.unpack(formatter_int, image.read(headerIntNumber*numberOfBytes_int))[:4]
    header_char = struct.unpack(formatter_char, image.read(headerCharNumber * numberOfBytes_char))

    ##get type then use numpy fromfile

    # print dimx, dimy, dimz, data_type
    # print header_char
    # print struct.pack("=c",'l')
    image.close()

    pass
# print struct.pack("=cc",'l','l')
# print struct.pack('hhl', 1, 2, 3)
myReadMRC('virusFBP.mrc')
# b = np.random.rand(5,5)
# print b
# np.save('b.npy',b)
# a = np.load('b.npy')
# print a

# nv = (a != 0).astype(float)
# nv2 = (a != 0).astype(float)
# print np.max(nv)
# nv += nv[::-1,::-1]
# # nv += nv[:,:]
# print np.max(nv)
# nv2 += nv2[::-1,::-1]
# print np.allclose(nv2,nv)
# print np.max(nv2)


#

# a = 5
# b = a
# a = 6
# print(b)
#
# a = np.array(5)
# b = a
# a = 6
# print(b)

# nv += nv[::-1,::-1]
# print np.max(nv)

# nv3 = nv3 + nv3[::-1,::-1]
# print np.max(nv3)

# print np.max(nv2)
