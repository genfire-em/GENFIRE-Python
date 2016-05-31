from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as sfft
import os
import scipy.io
import pyfftw
import time
import GENFIRE_from_GUI_Input

arr = np.random.rand(5,5,5)
arr[1:3,1:3,1:3] = 1
print arr[1:3,1:3,1:3]
print arr[(range(1,3),(range(1,3)),(range(1,3)))]
print "here",arr[(1,2),(1,2),(1,2)]