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

def myFun(inp):
    for i in range(0,1):
        a = i
        # a = i+3
        a = myStuff[0,i:i+100]
        myStuff[0,:] = 1
    return myStuff[0,inp]
    return inp

N = 100000
myStuff = np.random.rand(1,N)
print np.shape(myStuff)

print myFun(5)
#
# pool = Pool(processes=1)
# results = pool.map(myFun, xrange(0,N))
# # results = pool.map()
# print results
# outputs = [result for result in results]
# print outputs
# print "HERE,", np.shape(outputs)
tic = time.time()
pool = Pool(processes=4)
results = pool.map(myFun, xrange(0,N))
print "time1, ", time.time()-tic

tic = time.time()

pool = Pool(processes=1)
results2 = pool.map(myFun, xrange(0,N))
print "time2, ", time.time()-tic
print "sum",np.sum(np.array(results2)-np.array(results))





#
# N = 512
# NUMTHREADS = 8
# threads = [None] * NUMTHREADS
# #
# def proc(N):
#     print "one"
#     # time.sleep(1)
#     # a = np.random.rand(N,N,N)
#     print np.shape(np.random.rand(N,N,N))
#     print "made an a"
# def proc2(i):
#     print "two"
#     # b = np.random.rand(N,N,N)
#     print np.shape(np.random.rand(N,N,N))
#     print "made a b"
#
# tic = time.time()
# for i in range(0, NUMTHREADS):
#     proc(N)
# print "serial time," ,time.time()-tic
# tic2 = time.time()
# for i in range(0,NUMTHREADS):
#     threads[i] = threading.Thread(target=proc, args=(N,))
#     threads[i].start()
# # threads[1] = threading.Thread(target=proc, args=)
# # threads[1].start()
# for i in range(0,NUMTHREADS):
#     threads[i].join()
# print "parallel time,", time.time()-tic2
#
# from multiprocessing import Pool
# pool = Pool(processes=4)
# tic3 = time.time()
# r = pool.map(proc2,range(0,NUMTHREADS))
# print "parallel time2,", time.time()-tic3
#
# p = [None] * NUMTHREADS
# tic4 = time.time()
# for i in range(0,NUMTHREADS):
#     p[i] = Process(target=proc2, args=(i,))
#     p[i].start()
# for i in range(0,NUMTHREADS):
#     p[i].join()
# print "parallel time3,", time.time()-tic4
# #

########################################################
# print a[0:10]
# print b[0:10]


# def test(list):
#     list.append('ha')
# a = []
# a.append(1)
# test(a)
# print a

#
# #
# dim1= 100
# mat = np.random.rand(dim1,dim1,dim1)
# ind = np.arange(0, dim1**3,2)
# ind = np.unravel_index(ind, [dim1, dim1, dim1], order='F')
# print np.shape(ind)
# print ind[0:5]
# tic = time.time()
# mat[ind[0],ind[1],ind[2]] = 1
# print "time out," ,time.time()-tic
#
# def placeStuff(arr,myind,start,stop):
#     # arr = np.random.rand(dim1,dim1,dim1)
#     arr[myind[0][start:stop],myind[1][start:stop],myind[2][start:stop]] = 1
#
# # _start_new_thread = threading._start_new_thread
# tic2 = time.time()
# print "lol", np.shape(ind[:750000])
# print type(ind[:750000])
# # f = placeStuff(np.zeros((100,100,100)),ind[1:500000/2])
# f = placeStuff(np.zeros((100,100,100)),ind,0,750000)
# print f
# print "hey:", type(f)
# thread1 =  threading.Thread(target=f)
# # thread2 =  threading.Thread(target=placeStuff(np.zeros((100,100,100)),ind[500000/2:]))
# # print np.shape(ind[0,1:5])
# thread2 =  threading.Thread(target=placeStuff(np.zeros((1,dim1**3)),ind,750000/2,np.size(ind)))
#
# thread1.start()
# thread2.start()
# thread1.join()
# thread2.join()
# print "multi threading", time.time()-tic2

#
#
# # f = placeStuff(np.zeros((1,500000)),ind[:500000/2])
# # print f
# # print "hey:", type(f)
# # thread1 =  threading.Thread(target=f)
# # thread2 =  threading.Thread(target=placeStuff(np.zeros((1,500000)),ind[500000/2:]))
# #
# # thread1.start()
# # thread2.start()
# # thread1.join()
# # thread2.join()
# # print "multi threading", time.time()-tic2
#
#
#
# # import GENFIRE_GUI
#
# # print help(NavigationToolbar)
#
# # a = np.array([[1,2,3,4,5],[ 4,5,6,7,8]])
# # ind = ([0,0],[1,2])
# # ind2 = [[0,0],[1,2]]
# # print(a[ind])
# # print a[ind2]
#
# a = io.loadmat('mk2')['mk']
# b = misc.hermitianSymmetrize(a)
# c = {'b':b}
# io.savemat('testing',c)
#
# # a = np.random.rand(5,5,5) + 1j*np.random.rand(5,5,5)
# # b = misc.hermitianSymmetrize(a)
# # c = {'b':b}
# # io.savemat('testing2',c)
# #
# # b = []
# # print len(b)
# # print range(0,0)
# # print type(range(0,0))
# # a = range(0,0)
# # if a:
# #     print "a!"
# # print a
# # a[0]
# # if b:
# #     print "it is!"
# # b.append([1])
# # if b:
# #     print "now it really is"
# # # b.append(np.array([1,2,3]))
# # # b.append(np.array([5,6,7,8]))
# # # print b
# # # print type(b)
# # # print b[0]
# # # print type(b[0])
# # # print len(b)
# # print len(b[0])
# # print b[0][0]
# # print type(b[0][1])
# # a = [1,2,3]
# # print a[b[0][1]]
