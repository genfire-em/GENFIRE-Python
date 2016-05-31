import numpy as np
from PIL import Image
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt
from numbapro import vectorize


@vectorize(["float32(float32, float32)"], target='gpu')
def vectorAdd(a, b):
    return a + b


numProjections = 99
filebase = 'projections'
out = np.zeros((60,60,numProjections))
tic = time.time()

for j in range(numProjections):
    out[:, :, j] = np.array(Image.open(filebase + str(j) +'.tif'))

print "time 1",time.time()-tic
print np.shape(out)
b = out.copy()


def readProj(projNum):
    return np.array(Image.open(filebase + str(projNum) +'.tif'))


tic = time.time()
pool = Pool(4)
a = pool.map(readProj, range(numProjections))

for j  in range(numProjections):
    out[:, :, j] = a[j]

print time.time()-tic

print np.sum(abs(b-out)) / np.sum(b)

print np.allclose(b,out)
