import numpy as np
from GENFIRE import *
arr = myReadMRC('virusFBP.mrc',dtype='f4')
print arr.dtype
writeMRC('test.mrc',arr)
t = myReadMRC('test.mrc')
print np.allclose(arr,t)
print t[0,0,0]
print arr[0,0,0]
