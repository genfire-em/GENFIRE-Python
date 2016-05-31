import numpy as np

import time
import numba
N = 10000

def test(N):
    b = np.zeros((N,N))
    for j in range(N):
        for i in range(N):
            b[j,i] = np.random.rand()
    return b
tic = time.time()
b = test(N)
print time.time()-tic
print b[0,0]
print b[N-1,N-1]

N = 10000

@numba.autojit
def test2(N):
    b = np.zeros((N,N))
    for j in range(N):
        for i in range(N):
            b[j,i] = np.random.rand()
    return b
tic = time.time()
b = test(N)
print time.time()-tic
print b[0,0]
print b[N-1,N-1]


tic = time.time()
b = test2(N)
print time.time()-tic
print b[0,0]
print b[N-1,N-1]


tic = time.time()
b = test2(N)
print time.time()-tic
print b[0,0]
print b[N-1,N-1]




# def projConv2(projNums):
#     print "start"
#     out = np.zeros(256,256,N//4)
#
#     for projNum in range(projNums):
#         pj = projections[:, :, projNum]
#         out[:, :, projNum] = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(pj)))
#
#     return out


#
#
# proc = [None]*4
#
# tic = time.time()
# for i in range(4):
#     print i
#     proc[i] = Process(target = projConv2, args=(i,))
#     proc[i].start()
# for i in range(4):
#     proc[i].join()
# print time.time()-tic


# tic = time.time()
# pool = Pool(4)
# func = functools.partial(projConv,projections)
# results = pool.apply_async(func, (range(N),))
# print "par time," , time.time()-tic
# print np.shape(results)
# print type(results)
#
# print np.mean(np.array(proj[:, :, 0]))
# print np.mean(np.array(results[:,:,0]))
# #
# # print type(results[0])
# # print np.shape(results[0])

