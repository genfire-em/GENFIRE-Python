import numpy as np
def looper2(a):
    c= np.random.rand(a,a)
    b = 0
    for i in range(0,a):
        for j in range(0,a):
            b += c[i,j]
        # b +=1
    print b
