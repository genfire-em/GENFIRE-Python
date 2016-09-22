import GENFIRE
import numpy as np
import matplotlib.pyplot as plt

a = GENFIRE.fileio.readVolume('vesicle.mrc')
pj = GENFIRE.utility.calculateProjection_DFT(a,0,0,0,128,128)
plt.figure()
plt.imshow(pj)
plt.figure()
plt.imshow(np.sum(a,axis=2))
plt.show()
