from __future__ import division
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import time
import GENFIRE

def GENFIRE_from_GUI_Input(filename_projections,filename_angles,filename_support):


    import cProfile


    tic = time.clock()
    #######################################################################################################################
    ############################################### User Parameters  #####################g#################################
    #######################################################################################################################
    # filename_projections = 'projections.mat'  #filename of projections, which should be size NxNxN_projections where N_projections is the number of projections
    # filename_angles = 'angles.mat'  #angles can be either a 1xN_projections array containing a single tilt series, or 3xN_projections array containing 3 Euler angles for each projections in the form [phi;theta;psi]
    # filename_support = 'support60.mat'  #NxNxN binary array specifying a region of 1's in which the reconstruction can exist
    # filename_initialObject = ''  #initial object to use in reconstruction; comment this line out if you have no initial guess and want to start with all zeros
    filename_results = 'GENFIRE_rec_python.mat'  #filename to save results

    numIterations = 50  #number of iterations to run in iterative reconstruction
    oversamplingRatio = 3  #input projections will be padded internally to match this oversampling ratio. If you prepad your projections, set this to 1
    interpolationCutoffDistance = 0.7  #radius of spherical interpolation kernel (in pixels) within which to include measured datapoints
    #######################################################################################################################


    print(filename_projections)
    print(type(filename_projections))

    ### begin reconstruction ###
    projections = io.loadmat(filename_projections);projections = projections[projections.keys()[0]]
    angles = io.loadmat(filename_angles);angles = angles['angles']
    support = io.loadmat(filename_support); support = support[support.keys()[0]]

    dims = np.shape(projections)
    paddedDim = dims[0] * oversamplingRatio
    padding = int((paddedDim-dims[0])/2)
    # padding = padding.astype(int)
    support = np.pad(support,((padding,padding),(padding,padding),(padding,padding)),'constant')
    projections = np.pad(projections,((padding,padding),(padding,padding),(0,0)),'constant')
    # print np.shape(projections)

    #load initial object, or initialize it to zeros if none was given
    try:
        initialObject = io.loadmat(filename_initialObject); initialObject = initialObject[initialObject.keys()[0]]
        initialObject = np.pad(initialObject,((padding,padding),(padding,padding),(padding,padding)),'constant')
    except NameError:
        initialObject = np.zeros_like(support)
    print(type(angles))
    if np.shape(angles)[0] > 3:
        print ("angles dimension wrong")
        ##raise error about angles
    if np.shape(angles)[0] == 1:
        tmp = np.zeros([3, np.shape(angles)[1]])
        tmp[1, :] = angles
        angles = tmp
        del tmp

    ticGridding = time.time()
    measuredK = GENFIRE.fillInFourierGrid(projections, angles, interpolationCutoffDistance)
    print ("Gridding Time: ",time.time()-ticGridding)

    # mk = {'mk':measuredK}
    # io.savemat('mk2.mat', mk)

    K_indices = GENFIRE.generateKspaceIndices(support)
    Q_magnitudes = np.zeros_like(K_indices)
    Q_magnitudes[measuredK != 0] = 1-K_indices[measuredK != 0]

    # Q_magnitudes = {'resolutionFlags':Q_magnitudes}
    # io.savemat('resolutionFlags.mat',Q_magnitudes)

    # constraintEnforcementDelayFlags = np.concatenate((np.arange(0.95, -.25, -0.1), np.arange(-0.25, .95, .1)), axis=0)
    constraintEnforcementDelayFlags = np.array([-5, -5, -5, -5])
    R_freeInd_complex = []
    R_freeVals_complex = []
    # reconstructionOutputs = GENFIRE.GENFIRE_iterate(numIterations,initialObject,support,measuredK,resolutionFlags,constraintEnforcementDelayFlags,R_freeInd_complex,R_freeVals_complex);
    ticRecon = time.time()
    # cProfile.run('reconstructionOutputs = GENFIRE.GENFIRE_iterate(numIterations,np.fft.fftshift(initialObject),np.fft.fftshift(support),np.fft.fftshift(measuredK)[:, :, 0:(np.shape(measuredK)[-1]//2+1)],np.fft.fftshift(Q_magnitudes)[:, :, 0:(np.shape(measuredK)[-1]//2+1)],constraintEnforcementDelayFlags,R_freeInd_complex,R_freeVals_complex)')
    reconstructionOutputs = GENFIRE.GENFIRE_iterate(numIterations,np.fft.fftshift(initialObject),np.fft.fftshift(support),np.fft.fftshift(measuredK)[:, :, 0:(np.shape(measuredK)[-1]//2+1)],np.fft.fftshift(Q_magnitudes)[:, :, 0:(np.shape(measuredK)[-1]//2+1)],constraintEnforcementDelayFlags,R_freeInd_complex,R_freeVals_complex)
    print ("Reconstruction Time: ",time.time()-ticRecon)
    print ('GENFIRE: Reconstruction finished. Total computation time = ', time.clock()-tic)
    io.savemat(filename_results,reconstructionOutputs)


    # plt.figure(5)
    # plt.imshow(np.fft.ifftshift(reconstructionOutputs['reconstruction'])[:, :, 90])
    # #plt.imshow(abs((constraintConfidenceWeights)[:, :, 0]))
    # plt.draw()
    # plt.pause(.1)
if __name__ == "__main__":
    print("why am i here?")
    filename_projections = 'projections.mat'  #filename of projections, which should be size NxNxN_projections where N_projections is the number of projections
    filename_angles = 'angles.mat'  #angles can be either a 1xN_projections array containing a single tilt series, or 3xN_projections array containing 3 Euler angles for each projections in the form [phi;theta;psi]
    filename_support = 'support60.mat'  #NxNxN binary array specifying a region of 1's in which the reconstruction can exist

    GENFIRE_from_GUI_Input(filename_projections,filename_angles,filename_support)