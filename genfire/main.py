"""
* genfire.main *

The primary control module for running GENFIRE reconstructions.


Author: Alan (AJ) Pryor, Jr.
Jianwei (John) Miao Coherent Imaging Group
University of California, Los Angeles
Copyright 2015-2016. All rights reserved.
"""

from __future__ import division
import numpy as np
import genfire
import sys
import os
from genfire.reconstruct import ReconstructionParameters


def main_InteractivelySetParameters():
    #######################################################################################################################
    ############################################### User Parameters  ######################################################
    #######################################################################################################################

    # GENFIRE's reconstruction parameters can be edited here by the user and run interactively, any inputs provided
    # by either the command line or the GUI will override these momentarily

    filename_projections = '../data/projections.mat'  #filename of projections, which should be size NxNxN_projections where N_projections is the number of projections
    filename_angles = '../data/angles.mat'  #angles can be either a 1xN_projections array containing a single tilt series, or 3xN_projections array containing 3 Euler angles for each projections in the form [phi;theta;psi]
    filename_support = '../data/support60.mat'  #NxNxN binary array specifying a region of 1's in which the reconstruction can exist
    filename_initialObject = None  #initial object to use in reconstruction; set to None to provide no initial guess
    filename_results = 'GENFIRE_rec.mrc'  #filename to save results
    resolutionExtensionSuppressionState = 2 # 1) Turn on resolution extension/suppression, 2) No resolution extension/suppression, 3) Just resolution extension

    numIterations = 100  #number of iterations to run in iterative reconstruction
    oversamplingRatio = 3  #input projections will be padded internally to match this oversampling ratio. If you prepad your projections, set this to 1
    interpolationCutoffDistance = 0.7  #radius of spherical interpolation kernel (in pixels) within which to include measured datapoints
    doYouWantToDisplayFigure = True
    displayFigure = genfire.reconstruct.DisplayFigure()
    displayFigure.DisplayFigureON = doYouWantToDisplayFigure
    calculateRFree = True
    if filename_support is None:
        useDefaultSupport = True
    else:
        useDefaultSupport = False

    reconstruction_parameters                                      = ReconstructionParameters()
    reconstruction_parameters.projectionFilename                   = filename_projections
    reconstruction_parameters.angleFilename                        = filename_angles
    reconstruction_parameters.supportFilename                      = filename_support
    reconstruction_parameters.interpolationCutoffDistance          = interpolationCutoffDistance
    reconstruction_parameters.numIterations                        = numIterations
    reconstruction_parameters.oversamplingRatio                    = oversamplingRatio
    reconstruction_parameters.displayFigure                        = displayFigure
    reconstruction_parameters.calculateRfree                       = calculateRFree
    reconstruction_parameters.resolutionExtensionSuppressionState  = resolutionExtensionSuppressionState
    reconstruction_parameters.useDefaultSupport                    = useDefaultSupport
    if os.path.isfile(filename_results): # If a valid initial object was provided, use it
        reconstruction_parameters.initialObjectFilename           = filename_results

    main(reconstruction_parameters)

def main(reconstruction_parameters):
    import genfire.fileio

    filename_projections                    = reconstruction_parameters.projectionFilename
    filename_angles                         = reconstruction_parameters.angleFilename
    filename_support                        = reconstruction_parameters.supportFilename
    filename_results                        = reconstruction_parameters.resultsFilename
    numIterations                           = reconstruction_parameters.numIterations
    oversamplingRatio                       = reconstruction_parameters.oversamplingRatio
    interpolationCutoffDistance             = reconstruction_parameters.interpolationCutoffDistance
    displayFigure                           = reconstruction_parameters.displayFigure
    resolutionExtensionSuppressionState     = reconstruction_parameters.resolutionExtensionSuppressionState
    calculateRFree                          = reconstruction_parameters.calculateRfree
    useDefaultSupport                       = reconstruction_parameters.useDefaultSupport
    use_positivity                          = reconstruction_parameters.constraint_positivity
    use_support                             = reconstruction_parameters.constraint_support
    gridding_method                         = reconstruction_parameters.griddingMethod
    enforceResolutionCircle                 = reconstruction_parameters.enforceResolutionCircle
    permitMultipleGridding                  = reconstruction_parameters.permitMultipleGridding

    if reconstruction_parameters.isInitialObjectDefined:
            filename_initialObject          = reconstruction_parameters.initialObjectFilename
    else:
        filename_initialObject               = None

    ### begin reconstruction ###
    projections = genfire.fileio.loadProjections(filename_projections) # load projections into a 3D numpy array

    # get dimensions of array and determine the array size after padding
    dims = np.shape(projections)
    paddedDim = dims[0] * oversamplingRatio
    padding = int((paddedDim-dims[0])/2)

    # load the support, or generate one if none was provided
    if useDefaultSupport or filename_support == "":
        support = np.ones((dims[0],dims[0],dims[0]),dtype=float)
    else:
        support = (genfire.fileio.readVolume(filename_support) != 0).astype(bool)

    displayFigure.reconstructionDisplayWindowSize = np.shape(support) # this is used to show the central region of reconstruction

    # now zero-pad to match the oversampling ratio
    support = np.pad(support,((padding,padding),(padding,padding),(padding,padding)),'constant')
    projections = np.pad(projections,((padding,padding),(padding,padding),(0,0)),'constant')

    #load initial object, or initialize it to zeros if none was given
    if filename_initialObject is not None and os.path.isfile(filename_initialObject):
        initialObject = genfire.fileio.readVolume(filename_initialObject)
        initialObject = np.pad(initialObject,((padding,padding),(padding,padding),(padding,padding)),'constant')
    else:
        initialObject = np.zeros_like(support)

    # load angles and check that the dimensions match the number of provided projections and that they
    # are either 1 x num_projections or 3 x num_projections
    angles = genfire.fileio.loadAngles(filename_angles)
    if np.shape(angles)[1] > 3:
        raise ValueError("Error! Dimension of angles incorrect.")
    if np.shape(angles)[1] == 1:
        tmp = np.zeros([np.shape(angles)[1], 3])
        tmp[1, :] = angles
        angles = tmp
        del tmp

    # grid the projections
    if gridding_method == "DFT":
        measuredK = genfire.reconstruct.fillInFourierGrid_DFT(projections, angles, interpolationCutoffDistance, enforceResolutionCircle)
    else:
        measuredK = genfire.reconstruct.fillInFourierGrid(projections, angles, interpolationCutoffDistance, enforceResolutionCircle, permitMultipleGridding)

    # the grid is assembled with the origin at the geometric center of the array, but for efficiency in the
    # iterative algorithm the origin is shifted to array position [0,0,0] to avoid unnecessary fftshift calls
    measuredK = np.fft.ifftshift(measuredK)

    # create a map of the spatial frequency to be used to control resolution extension/suppression behavior
    K_indices = genfire.utility.generateKspaceIndices(support)
    K_indices = np.fft.fftshift(K_indices)
    resolutionIndicators = np.zeros_like(K_indices)
    resolutionIndicators[measuredK != 0] = 1-K_indices[measuredK != 0]

    # if calculating Rfree, setup some infrastructure
    if calculateRFree:
        R_freeInd_complexX = []
        R_freeInd_complexY = []
        R_freeInd_complexZ = []
        R_freeVals_complex = []
        shell_thickness_pixels = 1 # pixel thickness of an individual shell of Rfree points
        numberOfBins = int(round(dims[0]/2/shell_thickness_pixels)) # number of frequency bins. Rfree will be tracked within each shell separately
        percentValuesForRfree = 0.05 # percentage of measured points to withhold
        spatialFrequencyForRfree = np.linspace(0,1,numberOfBins+1)
        K_indicesSmall =(K_indices)[:, :, 0:(np.shape(measuredK)[-1]//2+1)]

        for shellNum in range(0,numberOfBins):
            # collect relevant points
            measuredPointInd_complex = np.where((measuredK[:, :, 0:(np.shape(measuredK)[-1]//2+1)] != 0) & (K_indicesSmall>=(spatialFrequencyForRfree[shellNum])) & (K_indicesSmall<=(spatialFrequencyForRfree[shellNum+1])))

            # randomly shuffle
            shuffledPoints = np.random.permutation(np.shape(measuredPointInd_complex)[1])
            measuredPointInd_complex = (measuredPointInd_complex[0][shuffledPoints], measuredPointInd_complex[1][shuffledPoints], measuredPointInd_complex[2][shuffledPoints])

            # determine how many values to take
            cutoffInd_complex = np.floor(np.shape(measuredPointInd_complex)[1] * percentValuesForRfree).astype(int)

            # collect the Rfree values and coordinates
            R_freeInd_complexX.append(measuredPointInd_complex[0][:cutoffInd_complex])
            R_freeInd_complexY.append(measuredPointInd_complex[1][:cutoffInd_complex])
            R_freeInd_complexZ.append(measuredPointInd_complex[2][:cutoffInd_complex])
            R_freeVals_complex.append(measuredK[R_freeInd_complexX[shellNum], R_freeInd_complexY[shellNum], R_freeInd_complexZ[shellNum] ])

            # delete the points from the measured data
            measuredK[R_freeInd_complexX[shellNum], R_freeInd_complexY[shellNum], R_freeInd_complexZ[shellNum]] = 0

        # create tuple of coordinates
        R_freeInd_complex = [[R_freeInd_complexX], [R_freeInd_complexY], [R_freeInd_complexZ]]
        del R_freeInd_complexX
        del R_freeInd_complexY
        del R_freeInd_complexZ
    else:
        R_freeInd_complex = []
        R_freeVals_complex = []

    if resolutionExtensionSuppressionState==1: # resolution extension/suppression
        constraintEnforcementDelayIndicators = np.array(np.concatenate((np.arange(0.95, -.25, -0.15), np.arange(-0.15, .95, .1)), axis=0))
    elif resolutionExtensionSuppressionState==2:# no resolution extension or suppression
        constraintEnforcementDelayIndicators = np.array([-999, -999, -999, -999])
    elif resolutionExtensionSuppressionState==3:# resolution extension only
        constraintEnforcementDelayIndicators = np.concatenate((np.arange(0.95, -.15, -0.15),[-0.15, -0.15, -0.15]))
    else:
        print("Warning! Input resolutionExtensionSuppressionState does not match an available option. Deactivating dynamic constraint enforcement and continuing.\n")
        constraintEnforcementDelayIndicators = np.array([-999, -999, -999, -999])

    reconstructionOutputs = genfire.reconstruct.reconstruct(numIterations, np.fft.fftshift(initialObject), np.fft.fftshift(support), (measuredK)[:, :, 0:(np.shape(measuredK)[-1] // 2 + 1)], (resolutionIndicators)[:, :, 0:(np.shape(measuredK)[-1] // 2 + 1)], constraintEnforcementDelayIndicators, R_freeInd_complex, R_freeVals_complex, displayFigure, use_positivity, use_support)

    # reclaim original array size. ncBig is center of oversampled array, and n2 is the half-width of original array
    ncBig = paddedDim//2
    n2 = dims[0]//2
    reconstructionOutputs['reconstruction'] = reconstructionOutputs['reconstruction'][ncBig-n2:ncBig+n2,ncBig-n2:ncBig+n2,ncBig-n2:ncBig+n2]
    genfire.fileio.saveResults(reconstructionOutputs, filename_results)

if __name__ == "__main__" and len(sys.argv) == 1:
    print ("starting with user parameters")
    main_InteractivelySetParameters()
elif __name__ == "__main__":
    if len(sys.argv) > 1: # Parse inputs provided either from the GUI or from the command line
        inputArgumentOptions = {"-p" :  "filename_projections",
                                "-a" :  "filename_angles",
                                "-s" :  "filename_support",
                                "-o" :  "filename_results",
                                "-i" :  "filename_initialObject",
                                "-r" :  "resolutionExtensionSuppressionState",
                                "-it":  "numIterations",
                                "-or":  "oversamplingRatio",
                                "-t" :  "interpolationCutoffDistance",
                                "-d" :  "displayFigure",
                                "-rf":  "calculateRFree"
                                }
        print (sys.argv[:])
        if len(sys.argv)%2==0:
            raise Exception("Number of input options and input arguments does not match!")
        for argumentNum in range(1,len(sys.argv),2):
            print (inputArgumentOptions[sys.argv[argumentNum]])
            print  (sys.argv[argumentNum+1])
            print (inputArgumentOptions[sys.argv[argumentNum]] + "=" + sys.argv[argumentNum+1])


            exec(inputArgumentOptions[sys.argv[argumentNum]] + "= '" + sys.argv[argumentNum+1] +"'")
            print("Setting argument %s from option %s equal to GENFIRE parameter %s " % (sys.argv[argumentNum+1],sys.argv[argumentNum], inputArgumentOptions[sys.argv[argumentNum]] ))

        numIterations = int(numIterations)
        # displayFigure = bool(displayFigure)
        doYouWantToDisplayFigure = bool(displayFigure)
        displayFigure = genfire.reconstruct.DisplayFigure()
        displayFigure.DisplayFigureON = doYouWantToDisplayFigure
        oversamplingRatio = float(oversamplingRatio)
        resolutionExtensionSuppressionState = int(resolutionExtensionSuppressionState)
        calculateRFree = bool(calculateRFree)
        try:
            main(filename_projections,
                 filename_angles,
                 filename_support,
                 filename_results,
                 numIterations,
                 oversamplingRatio,
                 interpolationCutoffDistance,
                 resolutionExtensionSuppressionState,
                 displayFigure,
                 calculateRFree,
                 filename_initialObject)
        except (NameError, IOError):
             main(filename_projections,
                  filename_angles,
                  filename_support,
                  filename_results,
                  numIterations,
                  oversamplingRatio,
                  interpolationCutoffDistance,
                  resolutionExtensionSuppressionState,
                  displayFigure,
                  calculateRFree)
