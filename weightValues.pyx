import numpy as np
cimport numpy as np

#This function takes in a numpy array of complex values from the FFT of projection data,
# the corresponding grid point for that coordinate, the distance to that grid point, and two
# objects describing the indices. "uniqueInd" is a list of the grid points being calculated
# without repitions, and "multiInd" is a list of indices within uniqueInd that indicates
# which values were repeated multiple times. The reason for this construct is that if a value
# only occurs once, then there is no need to go to the trouble of calculating a weighted average.
# These points are simply placed in the assembled grid in the main function prior to this one being called.


def weightValue(np.ndarray[long, ndim=1] multiInd,np.ndarray[long, ndim=1] uniqueInd, np.ndarray[double, ndim=1] masterDistances, np.ndarray[complex, ndim=1] masterVals):


    # declare static types for Cython variables
    cdef long numInd = np.size(multiInd)
    cdef long i
    cdef long j
    cdef long ind1
    cdef long ind2
    cdef long lengthOfVector
    cdef double distanceSum
    cdef np.ndarray[complex, ndim=1] valuesOut = np.zeros(numInd - 1, dtype=complex)
    cdef np.ndarray[complex, ndim=1] magnitudesOut = np.zeros(numInd - 1, dtype=complex)

    # These tmp variables will be used to hold values for each grid point. Here I
    # initialize them to the largest possible size they could be, which is if the entire
    # input is for one grid point. This is highly unlikely, but the point is declaring
    # a big array once and only using the first portion of it is much more efficient
    # then dynamic memory allocation within the inner loop
    cdef np.ndarray[complex, ndim=1] tmpValues = np.zeros(numInd - 1, dtype=complex)
    cdef np.ndarray[double, ndim=1] tmpWeights = np.zeros(numInd - 1, dtype=float)
    cdef np.ndarray[double, ndim=1] tmpDistances = np.zeros(numInd - 1, dtype=float)

    for i in range(0,numInd-1): # this loop is over each grid point

        # get starting and stopping points
        ind1 = uniqueInd[multiInd[i]]
        ind2 = uniqueInd[multiInd[i] + 1] - 1
        lengthOfVector = ind2 - ind1 + 1

        # initialize the sum of the distances (which is used in the weight normalization)
        distanceSum  = 0

        for j in range(0, lengthOfVector): #extract values and calculate distanceSum
            tmpValues[j] = masterVals[ind1 + j]
            tmpDistances[j] = masterDistances[ind1 + j] + 1e-30
            distanceSum += 1 / (masterDistances[ind1 + j] + 1e-30)

        for j in range(0, lengthOfVector):
            tmpWeights[j] = (1 / (masterDistances[ind1 + j] + 1e-30)) / distanceSum
            valuesOut[i] = valuesOut[i] + tmpWeights[j] * tmpValues[j]
            magnitudesOut[i] = magnitudesOut[i] + tmpWeights[j] * abs(tmpValues[j])

        # The magnitude of the final grid point is the weight average of the
        # constituent magnitudes. The phase is then given by the phase of the weighted
        # average of the complex values. We found this to give -slightly- better results
        # than just using the weight average of the complex value, most likely due to the fact
        # that any incoherent phases lead to dampening of the magnitudes. This technique
        # obviates this problem
        valuesOut[i] = magnitudesOut[i] * valuesOut[i] / abs(valuesOut[i])

    return valuesOut