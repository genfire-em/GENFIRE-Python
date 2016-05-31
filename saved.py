import numpy as np
cimport numpy as np
def weightValue(np.ndarray[int, ndim=1] multiInd,np.ndarray[int, ndim=1] uniqueInd, np.ndarray[double, ndim=1] masterDistances, np.ndarray[complex, ndim=1] masterVals):
    outputs = []
    for index in range(0,np.size(multiInd)):
        ind = multiInd[index]
        indVector = range(uniqueInd[ind],uniqueInd[ind+1])
        distances = np.array(masterDistances[indVector]+1e-20)
        complexVals = masterVals[indVector]
        sumDistances = np.sum(distances)
        weights = (((1/distances)*sumDistances) / (np.sum(1/distances)*sumDistances))
        voxel_Mag = np.sum(weights*abs(complexVals))
        voxel_Phase = np.sum(weights*complexVals)
        outputs.append(voxel_Mag*voxel_Phase/abs(voxel_Phase))
    return outputs

    # print complexVals[0]
    # d = []
    # print a
    # print b
    # d.append(a[0,0])
    # d.append(b[0])
    # print d




#
# def weightValue(int ind):
#     global distances, masterVals, masterDistances, uniqueInd, multiInd
#     ind = multiInd[0][index]
#     indVector = range(uniqueInd[ind],uniqueInd[ind+1])
#     distances = np.array(masterDistances[indVector]+1e-10)
#     complexVals = masterVals[indVector]
#     sumDistances = np.sum(distances)
#     weights = (((1/distances)*sumDistances) / (np.sum(1/distances)*sumDistances))
#     voxel_Mag = np.sum(weights*abs(complexVals))
#     voxel_Phase = np.sum(weights*complexVals)
#     return voxel_Mag*voxel_Phase/abs(voxel_Phase)
