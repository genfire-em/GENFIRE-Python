from __future__ import division
import numpy as np
import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
import scipy.fftpack as sfft
import os
import scipy.io as io
import pyfftw
import time
import GENFIRE_from_GUI_Input
import misc
import itertools
from multiprocessing import Pool,Array
PI = 3.14159265359

def fillInFourierGrid(projections,angles,interpolationCutoffDistance):
        print("correct version")
        # import time
        tic1 = time.time()
        # print angles
        dim1 = np.shape(projections)[0]
        dim2 = np.shape(projections)[1]
        if len(np.shape(projections))>2:
            numProjections = np.shape(projections)[2]
        else:
            numProjections = 1
        # print np.shape(projections)
        nc = np.round(dim1/2)
        n2 = nc
        # measuredY = zeros(1,size(projections,2)*size(projections,1),size(projections,3),'single');
        measuredX = np.zeros([dim1*dim2,numProjections])
        measuredY = np.zeros([dim1*dim2,numProjections])
        measuredZ = np.zeros([dim1*dim2,numProjections])
        kMeasured = np.zeros([dim1,dim1,numProjections], dtype=complex)
        confidenceWeights = np.zeros([dim1,dim1,numProjections]) #do I need this??
        # print "shape angles" ,np.shape(angles)
        ky,kx = np.meshgrid(np.arange(-n2,n2,1),np.arange(-n2,n2,1))
        Q = np.sqrt(ky**2+kx**2)/n2
        kx = np.reshape(kx, [1, dim1*dim2], 'F')
        ky = np.reshape(ky, [1, dim1*dim2], 'F')
        kz = np.zeros([1, dim1*dim1])
        for projNum in range(0, numProjections):
        #for projNum in range(0,1):
            # print "projNum", projNum
            phi = angles[0, projNum] * PI/180
            theta = angles[1, projNum] * PI/180
            psi = angles[2, projNum] * PI/180
            R = np.array([[np.cos(psi)*np.cos(theta)*np.cos(phi)-np.sin(psi)*np.sin(phi) ,np.cos(psi)*np.cos(theta)*np.sin(phi)+np.sin(psi)*np.cos(phi)   ,    -np.cos(psi)*np.sin(theta)],
            [-np.sin(psi)*np.cos(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi), -np.sin(psi)*np.cos(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi) ,   np.sin(psi)*np.sin(theta) ],
            [np.sin(theta)*np.cos(phi)                               , np.sin(theta)*np.sin(phi)                                ,              np.cos(theta)]])
            R = R.T
            # print R

            # print np.shape(kx)
            # print np.shape(kz)

            Kcoordinates = np.zeros([3, dim1*dim2])
            Kcoordinates[0, :] = kx
            Kcoordinates[1, :] = ky
            Kcoordinates[2, :] = kz

            # print kx[0,0:5]
            # print ky[0,0:5]
            # print np.shape(Kcoordinates)
            #
            # print Kcoordinates[0,0:5]
            # print Kcoordinates[1,0:5]
            # print Kcoordinates[2,0:5]
            rotkCoords = np.dot(R, Kcoordinates)
            # print rotkCoords[0, :]
            # print np.shape(Q)
            confidenceWeights[:, :, projNum] = np.ones_like(Q) #this implementation does not support individual projection weighting, so just set all weights to 1
            measuredX[:, projNum] = rotkCoords[0, :]
            measuredY[:, projNum] = rotkCoords[1, :]
            measuredZ[:, projNum] = rotkCoords[2, :]
            # print np.shape(np.fft.fftn(projections[:, :, projNum]))
            kMeasured[:, :, projNum] = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(projections[:, :, projNum])))
            # print kMeasured[0,0,0]
            # print kMeasured[1,0,0]
            # print 1/n2
            # print ky[0:5,0:5]

        # del projections, rotkCoords
        # print np.size(measuredX)
        measuredX = np.reshape(measuredX,[1, np.size(kMeasured)], 'F')
        measuredY = np.reshape(measuredY,[1, np.size(kMeasured)], 'F')
        measuredZ = np.reshape(measuredZ,[1, np.size(kMeasured)], 'F')
        kMeasured = np.reshape(kMeasured,[1, np.size(kMeasured)], 'F')
        confidenceWeights = np.reshape(confidenceWeights,[1, np.size(kMeasured)], 'F')
        notFlaggedIndices = kMeasured != -999
        measuredX = measuredX[notFlaggedIndices]
        measuredY = measuredY[notFlaggedIndices]
        measuredZ = measuredZ[notFlaggedIndices]
        kMeasured = kMeasured[notFlaggedIndices]
        confidenceWeights = confidenceWeights[notFlaggedIndices]

        masterInd = []
        masterVals = []
        masterDistances = []
        masterConfidenceWeights = []
        # shiftMax = int(round(interpolationCutoffDistance))
        shiftMax = 0

        print ("time 1" , time.time()-tic1)
        tic2 = time.time()
        for Yshift in range(-shiftMax, shiftMax+1):
            for Xshift in range(-shiftMax, shiftMax+1):
                for Zshift in range(-shiftMax, shiftMax+1):

                    tmpX = np.round(measuredX) + Xshift
                    tmpY = np.round(measuredY) + Yshift
                    tmpZ = np.round(measuredZ) + Zshift


                    tmpVals = kMeasured
                    tmpConfidenceWeights = confidenceWeights
                    distances = np.sqrt(abs(measuredX-tmpX)**2 + abs(measuredY-tmpY)**2 + abs(measuredZ-tmpZ)**2)
                    tmpX+=nc
                    tmpY+=nc
                    tmpZ+=nc

                    goodInd = (np.logical_not((tmpX > (dim1-1)) | (tmpX < 0) | (tmpY > (dim1-1)) | (tmpY < 0) | (tmpZ > (dim1-1)) | (tmpZ < 0))) & (distances <= interpolationCutoffDistance)
                    # masterInd+=np.ravel_multi_index((tmpX[goodInd], tmpY[goodInd], tmpZ[goodInd]),[dim1, dim1, dim1], order='F')
                    # masterVals+=tmpVals[goodInd]
                    # masterDistances+=distances[goodInd]
                    # masterConfidenceWeights+=tmpConfidenceWeights[goodInd]
                    # print (tmpX[goodInd], tmpY[goodInd], tmpZ[goodInd])
                    # print np.ravel_multi_index((tmpX[goodInd].astype(np.int64), tmpY[goodInd].astype(np.int64), tmpZ[goodInd].astype(np.int64)),[dim1, dim1, dim1], order='F')
                    # masterInd+=np.ndarray.tolist(np.ravel_multi_index((tmpX[goodInd].astype(np.int64), tmpY[goodInd].astype(np.int64), tmpZ[goodInd].astype(np.int64)),[dim1, dim1, dim1], order='F'))
                    # # masterInd+=np.ndarray.tolist(np.ravel_multi_index((tmpX[goodInd], tmpY[goodInd], tmpZ[goodInd]),[dim1, dim1, dim1], order='F'))
                    # masterVals+=np.ndarray.tolist(tmpVals[goodInd])
                    # masterDistances+=np.ndarray.tolist(distances[goodInd])
                    # masterConfidenceWeights+=np.ndarray.tolist(tmpConfidenceWeights[goodInd])

                    masterInd=np.append(masterInd, np.ravel_multi_index((tmpX[goodInd].astype(np.int64), tmpY[goodInd].astype(np.int64), tmpZ[goodInd].astype(np.int64)),[dim1, dim1, dim1], order='F'))
                    masterVals=np.append(masterVals, tmpVals[goodInd])
                    masterDistances=np.append(masterDistances, distances[goodInd])
                    masterConfidenceWeights=np.append(masterConfidenceWeights, tmpConfidenceWeights[goodInd])


                    # print goodInd[0:60]
                    t = 0;

        masterInd = np.array(masterInd).astype(np.int64)
        masterVals = np.array(masterVals)
        masterDistances = np.array(masterDistances)
        masterConfidenceWeights = np.array(masterConfidenceWeights)



        sortIndices = np.argsort(masterInd)
        masterInd = masterInd[sortIndices]
        masterVals = masterVals[sortIndices]
        masterDistances = masterDistances[sortIndices]
        masterConfidenceWeights = masterConfidenceWeights[sortIndices]
        # tmp2 = masterInd[sortIndices]


        halfwayCutoff = ((dim1+1)**3)//2+1


        masterVals = masterVals[masterInd <= halfwayCutoff]
        masterDistances = masterDistances[masterInd <= halfwayCutoff]
        masterConfidenceWeights = masterConfidenceWeights[masterInd <= halfwayCutoff]
        masterInd = masterInd[masterInd <= halfwayCutoff]


        uniqueVals, uniqueInd = np.unique(masterInd, return_index=True)

        uniqueInd = np.append(uniqueInd, 0)

        diffVec = np.diff(uniqueInd)
        singleInd = diffVec == 1
        # multiInd = diffVec != 1
        multiInd = np.where(diffVec != 1)
        # print masterInd[multiInd[0][0]-2:multiInd[0][0]+3]
        # measuredK = np.zeros([dim1, dim1, dim1], dtype=complex)
        measuredK = np.zeros([dim1**3], dtype=complex)

        # singleIndtoXYZ = np.unravel_index(uniqueVals[singleInd], [dim1, dim1, dim1], order='F')
        measuredK[uniqueVals[singleInd]] = masterVals[uniqueInd[0:-1][singleInd]]


        print ("time 2 ", time.time()-tic2)
        tic3 = time.time()
        tic3 = time.time()
        t1 = time.time()
# ####
#         def weightValue(index):
#             ind = multiInd[0][index]
#             indVector = range(uniqueInd[ind],uniqueInd[ind+1])
#             distances = np.array(masterDistances[indVector]+1e-10)
#             complexVals = masterVals[indVector]
#             sumDistances = np.sum(distances)
#             weights = (((1/distances)*sumDistances) / (np.sum(1/distances)*sumDistances))
#             voxel_Mag = np.sum(weights*abs(complexVals))
#             voxel_Phase = np.sum(weights*complexVals)
#             return voxel_Mag*voxel_Phase/abs(voxel_Phase)

        def weightValue(index):
            ind = multiInd[0][index]
            indVector = range(uniqueInd[ind],uniqueInd[ind+1])
            distances = np.array(masterDistances[indVector]+1e-10)
            complexVals = masterVals[indVector]
            sumDistances = np.sum(distances)
            weights = (((1/distances)*sumDistances) / (np.sum(1/distances)*sumDistances))
            voxel_Mag = np.sum(weights*abs(complexVals))
            voxel_Phase = np.sum(weights*complexVals)
            return voxel_Mag*voxel_Phase/abs(voxel_Phase)

        # for index in range(0,np.size(multiInd)):
        #     weightValue(index)
        #
        # debugFile = {'mk': measuredK}
        # scipy.io.savemat('debug.mat',debugFile)
        # vals = np.array(list(itertools.imap(weightValue,np.arange(0,np.size(multiInd)))))
        # v = list(vals)
        #
        # pool = Pool(processes=1)
        # multiInd = Array('d',multiInd)
        # results = pool.map(weightValue, np.arange(0,np.size(multiInd)))
        # results = pool.map()
        #
        # outputs = [results[0] for result in results]
        # print "HERE,", np.shape(outputs)

        time5 = time.time()
        vals = list(itertools.imap(weightValue,np.arange(0,np.size(multiInd))))
        print "BIG TIME2, " ,time.time()-time5
        testTime = time.time()
        # bigTime = time.time()
        measuredK[uniqueVals[multiInd[0][:]]] = vals
        # print "INDEX TIME", time.time()-bigTime
        measuredK = np.reshape(measuredK,[dim1,dim1,dim1],order='F')

        print ("time3 " , time.time()-tic3)
        measuredK[np.isnan(measuredK)] = 0
        measuredK = misc.hermitianSymmetrize(measuredK)

        print ("GENFIRE: Fourier grid assembled in %d seconds" % (time.time()-tic1))
        return measuredK

projections = io.loadmat('projections.mat')['projections']
dims = np.shape(projections)
paddedDim = dims[0] * 3
padding = int((paddedDim-dims[0])/2)
projections = np.pad(projections,((padding,padding),(padding,padding),(0,0)),'constant')
angles = io.loadmat('angles.mat')['angles']
icd = 0.7
fillInFourierGrid(projections,angles,icd)
# print locals()
