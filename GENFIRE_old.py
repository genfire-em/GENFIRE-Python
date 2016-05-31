from __future__ import division
import numpy as np
import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
import scipy.fftpack as sfft
import os
import scipy.io
import pyfftw
import time
import GENFIRE_from_GUI_Input
import misc
import itertools

# from weightVals import weightValue

from multiprocessing import Pool, Array


PI = 3.14159265359


if __name__ != "__main__":

    def GENFIRE_iterate(numIterations,initialObject,support,measuredK,constraintConfidenceWeights,confidenceCutoffs,R_freeInd_complex,R_freeVals_complex,displayFigure):
        # print "starting", confidenceCutoffs
        NUMTHREADS = 6
        bestErr = 1e30
        Rfree_complex = np.ones(numIterations)*-1
        errK = np.zeros(numIterations)
        errInd = measuredK != 0
        dims = np.shape(support)
        if R_freeInd_complex:
            Rfree_complex = np.zeros((np.shape(R_freeInd_complex)[2], numIterations))
            print "startdeb"
            print np.shape(R_freeInd_complex)
            print type(R_freeInd_complex)
            print np.shape(R_freeVals_complex)
            print type(R_freeVals_complex)
        if displayFigure.DisplayFigureON: #setup some indices for plotting.
            # print "figure plotting on"
            n_half_x = dims[0]/2 #this assumes even-sized arrays
            n_half_y = dims[1]/2
            n_half_z = dims[2]/2

            half_window_x = displayFigure.reconstructionDisplayWindowSize[0]//2
            half_window_y = displayFigure.reconstructionDisplayWindowSize[1]//2
            half_window_z = displayFigure.reconstructionDisplayWindowSize[2]//2


        if R_freeInd_complex:

            outputs = {'reconstruction':initialObject,'errK':errK,'R_free':Rfree_complex}
        else:
            outputs = {'reconstruction':initialObject,'errK':errK}


        iterationNumsToChangeCutoff = np.round(np.linspace(1, numIterations, num=np.size(confidenceCutoffs)))
        iterationNumsToChangeCutoff, uniqueIndices = np.unique(iterationNumsToChangeCutoff, return_index=True)
        iterationNumsToChangeCutoff = np.append(iterationNumsToChangeCutoff,1e30)
        confidenceCutoffs = confidenceCutoffs[uniqueIndices]
        # print "iterationNumsToChangeCutoff",iterationNumsToChangeCutoff
        currentCutoffNum = 0
        for iterationNum in range(1, numIterations+1):

            if iterationNum == iterationNumsToChangeCutoff[currentCutoffNum]:
                relevantCutoff = confidenceCutoffs[currentCutoffNum]
                constraintInd_complex = (constraintConfidenceWeights > relevantCutoff) * measuredK != 0

                bestErr = 1e30 #reset error
                currentCutoffNum+=1#update constraint set number


            if iterationNum%25==0:
                print ("iteration number: ", iterationNum)
            initialObject[initialObject<0] = 0 #enforce positivty
            initialObject = initialObject * support #enforce support

            if displayFigure.DisplayFigureON:
                if iterationNum % displayFigure.displayFrequency == 0:

                    plt.figure(1)
                    plt.subplot(233)
                    plt.imshow(np.squeeze(np.fft.ifftshift(initialObject)[n_half_x, n_half_y-half_window_y:n_half_y+half_window_y, n_half_z-half_window_z:n_half_z+half_window_z]))
                    plt.title("central YZ slice")
                    plt.subplot(232)
                    plt.imshow(np.squeeze(np.fft.ifftshift(initialObject)[n_half_x-half_window_x:n_half_x+half_window_x, n_half_y, n_half_z-half_window_z:n_half_z+half_window_z]))
                    plt.title("central XZ slice")
                    plt.subplot(231)
                    plt.title("central XY slice")
                    plt.imshow(np.squeeze(np.fft.ifftshift(initialObject)[n_half_x-half_window_x:n_half_x+half_window_x, n_half_y-half_window_y:n_half_y+half_window_y, n_half_z]))
                    plt.subplot(236)
                    plt.title("YZ projection")
                    plt.imshow(np.squeeze(np.sum(np.fft.ifftshift(initialObject)[n_half_x-half_window_x:n_half_x+half_window_x, n_half_y-half_window_y:n_half_y+half_window_y, n_half_z-half_window_z:n_half_z+half_window_z], axis=0)))
                    plt.subplot(235)
                    plt.title("XZ projection")
                    plt.imshow(np.squeeze(np.sum(np.fft.ifftshift(initialObject)[n_half_x-half_window_x:n_half_x+half_window_x, n_half_y-half_window_y:n_half_y+half_window_y, n_half_z-half_window_z:n_half_z+half_window_z], axis=1)))
                    plt.subplot(234)
                    plt.title("XY projection")
                    plt.imshow(np.squeeze(np.sum(np.fft.ifftshift(initialObject)[n_half_x-half_window_x:n_half_x+half_window_x, n_half_y-half_window_y:n_half_y+half_window_y, n_half_z-half_window_z:n_half_z+half_window_z], axis=2)))
                    #plt.imshow(abs((constraintConfidenceWeights)[:, :, 0]))
                    plt.get_current_fig_manager().window.setGeometry(25,25,400, 400)
                    plt.draw()

                    plt.figure(2)
                    plt.get_current_fig_manager().window.setGeometry(25,450,400, 400)
                    # mngr = plt.get_current_fig_manager()
                    # mngr.window.wm_geometry("+600+400")

                    # mngr.window.setGeometry(50,100,640, 545)
                    plt.hold(False)
                    plt.plot(range(0,numIterations),errK)
                    plt.title("K-space Error vs Iteration Number")
                    plt.xlabel("Spatial Frequency (% of Nyquist)")
                    plt.ylabel('Reciprocal Space Error')
                    plt.draw()

                    if R_freeInd_complex:
                        plt.figure(3)
                        mngr = plt.get_current_fig_manager()
                        mngr.window.setGeometry(450,25,400, 400)
                        plt.plot(range(0,numIterations),np.mean(Rfree_complex,axis=0))
                        plt.title("Mean R-free Value vs Iteration Number")
                        plt.xlabel("Iteration Num")
                        plt.ylabel('Mean R-free')
                        plt.draw()
                        plt.figure(4)
                        mngr = plt.get_current_fig_manager()
                        mngr.window.setGeometry(450,450,400, 400)
                        plt.hold(False)
                        print np.shape(np.mean(Rfree_complex,axis=1))
                        # print np.shape(range(0,np.shape(np.mean(Rfree_complex,axis=1))[1]))
                        X = np.linspace(0,1,np.shape(Rfree_complex)[0])
                        plt.plot(X, np.mean(Rfree_complex,axis=1))
                        plt.hold(False)
                        plt.title("Current Rfree Value vs Spatial Frequency")
                        plt.xlabel("Spatial Frequency (% of Nyquist)")
                        plt.ylabel('Rfree')
                        plt.draw()


                    plt.pause(.001)

            # k = pyfftw.interfaces.numpy_fft.rfftn(initialObject,overwrite_input=True)
            k = pyfftw.interfaces.numpy_fft.rfftn(initialObject,overwrite_input=True,threads=NUMTHREADS)
            print "SIZE K", np.shape(k)
            # k = np.fft.rfftn(initialObject)
            errK[iterationNum-1] = np.sum(abs(k[errInd]-measuredK[errInd]))/np.sum(abs(measuredK[errInd]))#monitor error
            # print errK[iterationNum-1]
            if errK[iterationNum-1] < bestErr:
                bestErr = errK[iterationNum-1]
                outputs['reconstruction'] = initialObject
            if R_freeInd_complex:
                for shellNum in range(0, np.shape(R_freeInd_complex)[2]):
                    # print "LEN",len(R_freeInd_complex)
                    # print np.shape(R_freeInd_complex)
                    # print len(R_freeInd_complex)
                    # print type(R_freeInd_complex)
                    # print np.shape(R_freeInd_complex)
                    # print "flag ", np.shape(R_freeInd_complex[0])
                    # tmpInd = R_freeInd_complex[shellNum, :, 0]
                    # print np.shape(R_freeInd_complex)
                    # tmpInd = R_freeInd_complex[shellNum][:]
                    # print "test test",np.shape(R_freeInd_complex)
                    tmpIndX = R_freeInd_complex[0][0][shellNum]
                    tmpIndY = R_freeInd_complex[1][0][shellNum]
                    tmpIndZ = R_freeInd_complex[2][0][shellNum]
                    # print "tmpIndZ shape" , np.shape(measuredK)
                    # print np.shape(tmpIndZ)
                    # print "MAX", np.max(np.max(np.max(tmpIndZ)))
                    # if shellNum==1 & iterationNum ==1:
                    #     deb = {'tmpIndX':tmpIndX,'tmpIndY':tmpIndY,'tmpIndZ':tmpIndZ, 'measuredK':measuredK,'k':k}
                    #     import scipy.io as io
                    #     io.savemat('debug3',deb)
                    #     print "success"
                    # print "tmpInd", np.shape(tmpInd)
                    # print "shape", np.shape(tmpInd)
                    # print type(tmpInd)
                    # print "DEB"
                    # print np.shape(tmpInd)
                    # print type(tmpInd)
                    tmpVals = R_freeVals_complex[shellNum]
                    # print "size k", np.shape(k)
                    # Rfree_complex[shellNum, iterationNum-1] = np.sum(abs(k[tmpInd] - tmpVals)) / np.sum(abs(tmpVals))
                    Rfree_complex[shellNum, iterationNum-1] = np.sum(abs(k[tmpIndX, tmpIndY, tmpIndZ] - tmpVals)) / np.sum(abs(tmpVals))



            # if R_freeInd_complex:
            #     Rfree_complex[iterationNum-1] = np.sum(abs(k[R_freeInd_complex[0,:],R_freeInd_complex[1,:] ,R_freeInd_complex[2,:] ]-R_freeVals_complex))/np.sum(abs(R_freeVals_complex))
            k[constraintInd_complex] = measuredK[constraintInd_complex]
            # initialObject = pyfftw.interfaces.numpy_fft.irfftn(k,overwrite_input=True)
            initialObject = pyfftw.interfaces.numpy_fft.irfftn(k,overwrite_input=True,threads=NUMTHREADS)

            # initialObject = np.fft.irfftn(k)

            # plt.figure(5)
        outputs['errK'] = errK
        if R_freeInd_complex:
            outputs['R_free'] = Rfree_complex
        outputs['reconstruction'] = np.fft.fftshift(outputs['reconstruction'])
        return outputs


    def generateKspaceIndices(obj):
        #testing
        dims = np.shape(obj)
        if len(dims) < 3:
            dims = dims + (0,)
        # print dims

        if dims[0] % 2 == 0:
            ncK0 = dims[0]/2
            vec0 = np.arange(-ncK0, ncK0, 1)/ncK0
        elif dims[0] == 1:
            vec0 = 0
            ncK0 = 1

        else:
            ncK0 = ((dims[0]+1)/2)-1
            #n2K0 = ncK0-1
            vec0 = np.arange(-ncK0, ncK0+1)/ncK0


        if dims[1] % 2 == 0:
            ncK1 = dims[1]/2
            vec1 = np.arange(-ncK1, ncK1, 1)/ncK1
        elif dims[1] == 1:
            vec1 = 0
            ncK1 = 1

        else:
            ncK1 = ((dims[1]+1)/2)-1
            #n2K1 = ncK1-1
            vec1 = np.arange(-ncK1, ncK1+1)/ncK1


        if dims[2] % 2 == 0:
            ncK2 = dims[2]/2
            vec2 = np.arange(-ncK2, ncK2, 1)/ncK2
        elif dims[2] == 1:
            vec2 = 0
            ncK2 = 1

        else:
            ncK2 = ((dims[2]+1)/2)-1
            #n2K2 = ncK2-1
            vec2 = np.arange(-ncK2, ncK2+1)/ncK2

            # print "bla"
        kx, ky, kz = np.meshgrid(vec1,vec0,vec2)
        Kindices = np.sqrt(kx**2 + ky**2 + kz**2)
        return Kindices






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

        vals = list(itertools.imap(weightValue,np.arange(0,np.size(multiInd))))
        testTime = time.time()
        measuredK[uniqueVals[multiInd[0][:]]] = vals
        measuredK = np.reshape(measuredK,[dim1,dim1,dim1],order='F')

        print ("time3 " , time.time()-tic3)
        measuredK[np.isnan(measuredK)] = 0
        measuredK = misc.hermitianSymmetrize(measuredK)

        print ("GENFIRE: Fourier grid assembled in %d seconds" % (time.time()-tic1))
        return measuredK




    def loadProjections(filename):

        filename, file_extension = os.path.splitext(filename)
        if file_extension == ".mat":
            print ("GENFIRE: reading projections from MATLAB file.\n")
            return myReadMATLAB(filename)
        elif file_extension == ".tif":
            print ("GENFIRE: reading projections from .tif file.\n")
            return myReadTiff(filename)
        elif file_extension == ".mrc":
            print ("GENFIRE: reading projections from .mrc file.\n")
            return myReadMRC(filename)
        else:
            raise Exception('GENFIRE: File format %s not supported.', file_extension)
        # fileIOdict =
        #     '.mat': myReadMATLAB(file_extension),
        #     '.tif': myReadTiff(file_extension),
        #     '.mrc': myReadMRC(file_extension)
        # }
        # if file_extension in fileIOdict:
        #     fileIOdict[file_extension]
        # else:
        #     raise Exception('GENFIRE: File format %s not supported.', file_extension)
        # print file_extension

    def myReadMATLAB(filename):
        projections = scipy.io.loadmat(filename);projections = projections[projections.keys()[0]]
        print ("MATLAB!")
        return projections

    def myReadTiff(filename):
        print ("tif!")
       # return projections

    def myReadMRC(filename):
        print ("mrc!")
        #return projections

    class DisplayFigure:
        def __init__(self):
            self.DisplayFigureON = False
            self.DisplayErrorFigureON = False
            self.displayFrequency = 5
            self.reconstructionDisplayWindowSize = 0