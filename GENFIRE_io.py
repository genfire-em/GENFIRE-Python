def readMAT(filename):
    import scipy.io
    import numpy as np
    import os
    """
    * readMAT *

    Read projections from a .mat file

    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved.

    :param filename: MATLAB file (.mat) containing projections
    :return: NumPy array containing projections
    """

    try: #try to open the projections as a stack
        projections = scipy.io.loadmat(filename)
        key = None
        for k in projections.keys():
            if k[0] != "_":
                key = k
                break

        projections = np.array(projections[key])
    except: ## -- figure out where error is thrown
         #check if the projections are in individual files
        flag = True
        filename_base, file_extension = os.path.splitext(filename)
        projectionCount = 1
        while flag: #first count the number of projections so the array can be initialized
            projectionCount = projectionCount
            nextFile = filename_base + str(projectionCount) + file_extension
            if os.path.isfile(nextFile):
                projectionCount += 1
            else:
                flag = False


        ## open first projection to get dimensions
        pj = scipy.io.loadmat(filename_base + str(1) + file_extension)
        pj = pj[projections.keys()[0]]
        dims = np.shape(pj)
        #initialize projection array
        projections = np.zeros((dims[0], dims[1], projectionCount),dtype=int)

        #now actually load in the tiff images
        for projNum in range(projectionCount):
            nextFile = filename_base + str(projNum) + file_extension
            pj = scipy.io.loadmat(filename_base + str(projNum) + file_extension)
            pj = pj[pj.keys()[0]]
            projections[:, :, projNum] = np.array(pj)

    return projections


def readTIFF(filename):
    """
    * readTIFF *

    Read (possibly multiple) TIFF images into a NumPy array

    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved.

    :param filename: Name of TIFF file or TIFF file basename to read. If the filename is a base then
    #       the images must begin with the string contained in filename followed by consecutive integers with
    #       no zero padding, i.e. foo1.tiff, foo2.tiff,..., foo275.tiff
    :return: NumPy array containing projections
    """
    import functools
    from PIL import Image
    import os
    import numpy as np
    from multiprocessing import Pool
    try:
        projections = np.array(Image.open(filename))
    except:
        flag = True
        filename_base, file_extension = os.path.splitext(filename)
        projectionCount = 1
        while flag: #first count the number of projections so the array can be initialized
            projectionCount = projectionCount
            nextFile = filename_base + str(projectionCount) + file_extension
            if os.path.isfile(nextFile):
                projectionCount += 1
            else:
                flag = False

        ## open first projection to get dimensions
        dims = np.shape(Image.open(filename_base + str(1) + file_extension))

        #initialize projection array
        projections = np.zeros((dims[0], dims[1], projectionCount),dtype=int)

        pool = Pool(4)
        func = functools.partial(readInTiffProjection, filename_base)
        pj = pool.map(func, range(projectionCount))
        for j  in range(projectionCount):
            projections[:, :, j] = pj[j]
    return projections

def readInTiffProjection(filename_base, fileNumber):
    """
    * readInTiffProjection *

    Reads and returns a single TIFF image as a NumPy array

    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved.

    :param filename_base: Base filename of TIFF
    :param fileNumber: Image number
    :return: Image in a 2D NumPy array
    """
    from PIL import Image
    import numpy as np
    nextFile = filename_base + str(fileNumber) + '.tif'
    return np.array(Image.open(nextFile))

def readMRC(filename, dtype=float,order="C"):
    """
    * readMRC *

    Read in a volume in .mrc file format. See http://bio3d.colorado.edu/imod/doc/mrc_format.txt

    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved.

    :param filename: Filename of .mrc
    :return: NumPy array containing the .mrc data
    """
    import numpy as np
    import struct
    headerIntNumber = 56
    sizeof_int = 4
    headerCharNumber = 800
    sizeof_char = 1
    with open(filename,'rb') as fid:
        int_header = struct.unpack('=' + 'i'*headerIntNumber, fid.read(headerIntNumber * sizeof_int))
        char_header = struct.unpack('=' + 'c'*headerCharNumber, fid.read(headerCharNumber * sizeof_char))
        dimx, dimy, dimz, data_flag= int_header[:4]
        print "reading, ", dimx,dimy,dimz
        if (data_flag == 0):
            datatype='u1'
        elif (data_flag ==1):
            datatype='i1'
        elif (data_flag ==2):
            datatype='f4'
        elif (data_flag ==3):
            datatype='c'
        elif (data_flag ==4):
            datatype='f4'
        elif (data_flag ==6):
            datatype='u2'
        else:
            raise ValueError("No supported datatype found!\n")
        print dimx,dimy,dimz
        print datatype
        return np.fromfile(file=fid, dtype=datatype,count=dimx*dimy*dimz).reshape((dimx,dimy,dimz),order=order).astype(dtype)

def writeMRC(filename, arr, datatype='f4'):
    """
    * writeMRC *

    Write a volume to .mrc file format. See http://bio3d.colorado.edu/imod/doc/mrc_format.txt

    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved

    :param filename: Filename of .mrc file to write
    :param arr: NumPy volume of data to write
    :param dtype: Type of data to write
    """
    import numpy as np
    dimx, dimy, dimz = np.shape(arr)
    if datatype != arr.dtype:
        arr = arr.astype(datatype)
    int_header = np.zeros(56,dtype='int32')

    if (datatype == 'u1'):
        data_flag = 0
    elif (datatype =='i1'):
        data_flag = 1
    elif (datatype =='f4'):
        data_flag = 2
    elif (datatype =='c'):
        data_flag = 3
    elif (datatype =='f4'):
        data_flag = 4
    elif (datatype =='u2'):
        data_flag = 6
    else:
        raise ValueError("No supported datatype found!\n")

    int_header[:4] = (dimx,dimy,dimz,data_flag)
    char_header = str(' '*800)
    with open(filename,'wb') as fid:
        fid.write(int_header.tobytes())
        fid.write(char_header)
        fid.write(arr.tobytes())