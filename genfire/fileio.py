"""
* genfire.fileio *

The primary file input/output module for GENFIRE.


Author: Alan (AJ) Pryor, Jr.
Jianwei (John) Miao Coherent Imaging Group
University of California, Los Angeles
Copyright 2015-2016. All rights reserved.
"""

import numpy as np

def readVolume(filename, order="C"):
    """
    * readVolume *

    Load volume into a numpy array

    :param filename: filename of volume
    :param order: "C" for row-major order (C-style), "F" for column-major (Fortran style)
    :return:

    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved.

    """
    import os
    base, ext = os.path.splitext(filename)
    if (ext == ".mrc"):
        return readMRC(filename, order=order)
    elif (ext == ".mat"):
        return readMAT_volume(filename)
    elif (ext == ".npy"):
        return np.load(filename)

def readMAT_volume(filename):

    #wrapper around scipy's loadmat

    import numpy as np
    import scipy.io
    data = scipy.io.loadmat(filename)
    var_name = [key for key in data if not key.startswith("__")]
    if len(var_name) > 1:
        raise IOError("Only 1 variable allowed per .MAT file")
    else:
        return np.array(data[var_name[0]])

def readNPY(filename, dtype=float, order="C"):
    import numpy as np
    return np.load(filename)

def readMRC(filename, dtype=float, order="C"):
    """
    * readMRC *

    Read in a volume in .mrc file format. See http://bio3d.colorado.edu/imod/doc/mrc_format.txt

    :param filename: Filename of .mrc
    :return: NumPy array containing the .mrc data

    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved.

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

        return np.fromfile(file=fid, dtype=datatype,count=dimx*dimy*dimz).reshape((dimx,dimy,dimz),order=order).astype(dtype)

def writeMRC(filename, arr, datatype='f4', order="C", pixel_size=1):
    """
    * writeMRC *

    Write a volume to .mrc file format. See http://bio3d.colorado.edu/imod/doc/mrc_format.txt
    This version is bare-bones and doesn't write out the full header -- just the critical bits and the
    volume itself

    :param filename: Filename of .mrc file to write
    :param arr: NumPy volume of data to write
    :param dtype: Type of data to write


    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved
    """
    import numpy as np

    dimx, dimy, dimz = np.shape(arr)

    if datatype != arr.dtype:
        arr = arr.astype(datatype)
    # int_header = np.zeros(56,dtype='int32') #must be 4-byte ints
    int_header1 = np.zeros(10,dtype='int32') #must be 4-byte ints
    float_header1 = np.zeros(6,dtype='float32') #must be 4-byte ints
    int_header2 = np.zeros(3,dtype='int32') #must be 4-byte ints
    float_header2 = np.zeros(3,dtype='float32') #must be 4-byte ints
    int_header3 = np.zeros(34,dtype='int32') #must be 4-byte ints


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
    int_header1[:4] = (dimx,dimy,dimz,data_flag)
    int_header1[7:10] = (dimx,dimy,dimz)
    float_header1[:3] = (pixel_size * dimx, pixel_size * dimy, pixel_size * dimz)
    int_header2[:3] = (1, 2, 3)
    float_header2[:3] = np.min(arr), np.max(arr), np.mean(arr)
    char_header = str(' '*800)
    with open(filename,'wb') as fid:
        fid.write(int_header1.tobytes())
        fid.write(float_header1.tobytes())
        fid.write(int_header2.tobytes())
        fid.write(float_header2.tobytes())
        fid.write(int_header3.tobytes())
        fid.write(char_header.encode('UTF-8'))
        fid.write(arr.tobytes(order=order))


def loadProjections(filename):
    """
    * loadProjections *

    Wrapper function for loading in projections of arbitrary (supported) extension

    :param filename: Filename of images to load
    :return: NumPy array containing projections


    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved.
    """
    import os
    filename, file_extension = os.path.splitext(filename)
    if file_extension == ".mat":
        return readMAT_volume(filename + file_extension)
    elif file_extension == ".tif":
        return readTIFF_projections(filename + file_extension)
    elif file_extension == ".mrc":
        return readMRC(filename + file_extension)
    elif file_extension == ".npy":
        return readNPY(filename + file_extension)
    else:
        raise Exception('File format %s not supported.', file_extension)

# def readMAT_projections(filename):
#     """
#     * readMAT *
#
#     Read projections from a .mat file
#
#     :param filename: MATLAB file (.mat) containing projections
#     :return: NumPy array containing projections
#
#
#     Author: Alan (AJ) Pryor, Jr.
#     Jianwei (John) Miao Coherent Imaging Group
#     University of California, Los Angeles
#     Copyright 2015-2016. All rights reserved.
#     """
#
#     import scipy.io
#     import numpy as np
#     import os
#     try: #try to open the projections as a stack
#         projections = scipy.io.loadmat(filename)
#         key = None
#         for k in projections.keys():
#             if k[0] != "_":
#                 key = k
#                 break
#
#         projections = np.array(projections[key])
#     except: ## -- figure out where error is thrown
#          #check if the projections are in individual files
#         flag = True
#         filename_base, file_extension = os.path.splitext(filename)
#         projectionCount = 1
#         while flag: #first count the number of projections so the array can be initialized
#             projectionCount = projectionCount
#             nextFile = filename_base + str(projectionCount) + file_extension
#             if os.path.isfile(nextFile):
#                 projectionCount += 1
#             else:
#                 flag = False
#
#
#         ## open first projection to get dimensions
#         pj = scipy.io.loadmat(filename_base + str(1) + file_extension)
#         pj = pj[projections.keys()[0]]
#         dims = np.shape(pj)
#         #initialize projection array
#         projections = np.zeros((dims[0], dims[1], projectionCount),dtype=int)
#
#         #now actually load in the tiff images
#         for projNum in range(projectionCount):
#             nextFile = filename_base + str(projNum) + file_extension
#             pj = scipy.io.loadmat(filename_base + str(projNum) + file_extension)
#             pj = pj[pj.keys()[0]]
#             projections[:, :, projNum] = np.array(pj)
#
#     return projections


def readTIFF_projections(filename):
    """
    * readTIFF *

    Read (possibly multiple) TIFF images into a NumPy array

    :param filename: Name of TIFF file or TIFF file basename to read. If the filename is a base then
    #       the images must begin with the string contained in filename followed by consecutive integers with
    #       no zero padding, i.e. foo1.tiff, foo2.tiff,..., foo275.tiff
    :return: NumPy array containing projections

    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved.
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

    :param filename_base: Base filename of TIFF
    :param fileNumber: Image number
    :return: Image in a 2D NumPy array

    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved.
    """
    from PIL import Image
    import numpy as np
    nextFile = filename_base + str(fileNumber) + '.tif'
    return np.array(Image.open(nextFile))

def loadAngles(filename):
    """
    * loadAngles *

    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved.

    :param filename:
    :return:
    """
    import os
    base,ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext == ".txt":
        return np.loadtxt(filename,dtype=float)
    elif ext== ".npy":
        return np.load(filename)
    elif ext==".mat":
        from genfire.fileio import readVolume
        return readVolume(filename)
    else:
        raise IOError("Unsupported file extension \"{}\" for Euler angles".format(ext))


def saveResults(reconstruction_outputs, filename):
    """
    * saveResults *

    Helper function to save results of GENFIRE reconstruction

    :param reconstruction_outputs: dictionary containing reconstruction, reciprocal error (errK), and possible R_free
    :param filename: Output filename

    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved
    """
    import os
    fn, ext = os.path.splitext(filename)
    writeVolume(filename, reconstruction_outputs['reconstruction'])
    np.savetxt(fn+'_errK.txt',reconstruction_outputs['errK'])
    if 'R_free_total' in reconstruction_outputs.keys():
        np.savetxt(fn+'_Rfree_total.txt',reconstruction_outputs['R_free_total'])
        np.savetxt(fn+'_Rfree_bybin.txt',reconstruction_outputs['R_free_bybin'])

def writeVolume(filename, data, order="C"):
    """
    * writeVolume *

    Wrapper volume to file

    :param filename: output filename with valid extension
    :param data: numpy volume to write
    :param order: "C" for row-major order (C-style), "F" for column-major (Fortran style)
    """
    import os
    fn, ext = os.path.splitext(filename)
    if ext == ".mrc":
        writeMRC(filename, arr=data, order=order)
    elif ext == ".npy":
        import numpy as np
        np.save(filename,data)
    elif ext == ".mat":
        import scipy.io
        scipy.io.savemat(filename, {"data":data})
    else:
        raise IOError("Unsupported file extension \"{}\" for volume object".format(ext))
