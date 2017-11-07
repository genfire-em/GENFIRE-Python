"""
* genfire.utility *

This module contains various useful functions that do not fit into another category


Author: Alan (AJ) Pryor, Jr.
Jianwei (John) Miao Coherent Imaging Group
University of California, Los Angeles
Copyright 2015-2016. All rights reserved.
"""


from __future__ import division
import numpy as np
from scipy.interpolate import RegularGridInterpolator
PI = np.pi

try:
    import pyfftw

    #These are just wrappers around various flavors of the FFT
    def rfftn(arr, threads=6):
        return pyfftw.interfaces.numpy_fft.rfftn(arr,overwrite_input=True,threads=threads)
    def irfftn(arr, threads=6):
        return pyfftw.interfaces.numpy_fft.irfftn(arr,overwrite_input=True,threads=threads)
    def rfftn_fftshift(arr, threads=6):
        return pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.rfftn(pyfftw.interfaces.numpy_fft.ifftshift(arr),overwrite_input=True,threads=threads))
    def irfftn_fftshift(arr, threads=6):
        return pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.irfftn(pyfftw.interfaces.numpy_fft.ifftshift(arr),overwrite_input=True,threads=threads))
    def fftn(arr, threads=6):
        return pyfftw.interfaces.numpy_fft.fftn(arr,overwrite_input=True,threads=threads)
    def ifftn(arr, threads=6):
        return pyfftw.interfaces.numpy_fft.ifftn(arr,overwrite_input=True,threads=threads)
    def fftn_fftshift(arr, threads=6):
        return pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(pyfftw.interfaces.numpy_fft.ifftshift(arr),overwrite_input=True,threads=threads))
    def ifftn_fftshift(arr, threads=6):
        return pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.ifftshift(arr),overwrite_input=True,threads=threads))

except ImportError:
    print("No pyFFTW installation found. Installing pyFFTW will significantly improve the speed of GENFIRE. Consult the documentation for more information. Continuing with NumPy FFT...")
    import numpy as np
    def rfftn(arr):
        return np.fft.rfftn(arr)
    def irfftn(arr):
        return np.fft.irfftn(arr)
    def rfftn_fftshift(arr):
        return np.fft.fftshift(np.fft.rfftn(np.fft.ifftshift(arr)))
    def irfftn_fftshift(arr):
        return np.fft.fftshift(np.fft.irfftn(np.fft.ifftshift(arr)))
    def fftn(arr):
        return np.fft.fftn(arr)
    def ifftn(arr):
        return np.fft.ifftn(arr)
    def fftn_fftshift(arr):
        return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(arr)))
    def ifftn_fftshift(arr):
        return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(arr)))


def hermitianSymmetrize(volume):
    """
    * hermitianSymmetrize *

    Enforce Hermitian symmetry to volume

    :param volume: 3D volume to symmetrize
    :return: symmetrized volume
    
    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright (c) 2016. All Rights Reserved.
    """

    startDims = np.shape(volume) # initial dimensions

    # remember the initial dimensions for the end
    dimx = startDims[0]
    dimy = startDims[1]
    dimz = startDims[2]
    flag = False # flag to trigger copying to new odd dimensioned array

    #check if any dimension is odd
    if dimx % 2 == 0:
        dimx += 1
        flag = True

    if dimy % 2 == 0:
        dimy += 1
        flag = True

    if dimz % 2 == 0:
        dimz += 1
        flag = True

    if flag: # if any dimensions are even, create a new with all odd dimensions and copy volume
        newInput = np.zeros((dimx,dimy,dimz), dtype=complex) #new array
        newInput[:startDims[0], :startDims[1], :startDims[2]] = volume # copy values
        numberOfValues = (newInput != 0).astype(float) #track number of values for averaging
        newInput = newInput + np.conj(newInput[::-1, ::-1, ::-1]) # combine Hermitian symmetry mates
        numberOfValues = numberOfValues + numberOfValues[::-1, ::-1, ::-1] # track number of points included in each sum

        newInput[numberOfValues != 0] /= numberOfValues[numberOfValues != 0] # take average where two values existed
        newInput[np.isnan(newInput)] = 0 # Shouldn't be any nan, but just to be safe

        return newInput[:startDims[0], :startDims[1], :startDims[2]] # return original dimensions


    else: # otherwise, save yourself the trouble of copying the matrix over. See previous comments for line-by-line
        numberOfValues = (volume != 0).astype(float)
        volume = volume + np.conjugate(volume[::-1, ::-1, ::-1])
        numberOfValues = numberOfValues + numberOfValues[::-1, ::-1, ::-1]
        volume[numberOfValues != 0] /= numberOfValues[numberOfValues != 0]
        volume[np.isnan(volume)] = 0
        return volume


def smooth3D(object,resolutionCutoff):
    """
    * smooth3D *

    Low pass filter a 3D volume

    :param object: 3D volume
    :param resolutionCutoff: Fraction of Nyquist frequency to use as sigma for the Gaussian filter
    :return: smoothed volume

    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright (c) 2016. All Rights Reserved.
    """
    dims = np.shape(object)
    if dims[2] == 1:
        raise Exception('This is not a 3D object, use smooth2D instead.')
    Xcenter = round(dims[0]//2)
    Ycenter = round(dims[1]//2)
    Zcenter = round(dims[2]//2)

    # These  next few lines are easier to read, but in the actual code I just directly
    # place the np.arange terms in to avoid creating unnecessary variables
    # Xvec = np.arange(0,dims[0])-Xcenter
    # Yvec = np.arange(0,dims[1])-Ycenter
    # Zvec = np.arange(0,dims[2])-Zcenter
    # xx, yy, zz = np.meshgrid(Xvec, Yvec, Zvec)

    xx, yy, zz = np.meshgrid(np.arange(0,dims[0])-Xcenter, np.arange(0,dims[1])-Ycenter, np.arange(0,dims[2])-Zcenter)
    sigma  = dims[0]/2*resolutionCutoff

    #construct the filter and normalize
    K_filter = np.exp(-(((xx)**2 + (yy)**2 + (zz) **2)/(2*sigma*sigma)))
    K_filter /= np.max(np.max(np.max(abs(K_filter))))

    # take FFT and multiply by filter
    # kbinned = pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(object,overwrite_input=True)) * K_filter
    kbinned = fftn_fftshift(object) * K_filter

    # Assuming the input is real, the output will be approximately real, but will have some
    # tiny imaginary part due to rounding errors. This function would work for smoothing a
    # complex object, but this return statement would need to be modified
    # return np.real(pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.ifftshift(kbinned)))
    return np.real(fftn_fftshift(kbinned))



def smooth2D(object,resolutionCutoff):
    """
    * smooth2D *

    Low pass filter a 2D image

    :param object: 2D image
    :param resolutionCutoff: Fraction of Nyquist frequency to use as sigma for the Gaussian filter
    :return: smoothed image

    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright (c) 2016. All Rights Reserved.
    """
    dims = np.shape(object)
    if len(dims)>2:
        raise Exception('This is a 3D object, use smooth3D instead.')
    Xcenter = round(dims[0]//2)
    Ycenter = round(dims[1]//2)

    # These  next few lines are easier to read, but in the actual code I just directly
    # place the np.arange terms in to avoid creating unnecessary variables
    # Xvec = np.arange(0,dims[0])-Xcenter
    # Yvec = np.arange(0,dims[1])-Ycenter
    # xx, yy = np.meshgrid(Xvec, Yvec)

    xx, yy = np.meshgrid(np.arange(0,dims[0])-Xcenter, np.arange(0,dims[1])-Ycenter)
    sigma  = dims[0]/2*resolutionCutoff

    #construct the filter and normalize
    K_filter = np.exp(-(((xx)**2 + (yy)**2)/(2*sigma*sigma)))
    K_filter /= np.max(np.max(np.max(abs(K_filter))))

    # take FFT and multiply by filter
    # kbinned =  pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(object,overwrite_input=True)) * K_filter
    kbinned = fftn_fftshift(object) * K_filter
    # Assuming the input is real, the output will be approximately real, but will have some
    # tiny imaginary part due to rounding errors. This function would work for smoothing a
    # complex object, but this return statement would need to be modified
    # return np.real(pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.ifftshift(kbinned)))
    return np.real(ifftn_fftshift(kbinned))


def calculateProjection_interp(modelK, phi, theta, psi):
    """
    * calculateProjection_interp *

    Calculate a projection of a 3D volume from it's oversampled Fourier transform by interpolating
    the central slice at the orientation determined by Euler angles (phi, theta, psi)

    :param modelK: numpy array holding the oversampled FFT of the model
    :param phi: euler angle 1
    :param theta: euler angle 2
    :param psi: euler angle 3
    :return: projection
    """

    # get dimensions
    dims = np.shape(modelK)
    Xcenter = round(dims[0]//2)
    Ycenter = round(dims[1]//2)
    Zcenter = round(dims[2]//2)

    # convert angles to radians
    phi *= PI/180
    theta *= PI/180
    psi *= PI/180

    # construct rotation matrix
    R = np.array([[np.cos(psi)*np.cos(theta)*np.cos(phi)-np.sin(psi)*np.sin(phi) , np.cos(psi)*np.cos(theta)*np.sin(phi)+np.sin(psi)*np.cos(phi)   ,    -np.cos(psi)*np.sin(theta)],
    [-np.sin(psi)*np.cos(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi), -np.sin(psi)*np.cos(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi) ,   np.sin(psi)*np.sin(theta) ],
    [np.sin(theta)*np.cos(phi)                               , np.sin(theta)*np.sin(phi)                                ,              np.cos(theta)]])

    R = R.T

    kx = np.arange(0, dims[0])-Xcenter
    ky = np.arange(0, dims[1])-Ycenter
    kz = np.arange(0, dims[2])-Zcenter

    # construct interpolator function that does the actual computation
    interpolator = RegularGridInterpolator((kx, ky, kz), modelK, bounds_error=False, fill_value=0)

    # build coordinates of the slice we want to calculate
    kx_slice, ky_slice, kz_slice = np.meshgrid((np.arange(0, dims[0])-Xcenter), (np.arange(0, dims[1])-Ycenter), 0)

    # rotate coordinates
    rotKCoords = np.zeros([3, np.size(kx_slice)])
    rotKCoords[0, :] = np.reshape(kx_slice, [1, np.size(kx_slice)])
    rotKCoords[1, :] = np.reshape(ky_slice, [1, np.size(ky_slice)])
    rotKCoords[2, :] = np.reshape(kz_slice, [1, np.size(kz_slice)])
    rotKCoords = np.dot(R, rotKCoords)

    # calculate projection
    projection = interpolator(rotKCoords.T)
    projection = np.reshape(projection, [dims[0], dims[1]], order='F')

    return np.real(ifftn_fftshift(projection))

def getProjectionInterpolator(modelK):
    """
    * generateKspaceIndices *

    Maps the radial coordinate indices in the matrix obj

    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved.


    :param obj: Matrix of size to be mapped
    :return: 3D indices for each voxel in the volume
    """

    # get dimensions
    dims = np.shape(modelK)
    Xcenter = round(dims[0]//2)
    Ycenter = round(dims[1]//2)
    Zcenter = round(dims[2]//2)

    kx = np.arange(0, dims[0])-Xcenter
    ky = np.arange(0, dims[1])-Ycenter
    kz = np.arange(0, dims[2])-Zcenter

    # construct interpolator function that does the actual computation
    return RegularGridInterpolator((kx, ky, kz), modelK, bounds_error=False, fill_value=0)

def calculateProjection_interp_fromInterpolator(interpolator, phi, theta, psi, dims):
    """
    * calculateProjection_interp_fromInterpolator *

    Calculate a projection from precomputed interpolator object

    :param interpolator: RegularGridInterpolator object from scipy.
    :param phi: euler angle 1
    :param theta: euler angle 2
    :param psi: euler angle 3
    :param dims: dimensions of the object

    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved.
    """

    # convert to radians
    phi *= PI/180
    theta *= PI/180
    psi *= PI/180

    Xcenter = round(dims[0]//2)
    Ycenter = round(dims[1]//2)
    Zcenter = round(dims[2]//2)

    # construct rotation matrix
    R = np.array([[np.cos(psi)*np.cos(theta)*np.cos(phi)-np.sin(psi)*np.sin(phi) , np.cos(psi)*np.cos(theta)*np.sin(phi)+np.sin(psi)*np.cos(phi)   ,    -np.cos(psi)*np.sin(theta)],
    [-np.sin(psi)*np.cos(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi), -np.sin(psi)*np.cos(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi) ,   np.sin(psi)*np.sin(theta) ],
    [np.sin(theta)*np.cos(phi)                               , np.sin(theta)*np.sin(phi)                                ,              np.cos(theta)]])

    R = R.T

    # build coordinates of the slice we want to calculate
    kx_slice, ky_slice, kz_slice = np.meshgrid((np.arange(0, dims[0])-Xcenter), (np.arange(0, dims[1])-Ycenter), 0)

    # rotate coordinates
    rotKCoords = np.zeros([3, np.size(kx_slice)])
    rotKCoords[0, :] = np.reshape(kx_slice, [1, np.size(kx_slice)])
    rotKCoords[1, :] = np.reshape(ky_slice, [1, np.size(ky_slice)])
    rotKCoords[2, :] = np.reshape(kz_slice, [1, np.size(kz_slice)])
    rotKCoords = np.dot(R, rotKCoords)

    # calculate projection
    projection = interpolator(rotKCoords.T)
    projection = np.reshape(projection, [dims[0], dims[1]], order='F')

    return np.real(ifftn_fftshift(projection))


def calculateProjection_DFT(model, phi, theta, psi, out_dimx, out_dimy):
    """
    * calculateProjection_interp *

    Calculate a projection of a 3D volume using the Discrete Fourier Transform (DFT)

    :param model: numpy array holding the model
    :param phi: euler angle 1
    :param theta: euler angle 2
    :param psi: euler angle 3
    :param out_dimx: output projection size in x
    :param out_dimy: output projection size in y
    :return: projection
    """

    # get dimensions
    dims = np.shape(model)
    Xcenter = round(dims[0]//2)
    Ycenter = round(dims[1]//2)
    Zcenter = round(dims[2]//2)

    KXcenter = round(out_dimx//2)
    KYcenter = round(out_dimy//2)

    # convert to radians
    phi *= PI/180
    theta *= PI/180
    psi *= PI/180

    R = np.array([[np.cos(psi)*np.cos(theta)*np.cos(phi)-np.sin(psi)*np.sin(phi) , np.cos(psi)*np.cos(theta)*np.sin(phi)+np.sin(psi)*np.cos(phi)   ,    -np.cos(psi)*np.sin(theta)],
    [-np.sin(psi)*np.cos(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi), -np.sin(psi)*np.cos(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi) ,   np.sin(psi)*np.sin(theta) ],
    [np.sin(theta)*np.cos(phi)                               , np.sin(theta)*np.sin(phi)                                ,              np.cos(theta)]])

    R = R.T

    # build coordinates of 3D FFT

    X = np.arange(0, dims[0])-Xcenter
    Y = np.arange(0, dims[1])-Ycenter
    Z = np.arange(0, dims[2])-Zcenter
    X, Y, Z = np.meshgrid(X,Y,Z)
    Y = Y.astype(float)
    X = X.astype(float)
    Z = Z.astype(float)

    KX = np.arange(0, out_dimx)-KXcenter
    KY = np.arange(0, out_dimy)-KYcenter
    KY, KX, KZ = np.meshgrid(KY, KX, 0)
    KY = KY.astype(float)
    KX = KX.astype(float)
    KZ = KZ.astype(float)

    # build coordinates of the slice we want to calculate
    # kx_slice, ky_slice, kz_slice = np.meshgrid((np.arange(0, dims[0])-Xcenter), (np.arange(0, dims[1])-Ycenter), 0)

    # rotate coordinates
    rotKCoords = np.zeros([3, np.size(KX)])
    rotKCoords[0, :] = np.reshape(KX, [1, np.size(KX)])
    rotKCoords[1, :] = np.reshape(KY, [1, np.size(KY)])
    rotKCoords[2, :] = np.reshape(KZ, [1, np.size(KZ)])
    rotKCoords = np.dot(R, rotKCoords)

    rotKX = rotKCoords[0, :]
    rotKY = rotKCoords[1, :]
    rotKZ = rotKCoords[2, :]

    projection = np.zeros(out_dimx*out_dimy,dtype=complex)
    ind = np.where(model!=0) # only consider nonzero indices for speed
    model = model[ind]
    X = X[ind]
    Y = Y[ind]
    Z = Z[ind]
    for i in range(np.size(projection)): # compute DFT
        projection[i] = np.sum( model * np.exp( -2*PI*1j * \
                          ( rotKX[i]*X/out_dimx + rotKY[i]*Y/out_dimy + rotKZ[i]*Z/out_dimx ) ) )
    projection = np.reshape(projection, (out_dimx, out_dimy), order='F')

    # return np.real(pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.ifftshift(projection))))
    return np.real(ifftn_fftshift(projection))


def generateKspaceIndices(obj):
    """
    * generateKspaceIndices *

    Maps the radial coordinate indices in the matrix obj

    Author: Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved.


    :param obj: Matrix of size to be mapped
    :return: 3D indices for each voxel in the volume
    """

    dims = np.shape(obj)
    if len(dims) < 3:
        dims = dims + (0,)

    if dims[0] % 2 == 0:
        ncK0 = dims[0]/2
        vec0 = np.arange(-ncK0, ncK0, 1)/ncK0
    elif dims[0] == 1:
        vec0 = 0
        ncK0 = 1

    else:
        ncK0 = ((dims[0]+1)/2)-1
        vec0 = np.arange(-ncK0, ncK0+1)/ncK0


    if dims[1] % 2 == 0:
        ncK1 = dims[1]/2
        vec1 = np.arange(-ncK1, ncK1, 1)/ncK1
    elif dims[1] == 1:
        vec1 = 0
        ncK1 = 1

    else:
        ncK1 = ((dims[1]+1)/2)-1
        vec1 = np.arange(-ncK1, ncK1+1)/ncK1


    if dims[2] % 2 == 0:
        ncK2 = dims[2]/2
        vec2 = np.arange(-ncK2, ncK2, 1)/ncK2
    elif dims[2] == 1:
        vec2 = 0
        ncK2 = 1

    else:
        ncK2 = ((dims[2]+1)/2)-1
        vec2 = np.arange(-ncK2, ncK2+1)/ncK2

    kx, ky, kz = np.meshgrid(vec1,vec0,vec2)
    Kindices = np.sqrt(kx**2 + ky**2 + kz**2)
    return Kindices

def pointToPlaneDistance(points, norm_vec):
    """
    * pointToPlaneDistance *

    compute distance from point to a plane with normal vector norm_vec passing through origin

    :param points: num_points x 3 array of (x,y,z) points
    :param norm_vec: (a,b,c) normal vector defining the plane ax + by + cz = 0
    :return: numpy array containing distance to plane

    Author: Yongsoo Yang
    Transcribed from MATLAB by Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved.
    """
    from numpy.linalg import norm
    norm_vec = norm_vec / norm(norm_vec)
    if np.ndim(points)>1:
        distances = np.abs(np.sum(points*norm_vec,axis=1))
        return distances
    return np.abs(np.dot(points,norm_vec))

def pointToPlaneClosest(points, norm_vec, distances):
    """
    * pointToPlaneClosest *

    compute the closest point in the plane defined by normal vector norm_vec to each
    point in points

    :param points: num_points x 3 array of (x,y,z) points
    :param norm_vec: (a,b,c) normal vector defining the plane ax + by + cz = 0
    :param distances: magnitude of distance to plane (see pointToPlaneDistance)
    :return:  numpy array containing the coordinates of the closest points

    Author: Yongsoo Yang
    Transcribed from MATLAB codes by Alan (AJ) Pryor, Jr.
    Jianwei (John) Miao Coherent Imaging Group
    University of California, Los Angeles
    Copyright 2015-2016. All rights reserved.
    """
    if np.ndim(points)>1: # use NumPy broadcasting for speed
        closest_points = points + ( (distances - np.sum(points * norm_vec)) * np.reshape(norm_vec,(3,1)) / np.dot(norm_vec,norm_vec) ).T
        return closest_points
    return points + (distances- np.dot(points, norm_vec)) * norm_vec

def printStringOrNumpyArray(value, name):
    if isinstance(value, np.ndarray):
        print("{} = {} NumPy array of type {}".format(name, value.shape, value.dtype))
    else:
        print("{} = {}".format(name, value))