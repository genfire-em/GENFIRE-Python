"""
* utility *

This module contains various useful functions that do not fit into another category


Author: Alan (AJ) Pryor, Jr.
Jianwei (John) Miao Coherent Imaging Group
University of California, Los Angeles
Copyright 2015-2016. All rights reserved.
"""


from __future__ import division
import numpy as np
import pyfftw
from scipy.interpolate import RegularGridInterpolator
PI = np.pi

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
        numberOfValues = (volume != 0).astype(int)
        volume += np.conjugate(volume[::-1, ::-1, ::-1])
        numberOfValues += numberOfValues[::-1, ::-1, ::-1]
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
    kbinned = pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(object,overwrite_input=True)) * K_filter

    # Assuming the input is real, the output will be approximately real, but will have some
    # tiny imaginary part due to rounding errors. This function would work for smoothing a
    # complex object, but this return statement would need to be modified
    return np.real(pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.ifftshift(kbinned)))




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
    kbinned =  pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(object,overwrite_input=True)) * K_filter

    # Assuming the input is real, the output will be approximately real, but will have some
    # tiny imaginary part due to rounding errors. This function would work for smoothing a
    # complex object, but this return statement would need to be modified
    return np.real(pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.ifftshift(kbinned)))



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

    # projection = None
    dims = np.shape(modelK)
    Xcenter = round(dims[0]//2)
    Ycenter = round(dims[1]//2)
    Zcenter = round(dims[2]//2)

    # X, Y, Z = np.meshgrid(np.arange(1,dims[0]-Xcenter), np.arange(1,dims[1]-Ycenter), 0)
    phi *= PI/180
    theta *= PI/180
    psi *= PI/180

    R = np.array([[np.cos(psi)*np.cos(theta)*np.cos(phi)-np.sin(psi)*np.sin(phi) , np.cos(psi)*np.cos(theta)*np.sin(phi)+np.sin(psi)*np.cos(phi)   ,    -np.cos(psi)*np.sin(theta)],
    [-np.sin(psi)*np.cos(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi), -np.sin(psi)*np.cos(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi) ,   np.sin(psi)*np.sin(theta) ],
    [np.sin(theta)*np.cos(phi)                               , np.sin(theta)*np.sin(phi)                                ,              np.cos(theta)]])

    R = R.T

    # build coordinates of 3D FFT
    # kx, ky, kz = np.meshgrid(np.arange(1,dims[0]-Xcenter), np.arange(1,dims[1]-Ycenter), np.arange(1,dims[2]-Zcenter))

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

    projection = interpolator(rotKCoords.T)
    projection = np.reshape(projection, [dims[0], dims[1]], order='F')


    return np.real(pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.ifftshift(projection))))


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
    # X, Y, Z = np.meshgrid(np.arange(1,dims[0]-Xcenter), np.arange(1,dims[1]-Ycenter), 0)
    phi *= PI/180
    theta *= PI/180
    psi *= PI/180

    Xcenter = round(dims[0]//2)
    Ycenter = round(dims[1]//2)
    Zcenter = round(dims[2]//2)

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

    projection = interpolator(rotKCoords.T)
    projection = np.reshape(projection, [dims[0], dims[1]], order='F')


    return np.real(pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.ifftn(pyfftw.interfaces.numpy_fft.ifftshift(projection))))

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