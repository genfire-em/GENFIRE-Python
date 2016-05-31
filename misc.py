from __future__ import division
import numpy as np
import pyfftw
import matplotlib.pyplot as plt
from scipy import io
from scipy.interpolate import RegularGridInterpolator
import time
PI = 3.14159265359



## hermitianSymmetrize ##

## Applies Hermitian symmetry to "input". If one symmetry mate is not equal to the complex conjugate of the other
## their average is taken. If only one of them exists (is nonzero), then the one value is used. If neither exists
## the value remains 0. In terms of implementation, this function produces Hermitian symmetry by adding the object
## to its complex conjugate with the indices reversed. This requires the array to be odd, so there is also a check
## to make the array odd and then take back the original size at the end, if necessary.

## Author: AJ Pryor
## Jianwei (John) Miao Coherent Imaging Group
## University of California, Los Angeles
## Copyright (c) 2016. All Rights Reserved.
def hermitianSymmetrize(input):

    startDims = np.shape(input) # initial dimensions

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

    if flag: # if any dimensions are even, create a new with all odd dimensions and copy input
        newInput = np.zeros((dimx,dimy,dimz), dtype=complex) #new array
        newInput[:startDims[0], :startDims[1], :startDims[2]] = input # copy values
        numberOfValues = (newInput != 0).astype(float) #track number of values for averaging
        newInput = newInput + np.conj(newInput[::-1, ::-1, ::-1]) # combine Hermitian symmetry mates
        numberOfValues = numberOfValues + numberOfValues[::-1, ::-1, ::-1] # track number of points included in each sum

        newInput[numberOfValues != 0] /= numberOfValues[numberOfValues != 0] # take average where two values existed
        newInput[np.isnan(newInput)] = 0 # Shouldn't be any nan, but just to be safe

        debug = {'numberOfValues':numberOfValues,'newInput':newInput}
        io.savemat('debugging',debug)
        return newInput[:startDims[0], :startDims[1], :startDims[2]] # return original dimensions


    else: # otherwise, save yourself the trouble of copying the matrix over. See previous comments for line-by-line
        numberOfValues = (input != 0).astype(int)
        input += np.conjugate(input[::-1, ::-1, ::-1])
        numberOfValues += numberOfValues[::-1, ::-1, ::-1]
        input[numberOfValues != 0] /= numberOfValues[numberOfValues != 0]
        input[np.isnan(input)] = 0
        return input




## smooth3D ##

## Smooths the real space object "object" by applying a Gaussian filter with standard
## deviation determined by "resolutionCutoff" which is a parameter expressed as a fraction
## of Nyquist frequency that determines the point at which the Gaussian filter decreases
## by 1/e

## Author: AJ Pryor
## Jianwei (John) Miao Coherent Imaging Group
## University of California, Los Angeles
## Copyright (c) 2016. All Rights Reserved.

def smooth3D(object,resolutionCutoff):
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



## smooth2D ##

## Smooths the 2D real space object "object" by applying a Gaussian filter with standard
## deviation determined by "resolutionCutoff" which is a parameter expressed as a fraction
## of Nyquist frequency that determines the point at which the Gaussian filter decreases
## by 1/e

## Author: AJ Pryor
## Jianwei (John) Miao Coherent Imaging Group
## University of California, Los Angeles
## Copyright (c) 2016. All Rights Reserved.

def smooth2D(object,resolutionCutoff):
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

def calculateProjection_interp_pyfftw(modelK, phi, theta, psi):

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


    return np.real( pyfftw.interfaces.numpy_fft.fftshift( pyfftw.interfaces.numpy_fft.ifftn( pyfftw.interfaces.numpy_fft.ifftshift(projection))))

