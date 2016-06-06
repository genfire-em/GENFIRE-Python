"""

M180G Phase Retrieval
Author: AJ Pryor
Jianwei (John) Miao Coherent Imaging Group
University of California, Los Angeles
Copyright (c) 2016. All Rights Reserved.


"""

import numpy as np
import matplotlib

SATURATED_INTENSITY_THRESHOLD = 60000

def my_fft(arr):
    #computes forward FFT of arr and shifts origin to center of array
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(arr)))

def my_ifft(arr):
    #computes inverse FFT of arr and shifts origin to center of array
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(arr)))

def smooth(obj, sigma_fraction):
    obj = my_fft(obj)
    dimx, dimy = np.shape(obj)
    center_x, center_y = dimx//2, dimy//2
    sigma_x, sigma_y = sigma_fraction*dimx/(2), sigma_fraction*dimy/(2)
    yy, xx = np.meshgrid(np.arange(dimy)-center_y, np.arange(dimx)-center_x)
    kfilter = np.exp(- (xx**2/(2*sigma_x**2) + yy**2/(2*sigma_y**2)) )
    return np.real(my_ifft( obj * kfilter ))

def subtractBackground(diff_pat, bg_filename, scaling_factor = 1):
    from PIL import Image
    BG = Image.open(bg_filename)
    BG = np.array(BG,dtype=float)
    diff_pat -= (BG*scaling_factor) # shorthand for self.data = self.data - BG
    diff_pat[(diff_pat <= 0 )] = -1 # any negative values are to be flagged as "missing" with a -1
    return diff_pat

def subtractBackground_arr(diff_pat, bg_array, scaling_factor = 1):
    diff_pat -= (bg_array*scaling_factor) # shorthand for self.data = self.data - BG
    diff_pat[(diff_pat <= 0 )] = -1 # any negative values are to be flagged as "missing" with a -1
    return diff_pat

def combine_images(directory_name, output_filename = None):
    """loop over all tiff images in directory_name and average them"""
    if directory_name[-1] != '/':
        directory_name += '/'
    import os
    from PIL import Image
    image_count = 0
    average_image = None
    for filename in os.listdir(directory_name):
        file,ext = os.path.splitext(filename)
        if ext==".tif" or ext==".tiff" or ext==".TIFF" or ext==".TIF":
            # print(directory_name + filename)
            if image_count == 0:
                average_image = np.array(Image.open(directory_name + filename),dtype=float)
            else:
                average_image += np.array(Image.open(directory_name + filename),dtype=float)

            image_count+=1
    try:
        print ('Count = ' , image_count)
        average_image/=image_count # take average
    except TypeError:
        print ("\n\n\n\nNO VALID TIFF IMAGES IN DIRECTORY!\n\n\n")
        raise
    if output_filename is not None:
        np.save(output_filename,average_image)
    return average_image

def scale_and_combine_images(directory_name, output_filename = None):
    """loop over all tiff images in directory_name, scale them to each other, and average"""
    if directory_name[-1] != '/':
        directory_name += '/'
    import os
    from PIL import Image
    image_count = 0
    average_image = None
    number_of_values = None
    for filename in os.listdir(directory_name):
        file,ext = os.path.splitext(filename)
        if ext==".tif" or ext==".tiff" or ext==".TIFF" or ext==".TIF":
            if file != "BG":
                print("Loading {0}".format(directory_name + filename))
                if image_count == 0: #initialize image
                    average_image = np.array(Image.open(directory_name + filename),dtype=float)
                    average_image [average_image > SATURATED_INTENSITY_THRESHOLD] = -1
                    number_of_values = (average_image > 0).astype(int)

                else:
                    #get current average image, must make copy since numpy arrays are references
                    # reference_image = np.copy(np.divide(average_image, image_count))
                    reference_image = np.copy(average_image)
                    reference_image [number_of_values > 0 ] /= number_of_values[number_of_values > 0 ]
                    #mask out the saturated region
                    reference_image [reference_image > SATURATED_INTENSITY_THRESHOLD] = -1
                    #load the current image to scale
                    current_image = np.array(Image.open(directory_name + filename),dtype=float)
                    tmp_image = np.copy(current_image)

                    tmp_image [tmp_image > SATURATED_INTENSITY_THRESHOLD] = -1
                    common_pixels = ( (reference_image > 0 ) & (tmp_image > 0) )
                    #scale factor is ratio of the mean intensities in the common region to both images
                    scale_factor = np.mean(reference_image[common_pixels])/np.mean(tmp_image[common_pixels])
                    average_image[tmp_image > 0 ] += scale_factor*tmp_image[tmp_image > 0 ]
                    number_of_values[tmp_image > 0] += 1
                    # average_image += (scale_factor*current_image)
                    print ("scale factor = " , scale_factor)
                image_count+=1 #track number of images added together
    try:
        print ('Total Number of Images Averaged = ' , image_count)
        # average_image/=image_count # take average
        average_image[number_of_values > 0 ] /=number_of_values[number_of_values > 0 ]
        average_image [average_image > SATURATED_INTENSITY_THRESHOLD] = -1
    except TypeError: # if no images were in the directory, raise an exception
        print ("\n\n\n\nNO VALID TIFF IMAGES IN DIRECTORY!\n\n\n")
        raise
    if output_filename is not None: #user can save the image for later use
        np.save(output_filename,average_image)
    return average_image






def scale_and_combine_images_multiple_directories_subtractBG(directory_names, bg_scale_factors=None):
    from glob import glob
    import os
    diff_pat_list = []
    if bg_scale_factors is None:
        bg_scale_factors = [1] * len(directory_names) #use 1 if no scaling factor provided
    for directory_name, bg_scale_factor in zip(directory_names, bg_scale_factors):
        if directory_name[-1] != '/':
            directory_name += '/'
        diff_pat = scale_and_combine_images(directory_name)
        BGfile = glob(directory_name + "*BG*")
        if BGfile is not None:
            if len(BGfile) > 1:
                raise ValueError("More than one BG frame found, only one should be provided!\n\n")
            diff_pat_list.append(subtractBackground(diff_pat, BGfile[0], bg_scale_factor))
    return scale_and_combine_images_in_list(diff_pat_list)

def scale_and_combine_images_multiple_directories(directory_names, bg_scale_factors=None):
    from glob import glob
    import os
    diff_pat_list = []
    if bg_scale_factors is None:
        bg_scale_factors = [1] * len(directory_names) #use 1 if no scaling factor provided
    for directory_name, bg_scale_factor in zip(directory_names, bg_scale_factors):
        if directory_name[-1] != '/':
            directory_name += '/'
        diff_pat = scale_and_combine_images(directory_name)
        diff_pat_list.append(diff_pat)

    return scale_and_combine_images_in_list(diff_pat_list)
def scale_and_combine_images_in_list(image_list, output_filename = None):
    """loop over all images in image_list, scale them to each other, and average"""
    if image_list is None:
        raise AttributeError("\n\n\nNo images in list!\n\n\n")
    image_count = 1
    average_image = image_list[0]
    average_image [average_image > SATURATED_INTENSITY_THRESHOLD] = -1
    number_of_values = (average_image > 0).astype(int)
    if len(image_list) > 1:
        for current_image in image_list[1:]:
            #get current average image, must make copy since numpy arrays are references
            reference_image = np.copy(average_image)
            reference_image [number_of_values > 0 ] /= number_of_values[number_of_values > 0 ]        #mask out the saturated region
            reference_image [reference_image > SATURATED_INTENSITY_THRESHOLD] = -1
            tmp_image = np.copy(current_image)

            tmp_image [tmp_image > SATURATED_INTENSITY_THRESHOLD] = -1
            common_pixels = ( (reference_image > 0 ) & (tmp_image > 0) )
            #scale factor is ratio of the mean intensities in the common region to both images
            scale_factor = np.mean(reference_image[common_pixels])/np.mean(tmp_image[common_pixels])
            average_image[tmp_image > 0 ] += scale_factor*tmp_image[tmp_image > 0 ]
            number_of_values[tmp_image > 0] += 1
            # average_image += (scale_factor*current_image)
            print ("scale factor = " , scale_factor)
            image_count+=1 #track number of images added together
        # print ('Count = ' , image_count)
        # average_image/=image_count # take average
        average_image[number_of_values > 0 ] /=number_of_values[number_of_values > 0 ]
        average_image [average_image > SATURATED_INTENSITY_THRESHOLD] = -1
        if output_filename is not None: #user can save the image for later use
            np.save(output_filename,average_image)
    return average_image


# This defines a decorator to time other functions
def timeThisFunction(function):
    from time import time
    def inner(*args, **kwargs):
        t0 = time()
        value = function(*args,**kwargs)
        print("Function call to \"{0:s}\" completed in {1:.3f} seconds".format(function.__name__,time()-t0))
        return value
    return inner



class DiffractionPattern(object):
    "Base class for containing a diffraction pattern"

    def __init__(self, data):
        "Build a diffraction pattern object from an existing numpy array representing the diffraction data"

        self.data = data
        dims = np.shape(self.data) # get initial dimensions of array


    @staticmethod # static method means the following is just a regular function inside of the class definition
    def fromTIFF(filename):
        "Factory method for constructing a DiffractionPattern object from a tiff image"
        from PIL import Image
        data = Image.open(filename) # load in data
        return DiffractionPattern(np.array(data,dtype=float)) #return a new DiffractionPattern object

    @timeThisFunction # adds a timer to the function
    def maskBrightValues(self, threshold_fraction = 0.98):
        # flag pixels at or near saturation limit as unmeasured
        # self.data [self.data > (threshold_fraction * np.amax(self.data))] = -1
        self.data [self.data > SATURATED_INTENSITY_THRESHOLD] = -1

    # @timeThisFunction # adds a timer to the function
    def hermitianSymmetrize(self):
        """Enforce Hermitian symmetry (centrosymmetry) on the diffraction pattern"""
        print ("Using the slower version of hermitianSymmetrize , this may take a while....")

        # define center, "//" performs floor division. This idiom works regardless
        # of whether the dimension is odd or even.
        dimX, dimY = np.shape[ self.data]
        centerX = dimX//2
        centerY = dimY//2

        # Now we want to loop over the array and combine each pixel with its symmetry mate. If
        # the dimension of our array is even, then the pixels at position "0" do not have a symmetry mate,
        # so we must check for that. Otherwise we will get an error for trying to index out of bounds
        if dimX % 2 == 0: # the "%" performs modular division, which gets the remainder
            startX = 1
        else:
            startX = 0

        if dimY % 2 == 0:
            startY = 1
        else:
            startY = 0

        # Now that we have the housekeeping stuff out of the way, we can actually do the loop
        # We have to keep up with two sets of coordinates -> (X,Y) refers to the
        # position of where the value is located in the array and counts 0, 1, 2, etc.
        # On the other hand,(centeredX, centeredY) gives the coordinates relative to the origin
        # so that we can find the Hermitian symmetry mate, which is at (-centeredX, -centeredY)
        for X in range(startX, dimX):
            for Y in range(startY, dimY):

                # for each pixel X, Y, get the centered coordinate
                centeredX = X - centerX
                centeredY = Y - centerY

                # get the array coordinate of the symmetry mate and shift back by the center
                symmetry_X = (-1 * centeredX) + centerX
                symmetry_Y = (-1 * centeredY) + centerY

                # get the values from the array
                val1 = self.data[X, Y]
                val2 = self.data[symmetry_X, symmetry_Y]

                # if both values exist, take their average. If only one exists, use it for both. If
                # neither exists, then the final value is unknown (so we do nothing)

                if (val1 != -1) and (val2 != -1): #if both exist, take the average
                    self.data[X, Y] = (val1 + val2) / 2
                    self.data[symmetry_X, symmetry_Y] = (val1 + val2) / 2
                elif (val1 == -1): #if val1 does not exist, use val2 for both
                    self.data[X, Y] = val2
                    self.data[symmetry_X, symmetry_Y] = val2
                else: #then val2 must not exist
                    self.data[X, Y] = val1
                    self.data[symmetry_X, symmetry_Y] = val1
                self.data[self.data == 0] = -1

    @timeThisFunction # adds a timer to the function
    def hermitianSymmetrize_express(self):
        """
        Functions the same as hermitianSymmetrize, except is ~10,000 times faster, but more cryptic

        Applies Hermitian symmetry (centrosymmetry) to the diffraction pattern. If one symmetry mate is not equal to the complex conjugate of the other
        their average is taken. If only one of them exists (is nonzero), then the one value is used. If neither exists
        the value remains 0. In terms of implementation, this function produces Hermitian symmetry by adding the object
        to its complex conjugate with the indices reversed. This requires the array to be odd, so there is also a check
        to make the array odd and then take back the original size at the end, if necessary.

        """
        startDims = np.shape(self.data) # initial dimensions

        # remember the initial dimensions for the end
        dimx = startDims[0]
        dimy = startDims[1]
        flag = False # flag to trigger copying to new odd dimensioned array

        #check if any dimension is odd
        if dimx % 2 == 0:
            dimx += 1
            flag = True

        if dimy % 2 == 0:
            dimy += 1
            flag = True

        if flag: # if any dimensions are even, create a new with all odd dimensions and copy array
            newInput = np.zeros((dimx,dimy), dtype=float) #new array
            newInput[:startDims[0], :startDims[1]] = self.data # copy values
            newInput[newInput == -1] = 0
            numberOfValues = (newInput != 0).astype(float) #track number of values for averaging
            newInput = newInput + newInput[::-1, ::-1] # combine Hermitian symmetry mates
            numberOfValues = numberOfValues + numberOfValues[::-1, ::-1] # track number of points included in each sum
            newInput[numberOfValues != 0] =  newInput[numberOfValues != 0] / numberOfValues[numberOfValues != 0] # take average where two values existed
            self.data = newInput[:startDims[0], :startDims[1]] # return original dimensions


        else: # otherwise, save yourself the trouble of copying the matrix over. See previous comments for line-by-line
            self.data[self.data == -1] = 0 #temporarily remove flags
            numberOfValues = (self.data != 0).astype(int)
            self.data = self.data + self.data[::-1, ::-1]
            numberOfValues = numberOfValues + numberOfValues[::-1, ::-1]
            self.data[numberOfValues != 0] = self.data[numberOfValues != 0] / numberOfValues[numberOfValues != 0]
        self.data[self.data == 0] = -1 # reflag

    @timeThisFunction # adds a timer to the function
    def correctCenter(self,search_box_half_size = 10):
        "This method optimizes the location of the diffraction pattern's center and shifts it accordingly \
         It does so by searching a range of centers determined by search_box_half_size. For each center, the \
         error between centrosymmetric partners is checked. The optimized center is the position which    \
         minimizes this error"
        import matplotlib.pyplot as plt

        h = plt.figure()
        plt.imshow(self.data)
        plt.title('Double-click the center')
        # plot.show()
        # plt.get_current_fig_manager().window.setGeometry(25,25,750, 750)
        center_guess_y, center_guess_x = (plt.ginput(1)[0])
        center_guess_x = int(center_guess_x)
        center_guess_y = int(center_guess_y)
        plt.close(h)


        dimX, dimY = np.shape(self.data)
        # If guesses for the center aren't provided, use the center of the array as a guess

        originalDimx = dimX
        originalDimy = dimY

        if center_guess_x is None:
            center_guess_x = dimX // 2

        if center_guess_y is None:
            center_guess_y = dimY // 2

        bigDimX = max(center_guess_x,originalDimx-center_guess_x-1)
        bigDimY = max(center_guess_y,originalDimy-center_guess_y-1)

        padding_1_x = abs(center_guess_x-bigDimX) + search_box_half_size
        padding_2_x = abs( (originalDimx - center_guess_x - 1) - bigDimX)+ search_box_half_size

        padding_1_y = abs(center_guess_y-bigDimY)+ search_box_half_size
        padding_2_y = abs( (originalDimy - center_guess_y - 1) - bigDimY)+ search_box_half_size


        self.data = np.pad(self.data,((padding_1_x, padding_2_x),(padding_1_y, padding_2_y)),mode='constant')
        dimx, dimy = np.shape(self.data) # initial dimensions
        startDims = (dimx, dimy)
        center_guess_x = dimx//2
        center_guess_y = dimy//2

        flag = False # flag to trigger copying to new odd dimensioned array

        #check if any dimension is odd
        if dimx % 2 == 0:
            dimx += 1
            flag = True

        if dimy % 2 == 0:
            dimy += 1
            flag = True

        if flag: # if any dimensions are even, create a new with all odd dimensions and copy array
            temp_data = np.zeros((dimx,dimy), dtype=float) #new array
            temp_data[:startDims[0], :startDims[1]] = self.data # copy values
            input = temp_data

        else:
            temp_data = self.data

        temp_data[temp_data == -1 ] = 0 # remove flags

        #initialize minimum error to a large value
        best_error = 1e30

        #initialize the best shifts to be 0
        bestShiftX = 0
        bestShiftY = 0

        #loop over the various center positions
        for xShift in range(-search_box_half_size,search_box_half_size+1):
            for yShift in range(-search_box_half_size,search_box_half_size+1):

                #shift the data
                temp_array = np.roll(temp_data,xShift,axis=0)
                temp_array = np.roll(temp_array,yShift,axis=1)
                temp_array_reversed = temp_array[::-1, ::-1]

                numberOfValues = (temp_array != 0).astype(float)
                numberOfValues =  numberOfValues + numberOfValues[::-1, ::-1]
                difference_map = np.abs(temp_array - temp_array_reversed)

                normalization_term = np.sum(abs(temp_array[numberOfValues == 2]))
                error_between_symmetry_mates = np.sum(difference_map[numberOfValues == 2]) / normalization_term
                if error_between_symmetry_mates < best_error:
                    best_error = error_between_symmetry_mates
                    bestShiftX = xShift
                    bestShiftY = yShift
        self.data = np.roll(self.data, bestShiftX, axis=0)
        self.data = np.roll(self.data, bestShiftY, axis=1)
        self.data = self.data[ search_box_half_size : -search_box_half_size, search_box_half_size:-search_box_half_size ]

    @timeThisFunction
    def makeArraySquare(self):
        """ Pad image to square array size that is the nearest even number greater than or equal to the current dimensions"""

        dimx, dimy = np.shape(self.data)
        new_dim = max(dimx,dimy) + (max(dimx,dimy)%2) # Take the ceiling even value above the larger dimension
        padding_x = ((new_dim - dimx)//2, (new_dim - dimx)//2 + (new_dim - dimx)%2 )
        padding_y = ((new_dim - dimy)//2, (new_dim - dimy)//2 + (new_dim - dimy)%2 )
        self.data = np.pad(self.data,(padding_x, padding_y), mode='constant')
        self.data [ self.data ==0] = -1

    @timeThisFunction # adds a timer to the function
    def subtractBackground(self, bg_filename, scaling_factor = 1):
        from PIL import Image
        BG = Image.open(bg_filename)
        BG = np.array(BG,dtype=float)
        self.data -= (BG*scaling_factor) # shorthand for self.data = self.data - BG
        self.data[(self.data <= 0 )] = -1 # any negative values are to be flagged as "missing" with a -1

    @timeThisFunction # adds a timer to the function
    def convertToFourierModulus(self):
        self.data[self.data != -1] = np.sqrt(self.data[self.data != -1])

    @timeThisFunction
    def binImage(self, bin_factor_x=1, bin_factor_y=1, fraction_required_to_keep = 0.5):
        # bin an image by bin_factor_x in X and bin_factor_y in Y by averaging all pixels in an bin_factor_x by bin_factor_y rectangle
        # This is accomplished using convolution followed by downsampling, with the downsampling chosen so that the center
        # of the binned image coincides with the center of the original unbinned one.

        from scipy.signal import convolve2d
        self.data [self.data <0 ] = 0 # temporarily remove flags
        numberOfValues = (self.data != 0).astype(int) # record positions that have a value
        binning_kernel = np.ones((bin_factor_x, bin_factor_y), dtype=float) # create binning kernel (all values within this get averaged)
        self.data = convolve2d(self.data, binning_kernel, mode='same') # perform 2D convolution
        numberOfValues = convolve2d(numberOfValues, binning_kernel, mode='same') # do the same with the number of values
        self.data[ numberOfValues > 1 ] = self.data[ numberOfValues > 1 ] / numberOfValues[ numberOfValues > 1 ] # take average, accounting for how many datapoints went into each point
        self.data[ numberOfValues < (bin_factor_x * bin_factor_y * fraction_required_to_keep)] = -1 # if too few values existed for averaging because too many of the pixels were unknown, make the resulting pixel unknown
        dimx, dimy = np.shape(self.data) # get dimensions
        centerX = dimx//2 # get center in X direction
        centerY = dimy//2 # get center in Y direction

        # Now take the smaller array from the smoothed large one to obtain the final binned image. The phrase "centerX % bin_factor_x"
        # is to ensure that the subarray we take includes the exact center of the big array. For example if our original image is
        # 1000x1000 then the central pixel is at position 500 (starting from 0). If we are binning this by 5 we want a 200x200 array
        # where the new central pixel at x=100 corresponds to the old array at x=500, so "centerX % bin_factor_x" ->
        # 500 % 5 = 0, so we would be indexing 0::5 = [0, 5, 10, 15..., 500, 505...] which is what we want. The same scenario with a
        # 1004x1004 image needs the center of the 200x200 array to be at x=502, and 502 % 5 = 2 and we index
        # 2::5 = [2,7,12..., 502, 507 ...]
        self.data = self.data[ centerX % bin_factor_x :: bin_factor_x, centerY % bin_factor_y :: bin_factor_y ]

class HIOReconstruction(object):
    """Basic class for performing phase retrieval of a diffraction pattern with the
    Hybrid Input-Output (HIO) method.

    ATTRIBUTES:

    diffraction_pattern -> a DiffractionPattern object that contains a diffraction pattern stored in a numpy array
        and information about it

    num_iterations -> integer number of iterations to run in the reconstruction

    support -> numpy array of 1's and 0's where the region of 1's defines the boundary of the reconstructed object

    METHODS:
    reconstruct -> run HIO reconstruction with the current parameters stored as attributes in the HIOReconstruction object
        The results are stored in self.reconstruction and self.errK

    """

    def __init__(self, diffraction_pattern, num_iterations = 500, support = None, initial_object = None):
        self.diffraction_pattern = diffraction_pattern # must be a DiffractionPattern object
        self.num_iterations = num_iterations
        self.support = support
        self.best_obj = None

        dimX, dimY = np.shape(self.diffraction_pattern.data)
        # Initialize array to hold reconstruction
        self.reconstruction = np.zeros((dimX, dimY), dtype=float)
        self.best_object = np.copy(self.reconstruction)
        self.display_results_during_reconstruction = False

        # If no initial object was provided, default to zeros
        if initial_object is None:
            self.initial_object = np.zeros((dimX, dimY), dtype=float)
        else:
            self.initial_object = initial_object

    @timeThisFunction # adds a timer to the function
    def reconstruct(self):
        """Primary reconstruction function"""
        print ('Reconstructing with HIO...')
        #setup the region to display by finding the rectangular boundaries of the support
        coords = np.where(self.support != 0)
        self.x_start = np.min(coords[0])
        self.x_stop = np.max(coords[0])
        self.y_start = np.min(coords[1])
        self.y_stop = np.max(coords[1])
        del coords

        beta = 0.9 # HIO beta parameter

        # Get dimensions
        dimX, dimY = np.shape(self.diffraction_pattern.data)

        # Initialize object that will hold the previous object for each iteration.
        previous_object = np.copy(self.initial_object)

        # Make a copy of the magnitudes to create the initial object
        initial_magnitudes = np.copy(self.diffraction_pattern.data)

        # Use 0 for the magnitude wherever we don't have a measurement
        initial_magnitudes[ self.diffraction_pattern.data == -1 ] = 0

        # Generate random initial phases
        random_phases = 2*np.pi*np.random.rand( dimX, dimY )

        # Obtain starting point for the reconstruction with these magnitudes and phases
        current_object = np.real(my_ifft(initial_magnitudes * np.exp(1j*random_phases)))

        # Make a mask where of the magnitudes that we will be replacing. This prevents us
        # having to recalculate this every iteration
        measured_data_mask = self.diffraction_pattern.data != -1

        # Initalize error array
        self.errK = np.zeros(self.num_iterations, dtype=float)

        best_error = 1e30 #initialize error to very high value

        # Begin primary loop
        for iterationNumber in range(self.num_iterations):
            if iterationNumber % 50 == 1:
                print("Iteration Number: {0} \t Lowest Error = {1:.2f}".format(iterationNumber, best_error))
            # Get indices of pixels that violated the constraints
            ind_to_change = (current_object < 0) | (self.support == 0)

            # Apply HIO update
            current_object [ind_to_change] = previous_object[ind_to_change] - beta * current_object[ind_to_change]

            # Update the old object
            previous_object = current_object

            # Take forward FFT of current_object, which is the reconstruction and current_object * self.support,
            # which is used to monitor the error
            k = my_fft(current_object)
            k_masked =  my_fft(current_object * self.support)

            # Compute error
            errK = np.sum(abs(abs(k_masked[measured_data_mask]) - \
              self.diffraction_pattern.data[measured_data_mask])) / np.sum(abs(self.diffraction_pattern.data[measured_data_mask]))
            self.errK[ iterationNumber ] = errK

            # If the current error is the lowest thus far, save that reconstruction
            if errK < best_error:
                best_error = errK
                self.best_obj = current_object
                self.reconstruction = current_object * self.support

            # Replace the magnitudes with measured ones (Fourier constraint)
            k[measured_data_mask] = self.diffraction_pattern.data[measured_data_mask]*np.exp(1j*np.angle(k[measured_data_mask]))

            # Obtain a new object by taking the inverse FFT
            current_object = np.real(my_ifft(k))

            # Update display if necessary
            if self.display_results_during_reconstruction & (iterationNumber%50==0):
                self.displayResults()

    def displayResults(self):
        # Displays the current reconstruction and plots the error
        from matplotlib import pyplot as plt
        plt.figure(101)
        plt.subplot(121)
        plt.imshow(self.reconstruction[self.x_start : self.x_stop, self.y_start : self.y_stop])
        plt.subplot(122)
        plt.imshow(self.best_object)
        plt.get_current_fig_manager().window.setGeometry(25,100,500, 500)
        plt.draw()
        plt.title('Reconstruction')

        plt.figure(102)
        plt.plot(range(self.num_iterations),self.errK,'ko')
        plt.title("Reciprocal Error")
        plt.get_current_fig_manager().window.setGeometry(500,100,500, 500)
        plt.draw()
        plt.pause(1e-12)


    def saveResults(self,output_filename):
        np.savez(output_filename,reconstruction=self.reconstruction, errK=self.errK)

class OSSReconstruction(HIOReconstruction): # the (HIOReconstruction) means OSSReconstruction inherits from HIOReconstruction
    """Basic class for performing phase retrieval of a diffraction pattern with the
    OverSampling Smoothness (OSS) method.

    ATTRIBUTES:

    diffraction_pattern -> a DiffractionPattern object that contains a diffraction pattern stored in a numpy array
        and information about it

    num_iterations -> integer number of iterations to run in the reconstruction

    support -> numpy array of 1's and 0's where the region of 1's defines the boundary of the reconstructed object

    METHODS:
    reconstruct -> run OSS reconstruction with the current parameters stored as attributes in the HIOReconstruction object
        The results are stored in self.reconstruction and self.errK

    """

    def __init__(self, diffraction_pattern, num_iterations = 500, support = None, initial_object = None):
        # matplotlib.use("Qt4Agg")
        super(OSSReconstruction, self).__init__(diffraction_pattern, num_iterations, support, initial_object)
        self.number_of_filters = 10
        self.filter_start_sigma =  5
        self.filter_end_sigma = .5

    def reconstruct(self):
        """Primary reconstruction function"""
        print ('Reconstructing with OSS...')
        #setup the region to display by finding the rectangular boundaries of the support
        coords = np.where(self.support != 0)
        self.x_start = np.min(coords[0])
        self.x_stop = np.max(coords[0])
        self.y_start = np.min(coords[1])
        self.y_stop = np.max(coords[1])
        del coords

        beta = 0.9 # HIO beta parameter

        # Get dimensions
        dimX, dimY = np.shape(self.diffraction_pattern.data)

        # Initialize object that will hold the previous object for each iteration.
        previous_object = np.copy(self.initial_object)

        # Make a copy of the magnitudes to create the initial object
        initial_magnitudes = np.copy(self.diffraction_pattern.data)

        # Use 0 for the magnitude wherever we don't have a measurement
        initial_magnitudes[ self.diffraction_pattern.data == -1 ] = 0

        # Generate random initial phases
        random_phases = 2*np.pi*np.random.rand( dimX, dimY )

        # Obtain starting point for the reconstruction with these magnitudes and phases
        current_object = np.real(my_ifft(initial_magnitudes * np.exp(1j*random_phases)))

        # Make a mask where of the magnitudes that we will be replacing. This prevents us
        # having to recalculate this every iteration
        measured_data_mask = self.diffraction_pattern.data != -1

        # Initalize error array
        self.errK = np.zeros(self.num_iterations, dtype=float)

        best_error = 1e30 #initialize error to very high value

        filter_sigmas = np.linspace(self.filter_start_sigma, self.filter_end_sigma, self.number_of_filters)
        iterations_for_each_filter = np.diff(np.round(np.linspace(0,self.num_iterations,self.number_of_filters + 1)))
        # Begin primary loop
        total_iteration_number = 0
        for filter_number in range(self.number_of_filters):
            current_sigma_fraction = filter_sigmas[filter_number]
            if filter_number > 0:
                current_object = self.best_object
            for iterationNumber in range(iterations_for_each_filter[filter_number].astype(int)):

                #isolate object from the outside region that will be smoothed
                object_within_support = current_object * self.support

                #apply beta constraint as in HIO
                current_object = previous_object - beta * current_object

                #enforce positivity
                object_within_support[ object_within_support < 0 ] = current_object[ object_within_support < 0 ]

                #zero the region inside the support to prevent "bleedout" of density into the surroundings
                current_object [self.support >0] = 0

                #smooth exterior region
                current_object = smooth(current_object, current_sigma_fraction)

                #place object within support back into the reconstruction with smoothed background
                current_object[ self.support > 0 ] = object_within_support[ self.support > 0 ]

                # Update the old object
                previous_object = current_object

                # Take forward FFT of current_object, which is the reconstruction and current_object * self.support,
                # which is used to monitor the error
                k = my_fft(current_object)
                k_masked =  my_fft(current_object * self.support)

                # Compute error
                errK = np.sum(abs(abs(k_masked[measured_data_mask]) - \
                  self.diffraction_pattern.data[measured_data_mask])) / np.sum(abs(self.diffraction_pattern.data[measured_data_mask]))
                self.errK[ total_iteration_number ] = errK

                # If the current error is the lowest thus far, save that reconstruction
                if errK < best_error:
                    best_error = errK
                    self.best_object = current_object
                    self.reconstruction = current_object * self.support

                # Replace the magnitudes with measured ones (Fourier constraint)
                k[measured_data_mask] = self.diffraction_pattern.data[measured_data_mask]*np.exp(1j*np.angle(k[measured_data_mask]))

                # Obtain a new object by taking the inverse FFT
                current_object = np.real(my_ifft(k))

                # Update display if necessary
                if self.display_results_during_reconstruction & (total_iteration_number%50==0):
                    self.displayResults()
                if total_iteration_number % 50 == 1:
                    print("Iteration Number: {0} \t Lowest Error = {1:.2f} \t Current Error = {2:.2f}".format(total_iteration_number, best_error, self.errK[total_iteration_number]))
                total_iteration_number+=1

class PtychographyReconstruction(object):
    def __init__(self, diffraction_pattern_stack, aperture_positions, reconstructed_pixel_size = 1, num_iterations = 100, aperture_guess = None, initial_object = None):
        self.diffraction_pattern_stack = diffraction_pattern_stack # NumPy array of dimension N x N x number_of_patterns

        self.num_iterations = num_iterations
        self.aperture_guess = aperture_guess # initial guess of the aperture
        dp_dimX, dp_dimY, number_of_patterns = np.shape(self.diffraction_pattern_stack)

        # Adjust the aperture positions. Convert into pixels, center the origin at 0 and
        # add an offset of size (dp_dimX, dp_dimY) as a small buffer
        aperture_pos_X, aperture_pos_Y = zip(*aperture_positions)
        min_x_pos = min(aperture_pos_X) / reconstructed_pixel_size
        min_y_pos = min(aperture_pos_Y) / reconstructed_pixel_size
        aperture_pos_X = [int(pos/reconstructed_pixel_size - min_x_pos) + dp_dimX for pos in aperture_pos_X]
        aperture_pos_Y = [int(pos/reconstructed_pixel_size - min_y_pos) + dp_dimY for pos in aperture_pos_Y]
        self.aperture_positions = [pair for pair in zip(aperture_pos_X, aperture_pos_Y)]
        self.number_of_apertures = len(self.aperture_positions)
        # determine size of the macro reconstruction
        big_dim_X, big_dim_Y = max(aperture_pos_X) + dp_dimX, max(aperture_pos_Y)+ dp_dimY

        # Initialize array to hold reconstruction
        self.reconstruction = np.zeros((big_dim_X, big_dim_Y), dtype=complex)
        self.display_results_during_reconstruction = False

        if aperture_guess is None:
            self.aperture = np.ones((dp_dimX, dp_dimX), dtype=complex)
        else:
            self.aperture = aperture_guess
        # If no initial object was provided, default to zeros
        if initial_object is None:
            self.initial_object = np.zeros((big_dim_X, big_dim_Y), dtype=complex)
        else:
            self.initial_object = initial_object

class EPIEReconstruction(PtychographyReconstruction):
    def reconstruct(self):
        print ("DIMENSIONS = " , np.shape(self.diffraction_pattern_stack))
        aperture_update_start = 5 # start updating aperture on iteration 5
        beta_object = 0.9
        beta_aperture = 0.9
        dp_dimX, dp_dimY, number_of_patterns = np.shape(self.diffraction_pattern_stack)
        x_crop_vector = np.arange(dp_dimX) - dp_dimX//2
        minX,maxX = np.min(x_crop_vector), np.max(x_crop_vector) + 1
        y_crop_vector = np.arange(dp_dimY) - dp_dimY//2
        minY,maxY = np.min(y_crop_vector), np.max(y_crop_vector) + 1


        for iteration in range(self.num_iterations):
            print ("iteration = " , iteration)
            # randomly loop over the apertures each iteration
            for cur_apert_num in np.random.permutation(range(self.number_of_apertures)):
                # crop out the relevant sub-region of the reconstruction
                x_center, y_center = self.aperture_positions[cur_apert_num][0] , self.aperture_positions[cur_apert_num][1]
                r_space = self.reconstruction[ minX+x_center:maxX+x_center, minY+y_center:maxY+y_center ]
                buffer_r_space = np.copy(r_space)

                buffer_exit_wave = r_space * self.aperture
                update_exit_wave = my_fft(np.copy(buffer_exit_wave))
                current_dp = self.diffraction_pattern_stack[:, :, cur_apert_num]
                update_exit_wave[ current_dp != -1 ] = abs(current_dp[current_dp != -1])\
                                                     * np.exp(1j*np.angle(update_exit_wave[current_dp != -1]))


                update_exit_wave = my_ifft(update_exit_wave)
                # max_ap = np.max(np.abs(self.aperture))
                # norm_factor = beta_object / max_ap**2
                diff_wave = (update_exit_wave - buffer_exit_wave)
                new_r_space = buffer_r_space + diff_wave * \
                                    np.conjugate(self.aperture) * beta_object / np.max(np.abs(self.aperture))**2
                self.reconstruction[ minX+x_center:maxX+x_center, minY+y_center:maxY+y_center ] = new_r_space

            if iteration > aperture_update_start:
                # norm_factor_apert = beta_aperture / np.max(np.abs(r_space))**2
                self.aperture = self.aperture + beta_aperture / np.max(np.abs(r_space))**2 * \
                    np.conjugate(buffer_r_space)*diff_wave
            if iteration % 5 == 0:
                self.displayResults()

    def displayResults(self):
        # Displays the current reconstruction
        from matplotlib import pyplot as plt
        plt.figure(101)
        plt.subplot(221)
        plt.imshow(np.abs(self.reconstruction))
        # plt.get_current_fig_manager().window.setGeometry(25,100,500, 500)
        plt.draw()
        plt.title('Reconstruction Magnitude')
        plt.subplot(222)
        plt.imshow(np.angle(self.reconstruction))
        # plt.get_current_fig_manager().window.setGeometry(25,100,500, 500)
        plt.draw()
        plt.title('Reconstruction Phase')
        plt.subplot(223)
        plt.imshow(np.abs(self.aperture))
        plt.title("Aperture Magnitude")
        # plt.get_current_fig_manager().window.setGeometry(500,100,500, 500)
        plt.draw()
        plt.subplot(224)
        plt.imshow(np.angle(self.aperture))
        plt.title("Aperture Phase")
        # plt.get_current_fig_manager().window.setGeometry(500,100,500, 500)
        plt.draw()
        plt.pause(1e-30)

