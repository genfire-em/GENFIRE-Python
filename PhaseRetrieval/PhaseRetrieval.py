import numpy as np
import matplotlib
matplotlib.use("Qt4Agg")
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

    def maskBrightValues(self, threshhold_value = 0.0):
        # flag pixels at or near saturation limit as unmeasured
        self.data [self.data > threshhold_value] = -1

    def hermitianSymmetrize(self):
        """
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

    def correctCenter(self,search_box_half_size = 10, center_guess_x=None, center_guess_y=None):
        "This method optimizes the location of the diffraction pattern's center and shifts it accordingly \
         It does so by searching a range of centers determined by search_box_half_size. For each center, the \
         error between centrosymmetric partners is checked. The optimized center is the position which    \
         minimizes this error"

        # plt.title('Double-click the center')

        # plt.show()
        # plt.draw()

        # plt.get_current_fig_manager().window.setGeometry(25,25,750, 750)

        if center_guess_x is None or center_guess_y is None:
            import matplotlib.pyplot as plt
            h = plt.figure(999)
            plt.imshow(self.data)
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
        print ("center optimized")

    def makeArraySquare(self):
        """ Pad image to square array size that is the nearest even number greater than or equal to the current dimensions"""

        dimx, dimy = np.shape(self.data)
        new_dim = max(dimx,dimy) + (max(dimx,dimy)%2) # Take the ceiling even value above the larger dimension
        padding_x = ((new_dim - dimx)//2, (new_dim - dimx)//2 + (new_dim - dimx)%2 )
        padding_y = ((new_dim - dimy)//2, (new_dim - dimy)//2 + (new_dim - dimy)%2 )
        self.data = np.pad(self.data,(padding_x, padding_y), mode='constant')
        self.data [ self.data ==0] = -1

    def subtractBackground(self, bg_filename, scaling_factor = 1):
        from PIL import Image
        BG = Image.open(bg_filename)
        BG = np.array(BG,dtype=float)
        self.data -= (BG*scaling_factor) # shorthand for self.data = self.data - BG
        self.data[(self.data <= 0 )] = -1 # any negative values are to be flagged as "missing" with a -1

    def convertToFourierModulus(self):
        self.data[self.data != -1] = np.sqrt(self.data[self.data != -1])

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
