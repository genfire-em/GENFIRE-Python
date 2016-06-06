from PhaseRetrieval import *

def scale_and_combine_fits_images(directory_name, output_filename = None):
    """loop over all tiff images in directory_name, scale them to each other, and average"""
    if directory_name[-1] != '/':
        directory_name += '/'
    import os
    from astropy.io import fits
    image_count = 0
    average_image = None
    number_of_values = None
    for filename in os.listdir(directory_name):
        file,ext = os.path.splitext(filename)
        if ext==".fits" or ext==".FITS":
            if file != "BG":
                print("Loading {0}".format(directory_name + filename))
                if image_count == 0: #initialize image
                    # average_image = np.array(Image.open(directory_name + filename),dtype=float)
                    hdulist = fits.open(directory_name + filename)
                    average_image = np.array(hdulist[0].data,dtype=float)
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
                    # current_image = np.array(Image.open(directory_name + filename),dtype=float)
                    hdulist = fits.open(directory_name + filename)
                    current_image = np.array(hdulist[0].data,dtype=float)
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

def subtractBackground_fits(img, bg_filename, scaling_factor = 1):
        from astropy.io import fits
        hdulist = fits.open(bg_filename)
        BG = np.array(hdulist[0].data,dtype=float)
        BG = np.array(BG,dtype=float)
        img -= (BG*scaling_factor) # shorthand for self.data = self.data - BG
        img[(img <= 0 )] = -1 # any negative values are to be flagged as "missing" with a -1
        return img