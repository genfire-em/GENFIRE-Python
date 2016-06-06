import scipy.io as io
from PhaseRetrieval import *
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt


# diffraction_pattern_directory_name = './images/'
diffraction_pattern_directory_name = ['./data1', './data2']

background_filename = './BG/BG.tiff'
output_filename = 'output.npz'
support_size_X = 15
support_size_Y = 15
saturated_pixel_percentage_threshold = 0.45 # pixels higher than this percentage of the maximum will be discarded
turn_figure_on = True # True/False: do you want to display the figure during the reconstruction?
bin_factor_x = 2
bin_factor_y = 2
num_iterations = 1000
phase_retrieval_algorithm = 1 # 0 for HIO; 1 for OSS


diffraction_pattern = np.load('diffraction_pattern.npy')

# initialize flags
diffraction_pattern[ diffraction_pattern < 0 ] = -1

# instantiate a DiffractionPattern
my_diffraction_pattern = DiffractionPattern(diffraction_pattern)

# flag saturated pixels
my_diffraction_pattern.maskBrightValues(saturated_pixel_percentage_threshold)

# subtract background
# if type(diffraction_pattern_directory_name) is not list: # the multi directory version already subtracts BG
my_diffraction_pattern.subtractBackground(background_filename,scaling_factor=1.2)

# optimize center
my_diffraction_pattern.correctCenter()

# make pad array such that the dimensions (and therefore the reconstructed pixes) are square
my_diffraction_pattern.makeArraySquare()

# bin pattern
my_diffraction_pattern.binImage( bin_factor_x, bin_factor_y )

# take square root
# my_diffraction_pattern.convertToFourierModulus()

# create support
support = np.zeros(np.shape(my_diffraction_pattern.data),dtype=float)
support[:support_size_X, :support_size_Y] = 1

# instantiate reconstruction object
if phase_retrieval_algorithm == 0:
    my_reconstruction = HIOReconstruction(my_diffraction_pattern,num_iterations=num_iterations,support=support)
else:
    my_reconstruction = OSSReconstruction(my_diffraction_pattern,num_iterations=num_iterations,support=support)

my_reconstruction.display_results_during_reconstruction = turn_figure_on

# display processed diffraction pattern
plt.figure(100)
plt.imshow(my_reconstruction.diffraction_pattern.data)
plt.draw()
plt.get_current_fig_manager().window.setGeometry(1000,100,500, 500)
plt.title('Processed Diffraction Pattern')

# run reconstruction
my_reconstruction.reconstruct()

# save results
my_reconstruction.saveResults(output_filename)

plt.figure(101)
plt.subplot(121)
plt.imshow(my_reconstruction.reconstruction[my_reconstruction.x_start : my_reconstruction.x_stop, my_reconstruction.y_start : my_reconstruction.y_stop])
plt.subplot(122)
plt.imshow(my_reconstruction.reconstruction)
plt.get_current_fig_manager().window.setGeometry(25,100,500, 500)
plt.draw()
plt.title('Reconstruction')

plt.figure(102)
plt.plot(range(my_reconstruction.num_iterations),my_reconstruction.errK,'ko')
plt.title("Reciprocal Error")
plt.get_current_fig_manager().window.setGeometry(500,100,500, 500)
plt.show()
