from setuptools import find_packages,setup, Extension
from subprocess import call
from Cython.Build import cythonize


import numpy
extensions = [
		Extension('weightValues',['./GENFIRE/weightValues.pyx'], include_dirs=[numpy.get_include()])]

setup(
		ext_modules=cythonize(extensions)
		)
		

setup(
	name 				= "GENFIRE",
	packages		    = find_packages(),
	version 			= "1.0",
	description 		= "GENeralized Fourier Iterative REconstruction",
	author 				= "Alan Pryor, Jr. (AJ)",
    author_email 		= "apryor6@gmail.com",
    install_requires	= ['Pillow','numpy','matplotlib','scipy', 'Cython', 'pyFFTW','pyparsing','six']
)


