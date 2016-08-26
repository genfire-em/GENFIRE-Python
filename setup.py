from setuptools import find_packages,setup, Extension
from subprocess import call

setup(
	name 				= "GENFIRE",
	packages		    = find_packages(),
	version 			= "1.0",
	description 		= "GENeralized Fourier Iterative REconstruction",
	author 				= "Alan (AJ) Pryor, Jr. ",
    author_email 		= "apryor6@gmail.com",
    install_requires	= ['Pillow','numpy','matplotlib','scipy', 'Cython', 'pyFFTW','pyparsing','six'],
	scripts				= ['bin/genfire']
)


