from setuptools import find_packages,setup, Extension
from subprocess import call
import sys

setup(
	name 				= "genfire",
	packages		    = find_packages(),
	version 			= "1.1.11",
	description 		= "GENeralized Fourier Iterative REconstruction",
	author 				= "Alan (AJ) Pryor, Jr.",
    author_email 		= "apryor6@gmail.com",
    install_requires	= ['Pillow>=4.1.1','numpy>=1.12.1','matplotlib>=2.0.2','scipy>=0.19.0', 'pyparsing>=2.2.0','six>=1.10.0', 'PyQt5>=5.5.0'],
	scripts				= ['bin/genfire']
)


