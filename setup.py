from setuptools import find_packages,setup
from subprocess import call
#call(["$pip install scipy"])

setup(
	name = "GENFIRE",
	packages = find_packages(),
	version = "0.0",
	description = "GENeralized Fourier Iterative REconstruction",
	author = "Alan Pryor, Jr. (AJ)",
    author_email = "apryor6@gmail.com",
    setup_requires=['numpy'],
    install_requires=['Pillow','numpy','matplotlib','scipy']
)
#call(["pip install scipy"])

