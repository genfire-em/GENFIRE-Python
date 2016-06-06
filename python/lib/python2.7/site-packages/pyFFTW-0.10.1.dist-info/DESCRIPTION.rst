pyFFTW is a pythonic wrapper around `FFTW <http://www.fftw.org/>`_, the
speedy FFT library. The ultimate aim is to present a unified interface for all
the possible transforms that FFTW can perform.

Both the complex DFT and the real DFT are supported, as well as arbitrary
axes of abitrary shaped and strided arrays, which makes it almost
feature equivalent to standard and real FFT functions of ``numpy.fft`` 
(indeed, it supports the ``clongdouble`` dtype which ``numpy.fft`` does not).

Operating FFTW in multithreaded mode is supported.

A comprehensive unittest suite can be found with the source on the github 
repository.

To build for windows from source, download the fftw dlls for your system
and the header file from here (they're in a zip file):
http://www.fftw.org/install/windows.html and place them in the pyfftw
directory. The files are libfftw3-3.dll, libfftw3l-3.dll, libfftw3f-3.dll 
and libfftw3.h.

Under linux, to build from source, the FFTW library must be installed already.
This should probably work for OSX, though I've not tried it.

Numpy is a dependency for both.

The documentation can be found 
`here <http://hgomersall.github.com/pyFFTW/>`_, and the source
is on `github <https://github.com/hgomersall/pyFFTW>`_.


