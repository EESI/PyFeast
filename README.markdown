====
PyFeast
====
Python Interface to the FEAST Feature Selection Toolbox

About
====
This set of scripts provides an interface to the FEAST feature selection
toolbox, originally written in C with a Mex interface to Matlab. Python 
2.7 is required, along with Numpy. The feast.py module provides an inter-
face to all the functionality of the FEAST implementation that was provided
with the original Matlab interface. 

Installation
====
To install the FEAST interface, you'll need to build and install the libraries 
first, and then install python.

Make MIToolbox and install it:

$ cd FEAST/MIToolbox
$ make
$ sudo make install

Make FSToolbox and install it:

$ cd FEAST/FSToolbox
$ make
$ sudo make install

Install our PyFeast module

$ python ./setup.py build
$ sudo python ./setup.py install


Demonstration
====
See test/test.py for an example with uniform data and an image
data set. The image data set was collected from the digits example in 
the Scikits-Learn toolbox.

To Do
====
1) Add a setup.py script that can manage the build and installation of the
   Python interface to FEAST. 
2) Add in the rest of the functionality to feast.py. Add subsequent 
   functionality into the demo. 
3) Integrate the module into KBase!
4) Clean up the paths
