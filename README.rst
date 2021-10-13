lbnl-ccd v 0.1
==============

Contains code to analyze data from a 4kx4k LBNL CCD. To set it up, just clone this repository and run `python setup.py install` or `python setup.py develop` (ifou want to work on the code development).

Currently has only a few basic functions in one module called `analysis`.

Python Dependencies
-------------------
Using `Anaconda <https://www.continuum.io/downloads/>`_ is recommended.

lbnl-ccd depends on the following list of Python packages.

* `python <http://www.python.org/>`_ versions 3.8 or later
* `numpy <http://www.numpy.org/>`_ version 1.17 or later
* `astropy <http://www.astropy.org/>`_ version 4.0 or later
* `scipy <http://www.scipy.org/>`_ version 1.2 or later
* `ccdproc <https://github.com/astropy/ccdproc>`_ version 1.0 or later

Usage
-----
In a python terminal or notebook:
    import lbnlccd
