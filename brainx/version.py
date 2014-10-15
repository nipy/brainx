"""brainx version/release information"""

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = '' # use '' for first of series, number for 1 and above
_version_extra = 'dev'
#_version_extra = '' # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

classifiers = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description = "Brainx: timeseries analysis for neuroscience data"

# Note: this long_description is actually a copy/paste from the top-level
# README.txt, so that it shows up nicely on PyPI.  So please remember to edit
# it only in one place and sync it correctly.
long_description = """
================================================
 Brainx: network analysis for neuroimaging data
================================================

Brainx provides a set of tools, based on the NetworkX graph theory package, for
the analysis of graph properties of neuroimaging data.


Installation
============

For a normal installation, simply type::

  python setup.py install [other options here]

To install using setuptools support, use::

  python setup_egg.py install [other options here]

For example, to install using a development-mode setup in your personal user
directory, use::

  python setup_egg.py develop --prefix=$HOME/.local

  
License information
===================

Brainx is licensed under the terms of the new BSD license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.
"""

# Other constants for distutils setup() call

name                = "brainx"
maintainer          = "Nipy Developers"
maintainer_email    = "nipy-devel@neuroimaging.scipy.org"
url                 = "http://nipy.org/brainx"
download_url        = "http://github.com/nipy/brainx/downloads"
license             = "Simplified BSD"
author              = "Brainx developers"
author_email        = "nipy-devel@neuroimaging.scipy.org"
platforms           = "OS Independent"
version             = __version__
packages            = ['brainx',
                       'brainx.tests',
                       ]
package_data        = {"brainx": ["LICENSE"]}
install_requires    = ["numpy", "matplotlib", "scipy", "networkx"]
test_requires       = ["nose", "runipy", "ipython"]
