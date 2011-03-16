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


Testing
=======

To run the test suite, once you have installed it as per the above
instructions,  simply use::

  nosetests brainx

or for more informative details::

  nosetests -vvs brainx

For further information, type ``nosetests -h``.


License information
===================

Brainx is licensed under the terms of the new BSD license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.
