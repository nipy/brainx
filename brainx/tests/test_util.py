"""Tests for the util module"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Stdlib
import sys

# Third party
import networkx as nx
import nose.tools as nt
import numpy as np
import numpy.testing as npt

# Hack until we make brainx an actually installable package
sys.path.append('..')
import util
reload(util)  # for interactive testing

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def assert_graphs_equal(g,h):
    """Trivial 'equality' check for graphs"""
    if not(g.nodes()==h.nodes() and g.edges()==h.edges()):
        raise AssertionError("Graphs not equal")


def test_regular_lattice():
    for n in [8,11,16]:
        # Be careful not to try and run with k > n-1, as the naive formula
        # below becomes invalid.
        for k in [2,4,7]:
            a = util.regular_lattice(n,k)
            msg = 'n,k = %s' % ( (n,k), )
            nedge = n * (k/2)  # even part of k
            yield nt.assert_equal,a.number_of_edges(),nedge,msg
