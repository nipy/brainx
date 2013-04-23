"""Tests for the util module"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------


# Third party
import nose.tools as nt
import numpy as np
import numpy.testing as npt

# Our own
from brainx import util

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
            nt.assert_equal,a.number_of_edges(),nedge,msg

def test_diag_stack():
    """Manual verification of simple stacking."""
    a = np.empty((2,2))
    a.fill(1)
    b = np.empty((3,3))
    b.fill(2)
    c = np.empty((2,3))
    c.fill(3)

    d = util.diag_stack((a,b,c))

    d_true = np.array([[ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  2.,  2.,  2.,  0.,  0.,  0.],
       [ 0.,  0.,  2.,  2.,  2.,  0.,  0.,  0.],
       [ 0.,  0.,  2.,  2.,  2.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  3.,  3.,  3.],
       [ 0.,  0.,  0.,  0.,  0.,  3.,  3.,  3.]])

    npt.assert_equal(d, d_true)


def test_no_empty_modules():
    """Test the utility that validates partitions against empty modules.
    """
    a = {0: [1,2], 1:[3,4]}
    b = a.copy()
    b[2] = []
    util.assert_no_empty_modules(a)
    nt.assert_raises(ValueError, util.assert_no_empty_modules, b)
    
