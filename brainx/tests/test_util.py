"""Tests for the util module"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import unittest

# Third party
import nose.tools as nt
import numpy as np
import numpy.testing as npt

# Our own
from brainx import util

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def test_slice_data():
    subcond, blocks, subjects, nodes = 5, 10, 20, 4 
    data_4d = np.ones((blocks, subjects, nodes, nodes))
    data_5d = np.ones((subcond, blocks, subjects, nodes, nodes))
    sym_4d = util.slice_data(data_4d, subjects - 1 , blocks - 1 )
    sym_5d = util.slice_data(data_5d, subjects -1 , blocks-1, subcond-1)
    npt.assert_equal(sym_4d.shape, (nodes, nodes))
    npt.assert_equal(sym_5d.shape, (nodes, nodes))
    npt.assert_raises(IndexError, util.slice_data, data_5d, subjects, blocks)


def test_all_positive():
    jnk = np.random.random(40)
    npt.assert_equal(util.all_positive(jnk), True)
    # zeros counted as positive
    jnk[0] = 0
    npt.assert_equal(util.all_positive(jnk), True)
    # find real negative
    jnk = jnk - 0.5
    npt.assert_equal(util.all_positive(jnk), False) 

def test_make_cost_thresh_lookup():
    adj_mat = np.zeros((10,10))
    ind = np.triu_indices(10,1)
    thresholds = np.linspace(.1, .8, 45)
    adj_mat[ind] = thresholds
    lookup = util.make_cost_thresh_lookup(adj_mat)

    npt.assert_equal(sorted(thresholds, reverse=True), lookup.weight)
    npt.assert_equal(lookup[0].cost < lookup[-1].cost, True) 
    # costs in ascending order
    ## last vector is same as second vector rounded to 2 decimals
    npt.assert_almost_equal(lookup.actual_cost, lookup.cost, decimal=2)

def test_cost_size():
    n_nodes = 5
    ## NOTE DeprecationWarnings are ignored by default in 2.7
    #npt.assert_warns(UserWarning, util.cost_size, n_nodes)



class TestCost2Thresh(unittest.TestCase):
    def setUp(self):
        nnodes, nsub, nblocks, nsubblocks = 45, 20, 6, 2
        prng = np.random.RandomState(42)
        self.data_5d = prng.random_sample((nsubblocks, nblocks, 
                nsub, nnodes, nnodes))
        ind = np.triu_indices(nnodes, k=1)
        nedges = (np.empty((nnodes, nnodes))[ind]).shape[0]
        costs, _, _ = util.cost_size(nnodes)
        self.nedges = nedges
        self.costs = costs
        self.lookup = np.zeros((nsubblocks, nblocks, nsub,2, nedges))
        bigcost =np.tile(costs[1:], nblocks*nsubblocks*nsub)
        bigcost.shape = (nsubblocks, nblocks, nsub, nedges)
        self.lookup[:,:,:,1,:] = bigcost
        for sblock in range(nsubblocks):
            for block in range(nblocks):
                for sid in range(nsub):
                    tmp = self.data_5d[sblock, block, sid]
                    self.lookup[sblock,block,sid,0,:] = sorted(tmp[ind], 
                            reverse=True)
        
    def test_cost2thresh2(self):
        thr = util.cost2thresh2(self.costs[100], 0,0,0,self.lookup)
        real_thr = self.lookup[0,0,0,0,100-1]
        npt.assert_almost_equal(thr, real_thr, decimal=7)

    def test_cost2thresh(self):
        lookup = self.lookup[0].squeeze()
        thr = util.cost2thresh(self.costs[100],0,0,lookup)
        real_thr = lookup[0,0,0,100-1]# costs padded by zero
        npt.assert_almost_equal(thr, real_thr, decimal=7)

    def test_format_matrix(self):
        thresh_matrix = util.format_matrix2(self.data_5d, 0,0,0,
                self.lookup, self.costs[100])
        print thresh_matrix.sum()
        npt.assert_equal(thresh_matrix.sum(), 100 -1)

    def test_threshold_adjacency_matrix(self):
        adj_matrix = self.data_5d[0,0,0].squeeze()
        mask, real_cost = util.threshold_adjacency_matrix(adj_matrix, 0)
        npt.assert_equal(mask.sum(), 0)
        npt.assert_equal(real_cost, 0)
        mask, real_cost = util.threshold_adjacency_matrix(adj_matrix, .9)
        npt.assert_equal(mask.sum(), 1840)
        npt.assert_equal(real_cost, 0.9)



def test_apply_cost():
    corr_mat = np.array([[0.0, 0.5, 0.3, 0.2, 0.1],
                         [0.5, 0.0, 0.4, 0.1, 0.2],
                         [0.3, 0.4, 0.0, 0.7, 0.2],
                         [0.2, 0.1, 0.7, 0.0, 0.4],
                         [0.1, 0.2, 0.2, 0.4, 0.0]])
    # A five-node undirected graph has ten possible edges.  Thus, the result
    # here should be a graph with five edges.
    possible_edges = 10
    cost = 0.5
    thresholded_corr_mat, threshold = util.apply_cost(corr_mat, cost,
                                                      possible_edges)
    nt.assert_true(np.allclose(thresholded_corr_mat,
                               np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.5, 0.0, 0.0, 0.0, 0.0],
                                         [0.3, 0.4, 0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.7, 0.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.4, 0.0]])))
    nt.assert_almost_equal(threshold, 0.3)
    # Check the case in which cost requires that one of several identical edges
    # be kept and the others removed.  apply_cost should keep all of these
    # identical edges.
    #
    # To test this, I need to update only a value in the lower triangle.  The
    # function zeroes out the upper triangle immediately.
    corr_mat[2, 0] = 0.2
    thresholded_corr_mat, threshold = util.apply_cost(corr_mat, cost,
                                                      possible_edges)
    nt.assert_true(np.allclose(thresholded_corr_mat,
                               np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.5, 0.0, 0.0, 0.0, 0.0],
                                         [0.2, 0.4, 0.0, 0.0, 0.0],
                                         [0.2, 0.0, 0.7, 0.0, 0.0],
                                         [0.0, 0.2, 0.2, 0.4, 0.0]])))
    nt.assert_almost_equal(threshold, 0.2)


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

def test_rescale_arr():
    array = np.arange(5)
    scaled = util.rescale_arr(array, 3, 6)
    npt.assert_equal(scaled.min(), 3)
    scaled = util.rescale_arr(array, -10, 10)
    npt.assert_equal(scaled.min(), -10)
    npt.assert_equal(scaled.max(), 10)

def test_normalize():
    array = np.arange(5)
    result = util.normalize(array)
    npt.assert_equal(result.min(), 0)
    npt.assert_equal(result.max(), 1)
    npt.assert_raises(ValueError, util.normalize, array, 'blueberry', (0,2)) 


