"""Tests for the metrics module"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Third party
import networkx as nx
import nose.tools as nt
import numpy as np
import numpy.testing as npt

# Our own imports
from brainx import metrics

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def test_nodal_pathlengths():
    corr_mat = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.5, 0.0, 0.0, 0.0, 0.0],
                         [0.3, 0.4, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.7, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.4, 0.0]])
    n_nodes = 5
    g = nx.from_numpy_matrix(corr_mat)
    path_lengths = metrics.nodal_pathlengths(g, n_nodes)
    # Distances for all node pairs:
    # 0-1: 1    1-2: 1    2-3: 1    3-4: 1
    # 0-2: 1    1-3: 2    2-4: 2
    # 0-3: 2    1-4: 3
    # 0-4: 3
    desired = 1.0 / (n_nodes - 1) * np.array([1 + 1 + 2 + 3,
                                              1 + 1 + 2 + 3,
                                              1 + 1 + 1 + 2,
                                              2 + 2 + 1 + 1,
                                              3 + 3 + 2 + 1])
    npt.assert_array_almost_equal(path_lengths, desired)
    # Check how unreachability is handled.
    g.remove_edge(2, 3)
    # Now all nodes have at least one edge, but not all nodes are reachable
    # from all others.
    path_lengths = metrics.nodal_pathlengths(g, n_nodes)
    # Distances for all node pairs:
    # 0-1: 1    1-2: 1    2-3: Inf  3-4: 1
    # 0-2: 1    1-3: Inf  2-4: Inf
    # 0-3: Inf  1-4: Inf
    # 0-4: Inf
    desired = 1.0 / (n_nodes - 1) * np.array([1 + 1 + np.inf + np.inf,
                                              1 + 1 + np.inf + np.inf,
                                              1 + 1 + np.inf + np.inf,
                                              np.inf + np.inf + np.inf + 1,
                                              np.inf + np.inf + np.inf + 1])
    npt.assert_array_almost_equal(path_lengths, desired)


def test_nodal_efficiency():
    corr_mat = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.5, 0.0, 0.0, 0.0, 0.0],
                         [0.3, 0.4, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.7, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.4, 0.0]])
    n_nodes = 5
    g = nx.from_numpy_matrix(corr_mat)
    n_eff_array = metrics.nodal_efficiency(g, n_nodes)
    # Distances for all node pairs:
    # 0-1: 1    1-2: 1    2-3: 1    3-4: 1
    # 0-2: 1    1-3: 2    2-4: 2
    # 0-3: 2    1-4: 3
    # 0-4: 3
    desired = 1.0 / (n_nodes - 1) * np.array([1 + 1 + 1 / 2.0 + 1 / 3.0,
                                              1 + 1 + 1 / 2.0 + 1 / 3.0,
                                              1 + 1 + 1 + 1 / 2.0,
                                              1 / 2.0 + 1 / 2.0 + 1 + 1,
                                              1 / 3.0 + 1 / 3.0 + 1 / 2.0 + 1])
    npt.assert_array_almost_equal(n_eff_array, desired)
    # Check how unreachability is handled.
    g.remove_edge(2, 3)
    # Now all nodes have at least one edge, but not all nodes are reachable
    # from all others.
    n_eff_array = metrics.nodal_efficiency(g, n_nodes)
    # Distances for all node pairs:
    # 0-1: 1    1-2: 1    2-3: Inf  3-4: 1
    # 0-2: 1    1-3: Inf  2-4: Inf
    # 0-3: Inf  1-4: Inf
    # 0-4: Inf
    desired = (1.0 / (n_nodes - 1) *
               np.array([1 + 1 + 1 / np.inf + 1 / np.inf,
                         1 + 1 + 1 / np.inf + 1 / np.inf,
                         1 + 1 + 1 / np.inf + 1 / np.inf,
                         1 / np.inf + 1 / np.inf + 1 / np.inf + 1,
                         1 / np.inf + 1 / np.inf + 1 / np.inf + 1]))
    npt.assert_array_almost_equal(n_eff_array, desired)


def test_path_lengths():
    """Very primitive tests, just using complete graphs which are easy.  Better
    than nothing..."""
    for nnod in [2,4,6]:
        g = nx.complete_graph(nnod)
        nedges = nnod*(nnod-1)/2
        path_lengths = metrics.path_lengths(g)
        # Check that we get the right size array
        nt.assert_equals(nedges, len(path_lengths))
        # Check that all lengths are 1
        pl_true = np.ones_like(path_lengths)
        npt.assert_equal(pl_true, path_lengths)
        
