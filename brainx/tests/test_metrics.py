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
        
