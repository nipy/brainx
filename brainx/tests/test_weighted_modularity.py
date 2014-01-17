"""Tests for the weighted_modularity module"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import os
import unittest


# Third party
import networkx as nx
import nose.tools as nt
import numpy as np
import numpy.testing as npt

# Our own
from .. import util
from .. import weighted_modularity as wm


def get_test_data():
	""" grabs local txt file with adj matrices
    Returns
    =======
    graph : networkx graph
    community : list of sets
    """
	pth, _ = os.path.split(__file__)
	testdir = os.path.join(pth, 'tdata_corr_txt')
	data_file = os.path.join(testdir, '101_Block01.txt')
	mat = np.loadtxt(data_file)
	graph = nx.from_numpy_matrix(mat)
	# graph has 85 nodes, make generic community
	community = [set(range(42)), set(range(42,86))]
	return graph, community

class TestPartition(unittest.TestCase):

    def setUp(self):
        ## generate a default graph and community
        graph, community = get_test_data()
        self.graph = graph
        self.community = community

    def test_init(self):
        part = wm.Partition(self.graph)
        self.assertEqual(type(part.degrees), type({}))
        npt.assert_array_almost_equal(part.total_edge_weight, 1485.466536)
        self.assertEquals(part.graph, self.graph)
        # generated communities
        comm = [set([node]) for node in self.graph.nodes()]
        self.assertEqual(part.community, comm)

    def test_community_degree(self):
        ## if no community, method will raise error
        part = wm.Partition(self.graph)
        part = wm.Partition(self.graph, self.community)
        cdegree = part.community_degree()
        self.assertEqual(round(cdegree[0]), 1444.0)


    def test_set_community(self):
        part = wm.Partition(self.graph, self.community)
        self.assertEqual(part.community, self.community)
        with self.assertRaises(TypeError):
            # raise error if not list of sets
            part.set_community(part.community[0])
        with self.assertRaises(TypeError):
            part.set_community('a')
        with self.assertRaises(ValueError):
            ## missing nodes
            comm = self.graph.nodes()[:-3]
            part.set_community([set(comm)])

    def test_allnodes_in_community(self):
        """checks communities contain all nodes
        with no repetition"""
        part = wm.Partition(self.graph)
        self.assertTrue(part._allnodes_in_community(self.community))
        self.assertFalse(part._allnodes_in_community([self.community[0]]))


    def test_get_node_community(self):
        part = wm.Partition(self.graph, self.community)
        self.assertEqual(part.get_node_community(0), 0)
        self.assertEqual(part.get_node_community(self.graph.nodes()[-1]),1)
        with self.assertRaises(ValueError):
            part.get_node_community(-1)
        part = wm.Partition(self.graph)
        self.assertEqual(part.get_node_community(0), 0)


def test_modularity():
    graph, comm = get_test_data()
    part = wm.Partition(graph, comm)
    npt.assert_almost_equal(wm.modularity(part), -0.00192480)

def test_meta_graph():
    graph, community = get_test_data()
    part = wm.Partition(graph)
    metagraph = wm.meta_graph(part)
    ## each node is a comm, so no change to metagraph
    npt.assert_equal(metagraph.nodes(), graph.nodes())
    ## two communitties
    part = wm.Partition(graph, community)
    metagraph = wm.meta_graph(part)
    npt.assert_equal(metagraph.nodes(), [0,1])
    npt.assert_equal(metagraph.edges(), [(0,0),(0,1), (1,1)])
    ## weight should not be lost between graphs
    npt.assert_almost_equal(metagraph.size(weight='weight'),
        graph.size(weight='weight'))

def test_nodeweights_by_community():
    graph, community = get_test_data()
    part = wm.Partition(graph) # one comm per node
    cweights2node = wm._nodeweights_by_community(part,0)
    # self loops not counted to weight to self community should be 0
    npt.assert_equal(cweights2node[0],0)
    part = wm.Partition(graph, community)
    cweights2node = wm._nodeweights_by_community(part,0)
    npt.assert_equal(len(cweights2node), 2)

def test_communities_without_node():
    graph, community = get_test_data()
    part = wm.Partition(graph) # one comm per node   
    node = 0
    updated_comm = wm._communities_without_node(part, node )
    npt.assert_equal(updated_comm[0], set([]))
    part = wm.Partition(graph, community)
    updated_comm = wm._communities_without_node(part, node )
    ## make sure we dont break community from original partition
    npt.assert_equal(part.community, community)
    npt.assert_equal(0 not in updated_comm[0], True)
    
def test_weight_alledges_tonode():
    graph, community = get_test_data()
    part = wm.Partition(graph) # one comm per node   
    node = 0    
    res = wm.weight_alledges_tonode(graph, node)
    npt.assert_almost_equal(res,36.9415167 )
