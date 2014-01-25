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
    mat[mat<0] = 0
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
        npt.assert_array_almost_equal(part.total_edge_weight, 1500.5653444)
        # generated communities
        comm = [set([node]) for node in self.graph.nodes()]
        self.assertEqual(part.community, comm)

    def test_community_degree(self):
        ## if no community, method will raise error
        part = wm.Partition(self.graph)
        part = wm.Partition(self.graph, self.community)
        cdegree = part.community_degree()
        self.assertEqual(round(cdegree[0]), 1462.0)


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
    npt.assert_almost_equal(wm.modularity(part), 0.0555463)


def test_total_links():
    graph, community = get_test_data()
    part = wm.Partition(graph) # one comm per node
    ## summ of all links in or out of community
    ## since one per scommunity, just weighted degree of each node
    tot_per_comm = wm.total_links(part)
    degw = graph.degree(weight='weight').values()
    npt.assert_equal(tot_per_comm, degw)
    ## This isnt true of we have communities with multiple nodes
    part_2comm = wm.Partition(graph, community)
    npt.assert_equal(part_2comm == degw, False)

def test_internal_links():
    graph, community = get_test_data()
    part = wm.Partition(graph) # one comm per node
    weights = wm.internal_links(part) 
    ## this inlcudes seld links so      


def test_dnodecom():
    graph, community = get_test_data()
    part = wm.Partition(graph) # one comm per node
    node = 0
    node2comm_weights = wm.dnodecom(node, part)
    # self loops not added to weight 
    # so community made only of node should be zero
    npt.assert_equal(node2comm_weights[0],0)
    # this should be equal to weight between two nodes
    neighbor = 1
    expected = graph[node][neighbor]['weight']
    npt.assert_equal(node2comm_weights[neighbor],expected)
    part = wm.Partition(graph, community)
    node2comm_weights = wm.dnodecom(node, part)
    npt.assert_equal(len(node2comm_weights), 2) 

def test_meta_graph():
    graph, community = get_test_data()
    part = wm.Partition(graph)
    metagraph,_ = wm.meta_graph(part)
    ## each node is a comm, so no change to metagraph
    npt.assert_equal(metagraph.nodes(), graph.nodes())
    ## two communitties
    part = wm.Partition(graph, community)
    metagraph,mapping = wm.meta_graph(part)
    npt.assert_equal(metagraph.nodes(), [0,1])
    npt.assert_equal(metagraph.edges(), [(0,0),(0,1), (1,1)])
    # mapping should map new node 0 to community[0]
    npt.assert_equal(mapping[0], community[0])
    ## weight should not be lost between graphs
    npt.assert_almost_equal(metagraph.size(weight='weight'),
        graph.size(weight='weight'))


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

def test_community_nodes_alledgesw():
    graph, community = get_test_data()
    part = wm.Partition(graph, community)    
    node = 0
    weights = wm._community_nodes_alledgesw(part, node)
    npt.assert_almost_equal(weights[0], 1424.0220362)
    ## test with possible empty node set
    part = wm.Partition(graph)
    weights = wm._community_nodes_alledgesw(part, node)
    npt.assert_equal(weights[0], 0)
    # other communities are made up of just one node
    npt.assert_equal(weights[1], graph.degree(weight='weight')[1])




def test_node_degree():
    graph, community = get_test_data()
    part = wm.Partition(graph) # one comm per node   
    node = 0    
    res = wm.node_degree(graph, node)
    npt.assert_almost_equal(res, 37.94151675 )

def test_combine():
    first = [set([0,1,2]), set([3,4,5]), set([6,7])]
    second = [set([0,2]), set([1])]
    npt.assert_raises(ValueError, wm._combine, second, first)
    res = wm._combine(first, second)
    npt.assert_equal(res, [set([0,1,2,6,7]), set([3,4,5])])


def test_calc_delta_modularity():
    graph, community = get_test_data()
    part = wm.Partition(graph) # one comm per node
    node = 0
    change = wm._calc_delta_modularity(node, part)
    npt.assert_equal(len(change), len(part.community))
    # change is an array
    npt.assert_equal(change.shape[0], len(part.community))
    npt.assert_equal(change[0] < change[1], True)
    # this is one comm per node, so once removed from own
    # comm, this delta_weight will be zero
    npt.assert_equal(change[node] , 0) 


def test_move_node():
    graph, community = get_test_data()
    part = wm.Partition(graph) # one comm per node 
    #move first node to second community 
    node = 0
    comm = 1
    newpart = wm._move_node(part, node, comm )
    npt.assert_equal(set([0,1]) in newpart.community, True)
    ## what happens if node or comm missing
    with npt.assert_raises(ValueError):
        newpart = wm._move_node(part, -1, comm) 
    invalid_community = len(part.community) + 1
    with npt.assert_raises(IndexError):
        newpart = wm._move_node(part, node, invalid_community)  