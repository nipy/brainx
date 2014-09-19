#test nodal roles 
import unittest
import networkx as nx
import nodal_roles as nr
import weighted_modularity as wm

class TestNodalRoles(unittest.TestCase):
	def test_participation_coefficient(self):
		graph = nx.Graph([(0,1)])
		graph.add_node(2)
		louvain = wm.LouvainCommunityDetection(graph)
		weighted_partitions = louvain.run()
		weighted_partition = weighted_partitions[0]
		with self.assertRaises(ValueError):
			nr.participation_coefficient(weighted_partition)
		graph = nx.Graph([(0,1),(1,2),(2,0),(3,4),(3,5),(4,5)])
		louvain = wm.LouvainCommunityDetection(graph)
		partition = louvain.run()[0]
	def test_within_community_degree(self):
		graph = nx.Graph([(0,1)])
		graph.add_node(2)
		louvain = wm.LouvainCommunityDetection(graph)
		weighted_partitions = louvain.run()
		weighted_partition = weighted_partitions[0]
		with self.assertRaises(ValueError):
			nr.within_community_degree(weighted_partition)
	def test_disconnected_communites(self):
		graph = nx.Graph([(0,1),(1,2),(2,0),(3,4),(3,5),(4,5)])
		louvain = wm.LouvainCommunityDetection(graph)
		partition = louvain.run()[0]
		wcd = nr.within_community_degree(partition)
		self.assertEqual(wcd, {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0})
		pc = nr.participation_coefficient(partition)
		self.assertEqual(pc, {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0})