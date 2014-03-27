#test nodal roles 
import unittest
import networkx as nx
import nodal_roles as nr
import weighted_modularity as wm

class TestNodalRoles(unittest.TestCase):
	def test_participation_coefficient_edgeless(self):
		graph = nx.Graph([(0,1)])
		graph.add_node(2)
		partition = wm.WeightedPartition(graph, communities=[ set([0,1]), set([2])])
		with self.assertRaises(ValueError):
			nr.participation_coefficient(partition)
	def test_within_community_degree_edgeless(self):
		graph = nx.Graph([(0,1)])
		graph.add_node(2)
		partition = wm.WeightedPartition(graph, communities=[ set([0,1]), set([2])])
		with self.assertRaises(ValueError):
			nr.within_community_degree(partition)
	def test_disconnected_communites(self):
		graph = nx.Graph([(0,1),(1,2),(2,0),(3,4),(3,5),(4,5)])
		partition = wm.WeightedPartition(graph, communities=[set([0, 1, 2]), set([3, 4, 5])])
		wcd = nr.within_community_degree(partition)
		self.assertAlmostEqual(wcd, {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0})
		pc = nr.participation_coefficient(partition)
		self.assertEqual(pc, {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0})
	def test_high_low_pc(self):
		graph = nx.Graph([(0,1),(1,2),(2,0),(0,3),(3,4),(3,5),(4,5)])
		partition = wm.WeightedPartition(graph, communities=[set([0, 1, 2]), set([3, 4, 5])])
		pc = nr.participation_coefficient(partition)
		self.assertAlmostEqual(pc,{0: 0.8888888888888888, 1: 0.0, 2: 0.0, 3: 0.8888888888888888, 4: 0.0, 5: 0.0})
	def test_high_low_wcd(self):
		graph = nx.Graph([(0,1),(0,2),(0,3),(0,4),(0,5),(6,7),(7,8),(8,6)])
		partition = wm.WeightedPartition(graph, communities=[set([0, 1, 2, 3, 4, 5]), set([8, 6, 7])])
		wcd = nr.within_community_degree(partition)
		self.assertAlmostEqual(wcd, {0: 3.8819660112501051, 1: -0.11803398874989512, 2: -0.11803398874989512, 3: -0.11803398874989512,4: -0.11803398874989512, 5: -0.11803398874989512, 6: 0.0, 7: 0.0, 8: 0.0})
