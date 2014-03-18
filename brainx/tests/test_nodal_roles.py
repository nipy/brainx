#test nodal roles 
import unittest
import networkx as nx
from .. import nodal_roles as nr
from .. import weighted_modularity as wm

class TestNodalRoles(unittest.TestCase):
	def test_participation_coefficient(self):
		graph = nx.Graph()
		graph.add_node(1)
		partition = wm.WeightedPartition(graph)
		with self.assertRaises(ValueError):
			nr.participation_coefficient(partition)