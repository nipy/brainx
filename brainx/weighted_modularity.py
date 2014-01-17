

import numpy as np
import networkx as nx
from . import util


class Partition:
    def __init__(self, graph, community=None):
        """ initialize partition of graph, with optional community
        defined

        Parameters
        ==========
        graph : networkx graph
        community : list of sets
            a list of sets with nodes in each set
            if community is None, will initialize with
            one community per node
        """
        self.graph = graph
        if community is None:
            self._community = self._init_communities_from_nodes()
        else:
            self.set_community(community)
        self.total_edge_weight = graph.size(weight='weight')
        self.degrees = graph.degree(weight = 'weight')

    @property
    def community(self):
        return self._community
    @community.setter
    def community(self, value):
        self._community = self.set_community(value)

    def _init_communities_from_nodes(self):
        """ creates a new community with one node per community
        eg nodes = [0,1,2] -> community = [set([0]), set([1]), set([2])]
        """
        return [set([node]) for node in self.graph.nodes()]


    def community_degree(self):
        """ calculates the joint degree of a community"""
        community_degrees = []
        for com in self.community:
            tmp = np.sum([self.graph.degree(weight='weight')[x] for x in com])
            community_degrees.append(tmp)
        return community_degrees

    def get_node_community(self, node):
        """returns the node's community"""
        try:
            return [val for val,x in enumerate(self.community) if node in x][0]
        except:
            if not node in self.graph.nodes():
                raise ValueError('node {0} not in graph for this '\
                    'partition'.format(node))
            else:
                raise StandardError('cannot find community for node '\
                    '{0}'.format(node))

    def set_community(self, community):
        """ set the partition community to the input community"""
        if self._allnodes_in_community(community):
            self._community = community
        else:
            raise ValueError('missing nodes {0}'.format(community))


    def _allnodes_in_community(self, community):
        """ checks all nodes are represented in communities, also catches
        duplicate nodes"""
        if not (isinstance(community, list) and \
            util._contains_only(community, set)):
            raise TypeError('community should be list of sets, not'\
                '{}'.format(community))  
        ## simple count to check for all nodes
        return len(self.graph.nodes()) == \
            len([item for com in community for item in com])

def _intersect_neighbors_community(part, node):
    """for a given partition, and node find the other nodes with
    edges to node in the community"""
    graph = part.graph
    neighbors = graph[node].keys()
    node_community = part.get_node_community(node)
    community = [x for x in part.community[node_community] if not x == node]
    # remove node so we can check for self-loops (in neighbors)
    intersect = list(set(neighbors) & set(community))
    return intersect



def modularity(partition):
    """Modularity of a graph with given partition
    using Newman 2004 Physical Review paper

    Parameters
    ==========
    partition : weighted graph partition object

    Returns
    =======
    modularity : float
        value reflecting the relation of within community connection
        to across community connections
    """
    if partition.graph.is_directed():
        raise TypeError('only valid on non directed graphs')
    graph = partition.graph
    community_degree = partition.community_degree()
    comm_within_weight = [0] * len(partition.community)
    for node in graph:
        within = _intersect_neighbors_community(partition, node)
        inweight = 0
        if node in within: # self loop
            inweight += graph[node][node]['weight']
            within.remove(node)
        inweight += np.sum([graph[node][other]['weight'] \
            for other in within]) / 2.0
        comm_within_weight[partition.get_node_community(node)] += inweight

    community_degree = np.array(partition.community_degree())
    full_weight = np.array(partition.total_edge_weight)
    modularity = np.sum((comm_within_weight / full_weight) - \
        (community_degree / (2 * full_weight))**2)
    return modularity

def meta_graph(partition):
    """ takes partition communities and creates a new meta graph where
    communities are now the nodes, the new edges are created based on the 
    node to node connections from original graph, and weighted accordingly,
    this includes self-loops"""
    metagraph = nx.Graph()
    # new nodes are communities
    newnodes = [val for val,_ in enumerate(partition.community)]
    metagraph.add_nodes_from(newnodes, weight=0.0) 

    for node1, node2, data in partition.graph.edges_iter(data=True):
        node1_community = partition.get_node_community(node1)
        node2_community = partition.get_node_community(node2)
        try:
            tmpw = metagraph[node1_community][node2_community]['weight']
        except:
            tmpw = 0
        metagraph.add_edge(
            node1_community,
            node2_community,
            weight = tmpw + data['weight'])

    return metagraph


def _nodeweights_by_community(part, node):
    """ looks for all neighbors to node, and builds a weight
    list that adds the connection weights to all neighbors in 
    each communityi
    refers to Ki,in in Blondel paper"""
    comm_weights = [0] * len(part.community)
    for neighbor, data in part.graph[node].items():
        if neighbor == node:
            continue
        tmpcomm = part.get_node_community(neighbor)
        comm_weights[tmpcomm] += data.get('weight',1)
    return comm_weights