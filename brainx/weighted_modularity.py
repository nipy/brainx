

import copy
import numpy as np
import networkx as nx
from . import util


class Partition(object):
    """Represent a weighted Graph Partition

       The main object keeping track of the nodes in each partition is the
       community attribute. 
    """
    def __init__(self, graph, community=None):
        """ initialize partition of graph, with optional community

        Parameters
        ==========
        graph : networkx graph
        
        community : list of sets
            a list of sets with nodes in each set
            if community is None, will initialize with
            one community per node
        """
        # assert graph has edge weights, and no negative weights
        mat = nx.adjacency_matrix(graph)
        if mat.min() < 0:
            raise ValueError('Graph has invalid negative weights')

        self.graph = nx.from_numpy_matrix(mat)
        if community is None:
            self._community = self._init_communities_from_nodes()
        else:
            self.set_community(community)
        self.total_edge_weight = graph.size(weight='weight')
        self.degrees = graph.degree(weight='weight')

    @property
    def community(self):
        """list of sets decribing the communities"""
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
            raise TypeError('community should be list of sets, not '\
                '{}'.format(community))  
        ## simple count to check for all nodes
        return len(self.graph.nodes()) == \
            len([item for com in community for item in com])


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
    
    m2 = partition.total_edge_weight
    internal_connect = np.array(internal_links(partition))
    total = np.array(total_links(partition))
    return np.sum(internal_connect / m2 - (total/(2*m2))**2)

def total_links(part):
    """ sum of all links inside or outside community
    no nodes are missing"""
    comm = part.community
    weights = [0] * len(comm)
    all_degree_weights = part.graph.degree(weight='weight')
    for node, weight in all_degree_weights.items():
        node_comm = part.get_node_community(node)
        weights[node_comm]+= weight
    return weights

def internal_links(part):
    """ sum of weighted links strictly inside each community
    includes self loops"""
    comm = part.community
    weights = [0] * len(comm)
    comm = part.community
    for val, nodeset in enumerate(comm):
        for node in nodeset:
            nodes_within = set([x for x in part.graph[node].keys() \
                if x in nodeset])
            if len(nodes_within) < 1:
                continue
            if node in nodes_within:
                weights[val]+= part.graph[node][node]['weight']
                nodes_within.remove(node)
            weights[val] += np.sum(part.graph[node][x]['weight']/ 2. \
                for x in nodes_within)
    return weights

    
def meta_graph(partition):
    """ takes partition communities and creates a new meta graph where
    communities are now the nodes, the new edges are created based on the 
    node to node connections from original graph, and weighted accordingly,
    this includes self-loops"""
    metagraph = nx.Graph()
    # new nodes are communities
    newnodes = [val for val,_ in enumerate(partition.community)]
    mapping = {val: nodes for val, nodes in enumerate(partition.community)}
    metagraph.add_nodes_from(newnodes, weight=0.0) 

    for node1, node2, data in partition.graph.edges_iter(data=True):
        node1_community = partition.get_node_community(node1)
        node2_community = partition.get_node_community(node2)
        try:
            tmpw = metagraph[node1_community][node2_community]['weight']
        except KeyError:
            tmpw = 0
        metagraph.add_edge(
            node1_community,
            node2_community,
            weight = tmpw + data['weight'])

    return metagraph, mapping


def _communities_without_node(part, node):
    """ returns a version of the partition with the node
    removed, may result in empty community"""
    node_comm = part.get_node_community(node)
    newpart = copy.deepcopy(part.community)
    newpart[node_comm].remove(node)
    return newpart


def _community_nodes_alledgesw(part, removed_node):
    """ returns the sum of all weighted edges to nodes in each
    community, once the removed_node is removed
    this refers to totc in Blondel paper"""
    comm_wo_node = _communities_without_node(part, removed_node)
    weights = [0] * len(comm_wo_node)
    ## make a list of all nodes degree weights
    all_degree_weights = part.graph.degree(weight='weight').values()
    all_degree_weights = np.array(all_degree_weights)
    for val, nodeset in enumerate(comm_wo_node):
        node_index = np.array(list(nodeset)) #index of nodes in community
        #sum the weighted degree of nodes in community
        if len(node_index)<1:
            continue
        weights[val] = np.sum(all_degree_weights[node_index])
    return weights  


def node_degree(graph, node):
    """ find the summed weight to node
    Ki in Blondel paper"""
    return graph.degree(weight='weight')[node]


def dnodecom(node, part):
    """ Find the number of links from node to each community"""
    comm_weights = [0] * len(part.community)
    for neighbor, data in part.graph[node].items():
        if neighbor == node:
            continue
        tmpcomm = part.get_node_community(neighbor)
        comm_weights[tmpcomm] += data.get('weight',1)
    return comm_weights



def gen_dendogram(graph, community=None, min=0.0000001):
    """generate dendogram based on muti-levels of partitioning"""

    if type(graph) != nx.Graph :
        raise TypeError("Bad graph type, use only non directed graph")

    #special case, when there is no link 
    #the best partition is everyone in its community
    if graph.number_of_edges() == 0 :
        return Partition(graph)
        
    current_graph = graph.copy()
    part = Partition(graph, community)
    # first pass
    mod = modularity(part)
    dendogram = list()
    new_part = _one_level(part)
    new_mod = modularity(new_part)

    dendogram.append(new_part)
    mod = new_mod
    current_graph, _ = meta_graph(new_part)
    
    while True :
        partition = Partition(current_graph)
        newpart = _one_level(partition)
        new_mod = modularity(newpart)
        if new_mod - mod < min :
            break

        dendogram.append(newpart)
        mod = new_mod
        current_graph,_ = meta_graph(newpart)
    return dendogram


def partitions_from_dendogram(dendo):
    """ returns community partitions based on results in dendogram
    """
    all_partitions = []
    init_part = dendo[0].community
    all_partitions.append(init_part)
    for comm in dendo[1:]:
        init_part = _combine(init_part, comm.community)
        all_partitions.append(init_part)
    return all_partitions


def _calc_delta_modularity(node, part):
    """calculate the increase(s) in modularity if node is moved to other
    communities
    deltamod = inC - totc * ki / total_weight"""
    noded = node_degree(part.graph, node)
    dnc = dnodecom(node, part)
    totc = _community_nodes_alledgesw(part, node)
    total_weight = part.total_edge_weight
    # cast to arrays to improve calc
    dnc = np.array(dnc)
    totc = np.array(totc)
    return dnc - totc*noded / (total_weight*2)


def _move_node(part, node, new_comm):
    """generate a new partition with node put into new_comm"""
    ## copy
    new_community = [x.copy() for x in part.community]
    ## update
    curr_node_comm = part.get_node_community(node)
    ## remove
    new_community[curr_node_comm].remove(node)
    new_community[new_comm].add(node)
    new_comm = [x for x in new_community if len(x) > 0]
    return Partition(part.graph, new_comm)


def _one_level(part, min_modularity= .0000001):
    """run one level of patitioning"""
    curr_mod = modularity(part)
    modified = True
    while modified:
        modified = False
        all_nodes = [x for x in part.graph.nodes()]
        np.random.shuffle(all_nodes)
        for node in all_nodes:
            node_comm = part.get_node_community(node)
            delta_mod = _calc_delta_modularity(node, part)
            #print node, delta_mod
            if delta_mod.max() <= 0.0:
                # no increase by moving this node
                continue
            best_comm = delta_mod.argmax()
            if not best_comm == node_comm: 
                new_part = _move_node(part, node, best_comm)
                part = new_part
                modified = True
        new_mod = modularity(part)
        change_in_modularity = new_mod - curr_mod
        if change_in_modularity < min_modularity:
            return part
    return part
            

def _combine(prev, next):
    """combines nodes in set based on next level
    community partition

    Parameters
    ==========
    prev : list of sets
        community partition
    next : list of sets
        next level community partition
    """
    expected_len = np.max([x for sublist in next for x in sublist])
    if not len(prev) == expected_len + 1:
        raise ValueError('Number of nodes in next does not'\
            ' match number of communities in prev')
    ret = []
    for itemset in next:
        newset = set()
        for tmps in itemset:
            newset.update(prev[tmps])
        ret.append(newset)
    return ret





