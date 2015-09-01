import copy
import numpy as np
import networkx as nx
import util


class WeightedPartition(object):
    """Represent a weighted GraphPartition

    The main object keeping track of the nodes in each partition is the
       communities attribute.
    """
    def __init__(self, graph, communities=None):
        """Initialize partition of graph, with optional communities

        Parameters
        ----------
        graph : networkx graph
        communities : list of sets, optional
            a list of sets with nodes in each set
            if communities is None, will initialize with
            one  per node

        Returns
        -------
        part : WeightedPartition object
        """
        self.graph = graph
        self._init_weight_attributes()
        if communities is None:
            self._communities = self._init_communities_from_nodes()
        else:
            self.set_communities(communities)
        ## whole-graph strength
        self.total_strength = graph.size(weight = 'weight')
        self.total_positive_strength = graph.size(weight = 'positive_weight')
        self.total_negative_strength = graph.size(weight = 'negative_weight')
        ## node strengths
        self.node_strengths = graph.degree(weight = 'weight')
        self.positive_node_strengths = graph.degree(weight = 'positive_weight')
        self.negative_node_strengths = graph.degree(weight = 'negative_weight')

    @property
    def communities(self):
        """List of sets decribing the communities"""
        return self._communities

    @communities.setter
    def communities(self, value):
        self.set_communities(value)

    def _init_communities_from_nodes(self):
        """Creates a new communities with one node per community
           eg nodes = [0,1,2] -> communities = [set([0]), set([1]), set([2])]
        """
        return [set([node]) for node in self.graph.nodes()]
        
    def _init_weight_attributes(self):
        """Adds edge attributes to graph to accomodate postive_weight and 
           negative_weight calculations"""
        weights = np.array(nx.get_edge_attributes(self.graph, 'weight').values(),\
                               dtype=float)
        nx.set_edge_attributes(self.graph, 'positive_weight',\
                               dict(zip(nx.get_edge_attributes(self.graph, 'weight').keys(),\
                                        weights * np.array(weights > 0., dtype=bool))))
        nx.set_edge_attributes(self.graph, 'negative_weight',\
                               dict(zip(nx.get_edge_attributes(self.graph, 'weight').keys(),\
                                       np.abs(weights * np.array(weights < 0., dtype=bool)))))

    def set_communities(self, communities):
        """Set the partition communities to the input communities"""
        if self._allnodes_in_communities(communities):
            self._communities = communities
        else:
            raise ValueError('missing nodes {0}'.format(communities))

    def get_node_community(self, node):
        """Returns the node's community"""
        try:
            return [val for val,x in enumerate(self.communities) if node in x][0]
        except IndexError:
            if not node in self.graph.nodes():
                raise ValueError('node:{0} is not in the graph'.format(node))
            else:
                raise StandardError('cannot find community for node '\
                    '{0}'.format(node))

    def _allnodes_in_communities(self, communities):
        """Checks all nodes are represented in communities, also catches
           duplicate nodes"""
        if not (isinstance(communities, list) and \
            util._contains_only(communities, set)):
            raise TypeError('communities should be list of sets, not '\
                '{}'.format(communities))
        ## simple count to check for all nodes
        return len(self.graph.nodes()) == \
            len([item for com in communities for item in com])

    def communities_positive_strength(self):
        """Calculate the joint positive strength of a community"""
        communities_positive_strengths = []
        for com in self.communities:
            tmp = np.sum([self.graph.degree(weight = 'positive_weight')[x] for x in com])
            communities_positive_strengths.append(tmp)
        return communities_positive_strengths

    def communities_negative_strength(self):
        """Calculate the joint negative strength of a community"""
        communities_negative_strengths = []
        for com in self.communities:
            tmp = np.sum([self.graph.degree(weight = 'negative_weight')[x] for x in com])
            communities_negative_strengths.append(tmp)
        return communities_negative_strengths

    def node_positive_strength(self, node):
        """Find the weighted sum of all positive node edges"""
        return self.graph.degree(weight = 'positive_weight')[node]
        
    def node_negative_strength(self, node):
        """Find the weighted sum of all negative node edges"""
        return self.graph.degree(weight = 'negative_weight')[node]

    def node_positive_strength_by_community(self, node):
        """Find the weighted sum of the positive edges from a node to each community
        Returns
        -------
        comm_positive_strengths : list
            list holding the strength of a node to each community
        """
        comm_positive_strengths = [0] * len(self.communities)
        for neighbor, data in self.graph[node].items():
            if neighbor == node:
                continue
            tmpcomm = self.get_node_community(neighbor)
            comm_positive_strengths[tmpcomm] += data.get('positive_weight', 1)
        return comm_positive_strengths

    def node_negative_strength_by_community(self, node):
        """Find the weighted sum of the negative edges from a node to each community
        Returns
        -------
        comm_negative_strengths : list
            list holding the strength of a node to each community
        """
        comm_negative_strengths = [0] * len(self.communities)
        for neighbor, data in self.graph[node].items():
            if neighbor == node:
                continue
            tmpcomm = self.get_node_community(neighbor)
            comm_negative_strengths[tmpcomm] += data.get('negative_weight', 1)
        return comm_negative_strengths
     
    def positive_strength_by_community(self):
        """ Weighted sum of all positive edges (both within and between communities)
        for each community
        Returns
        -------
        positive_strengths : list
            list is size of total number of communities"""
        comm = self.communities
        positive_strengths = [0] * len(comm)
        all_positive_strengths = self.graph.degree(weight = 'positive_weight')
        for node, strength in all_positive_strengths.items():
            node_comm = self.get_node_community(node)
            positive_strengths[node_comm] += strength
        return positive_strengths
        
    def negative_strength_by_community(self):
        """Weighted sum of all negative edges (both within and between communities)
           for each community
        Returns
        -------
        negative_strengths : list
            list is size of total number of communities"""
        comm = self.communities
        negative_strengths = [0] * len(comm)
        all_negative_strengths = self.graph.degree(weight = 'negative_weight')
        for node, strength in all_negative_strengths.items():
            node_comm = self.get_node_community(node)
            negative_strengths[node_comm] += strength
        return negative_strengths

    def positive_strength_within_community(self):
        """Weighted sum of the positive edges strictly inside each community
           including self loops"""
        comm = self.communities
        positive_strengths = [0] * len(comm)
        comm = self.communities
        for val, nodeset in enumerate(comm):
            for node in nodeset:
                nodes_within = set([x for x in self.graph[node].keys() \
                    if x in nodeset])
                if len(nodes_within) < 1:
                    continue
                if node in nodes_within:
                    positive_strengths[val] += self.graph[node][node]['positive_weight']
                    nodes_within.remove(node)
                positive_strengths[val] += np.sum(self.graph[node][x]['positive_weight'] / 2. \
                    for x in nodes_within)
        return positive_strengths
        
    def negative_strength_within_community(self):
        """Weighted sum of the negative edges strictly inside each community
           including self loops"""
        comm = self.communities
        negative_strengths = [0] * len(comm)
        comm = self.communities
        for val, nodeset in enumerate(comm):
            for node in nodeset:
                nodes_within = set([x for x in self.graph[node].keys() \
                    if x in nodeset])
                if len(nodes_within) < 1:
                    continue
                if node in nodes_within:
                    negative_strengths[val] += self.graph[node][node]['negative_weight']
                    nodes_within.remove(node)
                negative_strengths[val] += np.sum(self.graph[node][x]['negative_weight']/ 2. \
                    for x in nodes_within)
        return negative_strengths

    def modularity(self, qtype='pos'):
        """Calculates the proportion of within community edges compared to between community
           edges for all nodes in graph with given partition.

        Parameters
        ----------
        partition : WeightedPartition
        qtype : str
            type of normalization (see [2] for details)
            'pos' (Q_+]) 
            'neg' (Q_-)
            'smp' (Q_simple)
            'sta' (Q_*)
            'gja' (Q_GJA)

        Returns
        -------
        modularity : float
            value reflecting the relation of within community connections
            to across community connections

        References
        ----------
        .. [1] M. Newman, "Fast algorithm for detecting community structure
            in networks", Physical Review E vol. 69(6), 2004.
        .. [2] M. Rubinov and O. Sporns, "Weight-conserving characterization of
            complex functional brain networks", NeuroImage, vol. 56(4), 2011.
        .. [3] S. Gomez, P. Jensen, A. Arenas, "Analysis of community structure
            in networks of correlated data". Physical Review E vol. 80(1), 2009.
        """
        if self.graph.is_directed():
            raise TypeError('Only valid on non directed graphs')
        pos_m2 = self.total_positive_strength
        neg_m2 = self.total_negative_strength
        pos_win_community = np.array(self.positive_strength_within_community())
        neg_win_community = np.array(self.negative_strength_within_community())
        pos_tot_community = np.array(self.positive_strength_by_community())
        neg_tot_community = np.array(self.negative_strength_by_community())
        if qtype == 'pos':
            return np.sum(pos_win_community / pos_m2\
                              - (pos_tot_community / (2 * pos_m2)) ** 2)
        elif qtype == 'neg':
            return -(np.sum(neg_win_community / neg_m2\
                              - (neg_tot_community / (2 * neg_m2)) ** 2))
        elif qtype == 'smp':
            q_pos = self.modularity(qtype='pos')
            q_neg = self.modularity(qtype='neg')
            return q_pos + q_neg
        elif qtype == 'sta':
            q_pos = self.modularity(qtype='pos')
            q_neg = np.sum(neg_win_community / (pos_m2 + neg_m2)\
                               - (neg_tot_community / (2 * (pos_m2 + neg_m2))) ** 2)
            return q_pos - q_neg    
        elif qtype == 'gja':
            q_pos = np.sum(pos_win_community / (pos_m2 + neg_m2)\
                               - (pos_tot_community / (2 * pos_m2 + 2 * neg_m2)) ** 2)
            q_neg = np.sum(neg_win_community / (pos_m2 + neg_m2)\
                               - (neg_tot_community / (2 *  pos_m2 + 2 * neg_m2)) ** 2)
            return q_pos - q_neg


class LouvainCommunityDetection(object):
    """Uses the Louvain Community Detection algorithm to detect
       communities in networks

    Parameters
    ----------
    graph : netwrokx Graph object
    qtype : str
        type of normalization (see [2] for details)
        'pos' (Q_+]) 
        'neg' (Q_-)
        'smp' (Q_simple)
        'sta' (Q_*)
        'gja' (Q_GJA)
    communities : list of sets, optional
        initial identified communties
    minthr : float, optional
        minimum threshold value for change in modularity
        default(0.0000001)

    Methods
    -------
    run()
        run the algorithm to find partitions at multiple levels

    Examples
    --------
    >>> louvain = LouvainCommunityDetection(graph)
    >>> partitions = louvain.run()
    >>> ## best partition
    >>> partitions[-1].modularity()

    References
    ----------
    .. [1] VD Blondel, JL Guillaume, R Lambiotte, E Lefebvre, "Fast
        unfolding of communities in large networks", Journal of Statistical
        Mechanics: Theory and Experiment vol.10, P10008  2008.
    .. [2] M. Rubinov and O. Sporns, "Weight-conserving characterization of
        complex functional brain networks", NeuroImage, vol. 56(4), 2011.
    """

    def __init__(self, graph, qtype, communities=None, minthr=0.0000001):
        """initialize the algorithm with a graph and (optional) initial
           community partition , use minthr to provide a stopping limit
           for the algorith (based on change in modularity)"""
        self.graph = graph
        self.qtype = qtype
        self.initial_communities = communities
        self.minthr = minthr

    def run(self):
        """Run the algorithm to find partitions in graph

        Returns
        -------
        partitions : list
            a list containing instances of a WeightedPartition with the
            community partition reflecting that level of the algorithm
            The last item in the list is the final partition
            The first item was the initial partition
        """
        dendogram = self._gen_dendogram()
        partitions = self._partitions_from_dendogram(dendogram)
        return [WeightedPartition(self.graph, part) for part in partitions]


    def _gen_dendogram(self):
        """Generate dendogram based on muti-levels of partitioning"""
        if type(self.graph) != nx.Graph :
            raise TypeError("Bad graph type, use only non directed graph")

        ## special case, when there is no link
        ## the best partition is everyone in its communities
        if self.graph.number_of_edges() == 0 :
            raise IOError('graph has no edges why do you want to partition?')

        current_graph = self.graph.copy()
        part = WeightedPartition(self.graph, self.initial_communities)
        # first pass
        mod = part.modularity(qtype=self.qtype)
        dendogram = list()
        new_part = self._one_level(part, self.minthr)
        new_mod = new_part.modularity(qtype=self.qtype)
        dendogram.append(new_part)
        mod = new_mod
        current_graph, _ = meta_graph(new_part)

        while True :
            partition = WeightedPartition(current_graph)
            newpart = self._one_level(partition, self.minthr)
            new_mod = newpart.modularity(qtype=self.qtype)
            if new_mod - mod < self.minthr :
                break

            dendogram.append(newpart)
            mod = new_mod
            current_graph,_ = meta_graph(newpart)
        return dendogram

    def _one_level(self, part, min_modularity=0.0000001):
        """Run one level of patitioning"""
        curr_mod = part.modularity(qtype=self.qtype)
        modified = True
        while modified:
            modified = False
            all_nodes = [x for x in part.graph.nodes()]
            np.random.shuffle(all_nodes)
            for node in all_nodes:
                node_comm = part.get_node_community(node)
                delta_mod = self._calc_delta_modularity(node, part)
                ## print node, delta_mod
                if delta_mod.max() <= 0.0:
                    # no increase by moving this node
                    continue
                best_comm = delta_mod.argmax()
                if not best_comm == node_comm:
                    new_part = self._move_node(part, node, best_comm)
                    part = new_part
                    modified = True
            new_mod = part.modularity(qtype=self.qtype)
            change_in_modularity = new_mod - curr_mod
            if change_in_modularity < min_modularity:
                return part
        return part

    def _calc_delta_modularity(self, node, part, arg=None):
        """Calculate the increase(s) in modularity if node is moved to other
           communities
        deltamod = inC - totc * ki / total_weight"""
        pos_node_strength = part.node_positive_strength(node)
        neg_node_strength = part.node_negative_strength(node)
        pos_node_community_strength = np.array(part.node_positive_strength_by_community(node))
        neg_node_community_strength = np.array(part.node_negative_strength_by_community(node))
        pos_totc = np.array(self._communities_nodes_alledgesw(part, node, 'pos'))
        neg_totc = np.array(self._communities_nodes_alledgesw(part, node, 'neg'))
        pos_m = part.total_positive_strength
        neg_m = part.total_negative_strength
        if self.qtype or arg == 'pos':
             return pos_node_community_strength - \
                 pos_totc * pos_node_strength / (pos_m * 2)
        elif self.qtype or arg == 'neg':
             return - (neg_node_community_strength - \
                 neg_totc * neg_node_strength / (neg_m * 2))
        elif self.qtype == 'smp':
            pos_delta_modularity = self._calc_delta_modularity(node, part, 'pos')
            neg_delta_modularity = self._calc_delta_modularity(node, part, 'neg')
            return pos_delta_modularity + neg_delta_modularity
        elif self.qtype == 'sta':
            pos_delta_modularity = self._calc_delta_modularity(node, part, 'pos')
            neg_delta_modularity = (neg_node_community_strength - \
                neg_totc * neg_node_strength) * (neg_m / (pos_m + neg_m) * 2)
            return pos_delta_modularity - neg_delta_modularity
        elif self.qtype == 'gja':
            pos_delta_modularity = pos_node_community_strength - \
                pos_totc * pos_node_strength / ((pos_m + neg_m_) * 2)
            neg_delta_modularity = neg_node_community_strength - \
                neg_totc * neg_node_strength / ((pos_m + neg_m_) * 2)
            return pos_delta_modularity - neg_delta_modularity

    @staticmethod
    def _communities_without_node(part, node):
        """Return a version of the partition with the node
           removed, may result in empty communities"""
        node_comm = part.get_node_community(node)
        newpart = copy.deepcopy(part.communities)
        newpart[node_comm].remove(node)
        return newpart

    def _communities_nodes_alledgesw(self, part, removed_node, sign):
        """Return the sum of all weighted edges to nodes in each
           community, once the removed_node is removed
           this refers to totc in Blondel paper"""
        comm_wo_node = self._communities_without_node(part, removed_node)
        strengths = [0] * len(comm_wo_node)
        if sign == 'pos':
            all_strengths = list(part.graph.degree(weight='positive_weight').values())
        elif sign == 'neg':
            all_strengths = list(part.graph.degree(weight='negative_weight').values())
        ## make a list of all nodes strengths        
        all_strengths = np.array(all_strengths)
        for val, nodeset in enumerate(comm_wo_node):
            node_index = np.array(list(nodeset)) #index of nodes in community
            #sum the strength of nodes in community
            if len(node_index)< 1:
                continue
            strengths[val] = np.sum(all_strengths[node_index])
        return strengths

    @staticmethod
    def _move_node(part, node, new_comm):
        """Generate a new partition with node put into new community
           designated by index (new_comm) into existing part.communities"""
        ## copy
        new_community = [x.copy() for x in part.communities]
        ## update
        curr_node_comm = part.get_node_community(node)
        ## remove
        new_community[curr_node_comm].remove(node)
        new_community[new_comm].add(node)
        # remove any empty sets
        new_community = [x for x in new_community if len(x) > 0]
        return WeightedPartition(part.graph, new_community)

    def _partitions_from_dendogram(self, dendo):
        """Return community partitions based on results in dendogram"""
        all_partitions = []
        init_part = dendo[0].communities
        all_partitions.append(init_part)
        for partition in dendo[1:]:
            init_part = self._combine(init_part, partition.communities)
            all_partitions.append(init_part)
        return all_partitions

    @staticmethod
    def _combine(prev, next):
        """Combine nodes in sets (prev) based on mapping defined by
           (next) (which now treats a previous communitity as a node)
           but maintains specification of all original nodes

        Parameters
        ----------
        prev : list of sets
            communities partition
        next : list of sets
            next level communities partition

        Examples
        --------
        >>> prev = [set([0,1,2]), set([3,4]), set([5,6])]
        >>> next = [set([0,1]), set([2])]
        >>> result = _combine(prev, next)
        [set([0, 1, 2, 3, 4]), set([5,6])]
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


def meta_graph(partition):
    """Create a new graph object based on input graph and partition

    Takes WeightedPartition object with specified communities and
        creates a new graph object where
        1. communities are now the nodes in the new graph
        2. the new edges are created based on the node to node connections (weights)
           from communities in the original graph, and weighted accordingly,
           (this includes self-loops)

    Returns
    -------
    metagraph : networkX graph
    mapping : dict
        dict showing the mapping from newnode -> original community nodes
    """
    metagraph = nx.Graph()
    # new nodes are communities
    newnodes = [val for val,_ in enumerate(partition.communities)]
    mapping = {val:nodes for val,nodes in enumerate(partition.communities)}
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






