#Author: Maxwell Bertolero 

import numpy as np
from random import choice
import networkx as nx

def within_community_degree(weighted_partition):
    '''
    Computes the "within-module degree" (z-score) for each node (Guimera et al. 2005)

    ------
    Inputs
    ------
    partition = dictionary from modularity partition of graph using Louvain method

    ------
    Output
    ------
    Dictionary of the within-module degree of each node.

    '''
    wc_dict = {}
    for c, community in enumerate(weighted_partition.communities):
        community_degrees = []
        # community = list(community)
        for node in community: #get average within-community-degree
            community_degrees.append(weighted_partition.node_degree_by_community(node)[c])
        for node in community:
            within_community_degree = weighted_partition.node_degree_by_community(node)[c]
            std = np.std(community_degrees) # std of community's degrees
            mean = np.mean(community_degrees) # mean of community's degrees
            wc_dict[node] = (within_community_degree - mean / std) #zscore
    return wc_dict

def participation_coefficient(weighted_partition, catch_edgeless_node=True):
    '''
    Computes the participation coefficient for each node (Guimera et al. 2005).

    ------
    Inputs
    ------
    partition = modularity partition of graph

    ------
    Output
    ------
    Dictionary of the participation coefficient for each node.

    '''
    pc_dict = {}
    graph = weighted_partition.graph
    for node in graph:
        node_degree = weighted_partition.node_degree(node)
        if node_degree == 0.0: 
            if catch_edgeless_node:
                raise ValueError("Node {} is edgeless".format(node))
            pc_dict[node] = 0.0
            continue    
        deg_per_comm = weighted_partition.node_degree_by_community(node)
        node_comm = weighted_partition.get_node_community(node)
        deg_per_comm.pop(node_comm) 
        bc_degree = sum(deg_per_comm) #between community degree
        if bc_degree == 0.0:
            pc_dict[node] = 0.0
            continue
        pc = 1 - ((float(bc_degree) / float(node_degree))**2)
        pc_dict[node] = pc
    return pc_dict