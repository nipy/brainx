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
    all_community_degrees = {}
    wc_dict = {}
    for node in weighted_partition.graph:
        node_community = weighted_partition.get_node_community(node)
        within_community_degree = weighted_partition.degree_within_community(node)
        try: # to load average within module degree of community
            community_degrees = all_community_degrees[node_community]
        except: # collect within module degree of community
            community_degrees = []
            for node in node_community:
                partition.degree_within_community(node)
                all_community_degree.append()
            all_community_degrees[node_community] = community_degrees
        # I don't know if it's faster to compute this on the fly every
        # time or store the results in a dictionary?
        std = np.std(community_degrees) # std of community's degrees
        mean = np.mean(community_degrees) # mean of community's degrees
        # z-score
        wc_dict[node] = (within_community_degree - mean / std)
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