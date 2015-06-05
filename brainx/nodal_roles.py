#Author: Maxwell Bertolero, bertolero@berkeley.edu, bertolero@berkeley.edu

import numpy as np
from random import choice
import networkx as nx

def within_community_degree(weighted_partition, edgeless = np.nan, catch_edgeless_node=True):
    ''' Computes "within-module degree" (z-score) for each node (Guimera 2005, Nature)

    ------
    Parameters
    ------
    weighted_partition: Louvain Weighted Partition
        louvain = weighted_modularity.LouvainCommunityDetection(graph)
        weighted_partitions = louvain.run()
        weighted_partition = weighted_partition[0], where index is the partition level
    edgeless : int
        number to replace edgeless nodes with
        default = 0.0
    catch_edgeless_node: Boolean
        raise ValueError if node degree is zero
        default = True
        saves wcd of these nodes as edgeless variable if False

    ------
    Returns
    ------
    within_community_degree: dict
        Dictionary of the within community degree of each node.

    '''
    wc_dict = {}
    for c, community in enumerate(weighted_partition.communities):
        community_degrees = []
        for node in community: #get average within-community-degree
            node_degree = weighted_partition.node_degree(node)
            if node_degree == 0.0: #catch edgeless nodes, this shouldn't count towards avg wcd
                if catch_edgeless_node:
                    raise ValueError("Node {} is edgeless".format(node))
                continue
            community_degrees.append(weighted_partition.node_degree_by_community(node)[c])
        std = np.std(community_degrees) # std of community's degrees
        mean = np.mean(community_degrees) # mean of community's degrees
        for node in community: #get node's within_community-degree z-score
            if weighted_partition.node_degree(node) == 0:
                wc_dict[node] = edgeless
                continue
            within_community_degree = weighted_partition.node_degree_by_community(node)[c]
            if std == 0.0: #so we don't divide by 0
                wc_dict[node] = (float(within_community_degree) - float(mean)) #z_score
                continue
            wc_dict[node] = ((float(within_community_degree) - float(mean)) / std) #z_score
    return wc_dict

def participation_coefficient(weighted_partition, edgeless =np.nan, catch_edgeless_node=True):
    '''
    Computes the participation coefficient for each node (Guimera 2005, Nature)

    ------
    Parameters
    ------
    weighted_partition: Louvain Weighted Partition
        louvain = weighted_modularity.LouvainCommunityDetection(graph)
        weighted_partitions = louvain.run()
        weighted_partition = weighted_partition[0], where index is the partition level
    catch_edgeless_node: Boolean
        raise ValueError if node degree is zero
        default = True

    ------
    Returns
    ------
    participation_coefficient: dict
        Dictionary of the participation coefficient of each node.
    '''
    pc_dict = {}
    for node in weighted_partition.graph:
        node_degree = weighted_partition.node_degree(node)
        if node_degree == 0.0: 
            if catch_edgeless_node:
                raise ValueError("Node {} is edgeless".format(node))
            pc_dict[node] = edgeless
            continue    
        pc = 0.0
        for community_degree in weighted_partition.node_degree_by_community(node):
            pc = pc + ((float(community_degree)/float(node_degree))**2)
        pc = 1-pc
        pc_dict[node] = pc
    return pc_dict
