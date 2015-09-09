#Author: Maxwell Bertolero, bertolero@berkeley.edu, bertolero@berkeley.edu
#Author: Katelyn Arnemann, klarnemann@berkeley.edu


import numpy as np
from random import choice
import networkx as nx

def within_community_strength(weighted_partition, calc_type='pos', edgeless = np.nan, catch_edgeless_node=True):
    ''' Computes "within-module strength" (i.e. weighted degree) z-score for each node 
    (Guimera 2005, Nature)

    ------
    Parameters
    ------
    weighted_partition: Louvain Weighted Partition
        louvain = weighted_modularity.LouvainCommunityDetection(graph)
        weighted_partitions = louvain.run()
        weighted_partition = weighted_partition[0], where index is the partition level
    calc_type : str
        'all' = calculate within community strength on all weights
        'pos' = calculate within community strength on positive weights only
        'neg' = calculate within community strength on negative weights only
    edgeless : int
        number to replace edgeless nodes with
        default = 0.0
    catch_edgeless_node: Boolean
        raise ValueError if node strength is zero
        default = True
        saves wcd of these nodes as edgeless variable if False

    ------
    Returns
    ------
    within_community_strength: dict
        Dictionary of the within community strength of each node.

    '''
    wc_dict = {}
    for c, community in enumerate(weighted_partition.communities):
        community_strengths = []
        community_wc_dict = {}
        for node in community: #get average within-community strength (i.e. weighted degree)
            if calc_type == 'all':
                node_strength = weighted_partition.node_strengths(node)
                community_strengths.append(weighted_partition.node_strength_by_community(node)[c])
            elif calc_type == 'pos':
                node_strength = weighted_partition.node_positive_strengths(node)
                community_strengths.append(weighted_partition.node_positive_strength_by_community(node)[c])
            elif calc_type == 'neg':
                node_strength = weighted_partition.node_negative_strengths(node)
                community_strengths.append(weighted_partition.node_negative_strength_by_community(node)[c])
            else:
                raise ValueError('%s not supported; only all, pos, and neg options are supported.' % (calc_type))
            if node_strength == 0.0: #catch edgeless nodes, this shouldn't count towards avg wcd
                if catch_edgeless_node:
                    wc_dict[node] = edgeless
                    raise ValueError("Node {} is edgeless".format(node))
                continue
        std = np.std(community_strengths) # std of community's strengths
        mean = np.mean(community_strengths) # mean of community's strengths
        if std == 0.0: #so we don't divide by 0
            wc = ((float(community_strengths) - float(mean)) #z_score
            community_wc_dict = dict(zip(community, wc))
            wc_dict.update(community_wc_dict)
            continue
        wc = ((float(community_strengths) - float(mean)) / std) #z_score
        community_wc_dict = dict(zip(community, wc))
        wc_dict.update(community_wc_dict)
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
        raise ValueError if node strength is zero
        default = True

    ------
    Returns
    ------
    participation_coefficient: dict
        Dictionary of the participation coefficient of each node.
    '''
    pc_dict = {}
    for node in weighted_partition.graph:
        node_strength = weighted_partition.node_strength(node)
        if node_strength == 0.0: 
            if catch_edgeless_node:
                raise ValueError("Node {} is edgeless".format(node))
            pc_dict[node] = edgeless
            continue    
        pc = 0.0
        for community_strength in weighted_partition.node_strength_by_community(node):
            pc = pc + ((float(community_strength)/float(node_strength))**2)
        pc = 1-pc
        pc_dict[node] = pc
    return pc_dict
