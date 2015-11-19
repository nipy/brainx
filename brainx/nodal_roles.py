#Author: Maxwell Bertolero, bertolero@berkeley.edu, bertolero@berkeley.edu
#Author: Katelyn Arnemann, klarnemann@berkeley.edu

import numpy as np
from random import choice
import networkx as nx


def within_community_strength(weighted_partition, calc_type, edgeless = np.nan, catch_edgeless_node=True):
    ''' Computes "within-module strength" (i.e. weighted degree) z-score for each node 

    See:
    Guimera, Nature, 2005
    Rubinov & Sporns, NeuroImage, 2011

    ------
    Parameters
    ------
    weighted_partition: Louvain Weighted Partition
        louvain = weighted_modularity.LouvainCommunityDetection(graph)
        weighted_partitions = louvain.run()
        weighted_partition = weighted_partition[0], where index is the partition level
    calc_type : str
        'pos'
        'neg'
        'smp'
        'sta'
        'gja'
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
    within_community_strength : dict
        Dictionary of the within community strength of each node.

    '''
    wc_dict = {}
    for c, community in enumerate(weighted_partition.communities):
        community_strengths = []
        community_wc_dict = {}
        for node in community: #get average within-community strength (i.e. weighted degree)
            if calc_type == 'pos':
                node_strength = weighted_partition.node_positive_strength(node)
                community_strengths.append(weighted_partition.node_positive_strength_by_community(node)[c])
            elif calc_type == 'neg':
                node_strength = weighted_partition.node_negative_strength(node)
                community_strengths.append(weighted_partition.node_negative_strength_by_community(node)[c])
            elif calc_type == 'smp':
                pos_wc_dict =  within_community_strength(weighted_partition, calc_type='pos')
                neg_wc_dict =  within_community_strength(weighted_partition, calc_type='neg')
                keys = pos_wc_dict.keys()
                vals = pos_wc_dict.values() - neg_wc_dict.values()
                return dict(zip(keys,vals))
            elif calc_type == 'sta':
                pos_wc_dict =  within_community_strength(weighted_partition, calc_type='pos')
                neg_wc_dict =  within_community_strength(weighted_partition, calc_type='neg')
                keys = pos_wc_dict.keys()
                pos_m2 = weighted_partition.total_positive_strength
                neg_m2 = weighted_partition.total_negative_strength
                vals = pos_wc_dict.values - (neg_wc_dict.values * (neg_m2 / pos_m2 + neg_m2))
                return dict(zip(keys,vals))                
            elif calc_type == 'gja':
                pos_wc_dict =  within_community_strength(weighted_partition, calc_type='pos')
                neg_wc_dict =  within_community_strength(weighted_partition, calc_type='neg')
                keys = pos_wc_dict.keys()
                pos_m2 = weighted_partition.total_positive_strength
                neg_m2 = weighted_partition.total_negative_strength
                vals = (pos_wc_dict.values * (pos_m2 / (pos_m2 + neg_m2))) \
                    - (neg_wc_dict.values * (neg_m2 / (pos_m2 + neg_m2)))
                return dict(zip(keys,vals))   
            else:
                raise ValueError('%s not supported.' % (calc_type))
            if node_strength == 0.0: #catch edgeless nodes, this shouldn't count towards avg wcd
                if catch_edgeless_node:
                    wc_dict[node] = edgeless
                    raise ValueError("Node {} is edgeless".format(node))
                continue
        std = np.std(community_strengths) #std of community's strengths
        mean = np.mean(community_strengths) #mean of community's strengths
        community_strengths = np.array(community_strengths)
        if std == 0.0: #so we don't divide by 0
            wc = (community_strengths - float(mean)) #z_score
            community_wc_dict = dict(zip(community, wc))
            wc_dict.update(community_wc_dict)
            continue
        wc = ((community_strengths - float(mean)) / std) #z_score
        community_wc_dict = dict(zip(community, wc))
        wc_dict.update(community_wc_dict)
    return wc_dict


def participation_coefficient(weighted_partition, calc_type, edgeless =np.nan, catch_edgeless_node=True):
    '''
    Computes the participation coefficient for each node 
    
    See:
    Guimera, Nature, 2005
    Rubinov & Sporns, NeuroImage, 2011

    ------
    Parameters
    ------
    weighted_partition: Louvain Weighted Partition
        louvain = weighted_modularity.LouvainCommunityDetection(graph)
        weighted_partitions = louvain.run()
        weighted_partition = weighted_partition[0], where index is the partition level
    calc_type : str
        'pos'
        'neg'
        'smp'
        'sta'
        'gja'
    catch_edgeless_node: Boolean
        raise ValueError if node strength is zero
        default = True

    ------
    Returns
    ------
    participation_coefficient : dict
        Dictionary of the participation coefficient of each node.
    '''
    pc_dict = {}
    for node in weighted_partition.graph:
        if calc_type == 'pos':
            node_strength = weighted_partition.node_positive_strength(node)
            node_community_strengths = weighted_partition.node_positive_strength_by_community(node)
        elif calc_type == 'neg':
            node_strength = weighted_partition.node_negative_strength(node)
            node_community_strengths = weighted_partition.node_negative_strength_by_community(node)
        elif calc_type == 'smp':
            pos_pc = participation_coefficient(weighted_partition, calc_type='pos')
            neg_pc = participation_coefficient(weighted_partition, calc_type='neg')
            nodes = pos_pc.keys()
            pc = pos_pc.values() - neg_pc.values()
            return dict(zip(nodes, pc))
        elif calc_type == 'sta':
            pos_m2 = weighted_partition.total_positive_strength
            neg_m2 = weighted_partition.total_negative_strength
            pos_pc = participation_coefficient(weighted_partition, calc_type='pos')
            neg_pc = participation_coefficient(weighted_partition, calc_type='neg')
            nodes = pos_pc.keys()
            pc = pos_pc.values() - ((neg_m2 / (pos_m2 + neg_m2)) * neg_pc.values())
            return dict(zip(nodes, pc))
        elif calc_type == 'gja':
            pos_m2 = weighted_partition.total_positive_strength
            neg_m2 = weighted_partition.total_negative_strength
            pos_pc = participation_coefficient(weighted_partition, calc_type='pos')
            neg_pc = participation_coefficient(weighted_partition, calc_type='neg')
            nodes = pos_pc.keys()
            pc = ((pos_m2 / (pos_m2 + neg_m2)) * pos_pc.values()) \
                - ((neg_m2 / (pos_m2 + neg_m2)) * neg_pc.values())
            return dict(zip(nodes, pc))
        else:
            raise ValueError('%s not supported.' % (calc_type))
        if node_strength == 0.0: 
            if catch_edgeless_node:
                raise ValueError("Node {} is edgeless".format(node))
            pc_dict[node] = edgeless
            continue    
        pc = 0.0
        for community_strength in node_community_strengths:
            pc += ((float(community_strength) / float(node_strength)) ** 2)
        pc = 1 - pc
        pc_dict[node] = pc
    return pc_dict


def connection_strength(weighted_partition, calc_type):
    '''
    Computes the connection strength for each node.

    See:
    Guimera, Nature, 2005
    Rubinov & Sporns, NeuroImage, 2011

    ------
    Parameters
    ------
    weighted_partition: Louvain Weighted Partition
        louvain = weighted_modularity.LouvainCommunityDetection(graph)
        weighted_partitions = louvain.run()
        weighted_partition = weighted_partition[0], where index is the partition level
    calc_type : str
        'pos'
        'neg'
        'smp'
        'sta'
        'gja'

    ------
    Returns
    ------
    connection_strength : dict
        Dictionary of the connection strength of each node.
    '''
    nodes = weighted_partition.graph.nodes()
    n = len(nodes)
    if calc_type == 'pos':
        strengths = []
        for node in nodes:
            strengths.append((1/n-1)*node_positive_strength(node))
        return dict(zip(nodes, strengths))
    elif calc_type == 'neg':
        strengths = []
        for node in nodes:
            strengths.append((1/n-1)*node_negative_strength(node))
        return dict(zip(nodes, strengths))
    elif calc_type == 'smp':
        pos_strengths = connection_strength(weighted_partition, calc_type='pos')
        neg_strengths = connection_strength(weighted_partition, calc_type='neg')
        return dict(zip(nodes, pos_strengths - neg_strengths))
    elif calc_type == 'sta':
        pos_strengths = connection_strength(weighted_partition, calc_type='pos')
        neg_strengths = connection_strength(weighted_partition, calc_type='neg')
        pos_m2 = weighted_partition.total_positive_strength
        neg_m2 = weighted_partition.total_negative_strength
        strengths = pos_strengths - ((neg_m2 / (pos_m2 + neg_m2)) * neg_strenghts)
        return dict(zip(nodes, strenghts))
    elif calc_type == 'gja':
        pos_strengths = connection_strength(weighted_partition, calc_type='pos')
        neg_strengths = connection_strength(weighted_partition, calc_type='neg')
        pos_m2 = weighted_partition.total_positive_strength
        neg_m2 = weighted_partition.total_negative_strength
        strengths = ((pos_m2 / (pos_m2 + neg_m2)) * pos_strengths) \
            - ((neg_m2 / (pos_m2 + neg_m2)) * neg_strenghts)
        return dict(zip(nodes, strenghts))
    else:
        raise ValueError('%s not supported.' % (calc_type))


def connection_diversity(weighted_partition, calc_type):
    '''
    Computes the connection diversity for each node.

    ------
    Parameters
    ------
    weighted_partition: Louvain Weighted Partition
        louvain = weighted_modularity.LouvainCommunityDetection(graph)
        weighted_partitions = louvain.run()
        weighted_partition = weighted_partition[0], where index is the partition level
    calc_type : str
        'pos'
        'neg'
        'smp'
        'sta'
        'gja'

    ------
    Returns
    ------
    connection_diversity : dict
        Dictionary of the connection diversity of each node.
    '''
    nodes = weighted_partition.graph.nodes()
    m = len(weighted_partition.communities)
    if calc_type == 'pos':
        diversity = []
        for node in nodes:
            node_strength = weighted_partition.node_positive_strength(node)
            node_community_strength = weighted_partition.node_positive_strength_by_community(node)
            node_diversity = -(1 / np.log(m)) * \
                sum(node_community_strength * (np.log(node_community_strength)))
            diversity.append(node_diversity)
        return dict(zip(nodes, diversity))
    elif calc_type == 'neg':
        diversity = []
        for node in nodes:
            node_strength = weighted_partition.node_negative_strength(node)
            node_community_strength = weighted_partition.node_negative_strength_by_community(node)
            node_diversity = -(1 / np.log(m)) * \
                sum(node_community_strength * (np.log(node_community_strength)))
            diversity.append(node_diversity)
        return dict(zip(nodes, diversity))
    elif calc_type == 'smp':
        pos_diversity = connection_diversity(weighted_partition, 'pos')
        neg_diversity = connection_diversity(weighted_partition, 'neg')
        diversity = pos_diversity.values() -  neg_diversity.values()
        return dict(zip(nodes, diversity))
    elif calc_type == 'sta':
        pos_m2 = weighted_partition.total_positive_strength
        neg_m2 = weighted_partition.total_negative_strength
        pos_diversity = connection_diversity(weighted_partition, 'pos')
        neg_diversity = connection_diversity(weighted_partition, 'neg')
        diversity = pos_diversity.values() - ((neg_m2 / pos_m2 + neg_m2) * neg_diversity.values())
        return dict(zip(nodes, diversity))
    elif calc_type == 'gja':
        pos_m2 = weighted_partition.total_positive_strength
        neg_m2 = weighted_partition.total_negative_strength
        pos_diversity = connection_diversity(weighted_partition, 'pos')
        neg_diversity = connection_diversity(weighted_partition, 'neg')
        diversity = ((pos_m2 / pos_m2 + neg_m2) * pos_diversity.values() \
            - ((neg_m2 / pos_m2 + neg_m2) * neg_diversity.values())
        return dict(zip(nodes, diversity))
    else:
        raise ValueError('%s not supported.' % (calc_type))
    
