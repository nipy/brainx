#Author: Maxwell Bertolero 

import numpy as np
from random import choice
import networkx as nx

def within_module_degree(graph, partition, weighted = False):
    '''
    Computes the within-module degree for each node (Guimera et al. 2005)

    ------
    Inputs
    ------
    graph = Networkx Graph, unweighted, undirected.
    partition = dictionary from modularity partition of graph using Louvain method

    ------
    Output
    ------
    Dictionary of the within-module degree of each node.

    '''
    new_part = {}
    for m,n in zip(partition.values(),partition.keys()):
        try:
            new_part[m].append(n)
        except KeyError:
            new_part[m] = [n]
    partition = new_part
    wd_dict = {}

    #loop through each module, look at nodes in modules
    for m in partition.keys():
        mod_list = partition[m]
        mod_wd_dict = {}
        #get within module degree of each node
        for source in mod_list:
            count = 0
            for target in mod_list:
                if (source,target) in graph.edges() or (target,source) in graph.edges():
                    if weighted == True:
                        count += graph.get_edge_data(source,target)['weight']
                        count += graph.get_edge_data(target,source)['weight'] # i assume this will only get one weighted edge.
                    else:
                        count += 1
            mod_wd_dict[source] = count
        # z-score
        all_mod_wd = mod_wd_dict.values()
        avg_mod_wd = float(sum(all_mod_wd) / len(all_mod_wd))
        std = np.std(all_mod_wd)
        #add to dictionary
        for source in mod_list:
            wd_dict[source] = (mod_wd_dict[source] - avg_mod_wd) / std
    return wd_dict


def participation_coefficient(graph, partition):
    '''
    Computes the participation coefficient for each node (Guimera et al. 2005).

    ------
    Inputs
    ------
    graph = Networkx graph
    partition = modularity partition of graph

    ------
    Output
    ------
    Dictionary of the participation coefficient for each node.

    '''
    #this is because the dictionary output of Louvain is "backwards"
    new_part = {}
    for m,n in zip(partition.values(),partition.keys()):
        try:
            new_part[m].append(n)
        except KeyError:
            new_part[m] = [n]
    partition = new_part
    pc_dict = {}
    all_nodes = set(graph.nodes())
    # loop through modules
    if weighted == False:
        for m in partition.keys():
            #set of nodes in modules
            mod_list = set(partition[m])
            #set of nodes outside that module
            between_mod_list = list(set.difference(all_nodes, mod_list))
            for source in mod_list:
                #degree of node
                degree = float(nx.degree(G=graph, nbunch=source))
                count = 0
                # between module degree
                for target in between_mod_list:
                    if (source,target) in graph.edges() or(source,target) in graph.edges():
                        count += 1
                bm_degree = float(count)
                if bm_degree == 0.0:
                    pc = 0.0
                else:
                    pc = 1 - ((float(bm_degree) / float(degree))**2)
                pc_dict[source] = pc
        return pc_dict
        #this is because the dictionary output of Louvain is "backwards"
    if weighted == True:
        for m in partition.keys():
            #set of nodes in modules
            mod_list = set(partition[m])
            #set of nodes outside that module
            between_mod_list = list(set.difference(all_nodes, mod_list))
            for source in mod_list:
                #degree of node
                degree = 0
                edges = G.edges([source],data=True)
                for edge in edges:
                    degree += graph.get_edge_data(edge[0],edge[1])['weight']
                count = 0
                # between module degree
                for target in between_mod_list:
                    if (source,target) in graph.edges() or(source,target) in graph.edges():
                        count += graph.get_edge_data(source,target)['weight']
                        count += graph.get_edge_data(target,source)['weight'] # i assume this will only get one weighted edge.
                bm_degree = float(count)
                if bm_degree == 0.0:
                    pc = 0.0
                else:
                    pc = 1 - ((float(bm_degree) / float(degree))**2)
                pc_dict[source] = pc
        return pc_dict