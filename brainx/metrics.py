"""Compute various useful metrics.
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import networkx as nx
import numpy as np
from scipy import sparse

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def inter_node_distances(graph):
    """Compute the shortest path lengths between all nodes in graph.

    This performs the same operation as NetworkX's
    all_pairs_shortest_path_lengths with two exceptions: Here, self
    paths are excluded from the dictionary returned, and the distance
    between disconnected nodes is set to infinity.  The latter
    difference is consistent with the Brain Connectivity Toolbox for
    Matlab.

    Parameters
    ----------
    graph: networkx Graph
        An undirected graph.

    Returns
    -------
    lengths: dictionary
        Dictionary of shortest path lengths keyed by source and target.

    """
    lengths = nx.all_pairs_shortest_path_length(graph)
    node_labels = sorted(lengths)
    for src in node_labels:
        lengths[src].pop(src)
        for targ in node_labels:
            if src != targ:
                try:
                    lengths[src][targ]
                except KeyError:
                    lengths[src][targ] = np.inf
    return lengths


def compute_sigma(arr,clustarr,lparr):
    """ Function for computing sigma given a graph array arr and clust and lp
    arrays from a pseudorandom graph for a particular block b."""

    gc = arr['clust']#np.squeeze(arr['clust'])
    glp = arr['lp']#np.squeeze(arr['lp'])
    out = (gc/clustarr)/(glp/lparr)

        
    return out


def nodal_pathlengths(graph):
    """Compute mean path length for each node.

    Parameters
    ----------
    graph: networkx Graph
        An undirected graph.

    Returns
    -------
    nodal_means: numpy array
        An array with each node's mean shortest path length to all other
        nodes.  The array is in ascending order of node labels.

    Notes
    -----
    Per the Brain Connectivity Toolbox for Matlab, the distance between
    one node and another that cannot be reached from it is set to
    infinity.

    """
    lengths = inter_node_distances(graph)
    nodal_means = [np.mean(list(lengths[src].values())) for src in sorted(lengths)]
    return np.array(nodal_means)


def assert_no_selfloops(graph):
    """Raise an error if the graph graph has any selfloops.
    """
    if graph.nodes_with_selfloops():
        raise ValueError("input graph can not have selfloops")


def path_lengths(graph):
    """Compute array of all shortest path lengths for the given graph.

    The length of the output array is the number of unique pairs of nodes that
    have a connecting path, so in general it is not known in advance.

    This assumes the graph is undirected, as for any pair of reachable nodes,
    once we've seen the pair we do not keep the path length value for the
    inverse path.
    
    Parameters
    ----------
    graph : an undirected graph object.
    """

    assert_no_selfloops(graph)
    
    length = nx.all_pairs_shortest_path_length(graph)
    paths = []
    seen = set()
    for src,targets in length.items():
        seen.add(src)
        neigh = set(targets.keys()) - seen
        paths.extend(targets[targ] for targ in neigh)
    
    
    return np.array(paths) 


#@profile
def path_lengthsSPARSE(graph):
    """Compute array of all shortest path lengths for the given graph.

    XXX - implementation using scipy.sparse.  This might be faster for very
    sparse graphs, but so far for our cases the overhead of handling the sparse
    matrices doesn't seem to be worth it.  We're leaving it in for now, in case
    we revisit this later and it proves useful.

    The length of the output array is the number of unique pairs of nodes that
    have a connecting path, so in general it is not known in advance.

    This assumes the graph is undirected, as for any pair of reachable nodes,
    once we've seen the pair we do not keep the path length value for the
    inverse path.
    
    Parameters
    ----------
    graph : an undirected graph object.
    """

    assert_no_selfloops(graph)
    
    length = nx.all_pairs_shortest_path_length(graph)

    nnod = graph.number_of_nodes()
    paths_mat = sparse.dok_matrix((nnod,nnod))
    
    for src,targets in length.items():
        for targ,val in targets.items():
            paths_mat[src,targ] = val

    return sparse.triu(paths_mat,1).data


def glob_efficiency(graph):
    """Compute array of global efficiency for the given graph.

    Global efficiency: returns a list of the inverse path length matrix
    across all nodes The mean of this value is equal to the global efficiency
    of the network."""
    
    return 1.0/path_lengths(graph)


def nodal_efficiency(graph):
    """Return array with nodal efficiency for each node in graph.

    See Achard and Bullmore (2007, PLoS Comput Biol) for the definition
    of nodal efficiency.

    Parameters
    ----------
    graph: networkx Graph
        An undirected graph.

    Returns
    -------
    nodal_efficiencies: numpy array
        An array with the nodal efficiency for each node in graph, in
        the order specified by node_labels.  The array is in ascending
        order of node labels.

    Notes
    -----
    Per the Brain Connectivity Toolbox for Matlab, the distance between
    one node and another that cannot be reached from it is set to
    infinity.

    """
    lengths = inter_node_distances(graph)
    nodal_efficiencies = np.zeros(len(lengths), dtype=float)
    for src in sorted(lengths):
        inverse_paths = [1.0 / val for val in lengths[src].values()]
        nodal_efficiencies[src] = np.mean(inverse_paths)
    return nodal_efficiencies


def local_efficiency(graph):
    """Compute array of global efficiency for the given grap.h

    Local efficiency: returns a list of paths that represent the nodal
    efficiencies across all nodes with their direct neighbors"""

    nodepaths=[]
    length=nx.all_pairs_shortest_path_length(graph)
    for n in graph.nodes():
        nneighb= nx.neighbors(graph,n)
        
        paths=[]
        for src,targets in length.items():
            for targ,val in targets.items():
                val=float(val)
                if src==targ:
                    continue
                if src in nneighb and targ in nneighb:
                    
                    paths.append(1/val)
        
        p=np.array(paths)
        psize=np.size(p)
        if (psize==0):
            p=np.array(0)
            
        nodepaths.append(p.mean())
    
    return np.array(nodepaths)


def local_efficiency(graph):
    """Compute array of local efficiency for the given graph.

    Local efficiency: returns a list of paths that represent the nodal
    efficiencies across all nodes with their direct neighbors"""

    assert_no_selfloops(graph)

    nodepaths = []
    length = nx.all_pairs_shortest_path_length(graph)
    for n in graph:
        nneighb = set(nx.neighbors(graph,n))

        paths = []
        for nei in nneighb:
            other_neighbors = nneighb - set([nei])
            nei_len = length[nei]
            paths.extend( [nei_len[o] for o in other_neighbors] )

        if paths:
            p = 1.0 / np.array(paths,float)
            nodepaths.append(p.mean())
        else:
            nodepaths.append(0.0)
                
    return np.array(nodepaths)


def dynamical_importance(graph):
    """Compute dynamical importance for graph.

    Ref: Restrepo, Ott, Hunt. Phys. Rev. Lett. 97, 094102 (2006)
    """
    # spectrum of the original graph
    eigvals = nx.adjacency_spectrum(graph)
    lambda0 = eigvals[0]
    # Now, loop over all nodes in graph, and for each, make a copy of graph, remove
    # that node, and compute the change in lambda.
    nnod = graph.number_of_nodes()
    dyimp = np.empty(nnod,float)
    for n in range(nnod):
        gn = graph.copy()
        gn.remove_node(n)
        lambda_n = nx.adjacency_spectrum(gn)[0]
        dyimp[n] = lambda0 - lambda_n
    # Final normalization
    dyimp /= lambda0
    return dyimp


def weighted_degree(graph):
    """Return an array of degrees that takes weights into account.

    For unweighted graphs, this is the same as the normal degree() method
    (though we return an array instead of a list).
    """
    amat = nx.adj_matrix(graph).A  # get a normal array out of it
    return abs(amat).sum(0)  # weights are sums across rows


def graph_summary(graph):
    """Compute a set of statistics summarizing the structure of a graph.
    
    Parameters
    ----------
    graph : a graph object.

    threshold : float, optional

    Returns
    -------
      Mean values for: lp, clust, glob_eff, loc_eff, in a dict.
    """
    
    # Average path length
    lp = path_lengths(graph)
    clust = np.array(list(nx.clustering(graph).values()))
    glob_eff = glob_efficiency(graph)
    loc_eff = local_efficiency(graph)
    
    return dict( lp=lp.mean(), clust=clust.mean(), glob_eff=glob_eff.mean(),
                 loc_eff=loc_eff.mean() )


def nodal_summaryOut(graph):
    """Compute statistics for individual nodes.

    Parameters
    ----------
    graph: networkx graph
        An undirected graph.
        
    Returns
    -------
    dictionary
        The keys of this dictionary are lp (which refers to path
        length), clust (clustering coefficient), b_cen (betweenness
        centrality), c_cen (closeness centrality), nod_eff (nodal
        efficiency), loc_eff (local efficiency), and deg (degree).  The
        values are arrays (or lists, in some cases) of metrics, in
        ascending order of node labels.

    """
    lp = nodal_pathlengths(graph)
    clust_dict = nx.clustering(graph)
    clust = np.array([clust_dict[n] for n in sorted(clust_dict)])
    b_cen_dict = nx.betweenness_centrality(graph)
    b_cen = np.array([b_cen_dict[n] for n in sorted(b_cen_dict)])
    c_cen_dict = nx.closeness_centrality(graph)
    c_cen = np.array([c_cen_dict[n] for n in sorted(c_cen_dict)])
    nod_eff = nodal_efficiency(graph)
    loc_eff = local_efficiency(graph)
    deg_dict = graph.degree()
    deg = [deg_dict[n] for n in sorted(deg_dict)]
    return dict(lp=lp, clust=clust, b_cen=b_cen, c_cen=c_cen, nod_eff=nod_eff,
                loc_eff=loc_eff, deg=deg)
