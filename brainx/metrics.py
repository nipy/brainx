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
def compute_sigma(arr,clustarr,lparr):
    """ Function for computing sigma given a graph array arr and clust and lp
    arrays from a pseudorandom graph for a particular block b."""

    gc = arr['clust']#np.squeeze(arr['clust'])
    glp = arr['lp']#np.squeeze(arr['lp'])
    out = (gc/clustarr)/(glp/lparr)

        
    return out


def nodal_pathlengths(G, n_nodes):
    """Compute mean path length for each node.

    Parameters
    ----------
    G: networkx Graph
        An undirected graph.

    n_nodes: integer
        Number of nodes in G.

    Returns
    -------
    nodal_means: numpy array
        An array with each node's mean shortest path length to all other
        nodes.  The array is in ascending order of node labels.

    Notes
    -----
    This function assumes the nodes are labeled 0 to n_nodes - 1.

    Per the Brain Connectivity Toolbox for Matlab, the distance between
    one node and another that cannot be reached from it is set to
    infinity.

    """
    # float is the default dtype for np.zeros, but we'll choose it explicitly
    # in case numpy ever changes the default to something else.
    nodal_means = np.zeros(n_nodes, dtype=float)
    lengths = nx.all_pairs_shortest_path_length(G)
    # As stated in the Python documentation, "Keys and values are listed in an
    # arbitrary order which is non-random, varies across Python
    # implementations, and depends on the dictionary's history of insertions
    # and deletions."  Thus, we cannot assume we'd traverse the nodes in
    # ascending order if we were to iterate through 'lengths'.
    node_labels = range(n_nodes)
    for src in node_labels:
        source_lengths = []
        for targ in node_labels:
            if src != targ:
                try:
                    val = lengths[src][targ]
                except KeyError:
                    val = np.inf
                source_lengths.append(val)
        nodal_means[src] = np.mean(source_lengths)
    return nodal_means


def assert_no_selfloops(G):
    """Raise an error if the graph G has any selfloops.
    """
    if G.nodes_with_selfloops():
        raise ValueError("input graph can not have selfloops")


def path_lengths(G):
    """Compute array of all shortest path lengths for the given graph.

    The length of the output array is the number of unique pairs of nodes that
    have a connecting path, so in general it is not known in advance.

    This assumes the graph is undirected, as for any pair of reachable nodes,
    once we've seen the pair we do not keep the path length value for the
    inverse path.
    
    Parameters
    ----------
    G : an undirected graph object.
    """

    assert_no_selfloops(G)
    
    length = nx.all_pairs_shortest_path_length(G)
    paths = []
    seen = set()
    for src,targets in length.iteritems():
        seen.add(src)
        neigh = set(targets.keys()) - seen
        paths.extend(targets[targ] for targ in neigh)
    
    
    return np.array(paths) 


#@profile
def path_lengthsSPARSE(G):
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
    G : an undirected graph object.
    """

    assert_no_selfloops(G)
    
    length = nx.all_pairs_shortest_path_length(G)

    nnod = G.number_of_nodes()
    paths_mat = sparse.dok_matrix((nnod,nnod))
    
    for src,targets in length.iteritems():
        for targ,val in targets.items():
            paths_mat[src,targ] = val

    return sparse.triu(paths_mat,1).data


def glob_efficiency(G):
    """Compute array of global efficiency for the given graph.

    Global efficiency: returns a list of the inverse path length matrix
    across all nodes The mean of this value is equal to the global efficiency
    of the network."""
    
    return 1.0/path_lengths(G)


def nodal_efficiency(G, n_nodes):
    """Return array with nodal efficiency for each node in G.

    See Achard and Bullmore (2007, PLoS Comput Biol) for the definition
    of nodal efficiency.

    Parameters
    ----------
    G: networkx Graph
        An undirected graph.

    n_nodes: integer
        Number of nodes in G.

    Returns
    -------
    nodal_efficiencies: numpy array
        An array with the nodal efficiency for each node in G, in
        the order specified by node_labels.  The array is in ascending
        order of node labels.

    Notes
    -----
    This function assumes the nodes are labeled 0 to n_nodes - 1.

    Per the Brain Connectivity Toolbox for Matlab, the distance between
    one node and another that cannot be reached from it is set to
    infinity.

    """
    nodal_efficiencies = np.zeros(n_nodes, dtype=float)
    lengths = nx.all_pairs_shortest_path_length(G)
    node_labels = range(n_nodes)
    for src in node_labels:
        inverse_paths = []
        for targ in node_labels:
            if src != targ:
                try:
                    val = lengths[src][targ]
                except KeyError:
                    val = np.inf
                inverse_paths.append(1.0 / val)
        nodal_efficiencies[src] = np.mean(inverse_paths)
    return nodal_efficiencies


def local_efficiency(G):
    """Compute array of global efficiency for the given grap.h

    Local efficiency: returns a list of paths that represent the nodal
    efficiencies across all nodes with their direct neighbors"""

    nodepaths=[]
    length=nx.all_pairs_shortest_path_length(G)
    for n in G.nodes():
        nneighb= nx.neighbors(G,n)
        
        paths=[]
        for src,targets in length.iteritems():
            for targ,val in targets.iteritems():
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


#@profile
def local_efficiency(G):
    """Compute array of local efficiency for the given graph.

    Local efficiency: returns a list of paths that represent the nodal
    efficiencies across all nodes with their direct neighbors"""

    assert_no_selfloops(G)

    nodepaths = []
    length = nx.all_pairs_shortest_path_length(G)
    for n in G:
        nneighb = set(nx.neighbors(G,n))

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


def dynamical_importance(G):
    """Compute dynamical importance for G.

    Ref: Restrepo, Ott, Hunt. Phys. Rev. Lett. 97, 094102 (2006)
    """
    # spectrum of the original graph
    eigvals = nx.adjacency_spectrum(G)
    lambda0 = eigvals[0]
    # Now, loop over all nodes in G, and for each, make a copy of G, remove
    # that node, and compute the change in lambda.
    nnod = G.number_of_nodes()
    dyimp = np.empty(nnod,float)
    for n in range(nnod):
        gn = G.copy()
        gn.remove_node(n)
        lambda_n = nx.adjacency_spectrum(gn)[0]
        dyimp[n] = lambda0 - lambda_n
    # Final normalization
    dyimp /= lambda0
    return dyimp


def weighted_degree(G):
    """Return an array of degrees that takes weights into account.

    For unweighted graphs, this is the same as the normal degree() method
    (though we return an array instead of a list).
    """
    amat = nx.adj_matrix(G).A  # get a normal array out of it
    return abs(amat).sum(0)  # weights are sums across rows


def graph_summary(G):
    """Compute a set of statistics summarizing the structure of a graph.
    
    Parameters
    ----------
    G : a graph object.

    threshold : float, optional

    Returns
    -------
      Mean values for: lp, clust, glob_eff, loc_eff, in a dict.
    """
    
    # Average path length
    lp = path_lengths(G)
    clust = np.array(nx.clustering(G).values())
    glob_eff = glob_efficiency(G)
    loc_eff = local_efficiency(G)
    
    return dict( lp=lp.mean(), clust=clust.mean(), glob_eff=glob_eff.mean(),
                 loc_eff=loc_eff.mean() )


def nodal_summaryOut(G, n_nodes):
    """Compute statistics for individual nodes.

    Parameters
    ----------
    G: networkx graph
        An undirected graph.
        
    n_nodes: integer
        Number of nodes in G.

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
    lp = nodal_pathlengths(G, n_nodes)
    # As stated in the Python documentation, "Keys and values are listed in an
    # arbitrary order which is non-random, varies across Python
    # implementations, and depends on the dictionary's history of insertions
    # and deletions."  Thus, we cannot expect, e.g., nx.clustering(G)
    # to have the nodes listed in ascending order, as we desire.
    node_labels = range(n_nodes)
    clust_dict = nx.clustering(G)
    clust = np.array([clust_dict[n] for n in node_labels])
    b_cen_dict = nx.betweenness_centrality(G)
    b_cen = np.array([b_cen_dict[n] for n in node_labels])
    c_cen_dict = nx.closeness_centrality(G)
    c_cen = np.array([c_cen_dict[n] for n in node_labels])
    nod_eff = nodal_efficiency(G, n_nodes)
    loc_eff = local_efficiency(G)
    deg_dict = G.degree()
    deg = [deg_dict[n] for n in node_labels]
    return dict(lp=lp, clust=clust, b_cen=b_cen, c_cen=c_cen, nod_eff=nod_eff,
                loc_eff=loc_eff, deg=deg)
