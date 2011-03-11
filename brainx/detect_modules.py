#!/usr/bin/env python
"""Detect modules in a network.

Citation: He Y, Wang J, Wang L, Chen ZJ, Yan C, et al. (2009) Uncovering
Intrinsic Modular Organization of Spontaneous Brain Activity in Humans. PLoS
ONE 4(4): e5226. doi:10.1371/journal.pone.0005226

Comparing community structure identification
J. Stat. Mech. (2005) P0900
Leon Danon1,2, Albert Diaz-Guilera1, Jordi Duch2 and  Alex Arenas
Online at stacks.iop.org/JSTAT/2005/P09008
doi:10.1088/1742-5468/2005/09/P09008

"""

# Modules from the stdlib
import math
import random
import copy

# Third-party modules
import networkx as nx
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
import time

# Our own modules
import util

# Fancy testing support, eventually this will go into ipython (where it was
# written in the first place)
from decotest  import (as_unittest, ParametricTestCase, parametric)

#-----------------------------------------------------------------------------
# Class declarations
#-----------------------------------------------------------------------------
class GraphPartition(object):
    """Represent a graph partition."""

    def __init__(self, graph, index):
        """New partition, given a graph and a dict of module->nodes.

        Parameters
        ----------
        graph : network graph instance
          Graph to which the partition index refers to.
          
        index : dict
          A dict that maps module labels to sets of nodes, this describes the
          partition in full.

        Note
        ----
        The values in the index dict MUST be real sets, not lists.  No checks
        are made of this fact, but later the code relies on them being sets and
        may break in strange manners if the values were stored in non-set
        objects.
        """
        # Store references to the original graph and label dict
        self.index = copy.deepcopy(index)
        #self.graph = graph
        
        # We'll need the graph's adjacency matrix often, so store it once
        self.graph_adj_matrix = nx.adj_matrix(graph)
        
        # Just to be sure, we don't want to count self-links, so we zero out the
        # diagonal.
        util.fill_diagonal(self.graph_adj_matrix, 0)

        # Store statically a few things about the graph that don't change (as
        # long as the graph does not change
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = graph.number_of_edges()

        # Store the nodes as a set, needed for many operations
        self._node_set = set(graph.nodes())

        # Now, build the edge information used in modularity computations
        self.mod_e, self.mod_a = self._edge_info()
    

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.index)

    def _edge_info(self, mod_e=None, mod_a=None, index=None):
        """Create the vectors of edge information.
        
        Returns
        -------
          mod_e: diagonal of the edge matrix E

          mod_a: sum of the rows of the E matrix
        """
        num_mod = len(self)
        if mod_e is None: mod_e = [0] * num_mod
        if mod_a is None: mod_a = [0] * num_mod
        if index is None: index = self.index
        
        norm_factor = 1.0/(2.0*self.num_edges)
        mat = self.graph_adj_matrix
        set_nodes = self._node_set
        for m,modnodes in index.iteritems():
            #set_modnodes=set(modnodes)
            #btwnnodes   = list(set_nodes - modnodes)
            btwnnodes = list(set_nodes - set(index[m]))
            modnodes  = list(modnodes)
            #why isnt' self.index a set already?  graph_partition.index[m]
            #looks like a set when we read it in ipython
            mat_within  = mat[modnodes,:][:,modnodes]
            mat_between = mat[modnodes,:][:,btwnnodes]
            perc_within = mat_within.sum() * norm_factor
            perc_btwn   = mat_between.sum() * norm_factor
            mod_e[m] = perc_within #all of the E's
            mod_a[m] = perc_btwn+perc_within #all of the A's
            #mod_e.append(perc_within)
            #mod_a.append(perc_btwn+perc_within)

            
        return mod_e, mod_a

    def modularity_newman(self):
        """ Function using other version of expressing modularity, from the Newman papers (2004 Physical Review)

        Parameters:
        g = graph
        part = partition

        Returns:
        mod = modularity
        """
        return (np.array(self.mod_e) - (np.array(self.mod_a)**2)).sum()

    modularity = modularity_newman
    
    #modularity = modularity_guimera


    ## def modularity_guimera(self, g, part):
    ##     """This function takes in a graph and a partition and returns Newman's
    ##     modularity for that graph"""

    ##     """ Parameters
    ##     # g = graph part = partition; a dictionary that contains a list of
    ##     # nodes that make up that module"""

    ##     #graph values
    ##     num_mod = len(part)
    ##     L = nx.number_of_edges(g)
    ##     # construct an adjacency matrix from the input graph (g)
    ##     mat = nx.adj_matrix(g)

    ##     M = 0
    ##     # loop over the modules in the graph, create an adjacency matrix
    ##     for m, val in part.iteritems():
    ##         #create a 'sub mat'
    ##         submat = mat[val,:][:,val]

    ##         #make a graph
    ##         subg = nx.from_numpy_matrix(submat)

    ##         #calculate module-specific metrics
    ##         link_s = float(subg.number_of_edges())
    ##         deg_s = np.sum(nx.degree(g,val), dtype=float)

    ##         #compute modularity!
    ##         M += ((link_s/L) - (deg_s/(2*L))**2)

    ##     return M
    
    def compute_module_merge(self, m1, m2):
        """Merges two modules in a given partition.

        This updates in place the mod_e and mod_a arrays (both of which lose a
        row).  The new, merged module will be identified as m1.
        
        Parameters
        ----------
          
        Returns
        -------

          """
        # Below, we want to know that m1<m2, so we enforce that:
        if m1>m2:
            m1, m2 = m2, m1

        # Pull from m2 the nodes and merge them into m1
        merged_module = self.index[m1] | self.index[m2]
        
        #make an empty matrix for computing "modularity" level values
        e1 = [0]
        a1 = [0]
        e0, a0 = self.mod_e, self.mod_a
        
        # The values that change: _edge_info with arguments will update the e,
        # a vectors only for the modules in index
        e1, a1 = self._edge_info(e1, a1, {0:merged_module})
        
        # Compute the change in modularity
        delta_q =  (e1[0]-a1[0]**2) - \
            ( (e0[m1]-a0[m1]**2) + (e0[m2]-a0[m2]**2) )

        #print 'NEW: ',e1,a1,e0[m1],a0[m1],e0[m2],a0[m2]
  
        return merged_module, e1[0], a1[0], -delta_q, 'merge',m1,m2,m2

    
    def apply_module_merge(self, m1, m2, merged_module, e_new, a_new):
        """Merges two modules in a given partition.

        This updates in place the mod_e and mod_a arrays (both of which lose a
        row).  The new, merged module will be identified as m1.
        
        Parameters
        ----------
        XXX
        
        Returns
        -------
        XXX
          """

        # Below, we want to know that m1<m2, so we enforce that:
        if m1>m2:
            m1, m2 = m2, m1

        # Pull from m2 the nodes and merge them into m1
        self.index[m1] = merged_module
        del self.index[m2]

        # We need to shift the keys to account for the fact that we popped out
        # m2
        
        rename_keys(self.index,m2)
        
        self.mod_e[m1] = e_new
        self.mod_a[m1] = a_new
        self.mod_e.pop(m2)
        self.mod_a.pop(m2)
        
        
    def compute_module_split(self, m, n1, n2):
        """Splits a module into two new ones.

        This updates in place the mod_e and mod_a arrays (both of which lose a
        row).  The new, merged module will be identified as m1.
        
        Parameters
        ----------
        m : module identifier
        n1, n2 : sets of nodes
          The two sets of nodes in which the nodes originally in module m will
          be split.  Note: It is the responsibility of the caller to ensure
          that the set n1+n2 is the full set of nodes originally in module m.
          
        Returns
        -------
          The change in modularity resulting from the change
          (Q_final-Q_initial)"""

        # create a dict that contains the new modules 0 and 1 that have the sets n1 and n2 of nodes from module m.
        split_modules = {0: n1, 1: n2} 

        #make an empty matrix for computing "modularity" level values
        e1 = [0,0]
        a1 = [0,0]
        e0, a0 = self.mod_e, self.mod_a

        # The values that change: _edge_info with arguments will update the e,
        # a vectors only for the modules in index
        e1, a1 = self._edge_info(e1, a1, split_modules)

        # Compute the change in modularity
        delta_q =  ( (e1[0]-a1[0]**2) + (e1[1]- a1[1]**2) ) - \
            (e0[m]-a0[m]**2)
        
        return split_modules, e1, a1, -delta_q,'split',m,n1,n2

    
    def apply_module_split(self, m, n1, n2, split_modules, e_new, a_new):
        """Splits a module into two new ones.

        This updates in place the mod_e and mod_a arrays (both of which lose a
        row).  The new, merged module will be identified as m1.
        
        Parameters
        ----------
        m : module identifier
        n1, n2 : sets of nodes
          The two sets of nodes in which the nodes originally in module m will
          be split.  Note: It is the responsibility of the caller to ensure
          that the set n1+n2 is the full set of nodes originally in module m.
          
        Returns
        -------
          The change in modularity resulting from the change
          (Q_final-Q_initial)"""

        # To reuse slicing code, use m1/m2 lables like in merge code
        m1 = m
        m2 = len(self)
        
        #Add a new module to the end of the index dictionary
        self.index[m1] = split_modules[0] #replace m1 with n1
        self.index[m2] = split_modules[1] #add in new module, fill with n2
        
        self.mod_e[m1] = e_new[0]
        self.mod_a[m1] = a_new[0]
        self.mod_e.insert(m2,e_new[1])
        self.mod_a.insert(m2,a_new[1])
        
        #self.mod_e[m2] = e_new[1]
        #self.mod_a[m2] = a_new[1]

        #EN: Not sure if this is necessary, but sometimes it finds a partition with an empty module...
        
        # This checks whether there is an empty module. If so, renames the keys.
        if len(self.index[m1])<1:
            self.index.pop(m1)
            rename_keys(self.index,m1)
        # This checks whether there is an empty module. If so, renames the keys.
        if len(self.index[m2])<1:
            self.index.pop(m2)
            rename_keys(self.index,m2)


    def node_update(self, n, m1, m2):
        """Moves a single node within or between modules

        Parameters
        ----------
        n : node identifier
          The node that will be moved from module m1 to module m2
        m1 : module identifier
          The module that n used to belong to.
        m2 : module identifier
          The module that n will now belong to.

        Returns
        -------
          The change in modularity resulting from the change
          (Q_final-Q_initial)"""

        #Update the index with the change
        index = self.index
        index[m1].remove(n)
        index[m2].add(n)

        # This checks whether there is an empty module. If so, renames the keys.
        if len(self.index[m1])<1:
            self.index.pop(m1)
            rename_keys(self.index,m1)
            
        # Before we overwrite the mod vectors, compute the contribution to
        # modularity from before the change
        e0, a0 = self.mod_e, self.mod_a
        mod_old = (e0[m1]-a0[m1]**2) + (e0[m2]-a0[m2]**2)
        # Update in place mod vectors with new index
        self._edge_info(self.mod_e, self.mod_a, {m1:index[m1], m2:index[m2]})
        e1, a1 = self.mod_e, self.mod_a
        #Compute the change in modularity
        return (e1[m1]-a1[m1]**2) + (e1[m2]-a1[m2]**2) - mod_old

    def compute_node_update(self, n, m1, m2):
        """Moves a single node within or between modules

        Parameters
        ----------
        n : node identifier
          The node that will be moved from module m1 to module m2
        m1 : module identifier
          The module that n used to belong to.
        m2 : module identifier
          The module that n will now belong to.

        Returns
        -------
          The change in modularity resulting from the change
          (Q_final-Q_initial)"""
        
        n1 = self.index[m1]
        n2 = self.index[m2]

        node_moved_mods = {0: n1 - set([n]),1: n2 | set([n])}
            
        # Before we overwrite the mod vectors, compute the contribution to
        # modularity from before the change
        e1 = [0,0]
        a1 = [0,0]
        e0, a0 = self.mod_e, self.mod_a

        # The values that change: _edge_info with arguments will update the e,
        # a vectors only for the modules in index
        e1, a1 = self._edge_info(e1, a1, node_moved_mods)
        
        #Compute the change in modularity
        delta_q =  ( (e1[0]-a1[0]**2) + (e1[1]-a1[1]**2)) - \
            ( (e0[m1]-a0[m1]**2) + (e0[m2]-a0[m2]**2) )
        
        #print n,m1,m2,node_moved_mods,n1,n2
        return node_moved_mods, e1, a1, -delta_q, n, m1, m2

    def apply_node_update(self, n, m1, m2, node_moved_mods, e_new, a_new):
        """Moves a single node within or between modules

        Parameters
        ----------
        n : node identifier
          The node that will be moved from module m1 to module m2
        m1 : module identifier
          The module that n used to belong to.
        m2 : module identifier
          The module that n will now belong to.

        Returns
        -------
          The change in modularity resulting from the change
          (Q_final-Q_initial)"""

        
        
        self.index[m1] = node_moved_mods[0]
        self.index[m2] = node_moved_mods[1]
        
        # This checks whether there is an empty module. If so, renames the keys.
        if len(self.index[m1])<1:
            self.index.pop(m1)
            rename_keys(self.index,m1)
            
        self.mod_e[m1] = e_new[0]
        self.mod_a[m1] = a_new[0]
        self.mod_e[m2] = e_new[1]
        self.mod_a[m2] = a_new[1]

    def random_mod(self):
        """Makes a choice whether to merge or split modules in a partition
        
        Returns:
        -------
        if splitting: m1, n1, n2
          m1: the module to split
          n1: the set of nodes to put in the first output module
          n2: the set of nodes to put in the second output module

        if merging: m1, m2
          m1: module 1 to merge
          m2: module 2 to merge
        """

        # number of modules in the partition
        num_mods=len(self)
        
        
        # Make a random choice bounded between 0 and 1, less than 0.5 means we will split the modules
        # greater than 0.5 means we will merge the modules.
        
        if num_mods >= self.num_nodes-1:
            coin_flip = 1 #always merge if each node is in a separate module
        elif num_mods <= 2:
            coin_flip = 0 #always split if there's only one module
        else:
            coin_flip = random.random()
            

        #randomly select two modules to operate on
        rand_mods = np.random.permutation(range(num_mods))
        m1 = rand_mods[0]
        m2 = rand_mods[1]

        if coin_flip > 0.5:
            #merge
            #return self.module_merge(m1,m2)
            return self.compute_module_merge(m1,m2)
        else: 
            #split
            # cannot have a module with less than 1 node
            while len(self.index[m1]) <= 1:

                #reselect the first  module
                rand_mods = np.random.permutation(range(num_mods))
                m1 = rand_mods[0]
                #m1 = random.randint(0,num_mods)

            # list of nodes within that module
            list_nods = list(self.index[m1])

            # randomly partition the list of nodes into 2
            nod_split_ind = random.randint(1,len(list_nods)) #can't pick the first node as the division
            n1 = set(list_nods[:nod_split_ind])
            n2 = set(list_nods[nod_split_ind:])

            #We may want to return output of merging/splitting directly, but
            #for now we're returning inputs for those modules.
            
            return self.compute_module_split(m1,n1,n2)


    def random_mod_old(self):
        """Makes a choice whether to merge or split modules in a partition
        
        Returns:
        -------
        if splitting: m1, n1, n2
          m1: the module to split
          n1: the set of nodes to put in the first output module
          n2: the set of nodes to put in the second output module

        if merging: m1, m2
          m1: module 1 to merge
          m2: module 2 to merge
        """

        # number of modules in the partition
        num_mods=len(self)
        
        
        # Make a random choice bounded between 0 and 1, less than 0.5 means we will split the modules
        # greater than 0.5 means we will merge the modules.
        
        if num_mods >= self.num_nodes-1:
            coin_flip = 1 #always merge if each node is in a separate module
        elif num_mods <= 2:
            coin_flip = 0 #always split if there's only one module
        else:
            coin_flip = random.random()
            
        #randomly select two modules to operate on
        rand_mods = np.random.permutation(range(num_mods))
        m1 = rand_mods[0]
        m2 = rand_mods[1]

        if coin_flip > 0.5:
            #merge
            #return self.module_merge(m1,m2)
            return self.module_merge(m1,m2)
        else: 
            #split
            # cannot have a module with less than 1 node
            while len(self.index[m1]) <= 1:

                #reselect the first  module
                rand_mods = np.random.permutation(range(num_mods))
                m1 = rand_mods[0]
                #m1 = random.randint(0,num_mods)

            # list of nodes within that module
            list_nods = list(self.index[m1])

            # randomly partition the list of nodes into 2
            nod_split_ind = random.randint(1,len(list_nods)) #can't pick the first node as the division
            n1 = set(list_nods[:nod_split_ind])
            n2 = set(list_nods[nod_split_ind:])

            #We may want to return output of merging/splitting directly, but
            #for now we're returning inputs for those modules.
            
            return self.module_split(m1,n1,n2)
       
    def random_node(self):
        """ Randomly reassign one node from one module to another

        Returns:
        -------

        n: node to move
        m1: module node is currently in
        m2: module node will be moved to """

        # number of modules in the partition
        num_mods=len(self)
        if num_mods < 2:
            raise ValueError("Can not reassign node with only one module")

        # initialize a variable so we can search the modules to find one with
        # at least 1 node
        node_len = 0
        
        # select 2 random modules (the first must have at least 2 nodes in it)
        while node_len <= 1:
            
            # randomized list of modules
            rand_mods=np.random.permutation(range(num_mods))
            
            node_len = len(self.index[rand_mods[0]])
        
        m1 = rand_mods[0]
        m2 = rand_mods[1]

            
        # select a random node within one module
        node_list = list(self.index[m1])
        rand_perm = np.random.permutation(node_list)
        n = rand_perm[0]
        
        return self.compute_node_update(n,m1,m2)

    def store_best(self):
        """ Keeps the best partition stored for later. It should 'refresh' each time. """

        #attempting to initialize this every time this function is called...make sure this works
        self.bestindex = dict()
        
        #Store references to the original graph and label dict
        self.bestindex = copy.deepcopy(self.index)
        
        
        
#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------
def diag_stack(tup):
    """Stack arrays in sequence diagonally (block wise).
    
    Take a sequence of arrays and stack them diagonally to make a single block
    array.
    
    
    Parameters
    ----------
    tup : sequence of ndarrays
        Tuple containing arrays to be stacked. The arrays must have the same
        shape along all but the first two axes.
    
    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays.
    
    See Also
    --------
    hstack : Stack arrays in sequence horizontally (column wise).
    vstack : Stack arrays in sequence vertically (row wise).
    dstack : Stack arrays in sequence depth wise (along third dimension).
    concatenate : Join a sequence of arrays together.
    vsplit : Split array into a list of multiple sub-arrays vertically.
    
    
    Examples
    --------
    """
    # Find number of rows and columns needed
    shapes = np.array([a.shape for a in tup], int)
    sums = shapes.sum(0)
    nrow = sums[0]
    ncol = sums[1]
    out = np.zeros((nrow, ncol), tup[0].dtype)
    row_offset = 0
    col_offset = 0
    for arr in tup:
        nr, nc = arr.shape
        row_end = row_offset+nr
        col_end = col_offset+nc
        out[row_offset:row_end, col_offset:col_end] = arr
        row_offset, col_offset = row_end, col_end
    return out


def random_modular_graph(nnod, nmod, av_degree, between_fraction=0.0):
    """
    Parameters
    ----------

    nnod : int
      Total number of nodes in the graph.

    nmod : int
      Number of modules.  Note that nmod must divide nnod evenly.

    av_degree : int
      Average degree of the nodes.

    between_fraction : float
      A number in [0,1], indicating the fraction of edges in each module which
      are wired to go between modules.
    """
    # sanity checks:
    if nnod%nmod:
        raise ValueError("nmod must divide nnod evenly")

    # Compute the number of nodes per module
    nnod_mod = nnod/nmod

    # The average degree requested can't be more than what the graph can
    # support if it were to be fully dense
    if av_degree > nnod_mod - 1:
        e = "av_degree can not be larger than (nnod_mod-1) = %i" % (nnod_mod-1)
        raise ValueError(e)

    # Compute the probabilities to generate the graph with, both for
    # within-module (p_in) and between-modules (p_out):
    z_out = between_fraction*av_degree
    p_in = (av_degree-z_out)/(nnod_mod-1.0)
    p_out = float(z_out)/(nnod-nnod_mod)

    # Some sanity checks
    assert 0 <= p_in <=1, "Invalid p_in=%s, not in [0,1]" % p_in
    assert 0 <= p_out <=1, "Invalid p_out=%s, not in [0,1]" % p_out

    # Create initial matrix with uniform random numbers in the 0-1 interval.
    mat = util.symm_rand_arr(nnod)

    # Create the masking matrix
    blocks = [np.ones((nnod_mod, nnod_mod))] * nmod
    mask = diag_stack(blocks)

    # Threshold the random matrix to create an actual adjacency graph.

    # Emi's trick: we need to use thresholding in only certain parts of the
    # matrix, corresponding to where the mask is 0 or 1.  Rather than having a
    # complex indexing operation, we'll just multiply the numbers in one region
    # by -1, and then we can do the thresholding over negative and positive
    # values. As long as we correct for this, it's a much simpler approach.
    mat[mask==1] *= -1

    adj = np.zeros((nnod, nnod))
    # Careful to flip the sign of the thresholding for p_in, since we used the
    # -1 trick above
    adj[np.logical_and(0 >= mat, mat > -p_in)] = 1
    adj[np.logical_and(0 < mat, mat < p_out)] = 1

    # no self-links
    util.fill_diagonal(adj, 0)
    # Our return object is a graph, not the adjacency matrix
    return nx.from_numpy_matrix(adj)


def array_to_string(part):
    """The purpose of this function is to convert an array of numbers into
    a list of strings. Mainly for use with the plot_partition function that
    requires a dict of strings for node labels.

    """

    out_part=dict.fromkeys(part)
    
    for m in part.iterkeys():
        out_part[m]=str(part[m])
    
    return out_part


def rename_keys(dct, key):
    """This function reads in a partition and a single module to be
    removed,pops out the value(s) and shifts the key names accordingly.

    Parameters
    ----------
    XXX
    
    Returns
    -------
    XXX
    """
 
    for m in range(key,len(dct)):
        dct[m] = dct.pop(m+1)

def rand_partition(g):
    """This function takes in a graph and returns a dictionary of labels for
    each node. Eventually it needs to be part of the simulated annealing program,
    but for now it will just make a random partition."""

    num_nodes = g.number_of_nodes()

    # randomly select a number of modules
    num_mods = random.randint(1,num_nodes)

    # randomize the order of nodes into a list
    rand_nodes = np.random.permutation(num_nodes)

    # We'll use this twice below, don't re-generate it.
    mod_range = range(num_mods)
    
    # set up a dictionary containing each module and the nodes under it.
    # Note: the following loop *does* cover the entire range, even if it
    # doesn't appear obvious immediately.  The easiest way to see this is to
    # write the execution of the loop row-wise, assuming an ordered permutation
    # (rand_nodes), and then to read it column-wise.  It will be then obvious
    # that when each column ends at the last row, the next column starts with
    # the next node in the list, and no node is ever skipped.
    out = [set(rand_nodes[i::num_mods]) for i in mod_range]

##     # a simpler version of the partitioning

##     # We need to split the list of nodes into (num_mods) partitions which means we need (num_mods-1) slices.
##     # The slices need to be in increasing order so we can use them as indices
##     rand_slices=sort(np.random.permutation(rand_nodes)[:num_mods-1])

##     # initialize a dictionary
##     out = dict()
##     # initialize the first element of the node list
##     init_node=0
##     for m in range_mods:
        
##         #length of the current module
##         len_mod=rand_slices[s]-init_node
##         out[mod_ind] = rand_nodes[init_node:len_mod+init_node]
##         init_node=rand_slices[m]
        
    # The output is the final partition
    return dict(zip(mod_range,out))


def perfect_partition(nmod,nnod_mod):
    """This function takes in the number of modules and number of nodes per module
    and returns the perfect partition depending on the number of modules
    where the module number is fixed according to random_modular_graph()"""
    
    #empty dictionary to fill with the correct partition
    part=dict()
    #set up a dictionary containing each module and the nodes under it
    for m in range(nmod):
        part[m]=set(np.arange(nnod_mod)+m*nnod_mod) #dict([(nmod,nnod)])# for x in range(num_mods)])
        #print 'Part ' + str(m) + ': '+ str(part[m])
    
    return part

def plot_partition(g,part,title,fname='figure',nod_labels = None, pos = None,
    within_mod = 'none', part_coeff = 'none',les_dam='none'):
    """This function takes in a graph and a partition and makes a figure that
    has each node labeled according to its partition assignment"""
    
    #set up figure
    fig=plt.figure()
    plt.axis('off')
    
    #set up nodes
    nnod = g.number_of_nodes()

    if nod_labels == None:
        nod_labels = dict(zip(range(nnod),range(nnod)))
    else:
        nod_labels = dict(zip(range(nnod),nod_labels))

    nod_labels = array_to_string(nod_labels)

    #set up node locations
    if pos == None:
        pos=nx.circular_layout(g)
        
    #col=colors.cnames.keys()
    col = ['r','g','b','m','c','y']
    col2 = ['#000066','#000099','#660000','#CC6633','#FF0099','#FF00FF','#33FFFF','#663366','#FFCC33','#CCFF66','#FFCC99','#33CCCC','#FF6600','#FFCCFF','#CCFFFF','#CC6699','#CC9900','#FF6600','#99FF66','#CC0033','#99FFFF','#CC00CC','#CC99CC','#660066','#33CC66','#336699','#3399FF','#339900','#003300','#00CC00','#330033','#333399','#0033CC','#333333','#339966','#333300']
    
    niter = 0
    edge_list_between = []
    #for m,val in part.iteritems():
    for m in part:

        val = part[m]

        if niter <len(col):
            if within_mod == 'none': #note: assumes part_coeff also there
                for v in val:
                    if les_dam != 'none':
                        plt.scatter(pos[v][0],pos[v][1],s=100*les_dam[v],c='orange',marker=(4,1,0))
                nx.draw_networkx_nodes(g,pos,nodelist=list(val),node_color=col[niter],node_size=50)
            else:
                for v in val:
                    if les_dam != 'none':
                        plt.scatter(pos[v][0],pos[v][1],s=500*les_dam[v],c='orange',marker=(4,1,0))

                    if within_mod[v] > 1:
                        nx.draw_networkx_nodes(g,pos,nodelist=[v],node_color=col[niter],node_size=part_coeff[v] * 500+50,node_shape='s',linewidths=2)
                    else:
                        nx.draw_networkx_nodes(g,pos,nodelist=[v],node_color=col[niter],node_size=part_coeff[v] * 500+50,node_shape='o',linewidths=0.5)
        else:
            #print 'out of colors!!'
            if within_mod == 'none': #note: assumes part_coeff also there
                for v in val:
                    if les_dam != 'none':
                        plt.scatter(pos[v][0],pos[v][1],s=100*les_dam[v],c='orange',marker=(4,1,0))
                nx.draw_networkx_nodes(g,pos,nodelist=list(val),node_color=col2[niter],node_size=50)
            else:
                for v in val:
                    if les_dam != 'none':
                        plt.scatter(pos[v][0],pos[v][1],s=500*les_dam[v],c='orange',marker=(4,1,0))

                    if within_mod[v] > 1:
                        nx.draw_networkx_nodes(g,pos,nodelist=[v],node_color=col2[niter],node_size=part_coeff[v] * 500+50,node_shape='s',linewidths=2)
                    else:
                        nx.draw_networkx_nodes(g,pos,nodelist=[v],node_color=col2[niter],node_size=part_coeff[v] * 500+50,node_shape='o',linewidths=0.5)
                    

            
        val_array = np.array(val)
        edge_list_within = []
        for edg in g.edges():
            #temp = np.array(edge_list_between)
            n1_ind = np.where(val_array == edg[0])[0]
            n2_ind = np.where(val_array == edg[1])[0]
            #edg_ind = np.where(temp == edg)

            if len(n1_ind) > 0 and len(n2_ind) > 0:
                #add on the edge if it is within the partition
                edge_list_within.append(edg)
            elif len(n1_ind)>0 and len(n2_ind) == 0:
                #add on the edge if it hasn't been seen before
                edge_list_between.append(edg)
            elif len(n2_ind)>0 and len(n1_ind) == 0:
                edge_list_between.append(edg)
        

        if niter <len(col):
            nx.draw_networkx_edges(g,pos,edgelist=edge_list_within,edge_color=col[niter])
        else:
            nx.draw_networkx_edges(g,pos,edgelist=edge_list_within,edge_color=col2[niter])
        niter += 1
        
    nx.draw_networkx_labels(g,pos,nod_labels,font_size=6)    
    #nx.draw_networkx_edges(g,pos,edgelist=nx.edges(g))
    nx.draw_networkx_edges(g,pos,edgelist=edge_list_between,edge_color='k')

    plt.title(title)
    #plt.savefig(fname)
    #plt.close()
    #plt.show()

def compare_dicts(d1,d2):
    """Function that reads in two dictionaries of sets (i.e. a graph partition) and assess how similar they are.
    Needs to be updated so that it can adjust this measure to include partitions that are pretty close."""
    

    if len(d1)>len(d2):
        longest_dict=len(d1)
    else:
        longest_dict=len(d2)
    check=0
    #loop through the keys in the first dict
    for m1,val1 in d1.iteritems():
        #compare to the values in each key of the second dict
        for m2,val2 in d2.iteritems():
            if val1 == val2:
                check+=1
    return float(check)/longest_dict
        

def mutual_information(d1,d2):
    """Function that reads in two dictionaries of sets (i.e. a graph partition) and assess how similar they are using mutual information as in Danon, Diaz-Guilera, Duch & Arenas, J Statistical Mechanics 2005.
    
    Inputs:
    ------
    d1 = dictionary of 'real communities'
    d2 = dictionary of 'found communities'
    """
    
    dlist = [d1,d2]
    #first get rid of any empty values and relabel the keys accordingly
    new_d2 = dict()
    old_d2=d2
    for d in dlist:
        items = d.items()
        sort_by_length = [[len(v[1]),v[0]] for v in items]
        sort_by_length.sort()
        
        counter=0
        for i in range(len(sort_by_length)):
            #if the module is not empty...
            if sort_by_length[i][0]>0:
                new_d2[counter]=d[sort_by_length[i][1]]
                counter+=1

        d2=new_d2
    
    #define a 'confusion matrix' where rows = 'real communities' and columns = 'found communities'
    #The element of N (Nij) = the number of nodes in the real community i that appear in the found community j
    rows = len(d1)
    cols = len(d2)
    N = np.empty((rows,cols))
    rcol = range(cols)
    for i in range(rows):
        for j in rcol:
            #N[i,j] = len(d1[i] & d2[j])
            N[i,j] = len(set(d1[i]) & set(d2[j]))
         

    nsum_row = N.sum(0)[np.newaxis, :]
    nsum_col = N.sum(1)[:, np.newaxis]
    nn = nsum_row.sum()
    log = np.log
    nansum = np.nansum
    
    num = nansum(N*log(N*nn/(nsum_row*nsum_col)))
    den = nansum(nsum_row*log(nsum_row/nn)) + nansum(nsum_col*log(nsum_col/nn))

    return -2*num/den
        
def decide_if_keeping(dE,temperature):
    """Function which uses the rule from Guimera & Amaral (2005) Nature paper to decide whether or not to keep new partition

    Parameters:
    dE = delta energy 
    temperature = current state of the system
=
    Returns:
    keep = 1 or 0 to decide if keeping new partition """

    if dE <= 0:
        return True
    else:
        return random.random() < math.exp(-dE/temperature)

    
def simulated_annealing(g,temperature = 50, temp_scaling = 0.995, tmin=1e-5,
                        bad_accept_mod_ratio_max = 0.8 ,
                        bad_accept_nod_ratio_max = 0.8, accept_mod_ratio_min =
                        0.05, accept_nod_ratio_min = 0.05,
                        extra_info = False):

    """ This function does simulated annealing on a graph

    Parameters:
    g = graph #to anneal over
    temperature = 5777 #temperature of the sun in Kelvin, where we're starting
    tmin = 0.0 # minimum temperature
    n_nochanges = 25 # number of times to allow no change in modularity before
    breaking out of loop search

    Return:
    part = final partition
    M = final modularity """

    #Make a random partition for the graph
    nnod = g.number_of_nodes()
    nnod2 = nnod**2
    part = dict()
    #check if there is only one module or nnod modules
    while (len(part) <= 1) or (len(part) == nnod): 
        part = rand_partition(g)
    

    # make a graph partition object
    graph_partition = GraphPartition(g,part)
    
    # The number of times we switch nodes in a partition and the number of
    # times we modify the partition, at each temperature.  These values were
    # suggested by Guimera and Amaral, Nature 443, p895.  This is achieved
    # simply by running two nested loops of length nnod
    
    nnod = graph_partition.num_nodes
    rnod = range(nnod)

    #initialize some counters
    count = 0
    
    #Initialize empty lists for keeping track of values
    energy_array = []#negative modularity
    rej_array = []
    temp_array = []
    energy_best = 0
    
    energy = -graph_partition.modularity()
    energy_array.append(energy)

    while temperature > tmin:
        # Initialize counters
        bad_accept_mod = 0
        accept_mod = 0
        reject_mod = 0
        count_mod = 0
        count_bad_mod = 0.0001  # small offset to avoid occasional 1/0 errors
        
        for i_mod in rnod:
            # counters for module change attempts
            count_mod+=1
            count+=1
            
            # Assess energy change of a new partition without changing the partition
            calc_dict,e_new,a_new,delta_energy,movetype,p1,p2,p3 = graph_partition.random_mod()
            
            # Increase the 'count_bad_mod' if the new partition increases the energy
            if delta_energy > 0:
                count_bad_mod += 1
            
            # Decide whether the new partition is better than the old
            keep = decide_if_keeping(delta_energy,temperature)
            
            # Append the current temperature to the temp list
            temp_array.append(temperature)
            
            if keep:
                # this applies changes in place if energy decreased; the
                # modules will either be merged or split depending on a random
                # coin flip
                if movetype=='merge':
                    graph_partition.apply_module_merge(p1,p2,calc_dict,e_new,a_new)
                else:
                    graph_partition.apply_module_split(p1,p2,p3,calc_dict,e_new,a_new)
                
                # add the change in energy to the total energy
                energy += delta_energy
                accept_mod += 1 #counts times accept mod because lower energy
                
                # Increase the 'bad_accept_mod' if the new partition increases
                # the energy and was accepted
                if delta_energy > 0 :
                    bad_accept_mod += 1

                #maybe store the best one here too?
                #graph_partition.store_best()

            #else:
                #make a new graph partition with the last partition
                #reject_mod += 1
                #graph_partition = GraphPartition(g,graph_partition.index)

            if energy < energy_best:
                energy_best = energy
                graph_partition.store_best()
                
                
            energy_array.append(energy)   
            
            #break out if we are accepting too many "bad" options (early on)
            #break out if we are accepting too few options (later on)
            if count_mod > 10:
                bad_accept_mod_ratio =  float(bad_accept_mod)/(count_bad_mod)
                accept_mod_ratio = float(accept_mod)/(count_mod)
                #print 'ba_mod_r', bad_accept_mod_ratio  # dbg
                if (bad_accept_mod_ratio > bad_accept_mod_ratio_max) \
                        or (accept_mod_ratio < accept_mod_ratio_min):
                    #print 'MOD BREAK'
                    break

            bad_accept_nod = 0
            accept_nod = 0
            count_nod = 0
            count_bad_nod =  0.0001 # init at 1 to avoid 1/0 errors later
            
            for i_nod in rnod:
                count_nod+=1
                count+=1

                #if (np.mod(count,10000)==0) and (temperature < 1e-1):
                #    plot_partition(g,part,'../SA_graphs2/try'+str(count)+'.png')

                # Assess energy change of a new partition
                calc_dict,e_new,a_new,delta_energy,p1,p2,p3 = graph_partition.random_node()
                if delta_energy > 0:
                    count_bad_nod += 1
                temp_array.append(temperature)
                
                keep = decide_if_keeping(delta_energy,temperature)

                if keep:
                    
                    graph_partition.apply_node_update(p1,p2,p3,calc_dict,e_new,a_new)
                    energy += delta_energy
                    accept_nod += 1
                    if delta_energy > 0 :
                        bad_accept_nod += 1
                    #maybe store the best one here too?
                    #graph_partition.store_best()
                    
                #else:
                    #graph_partition = GraphPartition(g,graph_partition.index)

                if energy < energy_best:
                    energy_best = energy
                    graph_partition.store_best()
                    
                energy_array.append(energy)
                
                #break out if we are accepting too many "bad" options (early on)
                #break out if we are accepting too few options (later on)
                if count_nod > 10:
                    bad_accept_nod_ratio =  float(bad_accept_nod)/count_bad_nod
                    accept_nod_ratio = float(accept_nod)/(count_nod)
                    # if (bad_accept_nod_ratio > bad_accept_nod_ratio_max) \
#                         or (accept_nod_ratio < accept_nod_ratio_min):
#                         print 'nod BREAK'
#                         break
                    if (bad_accept_nod_ratio > bad_accept_nod_ratio_max):
                        #print 'too many accept'
                        break
                    if (accept_nod_ratio < accept_nod_ratio_min):
                        #print 'too many reject'
                        break
                    
                    if 0: #for debugging. 0 suppresses this for now.
                        print 'T: %.2e' % temperature, \
                            'accept nod ratio: %.2e ' %accept_nod_ratio, \
                            'bad accept nod ratio: %.2e' % bad_accept_nod_ratio, \
                            'energy: %.2e' % energy
         
        #print 'T: %.2e' % temperature, \
        #    'accept mod ratio: %.2e ' %accept_mod_ratio, \
        #    'bad accept mod ratio: %.2e' % bad_accept_mod_ratio, \
        #    'energy: %.2e' %energy, 'best: %.2e' %energy_best
        print 'T: %.2e' % temperature, \
            'energy: %.2e' %energy, 'best: %.2e' %energy_best
        temperature *= temp_scaling

    #NEED TO APPLY THE BEST PARTITION JUST IN CASE...
    #make a new graph object, apply the best partition
    #graph_partition.index = graph_partition.bestindex
    print graph_partition.modularity()
    graph_part_final = GraphPartition(g,graph_partition.bestindex)
    print graph_partition.modularity()
    print graph_part_final.modularity()
    
    if extra_info:
        extra_dict = dict(energy = energy_array, temp = temp_array)
        #return graph_partition, extra_dict
        return graph_part_final, extra_dict
    else:
        #return graph_partition
        print graph_part_final.modularity()
        return graph_part_final


#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------
import nose.tools as nt
import numpy.testing as npt


def test_random_modular_graph_between_fraction():
    """Test for graphs with non-zero between_fraction"""
    # We need to measure the degree within/between modules
    nnods = 120, 240, 360
    nmods = 2, 3, 4
    av_degrees = 8, 10, 16
    btwn_fractions = 0, 0.1, 0.3, 0.5

    for nnod in nnods:
        for nmod in nmods:
            for av_degree in av_degrees:
                for btwn_fraction in btwn_fractions:
                    g = random_modular_graph(nnod, nmod, av_degree,
                                             btwn_fraction)

                    # First, check the average degree.
                    av_degree_actual = np.mean(g.degree())
                    # Since we are generating random graphs, the actual average
                    # degree we get may be off from the reuqested one by a bit.  We
                    # allow it to be off by up to 1.
                    #print 'av deg:',av_degree, av_degree_actual  # dbg
                    yield (nt.assert_true, abs(av_degree-av_degree_actual) < 1, 
                           "av deg: %.2f  av deg actual: %.2f" % (av_degree,
                                                                  av_degree_actual))


                    # Now, check the between fraction
                    mat = nx.adj_matrix(g)

                    #compute the total number of edges in the real graph
                    nedg = nx.number_of_edges(g)
                    
                     # sanity checks:
                    if nnod%nmod:
                        raise ValueError("nmod must divide nnod evenly")

                    #Compute the of nodes per module
                    nnod_mod = nnod/nmod

                    #compute what the values are in the real graph
                    blocks = [np.ones((nnod_mod, nnod_mod))] * nmod
                    mask = diag_stack(blocks)
                    mask[mask==0] = 2
                    mask = np.triu(mask,1)
                    btwn_real = np.sum(mat[mask == 2].flatten())
                    btwn_real_frac = btwn_real / nedg
                    
                    #compare to what the actual values are
                    yield ( nt.assert_almost_equal, btwn_fraction,
                            btwn_real_frac, 1 )

    
def test_diag_stack():
    """Manual verification of simple stacking."""
    a = np.empty((2,2))
    a.fill(1)
    b = np.empty((3,3))
    b.fill(2)
    c = np.empty((2,3))
    c.fill(3)

    d = diag_stack((a,b,c))

    d_true = np.array([[ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  2.,  2.,  2.,  0.,  0.,  0.],
       [ 0.,  0.,  2.,  2.,  2.,  0.,  0.,  0.],
       [ 0.,  0.,  2.,  2.,  2.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  3.,  3.,  3.],
       [ 0.,  0.,  0.,  0.,  0.,  3.,  3.,  3.]])

    npt.assert_equal(d, d_true)


def test_modularity():
    """Test the values that go into the modularity calculation after randomly creating a graph"""
    # Given a partition with the correct labels of the input graph, verify that
    # the modularity returns 1

    # We need to measure the degree within/between modules
    nnods = 120, 240, 360
    nmods = 2, 3, 4
    av_degrees = 8, 10, 16

    for nnod in nnods:
        for nmod in nmods:
            for av_degree in av_degrees:
                g = random_modular_graph(nnod, nmod, av_degree)
                #Compute the of nodes per module
                nnod_mod = nnod/nmod
                #Make a "correct" partition for the graph
                part = perfect_partition(nmod,nnod_mod)
                #Make a graphpartition object
                graph_partition = GraphPartition(g,part)
                #call modularity
                mod_meas = graph_partition.modularity()
                mod_true = 1.0 - 1.0/nmod
                yield (npt.assert_almost_equal, mod_meas, mod_true, 2)


@parametric
def test_apply_module_merge():
    """Test the GraphPartition operation that merges modules so that it returns
    a change in modularity that reflects the difference between the modularity
    of the new and old parititions"""

    # nnod_mod, av_degrees, nmods
    networks = [ [3, [2], [3, 4]],
                 [4, [2, 3], [2, 4, 6]],
                 [8, [4, 6], [4, 6, 8]] ]

    for nnod_mod, av_degrees, nmods in networks:
        for nmod in nmods:
            nnod = nnod_mod*nmod
            for av_degree in av_degrees:
                
                g = random_modular_graph(nnod, nmod, av_degree)

                #Make a "correct" partition for the graph
                part = perfect_partition(nmod,nnod/nmod)

                #Make a random partition for the graph
                part_rand = dict()
                while len(part_rand) <= 1: #check if there is only one module
                    part_rand = rand_partition(g)
                
                #List of modules in the partition
                r_mod=range(len(part))

                #Loop through pairs of modules
                for i in range(1):
                    #select two modules to merge
                    mod_per = np.random.permutation(r_mod)
                    m1 = mod_per[0]; m2 = mod_per[1]
                    

                    #make a graph partition object
                    graph_partition = GraphPartition(g,part)
                    
                    #index of nodes within the original module (before split)
                    n1_init = list(graph_partition.index[m1])
                    n2_init = list(graph_partition.index[m2])
                    n_all_init = n1_init+n2_init

                    #calculate modularity before splitting
                    mod_init = graph_partition.modularity()

                    #merge modules
                    merge_module,e1,a1,delta_energy_meas,type,m1,m2,m2 = graph_partition.compute_module_merge(m1,m2)


                    graph_part2 = copy.deepcopy(graph_partition)
                    graph_part2.apply_module_merge(m1,m2,merge_module,e1,a1)
                    #index of nodes within the modules after merging
                    n_all = list(graph_part2.index[min(m1,m2)])

                    
                    # recalculate modularity after splitting
                    mod_new = graph_part2.modularity()

                    # difference between new and old modularity
                    delta_energy_true = -(mod_new - mod_init)

                    # Test the measured difference in energy against the
                    # function that calculates the difference in energy
                    yield npt.assert_almost_equal(delta_energy_meas, delta_energy_true)
                    # Check that the list of nodes in the two original modules
                    # is equal to the list of nodes in the merged module
                    n_all_init.sort()
                    n_all.sort()
                    yield npt.assert_equal(n_all_init, n_all)
                    
                    # Test that the keys are equivalent after merging modules 
                    yield npt.assert_equal(r_mod[:-1],
                                           sorted(graph_part2.index.keys()))


@parametric 
def test_apply_module_split():
    """Test the GraphPartition operation that splits modules so that it returns
    a change in modularity that reflects the difference between the modularity
    of the new and old parititions. Also test that the module that was split now contains the correct nodes."""

    #nnods = [120, 240, 360]
    #nmods = [2, 3, 4]
    #av_degree=8
    # nnod_mod, av_degrees, nmods
    networks = [ [3, [2], [2, 3, 4]],
                 [4, [2, 3], [2, 4, 6]],
                 [8, [4, 6], [4, 6, 8]] ]
    for nnod_mod, av_degrees, nmods in networks:
        for nmod in nmods:
            nnod = nnod_mod*nmod
            for av_degree in av_degrees:

                g = random_modular_graph(nnod, nmod, av_degree)

                #Make a "correct" partition for the graph
                part = perfect_partition(nmod,nnod/nmod)

                #Make a random partition for the graph
                part_rand = rand_partition(g)

                #List of modules in the partition
                r_mod=range(len(part_rand))

                #Module that we are splitting
                for m in r_mod[::10]:

                    graph_partition = GraphPartition(g,part_rand)

                    #index of nodes within the original module (before split)
                    n_init = list(graph_partition.index[m])

                    #calculate modularity before splitting
                    mod_init = graph_partition.modularity()

                    #assign nodes to 2 groups, take every other one
                    n1=list(graph_partition.index[m])[::2]
                    n2=list(graph_partition.index[m])[1::2]
                    #n_all_old = n1 + n2

                    # split modules
                    #delta_energy_meas = graph_partition.module_split(m,n1,n2)

                    split_modules,e1,a1,delta_energy_meas,type,m,n1,n2 = graph_partition.compute_module_split(m,n1,n2)

                    graph_part2 = copy.deepcopy(graph_partition)
                    graph_part2.apply_module_split(m,n1,n2,split_modules,e1,a1)
                    
                    #index of nodes within the modules after splitting
                    n1_new = list(graph_part2.index[m])
                    n2_new = list(graph_part2.index[len(graph_part2)-1])
                    n_all = n1_new + n2_new

                    # recalculate modularity after splitting
                    mod_new = graph_part2.modularity()

                    # difference between new and old modularity
                    delta_energy_true = -(mod_new - mod_init)

                    # Test that the measured change in energy by splitting a
                    # module is equal to the function output from module_split
                    yield npt.assert_almost_equal(delta_energy_meas,
                                                  delta_energy_true)
                    # Test that the nodes in the split modules are equal to the
                    # original nodes of the module
                    yield npt.assert_equal(n1, n1_new)
                    yield npt.assert_equal(n2, n2_new)
                    n_init.sort()
                    n_all.sort()

                    # Test that the initial list of nodes in the module are
                    # equal to the nodes in m1 and m2 (split modules)
                    yield npt.assert_equal(n_init,n_all)
               

                    
@parametric
def test_apply_node_move():
    """Test the GraphPartition operation that moves a single node so that it
    returns a change in modularity that reflects the difference between the
    modularity of the new and old parititions"""

    # nnod_mod, av_degrees, nmods
    networks = [ [3, [2], [2, 3, 4]],
                 [4, [2, 3], [2, 4, 6]],
                 [8, [4, 6], [4, 6, 8]] ]

    for nnod_mod, av_degrees, nmods in networks:
        for nmod in nmods:
            nnod = nnod_mod*nmod
            for av_degree in av_degrees:
    
                g = random_modular_graph(nnod, nmod, av_degree)

                #Make a "correct" partition for the graph
                part = perfect_partition(nmod,nnod/nmod)
                
                #Make a random partition for the graph
                part_rand = dict()
                while len(part_rand) <= 1: #check if there is only one module
                    part_rand = rand_partition(g)

                #List of modules in the partition
                r_mod=range(len(part_rand))

                #select two modules to change node assignments
                mod_per = np.random.permutation(r_mod)
                m1 = mod_per[0]; m2 = mod_per[1]

                #Make a graph_partition object
                graph_partition = GraphPartition(g,part_rand)
                
                #pick a random node to move between modules m1 and m2
                node_list=list(graph_partition.index[m1])
                nod_per = np.random.permutation(node_list)
                n = nod_per[0]
  
                #list of nodes within the original modules (before node move)
                n1_init = list(nod_per) #list(graph_partition.index[m1])
                n2_init = list(graph_partition.index[m2])
                n1_new = copy.deepcopy(n1_init)
                n2_new = copy.deepcopy(n2_init)

                # calculate modularity before node move
                mod_init = graph_partition.modularity()

                # move node
                #delta_energy_meas = graph_partition.node_update(n,m1,m2)

                node_moved_mods,e1,a1,delta_energy_meas,n,m1,m2 = graph_partition.compute_node_update(n,m1,m2)

                graph_part2 = copy.deepcopy(graph_partition)
                graph_part2.apply_node_update(n,m1,m2,node_moved_mods,e1,a1)
                # remove the first node from m1--because we defined n to equal the first element of the randomized node_list
                n1_new.pop(0)
                
                # append the node to m2
                n2_new.append(n)

                # recalculate modularity after splitting
                mod_new = graph_part2.modularity()
                
                # difference between new and old modularity
                delta_energy_true = -(mod_new - mod_init)
                print delta_energy_meas,delta_energy_true

                # Test that the measured change in energy is equal to the true change in
                # energy calculated in the node_update function
                yield npt.assert_almost_equal(delta_energy_meas, delta_energy_true)
                #yield npt.assert_equal(n1, n1_new)
                #yield npt.assert_equal(n2, n2_new)
                #yield npt.assert_equal(n_init.sort(),n_all.sort())



@parametric
def test_random_mod():
    """ Test the GraphPartition operation that selects random modules to merge
    and split
    XXX not working yet"""
    
    #nnod_mod, av_degrees, nmods
    networks = [ [4, [2, 3], [2, 4, 6]],
                 [8, [4, 6], [4, 6, 8]] ]
            
    n_iter = 100
    for nnod_mod, av_degrees, nmods in networks:
        for nmod in nmods:
            nnod = nnod_mod*nmod
            for av_degree in av_degrees:

                g = random_modular_graph(nnod, nmod, av_degree)
                part = dict()
                while (len(part) <= 1) or (len(part) == nnod):
                    part = rand_partition(g)

                graph_partition = GraphPartition(g,part)

                for i in range(n_iter):
                    graph_partition.random_mod()

                    #check that the partition has > 1 modules
                    true = len(graph_partition)>1
                    yield npt.assert_equal(true,1)

                    #check that the partition has < nnod modules
                    true = len(graph_partition)<nnod
                    yield npt.assert_equal(true,1)

                

def test_random_nod():
    """ Test the GraphPartition operation that selects random nodes to move
    between modules """


@parametric
def test_decide_if_keeping():
    """ Test the function which decides whether or not to keep the new
    partition"""

    dEs = [-.5,-.4,-.3,-.2,-.1,0,.1,.2,.3,.4,.5]
    temperatures = [.01,1,10,100]
    iter = 1000
    tolerance = 1

    for temp in temperatures:
        for dE in dEs:
            keep_list = np.empty(iter)
            for i in range(iter):
                keep_list[i] = float(decide_if_keeping(dE,temp))

            if dE <= 0:
                keep_correct = np.ones(iter)
                yield npt.assert_equal(keep_list,keep_correct)
            else:
                mean_keep = np.mean(keep_list)
                mean_correct = math.exp(-dE/temp)
                yield npt.assert_almost_equal(mean_keep,mean_correct, tolerance)


@parametric
def test_mutual_information():
    """ Test the function which returns the mutual information in two
    partitions"""

    #nnod_mod, av_degrees, nmods
    networks = [ [4, [2, 3], [2, 4, 6]],
                 [8, [4, 6], [4, 6, 8]],
                 [40, [20], [2]] ]

    tolerance = 2

    for nnod_mod, av_degrees, nmods in networks:
        for nmod in nmods:
            nnod = nnod_mod*nmod
            for av_degree in av_degrees:
                #make a graph object
                g = random_modular_graph(nnod, nmod, av_degree)

                #Compute the of nodes per module
                nnod_mod = nnod/nmod
                #Make a "correct" partition for the graph
                ppart = perfect_partition(nmod,nnod_mod)

                #graph_out, mod_array =simulated_annealing(g, temperature =
                #temperature,temp_scaling = temp_scaling, tmin=tmin)

                #test the perfect case for now: two of the same partition
                #returns 1
                mi  = mutual_information(ppart,ppart)
                yield npt.assert_equal(mi,1)

                #move one node and test that mutual_information comes out
                #correctly
                graph_partition = GraphPartition(g,ppart)
                graph_partition.node_update(0,0,1)

                mi2 = mutual_information(ppart,graph_partition.index)
                mi2_correct = mi
                
                yield npt.assert_almost_equal(mi2,mi2_correct,tolerance)

@parametric
def test_rename_keys():
    a = {0:0,1:1,2:2,4:4,5:5}
    rename_keys(a, 3)
    yield npt.assert_equal(a, {0:0,1:1,2:2,3:4,4:5})

    a = {0:0,1:1,3:3,}
    rename_keys(a, 2)
    yield npt.assert_equal(a, {0:0,1:1,2:3})

    a = {0:0,1:1,2:2,3:[]}
    rename_keys(a, 3)
    yield npt.assert_equal(a, {0:0,1:1,2:2})

def betweenness_to_modularity(g,ppart):
    """Function to convert between betweenness fractions and modularity
    Parameters:
    ----------
    g = graph object
    ppart = perfect partition
    
    Returns:
    --------
    mod = best modularity associated with this graph object
    """

    graph_partition = GraphPartition(g,ppart)
    return graph_partition.modularity()
    
    
def danon_test():
    """This test comes from Danon et al 2005. It will create the line plot of Mututal Information vs. betweenness fraction to assess the performance of the simulated annealing algorithm."""
    networks = [[32, [16], [6]]]
    btwn_fracs = [float(i)/100 for i in range(0,80,3)]

    temperature = 0.1
    temp_scaling = 0.9995
    tmin=1e-4

    num_reps = range(1)
    mi_arr=np.empty((len(btwn_fracs),len(num_reps)))

    #keep time
    for rep in num_reps:
        t1 = time.clock()
        for nnod_mod, av_degrees, nmods in networks:
            for nmod in nmods:
                nnod = nnod_mod*nmod
                for av_degree in av_degrees:
                    x_mod = []
                    for ix,btwn_frac in enumerate(btwn_fracs):
                        print 'btwn_frac: ',btwn_frac
                        g = random_modular_graph(nnod, nmod, av_degree,btwn_frac)
                        #Compute the # of nodes per module
                        nnod_mod = nnod/nmod
                        #Make a "correct" partition for the graph
                        ppart = perfect_partition(nmod,nnod_mod)

                        graph_out, graph_dict =simulated_annealing(g,
                        temperature = temperature,temp_scaling = temp_scaling,
                        tmin=tmin, extra_info = True)

                        #print "SA partition",graph_out.index
                        mi = mutual_information(ppart,graph_out.index)
                        t2 = time.clock()
                        print 'Elapsed time: ', (float(t2-t1)/60), ' minutes'
                        print 'partition similarity: ',mi
                        mi_arr[ix,rep] = mi
                        plot_partition(g,graph_out.index,'mi: '+ str(mi),'danon_test_6mod'+str(btwn_frac)+'_graph.png')
                        x_mod.append(betweenness_to_modularity(g,ppart))
                        
                    
                    mi_arr_avg = np.mean(mi_arr,1)
                    plt.figure()
                    plt.plot(btwn_fracs,mi_arr_avg)
                    plt.xlabel('Betweenness fraction')
                    plt.ylabel('Mutual information')
                    plt.savefig('danon_test_6mod/danontest_btwn.png')

                    plt.figure()
                    plt.plot(x_mod,mi_arr_avg)
                    plt.xlabel('Modularity')
                    plt.ylabel('Mutual information')
                    plt.savefig('danon_test_6mod/danontest_mod.png')
    
    #plt.figure()
    #plt.plot(graph_dict['energy'], label = 'energy')
    #plt.plot(graph_dict['temperature'], label = 'temperature')
    #plt.xlabel('Iteration')
    
    return mi_arr

   
    
#@parametric
def SA():
    """ Test the simulated annealing script"""

    
    #nnod_mod, av_degrees, nmods
    #networks = [ [4, [2, 3], [2, 4, 6]]]#,
    #networks =  [ [8, [4, 6], [4, 6, 8]]]
    #networks = [[40, [20], [2]]]
    networks = [[32, [16], [4]]]
    #networks = [[64, [12], [6]]]
    btwn_fracs = [0]
    temperature = 10
    temp_scaling = 0.9995
    tmin=1e-4
    nochange_ratio_min=0.01
    
    #keep time
    
    for nnod_mod, av_degrees, nmods in networks:
        for nmod in nmods:
            nnod = nnod_mod*nmod
            for av_degree in av_degrees:
                for btwn_frac in btwn_fracs:
                    t1=time.clock()
                    g = random_modular_graph(nnod, nmod, av_degree,btwn_frac)
                    #Compute the # of nodes per module
                    nnod_mod = nnod/nmod
                    #Make a "correct" partition for the graph
                    ppart = perfect_partition(nmod,nnod_mod)

                    graph_out, energy_array, rej_array, temp_array =simulated_annealing(g,
                    temperature = temperature,temp_scaling = temp_scaling,
                    tmin=tmin, nochange_ratio_min = nochange_ratio_min)

                    print "perfect partition", ppart
                    print "SA partition",graph_out.index
                    
                    t2 = time.clock()
                    print 'Elapsed time: ', float(t2-t1)/60, 'minutes'
                    print 'partition similarity: ',mutual_information(ppart,graph_out.index)
                    return graph_out, g, energy_array, rej_array, ppart, temp_array
    

#-----------------------------------------------------------------------------
# Main script
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    if 0:
        mi_arr = danon_test()  
    if 0:
        apply_module_merge()
        graph_out,g_out, energy_array,rej_array,ppart, temp_array = SA()
        plt.figure()
        plt.plot(temp_array)
        plt.title('Temp_array')
        plt.figure()
        plt.plot(energy_array)
        plt.title('Energy Array')
        plt.show()

        plot_partition(g_out,graph_out.index)
    
    plt.show()
