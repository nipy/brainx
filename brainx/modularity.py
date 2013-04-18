# encoding: utf-8
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
import copy

# Third-party modules
import networkx as nx
import numpy as np
import numpy.testing as npt
import numpy.linalg as nl
import scipy.linalg as sl

from matplotlib import pyplot as plt

# Our own modules
import util

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

        if self.num_edges == 0:
            raise ValueError("TODO: Cannot create a graph partition of only one node.")

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
            if np.isnan(mod_e[m]) or np.isnan(mod_a[m]):
                1/0

        return mod_e, mod_a

    def modularity_newman(self):
        """ Function using other version of expressing modularity, from the
        Newman papers (2004 Physical Review)

        Parameters:
        g = graph
        part = partition

        Returns:
        mod = modularity
        """
        if np.isnan((np.array(self.mod_e) - (np.array(self.mod_a)**2)).sum()):
            1/0
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
          m1: name (i.e., index) of one module
          m2: name (i.e., index) of the other module

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
          m1: name (i.e., index) of one module
          m2: name (i.e., index) of the other module
          merged_module: set of all nodes from m1 and m2
          e_new: mod_e of merged_module
          a_new: mod_a of merged_module

        Returns
        -------
          Does not return anything -- operates on self.mod_e and self.mod_a in
          place
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

        # create a dict that contains the new modules 0 and 1 that have the
        # sets n1 and n2 of nodes from module m.
        split_modules = {0: n1, 1: n2}

        #make an empty matrix for computing "modularity" level values
        e1 = [0,0]
        a1 = [0,0]
        e0, a0 = self.mod_e, self.mod_a

        # The values that change: _edge_info with arguments will update the e,
        # a vectors only for the modules in index
        e1, a1 = self._edge_info(e1, a1, split_modules)

        # Compute the change in modularity
        delta_q =  ( (e1[0]-a1[0]**2) + (e1[1]- a1[1]**2) ) - (e0[m]-a0[m]**2)

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

        #EN: Not sure if this is necessary, but sometimes it finds a partition
        #with an empty module...
        ## CG: should not be necessary... but may mess things up by renaming
        #keys in dictionary but not updating mod_e and mod_a.  Maybe we should
        #take care of this case earlier to ensure that it can not happen?
        #Otherwise need to create a new function to update/recompute mod_e and
        #mod_a.

        # This checks whether there is an empty module. If so, renames the keys.
        #if len(self.index[m1])<1:
        #    self.index.pop(m1)
        #    rename_keys(self.index,m1)
        # This checks whether there is an empty module. If so, renames the keys.
        #if len(self.index[m2])<1:
        #    self.index.pop(m2)
        #    rename_keys(self.index,m2)

        # For now, rather than renumbering, just check if this ever happens
        if len(self.index[m1])<1:
            raise ValueError('Empty module after module split, old mod')
        if len(self.index[m2])<1:
            raise ValueError('Empty module after module split, new mod')

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
            raise ValueError('Empty module after node move')
            #self.index.pop(m1)
            #rename_keys(self.index,m1)
            #if m1 < m2:
            #    m2 = m2 - 1 #only need to rename this index if m1 is before m2

        self.mod_e[m1] = e_new[0]
        self.mod_a[m1] = a_new[0]
        self.mod_e[m2] = e_new[1]
        self.mod_a[m2] = a_new[1]

        return m2

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

        if num_mods >= self.num_nodes-1: ### CG: why are we subtracting 1 here?
            coin_flip = 1 #always merge if each node is in a separate module
        elif num_mods <= 2: ### Why 2 and not 1?
            coin_flip = 0 #always split if there's only one module
        else:
            coin_flip = np.random.random()


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

                #reselect the first module
                rand_mods = np.random.permutation(range(num_mods))
                m1 = rand_mods[0]
                #m1 = random.randint(0,num_mods)
                ### CG: why not just work your way through the list?

            n1,n2 = self.determine_node_split(m1)

            #We may want to return output of merging/splitting directly, but
            #for now we're returning inputs for those modules.

            return self.compute_module_split(m1,n1,n2)

    def determine_node_split(self,m1):
        """ Determine hwo to split notes within a module
        """

        # list of nodes within that module
        list_nods = list(self.index[m1])

        # randomly partition the list of nodes into 2
        nod_split_ind = np.random.randint(1,len(list_nods)) #can't pick the first node as the division

        ### CG: but it's ok to put up to the last because
        ## np.random.randint is exclusive on the second number
        n1 = set(list_nods[:nod_split_ind]) #at least 1 large
        n2 = set(list_nods[nod_split_ind:]) #at least 1 large

        return n1,n2


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
    # within-module (p_in) and between-modules (p_out).  See ﻿[1] L. Danon,
    # A. Díaz-Guilera, J. Duch, and A. Arenas, “Comparing community structure
    # identifcation,” Journal of Statistical Mechanics: Theory and Experiment,
    # 2005. for definitions of these quantities.
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
    mask = util.diag_stack(blocks)

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
    adj[np.logical_and(-p_in < mat, mat <= 0)] = 1
    adj[np.logical_and(0 < mat, mat < p_out)] = 1

    # no self-links
    util.fill_diagonal(adj, 0)

    # Our return object is a graph, not the adjacency matrix
    return nx.from_numpy_matrix(adj)


def rename_keys(dct, key):
    """This function reads in a partition and a single module to be
    removed,pops out the value(s) and shifts the key names accordingly.

    Parameters
    ----------
    dct : dict
      Input dict with all integer keys.

    key : int
      Key after which all other keys are downshifted by one.

    Returns
    -------
    None.  The input dict is modified in place.
    """

    for m in range(key, len(dct)):
        try:
            dct[m] = dct.pop(m+1)
        except KeyError:
            # If we can't pop a key, it's simply missing from the dict and we
            # can safely ignore it.  This is likely to happen at the edge of
            # the dict, if the function is called on the last key.
            pass


def rand_partition(g, num_mods=None):
    """This function takes in a graph and returns a dictionary of labels for
    each node. Eventually it needs to be part of the simulated annealing
    program, but for now it will just make a random partition.

    Parameters
    ----------
    g : graph
      Graph for which the partition is to be computed.

    num_mods : optional, int
      If given, the random partition will have these many modules.  If not
      given, the number of modules in the partition will be chosen as at
      random, up to the number of nodes in the graph."""

    num_nodes = g.number_of_nodes()

    # randomly select a number of modules
    if num_mods is None:
        num_mods = np.random.randint(1, num_nodes)

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


    write_labels = False
    nnod = g.number_of_nodes()

    if nod_labels == None:
        nod_labels = dict(zip(range(nnod),range(nnod)))
    else:
        nod_labels = dict(zip(range(nnod),nod_labels))


    plt.figure()
    plt.subplot(111)
    plt.axis('off')

    if pos == None:
        pos=nx.circular_layout(g)

    #col=colors.cnames.keys()
    col = ['r','g','b','m','c','y']
    col2 = ['#000066','#000099','#660000','#CC6633','#FF0099','#FF00FF','#33FFFF','#663366','#FFCC33','#CCFF66','#FFCC99','#33CCCC','#FF6600','#FFCCFF','#CCFFFF','#CC6699','#CC9900','#FF6600','#99FF66','#CC0033','#99FFFF','#CC00CC','#CC99CC','#660066','#33CC66','#336699','#3399FF','#339900','#003300','#00CC00','#330033','#333399','#0033CC','#333333','#339966','#333300']

    niter = 0
    edge_list_between = []
    for m,val in part.iteritems():

        if niter <len(col):
            if within_mod == 'none': #note: assumes part_coeff also there
                for v in val:
                    if les_dam != 'none':
                        plt.scatter(pos[v][0],pos[v][1],s=100*les_dam[v],c='orange',marker=(10,1,0))

                nx.draw_networkx_nodes(g,pos,nodelist=list(val),node_color=col[niter],node_size=50)
            else:
                for v in val:
                    if les_dam != 'none':
                        plt.scatter(pos[v][0],pos[v][1],s=500*les_dam[v],c='orange',marker=(10,1,0))

                    if within_mod[v] > 1:
                        nx.draw_networkx_nodes(g,pos,nodelist=[v],node_color=col[niter],node_size=part_coeff[v] * 500+50,node_shape='s',linewidths=2)
                    else:
                        nx.draw_networkx_nodes(g,pos,nodelist=[v],node_color=col[niter],node_size=part_coeff[v] * 500+50,node_shape='o',linewidths=0.5)

        else:
            #print 'out of colors!!'
            if within_mod == 'none': #note: assumes part_coeff also there
                for v in val:
                    if les_dam != 'none':
                        plt.scatter(pos[v][0],pos[v][1],s=100*les_dam[v],c='orange',marker=(10,1,0))

                nx.draw_networkx_nodes(g,pos,nodelist=list(val),node_color=col2[niter],node_size=50)
            else:
                for v in val:
                    if les_dam != 'none':
                        plt.scatter(pos[v][0],pos[v][1],s=500*les_dam[v],c='orange',marker=(10,1,0))

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


    #nx.draw_networkx_edges(g,pos,edgelist=nx.edges(g))
    nx.draw_networkx_edges(g,pos,edgelist=edge_list_between,edge_color='k')
    if write_labels:
        nx.draw_networkx_labels(g,pos,nod_labels,font_size=6)

    #add loop for damage labels
    if les_dam != 'none':
        for m,val in part.iteritems():
            for v in val:
                if les_dam[v] > 0:
                    plt.scatter(pos[v][0],pos[v][1],s=500*les_dam[v]+100,c='orange',marker=(10,1,0))

    plt.title(title)
    #plt.savefig(fname)
    #plt.close()
    #plt.show()


def confusion_matrix(d1, d2):
    """Return the confusion matrix for two graph partitions.

    See Danon et al, 2005, for definition details.

    Parameters
    ----------
    d1 : dict
      dictionary with first partition.
    d2 : dict
      dictionary with second partition.

    Returns
    -------
    N : numpy 2d array.
      Confusion matrix for d1 and d2.
    """
    # define a 'confusion matrix' where rows = 'real communities' and columns =
    # 'found communities' The element of N (Nij) = the number of nodes in the
    # real community i that appear in the found community j

    # Compute the sets of the values of d1/d2 only once, to avoid quadratic
    # recomputation.
    rows = len(d1)
    cols = len(d2)

    sd1 = [set(d1[i]) for i in range(rows)]
    sd2 = [set(d2[j]) for j in range(cols)]

    N = np.empty((rows,cols))
    for i, sd1i in enumerate(sd1):
        for j, sd2j in enumerate(sd2):
            N[i,j] = len(sd1i & sd2j)

    return N


def mutual_information(d1, d2):
    """Mutual information between two graph partitions.

    Read in two dictionaries of sets (i.e. a graph partition) and assess how
    similar they are using mutual information as in Danon, Diaz-Guilera, Duch &
    Arenas, J Statistical Mechanics 2005.

    Parameters
    ----------
    d1 : dict
      dictionary of 'real communities'
    d2 : dict
      dictionary of 'found communities'

    Returns
    -------
    mi : float
      Value of mutual information between the two partitions.
    """
    log = np.log
    nansum = np.nansum

    N = confusion_matrix(d1, d2)

    nsum_row = N.sum(0)[np.newaxis, :]
    nsum_col = N.sum(1)[:, np.newaxis]

    # Sanity checks: a zero in either of these can only happen if there was an
    # empty module  in one of the input partitions.  Rather than manually check
    # the entire partitions, we look for this problem at this stage, and bail
    # if there was an empty module.
##     if (nsum_row==0).any():
##         raise ValueError("Empty module in second partition.")
##     if (nsum_col==0).any():
##         raise ValueError("Empty module in first partition.")

    # nn is the total number of nodes
    nn = nsum_row.sum()
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
        return np.random.random() < math.exp(-dE/temperature)


def simulated_annealing(g, p0=None, temperature = 50, temp_scaling = 0.995, tmin=1e-5,
                        bad_accept_mod_ratio_max = 0.8 ,
                        bad_accept_nod_ratio_max = 0.8, accept_mod_ratio_min =
                        0.05, accept_nod_ratio_min = 0.05,
                        extra_info = False,
                        debug = False):

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
    part = dict()
    #check if there is only one module or nnod modules
    while (len(part) <= 1) or (len(part) == nnod):
        part = rand_partition(g)

    # make a graph partition object
    if p0 is None:
        graph_partition = GraphPartition(g,part)
    else:
        graph_partition = p0

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

                if debug:
                    debug_partition = GraphPartition(g, graph_partition.index)
                    npt.assert_almost_equal(debug_partition.modularity(),
                                            graph_partition.modularity(), 11)
                    for mod in graph_partition.index:
                        if len(graph_partition.index[mod]) < 1:
                            raise ValueError('Empty module after module %s,SA' % (movetype))


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
                # Only compute this quantity after enough steps for these
                # ratios to make any sense (they are ~1 at the first step).
                bad_accept_mod_ratio =  float(bad_accept_mod)/(count_bad_mod)
                accept_mod_ratio = float(accept_mod)/(count_mod)
                #print 'ba_mod_r', bad_accept_mod_ratio  # dbg
                if (bad_accept_mod_ratio > bad_accept_mod_ratio_max) \
                        or (accept_mod_ratio < accept_mod_ratio_min):
                    #print 'MOD BREAK'
                    break

            # Second loop over node changes
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

                    nnn = graph_partition.apply_node_update(p1,p2,p3,calc_dict,e_new,a_new)
                    energy += delta_energy
                    accept_nod += 1
                    if delta_energy > 0 :
                        bad_accept_nod += 1
                    #maybe store the best one here too?
                    #graph_partition.store_best()
                    if debug:
                        debug_partition = GraphPartition(g,
                                                         graph_partition.index)
                        npt.assert_almost_equal(debug_partition.modularity(),
                                                graph_partition.modularity(), 11)


                        for mod in graph_partition.index:
                            if len(graph_partition.index[mod]) < 1:
                                raise ValueError('Empty module after ndoe move,SA')


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


    if debug:
        debug_partition = GraphPartition(g, graph_part_final.index)
        npt.assert_almost_equal(debug_partition.modularity(),
                                graph_part_final.modularity(), 11)

        for mod in graph_part_final.index:
            if len(graph_part_final.index[mod]) < 1:
                raise ValueError('LAST CHECK: Empty module after module %s,SA' % (movetype))

    if extra_info:
        extra_dict = dict(energy = energy_array, temp = temp_array)
        #return graph_partition, extra_dict
        return graph_part_final, extra_dict
    else:
        #return graph_partition
        #print graph_part_final.modularity()

        #check that the energy matches the computed modularity value of the partition
        finalmodval = graph_part_final.modularity()
        print finalmodval
        print -energy_best
        print graph_part_final.index

        print np.abs(finalmodval - (-energy_best))

        if np.abs(finalmodval - (-energy_best)) > 0.000001: #to account for float error
            raise ValueError('mismatch in energy and modularity')

        return graph_part_final,graph_part_final.modularity()


def modularity_matrix(g):
    """Modularity matrix of the graph.

    The eigenvector corresponding to the largest eigenvalue of the modularity
    matrix is analyzed to assign clusters.

    """
    A = np.asarray(nx.adjacency_matrix(g))
    k = np.sum(A, axis=0)
    M = np.sum(k) # 2x number of edges

    return A - ((k * k[:, None]) / float(M))


def newman_partition(g, max_div=np.inf):
    """Greedy estimation of optimal partition of a graph, using
    Newman (2006) spectral method.

    Parameters
    ----------
    g : NetworkX Graph
        Input graph.
    max_div : int
        Maximum number of times to sub-divide partitions.

    Returns
    -------
    p : GraphPartition
        Estimated optimal partitioning.

    """
    A = np.asarray(nx.adjacency_matrix(g))
    k = np.sum(A, axis=0)
    M = np.sum(A) # 2x number of edges
    B = modularity_matrix(g)

    p = range(len(g))

    def _divide_partition(p, max_div=np.inf):
        """
        Parameters
        ----------
        p : array of ints
            Node labels.
        B : ndarray
            Modularity matrix.

        Returns
        -------
        pp, qq : list of ints
            Partitioning of node labels.  If the partition is indivisible, then
            only `pp` is returned.

        """
        p = np.asarray(p)

        if max_div <= 0 or p.size == 1:
            return [p]

        # Construct the subgraph modularity matrix
        A_ = A[p, p[:, None]]
        k_ = np.sum(A_, axis=0)
        M_ = np.sum(k_)

        B_ = B[p, p[:, None]]
        B_ = B_ - np.diag(k_ - k[p] * M_ / float(M))

#        w, v = nl.eigh(B_)
        w, v = sl.eigh(B_, eigvals=(len(B_) - 2, len(B_) - 1))

        # Find the maximum eigenvalue of the modularity matrix
        # If it is smaller than zero, then we won't be able to
        # increase the modularity any further by partitioning.
        n = np.argsort(w)[-1]
        if w[n] <= 0:
            return [p]

        # Construct the partition vector s, that has value -1 corresponding
        # to nodes in the first partition and 1 for nodes in the second
        v_max = v[:, n]
        mask = (v_max < 0)
        s = np.ones_like(v_max)
        s[mask] = -1

        # Compute the increase in modularity due to this partitioning.
        # If it is less than zero, we should rather not have partitioned.
        q = s[None, :].dot(B_).dot(s)
        if q <= 0:
            return [p]

        # Make the partitioning, and subdivide each
        # partition in turn.

        out = []
        for pp in (p[mask], p[~mask]):
            out.extend(_divide_partition(pp, max_div - 1))

        return out

    p = _divide_partition(p, max_div)

    index = {}
    for k, nodes in enumerate(p):
        index[k] = set(nodes)

    return GraphPartition(g, index)


def adjust_partition(g, partition, max_iter=None):
    """Adjust partition, using the heuristic method described in Newman (2006),
    to have higher modularity.

    Parameters
    ----------
    g : NetworkX graph
        Input graph.
    partition : GraphPartition
        Existing partitioning.
    max_iter : int, optional
        Maximum number of improvement iterations.  By default,
        continue until 10 iterations without any improvement.

    Returns
    -------
    improved_partition : GraphPartition
        Partition with higher modularity.

    """
    nodes = g.nodes()
    P = set(range(len(partition)))

    node_map = {}
    for p in P:
        for node in partition.index[p]:
            node_map[node] = p

    L = len(nodes)
    no_improvement = 0
    iterations = 0
    max_iter = max_iter or np.inf
    best_modularity = partition.modularity()

    while nodes and no_improvement < 10 and iterations <= max_iter:
        moves = []
        move_modularity = []
        iterations += 1

        for n in nodes:
            for p in P.difference([node_map[n]]):
                moves.append((n, node_map[n], p))
                M = -partition.compute_node_update(n, node_map[n], p)[3]
                move_modularity.append(M)

        (n, p0, p1) = moves[np.argmax(move_modularity)]
        split_modules, e_new, a_new = partition.compute_node_update(n, p0, p1)[:3]
        partition.apply_node_update(n, p0, p1, split_modules, e_new, a_new)
        node_map[n] = p1
        nodes.remove(n)

        print '[%d/%d] -> %.4f' % (len(nodes), L, partition.modularity())

        M = partition.modularity()
        if M > best_modularity:
            gp_best = copy.deepcopy(partition)
            best_modularity = M
            no_improvement = 0
        else:
            no_improvement += 1

    return gp_best
