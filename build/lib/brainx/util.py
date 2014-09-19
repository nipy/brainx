"""Generic utilities that may be needed by the other modules.
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import warnings
import numpy as np
import networkx as nx

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def dictset_to_listset(dict_set):
    """ converts a dict of sets to a list of sets
    for converting partition.community objects"""
    if  isinstance(dict_set, dict) \
        and _contains_only(dict_set, set):
        return dict_set.values()
        
    raise ValueError('{0} is not a dict of sets'.format(dict_set))
   
def listset_to_dictset(list_set):
    """ converts a list of sets to a dict of sets
    for converting partition.community objects"""
    ## check input is dict of sets
    if isinstance(list_set, list) and \
        _contains_only(list_set, set):
        return {val: value for val, value in enumerate(list_set)}
    raise ValueError('{0} is not a list of sets'.format(list_set))

def _no_repeats_in_listlist(list_list):
    """ checks for duplicates in list of lists
    returns True or False"""
    if isinstance(list_list, list) and \
        _contains_only(list_list, list):
        allitems = [item for sublist in list_list for item in sublist]
        return len(allitems) == len(set(allitems))
    raise ValueError('{0} is not a list of lists'.format(list_list))

def _contains_only(container, type):
    """check that contents of a container are all of the same type"""
    try:
        container = container.values()  # dict
    except AttributeError:
        pass

    return all(isinstance(s, type) for s in container)

def listlist_to_listset(list_list):
    """ converts list of lists to a list of sets (with check)
    for converting partition.community objects"""
    if _no_repeats_in_listlist(list_list):
        return [set(x) for x in list_list] 
    else:
        raise ValueError('found duplicate(s) in {0}, cannot validly format to '\
            'list of sets'.format(list_list))

def slice_data(data, sub, block, subcond=None):
    """ pull symmetric matrix from data block (4D or 5D)
    
    Parameters
    ----------
    data : numpy array
        4D array (block, sub, nnode, nnode)
        5D array (subcond, block, sub, nnode, nnode)
    sub : int
        int representing subject to index in data
    block : int
        int representing block to index in data
    subcond : int
        int representing optional subcondition from 5D array

    Returns
    -------
    adjacency_matrix : numpy array
        symmetric numpy array (innode, nnode)
    """
    if subcond is None:
        return data[block, sub]
    return data[subcond, block, sub]


def format_matrix(data, s, b, lk, co, idc=[], costlist=[],
        nouptri=False, asbool=True):
    """ Function which thresholds the adjacency matrix for a particular 
    subject and particular block, using lookuptable to find thresholds, 
    cost value to find threshold, costlist
    (thresholds, upper-tris it) so that we can use it with simulated annealing

    Parameters
    -----------
    data : full data array 4D (block, sub, node, node)
    s : int
        subject
    b : int
        block
    lk : numpy array
        lookup table for study
    co : int
        cost value to threshold at
    idc : int
        index of ideal cost
    costlist : list
        list (size num_edges) with ordered values used to find 
        threshold to control number of edges
    nouptri : bool
        if False only keeps upper tri, True yields symmetric matrix
    asbool : bool
        if True return boolean mask, otherwise returns thesholded
        weight matrix
    """
    cmat = slice_data(data, s, b)
    th = cost2thresh(co,s,b,lk,idc,costlist) #get the right threshold
    cmat = thresholded_arr(cmat,th,fill_val=0)
    if not nouptri:
        cmat = np.triu(cmat,1)
    if asbool:
        return ~(cmat == 0)
    return cmat


def format_matrix2(data, s, sc, c, lk, co, idc=[],
        costlist=[], nouptri=False, asbool=True):
    """ Function which formats matrix for a particular subject and 
    particular block (thresholds, upper-tris it) so that we can 
    make a graph object out of it

    Parameters
    ----------
    data : numpy array
        full data array 5D (subcondition, condition, subject, node, node) 
    s : int
        index of subject
    sc : int
        index of sub condition
    c : int
        index of condition
    lk : numpy array
        lookup table for thresholds at each possible cost
    co : float
        cost value to threshold at
    idc : float
        ideal cost 
    costlist : list
        list of possible costs
    nouptri : bool
        False zeros out diag and below, True returns symmetric matrix
    asbool : bool
        If true returns boolean mask, otherwise returns thresholded w
        weighted matrix
    """
    cmat = slice_data(data, s, c, sc) 
    th = cost2thresh2(co,s,sc,c,lk,[],idc,costlist) #get the right threshold
    cmat = thresholded_arr(cmat,th,fill_val=0)
    if not nouptri:
        cmat = np.triu(cmat,1)
    if asbool:
        # return boolean mask
        return ~(cmat == 0)
    return cmat

def threshold_adjacency_matrix(adj_matrix, cost, uptri=False):
    """threshold adj_matrix at cost
    
    Parameters
    ----------
    adj_matrix : numpy array
        graph adjacency matrix
    cost : float
        user specified cost
    uptri : bool
        False returns symmetric matrix, True zeros out diagonal and below
    Returns
    -------
    thresholded : array of bools
        binary matrix thresholded to result in cost
    expected_cost : float
        the real cost value (closest to cost)
    """
    nnodes, _ = adj_matrix.shape
    ind = np.triu_indices(nnodes, 1)
    nedges = adj_matrix[ind].shape[0]
    lookup = make_cost_thresh_lookup(adj_matrix)
    cost_index = np.round(cost * float(nedges))
    thresh, expected_cost, round_cost = lookup[cost_index]
    adj_matrix = adj_matrix > thresh #threshold matrix
    np.fill_diagonal(adj_matrix, 0) #zero out diagonal
    if uptri: #also zero out below diagonal
        adj_matrix = np.triu(adj_matrix) 
    return adj_matrix, expected_cost 

def find_true_cost(boolean_matrix):
    """ when passed a boolean matrix, presumably from thresholding to 
    achieve a specific cost, this calculates the actual cost for 
    this thresholded array"""
    ind = np.triu_indices_from( boolean_matrix, 1)
    alledges = np.array(boolean_matrix)[ind].shape[0]
    found_edges = boolean_matrix[ind].sum()
    return float(found_edges) / alledges

def all_positive(adjacency_matrix):
    """ checks if edge values in adjacency matrix are all positive
    or positive and negative 
    Returns
    -------
    all_positive : bool
        True if all values are >=0
        False if values < 0
    """
    # add 1 so 0-> 1(True) , -1 -> 0 False
    signs = set( np.sign(adjacency_matrix) + 1 )
    return bool(sorted(signs)[0])


def make_cost_thresh_lookup(adjacency_matrix):
    """takes upper triangular (offset 1, no diagonal) of summetric 
    adjacency matrix, sorts (lowest -> highest)
    Returns
    -------
    lookup : numpy record array
        shape = number of edges
        'weight' is sorted weight values (largest -> smallest)
        'actual_cost' is cost at each weight (smallest -> largest)
        'cost' is 'actual_costs' rounded to two decimal points
    Example
    -------
    lookup = make_cost_thresh_lookup(adj_mat)
    lookup[100] 
      (0.3010111736597483, 0.704225352112676, 0.7)
    lookup[100].weight
       0.3010111736597483
    lookup[100].actual_cost
       0.704225352112676
    lookup[100].cost
       0.70

    """
    ## check for nan in matrix, sorting will behave badly if nan found
    if np.any(np.isnan(adjacency_matrix)):
        raise ValueError('NAN found in adjacency matrix, this will cause'\
                'improper behavior in sorting and improper results, '\
                'please remove all nan ')
    ind = np.triu_indices_from(adjacency_matrix, k = 1)
    edges = adjacency_matrix[ind]
    nedges = edges.shape[0]
    lookup = np.recarray((nedges), dtype = [('weight', float), 
                                            ('actual_cost', float), 
                                            ('cost', float)])
    lookup['weight'] = sorted(edges, reverse = True)
    lookup['actual_cost'] = np.arange(nedges) / float(nedges)
    lookup['cost'] = np.round(lookup['actual_cost'], decimals = 2)
    return lookup

def cost_size(nnodes):
    """create a list of actual costs, tot_edges, edges_short
    given a fixed number of nodes"""
    warnings.warn('this is no longer used: use make_cost_array')
     
    tot_edges = 0.5 * nnodes * (nnodes - 1)
    costs = np.array(range(int(tot_edges) + 1), dtype=float) / tot_edges
    edges_short = tot_edges / 2
    return costs, tot_edges, edges_short


def make_cost_array(n_nodes, cost=0.5):
    """Make cost array of length cost * (the number of possible edges).

    Parameters
    ----------
    n_nodes: integer
        Number of nodes in the graph.

    cost: float, optional
        Value between 0 and 1 (0.5 by default).  The length of
        cost_array will be set to cost * tot_edges.

    Returns
    -------
    cost_array: numpy array
        N+1-length array of costs, with N the number of possible
        undirected edges in the graph.  The costs range from 0 to 1 and
        are equally-spaced.

    tot_edges: float
        Number of possible undirected edges in the graph.

    Notes
    -----
    This is an edited version of the former function cost_size.

    """
    tot_edges = 0.5 * n_nodes * (n_nodes - 1)
    costs = np.array(range(int(tot_edges * cost)), dtype=float) / tot_edges
    return costs, tot_edges
    
def metrics_to_pandas():
    """docstring for metrics_to_pandas"""
    pass

def store_metrics(b, s, co, metd, arr):
    """Store a set of metrics into a structured array
    b = block
    s = subject
    co = cost? float
    metd = dict of metrics
    arr : array?"""

    if arr.ndim == 3:
        idx = b,s,co
    elif arr.ndim == 4:
        idx = b,s,co,slice(None)
    else:
        raise ValueError("only know how to handle 3 or 4-d arrays")
    
    for met_name, met_val in metd.iteritems():
        arr[idx][met_name] = met_val
    

def regular_lattice(n,k):
    """Return a regular lattice graph with n nodes and k neighbor connections.

    This graph consists of a ring with n nodes which then get connected to
    their k (k-1 if k is odd) nearest neighbors.

    This type of graph is the starting point for the Watts-Strogatz small-world
    model, where connections are then rewired in a second phase.

    XXX TODO Use as comparison, check networkx to see if its update worth redundancy
    """
    # Code simplified from the networkx.watts_strogatz_graph one
    G = nx.Graph()
    G.name="regular_lattice(%s,%s)"%(n,k)
    nodes = range(n) # nodes are labeled 0 to n-1
    # connect each node to k/2 neighbors
    for j in range(1, k/2+1):
        targets = nodes[j:] + nodes[:j] # first j nodes are now last in list
        G.add_edges_from(zip(nodes,targets))
    return G


def compile_data(input,tmslabel,mat_type,scale,data_type):
    """This function reads in data into a text file"""
    filename='Mean_'+data_type+'_'+tmslabel+'_'+mat_type+scale+'.txt'
    f=open(filename,'a')
    for i in range(0,len(input)):
        f.write('%s\t' %input[i])
    f.write('\n')
    f.close()


def arr_stat(x,ddof=1):
    """Return (mean,stderr) for the input array"""
    m = x.mean()
    std = x.std(ddof=ddof)
    return m,std


def threshold_arr(cmat, threshold=0.0, threshold2=None):
    """Threshold values from the input matrix.

    Parameters
    ----------
    cmat : array_like
        An array of numbers.
    
    threshold : float, optional
        If threshold2 is None, indices and values for elements of cmat
        greater than this value (0 by default) will be returned.  If
        threshold2 is not None, indices and values for elements of cmat
        less than this value (or greater than threshold2) will be
        returned.
      
    threshold2 : float, optional
        Indices and values for elements of cmat greater than this value
        (or less than threshold) will be returned.  By default,
        threshold2 is set to None and not used.

    Returns
    -------
    A tuple of length N + 1, where N is the number of dimensions in
    cmat.  The first N elements of this tuple are arrays with indices in
    cmat, for each dimension, corresponding to elements greater than
    threshold (if threshold2 is None) or more extreme than the two
    thresholds.  The last element of the tuple is an array with the
    values in cmat corresponding to these indices.
    
    Examples
    --------
    >>> a = np.linspace(0, 0.8, 7)
    >>> a
    array([ 0.    ,  0.1333,  0.2667,  0.4   ,  0.5333,
            0.6667,  0.8   ])
    >>> threshold_arr(a, 0.3)
    (array([3, 4, 5, 6]),
     array([ 0.4   ,  0.5333,  0.6667,  0.8       ]))

    With two thresholds:
    >>> threshold_arr(a, 0.3, 0.6)
    (array([0, 1, 2, 5, 6]),
     array([ 0.        ,  0.1333,  0.2667,  0.6667, 0.8       ]))

    """
    # Select thresholds.
    if threshold2 is None:
        th_low = -np.inf
        th_hi  = threshold
    else:
        th_low = threshold
        th_hi  = threshold2
    # Mask out the values we are actually going to use.
    idx = np.where((cmat < th_low) | (cmat > th_hi))
    vals = cmat[idx]
    return idx + (vals,)


def thresholded_arr(arr, threshold=0.0, threshold2=None, fill_val=np.nan):
    """Threshold values from the input matrix and return a new matrix.

    Parameters
    ----------
    arr : array_like
        An array of numbers.
    
    threshold : float, optional
        If threshold2 is None, elements of arr less than this value (0
        by default) will be filled with fill_val.  If threshold2 is not
        None, elements of arr greater than this value but less than
        threshold2 will be filled with fill_val.
      
    threshold2 : float, optional
        Elements of arr less than this value but greater than threshold
        will be filled with fill_val.  By default, high_thresh is set to
        None and not used.

    fill_val : float or numpy.nan, optional
        Value (np.nan by default) with which to fill elements below
        threshold or between threshold and threshold2.

    Returns
    -------
    a2 : array_like
        An array with the same shape as arr, but with values below
        threshold or between threshold and threshold2 replaced with
        fill_val.

    Notes
    -----
    arr itself is not altered.

    """
    a2 = np.empty_like(arr)
    a2.fill(fill_val)
    mth = threshold_arr(arr, threshold, threshold2)
    idx,vals = mth[:-1], mth[-1]
    a2[idx] = vals
    return a2


def normalize(arr,mode='direct',folding_edges=None):
    """Normalize an array to [0,1] range.

    By default, this simply rescales the input array to [0,1].  But it has a
    special 'folding' mode that allong absolute value of all values, in addition
    values between the folding_edges (low_cutoff, high_cutoff)  will be zeroed.

    Parameters
    ----------
    arr : 1d array
        assumes dtype == float, if int32, will raise ValueError

    mode : string, one of ['direct','folding']
        if direct rescale all values (pos and neg) between 0,1
        if folding, zeros out values between folding_values (inclusive)
        and normalizes absolute value of remaining values

    folding_edges : (float,float)
        (low_cutoff, high_cutoff) lower and upper values to zero out
        (values are inclusive)
        Only needed for folding mode, ignored in 'direct' mode.

    Examples
    --------
    >>> np.set_printoptions(precision=4)  # for doctesting
    >>> a = np.linspace(0.3,0.8,7)
    >>> normalize(a)
    array([ 0.    ,  0.1667,  0.3333,  0.5   ,  0.6667,  0.8333,  1.    ])
    >>> 
    >>> b = np.concatenate([np.linspace(-0.7,-0.3,4),
    ...                     np.linspace(0.3,0.8,4)] )
    >>> b
    array([-0.7   , -0.5667, -0.4333, -0.3   ,  0.3   ,  0.4667,  0.6333,  0.8   ])
    >>> normalize(b,'folding',[-0.3,0.3])
    array([ 0.8   ,  0.5333,  0.2667,  0.    ,  0.    ,  0.3333,  0.6667,  1.    ])
    >>> 
    >>> 
    >>> c = np.concatenate([np.linspace(-0.8,-0.3,4),
    ...                     np.linspace(0.3,0.7,4)] )
    >>> c
    array([-0.8   , -0.6333, -0.4667, -0.3   ,  0.3   ,  0.4333,  0.5667,  0.7   ])
    >>> normalize(c,'folding',[-0.3,0.3])
    array([ 1.    ,  0.7917,  0.5833,  0.    ,  0.    ,  0.5417,  0.7083,  0.875   ])
    """
    if mode == 'direct':
        return rescale_arr(arr,0,1)
    elif mode == 'folding':
        # cast folding_edges to floats in case inputs are ints
        low_cutoff, high_cutoff = [float(x) for x in folding_edges]
        amin, amax = arr.min(), arr.max()
        low_diff, high_diff = low_cutoff-amin, amax-high_cutoff 
        if low_diff < 0 or high_diff < 0:
            raise ValueError("folding edges must be within array range")
        mask = np.logical_and( arr >= low_cutoff, arr <= high_cutoff)
        out = arr.copy()
        out[mask] = 0
        return rescale_arr(np.abs(out), 0, 1)
    else:
        raise ValueError('Unknown mode %s: valid options("direct", "folding")')

def mat2graph(cmat,threshold=0.0,threshold2=None):
    """Make a weighted graph object out of an adjacency matrix.

    The values in the original matrix cmat can be thresholded out.  If only one
    threshold is given, all values below that are omitted when creating edges.
    If two thresholds are given, then values in the th2-th1 range are
    ommitted.  This allows for the easy creation of weighted graphs with
    positive and negative values where a range of weights around 0 is omitted.
    
    Parameters
    ----------
    cmat : 2-d square array
      Adjacency matrix.
    threshold : float
      First threshold.
    threshold2 : float
      Second threshold.

    Returns
    -------
    G : a NetworkX weighted graph object, to which a dictionary called
    G.metadata is appended.  This dict contains the original adjacency matrix
    cmat, the two thresholds, and the weights 
    """ 

    # Input sanity check
    nrow,ncol = cmat.shape
    if nrow != ncol:
        raise ValueError("Adjacency matrix must be square")

    row_idx, col_idx, vals = threshold_arr(cmat,threshold,threshold2)
    # Also make the full thresholded array available in the metadata
    cmat_th = np.empty_like(cmat)
    if threshold2 is None:
        cmat_th.fill(threshold)
    else:
        cmat_th.fill(-np.inf)
    cmat_th[row_idx,col_idx] = vals

    # Next, make a normalized copy of the values.  For the 2-threshold case, we
    # use 'folding' normalization
    if threshold2 is None:
        vals_norm = normalize(vals)
    else:
        vals_norm = normalize(vals,'folding',[threshold,threshold2])

    # Now make the actual graph
    G = nx.Graph(weighted=True)
    G.add_nodes_from(range(nrow))
    # To keep the weights of the graph to simple values, we store the
    # normalize ones in a separate dict that we'll stuff into the graph
    # metadata.
    
    normed_values = {}
    for i,j,val,nval in zip(row_idx,col_idx,vals,vals_norm):
        if i == j:
            # no self-loops
            continue
        G.add_edge(i,j,weight=val)
        normed_values[i,j] = nval

    # Write a metadata dict into the graph and save the threshold info there
    G.metadata = dict(threshold1=threshold,
                      threshold2=threshold2,
                      cmat_raw=cmat,
                      cmat_th =cmat_th,
                      vals_norm = normed_values,
                      )
    return G

# Backwards compatibility name
mkgraph = mat2graph

def mkdigraph(cmat,dmat,threshold=0.0,threshold2=None):
    """Make a graph object out of an adjacency matrix and direction matrix"""

    # Input sanity check
    nrow,ncol = cmat.shape
    if not nrow==ncol:
        raise ValueError("Adjacency matrix must be square")

    row_idx, col_idx, vals = threshold_arr(cmat,threshold,threshold2)

    # Now make the actual graph
    G = nx.DiGraph()
    G.add_nodes_from(range(nrow))

    for i,j,val in zip(row_idx,col_idx,vals):
        if dmat[i,j] > 0:
            G.add_edge(i,j,val)
        else:
            G.add_edge(j,i,val)

    return G


def rescale_arr(arr,amin,amax):
    """Rescale an array to a new range.

    Return a new array whose range of values is (amin,amax).

    Parameters
    ----------
    arr : array-like

    amin : float
      new minimum value

    amax : float
      new maximum value

    Examples
    --------
    >>> a = np.arange(5)

    >>> rescale_arr(a,3,6)
    array([ 3.  ,  3.75,  4.5 ,  5.25,  6.  ])
    """
    
    # old bounds
    m = arr.min()
    M = arr.max()
    # scale/offset
    s = float(amax-amin)/(M-m)
    d = amin - s*m
    
    # Apply clip before returning to cut off possible overflows outside the
    # intended range due to roundoff error, so that we can absolutely guarantee
    # that on output, there are no values > amax or < amin.
    return np.clip(s*arr+d,amin,amax)


# backwards compatibility only, deprecated
def replace_diag(arr,val=0):
    fill_diagonal(arr,val)
    return arr


def cost2thresh(cost, sub, bl, lk, idc=[], costlist=[]):
    """Return the threshold associated with a particular cost.

    The cost is assessed with regard to block 'bl' and subject 'sub'.
    
    Parameters
    ----------
    cost: float
        Cost value for which the associated threshold will be returned.

    sub: integer
        Subject number.

    bl: integer
        Block number.

    lk: numpy array
        Lookup table with blocks X subjects X 2 (threshold or cost, in
        that order) X thresholds/costs.  Each threshold is a value
        representing the lowest correlation value accepted.  They are
        ordered from least to greatest.  Each cost is the fraction of
        all possible edges that exists in an undirected graph made from
        this block's correlations (thresholded with the corresponding
        threshold).

    idc: integer or empty list, optional
        Index in costlist corresponding to cost currently being
        processed.  By default, idc is an empty list.

    costlist: array_like
        List of costs that are being queried with the current function
        in order.

    Returns
    -------
    th: float
        Threshold value in lk corresponding to the supplied cost.  If
        multiple entries matching cost exist, the smallest threshold
        corresponding to these is returned.  If no entries matching cost
        are found, return the threshold corresponding to the previous
        cost in costlist.

    Notes
    -----
    The supplied cost must exactly match an entry in lk for a match to
    be registered.

    """
    return cost2thresh2(cost, sub, bl, axis0=None, 
            lk=lk, last = None, idc=idc,costlist = costlist)

def cost2thresh2(cost, sub, axis1, axis0, lk, 
        last = None, idc = [], costlist=[]):
    """A definition for loading the lookup table and finding the threshold 
    associated with a particular cost for a particular subject in a 
    particular block of data
    
    Inputs
    ------
    cost : float
        cost value for which we need the associated threshold
    sub : int 
        (axis -2) subject number
    axis1 : int
        axis 1 into lookup (eg block number or condition)
    axis0 : int
        axis 0 into lookup (eg subcondition)
    lk : numpy array 
        lookup table (axis0 x axis1  x subject x 2 )
    last : None
        NOT USED last threshold value
    idc : int or empty list
        Index in costlist corresponding to cost currently being
        processed.  By default, idc is an empty list.
    costlist : array-like
        List of costs that are being queried with the current function
        in order.
    
    Returns
    -------
    threshold : float
        threshold value for this cost"""

    subject_lookup = slice_data(lk, sub, axis1, subcond=axis0) 
    index = np.where(subject_lookup[1] == cost)
    threshold = subject_lookup[0][index]
    
    if len(threshold) > 1:
        threshold = threshold[0] 
        #if there are multiple thresholds, go down to the lower cost 
        ####Is this right?!!!####
        print('Subject %s has multiple thresholds at cost %s'%(sub, cost))
        print('index 1: %s, index 2: %s'%(axis1, axis0))
    elif len(threshold) < 1:
        idc = idc-1
        newcost = costlist[idc]
        threshold = cost2thresh2(newcost, sub, axis1, axis0, lk, 
                                 idc=idc, costlist = costlist) 
        print(' '.join(['Subject %s does not have cost at %s'%(sub, cost),
                        'index 1: %s, index 2: %s'%(axis1, axis0),
                        'nearest cost %s being used'%(newcost)]))
    else:
        threshold = threshold[0]
      
    return threshold


def apply_cost(corr_mat, cost, tot_edges):
    """Threshold corr_mat to achieve cost.

    Return the thresholded matrix and the threshold value.  In the
    thresholded matrix, the main diagonal and upper triangle are set to
    0, so information is held only in the lower triangle.

    Parameters
    ----------
    corr_mat: array_like
        Square matrix with ROI-to-ROI correlations.

    cost: float
        Fraction of all possible undirected edges desired in the
        thresholded matrix.

    tot_edges: integer
        The number of possible undirected edges in a graph with the
        number of nodes in corr_mat.

    Returns
    -------
    thresholded_mat: array_like
        Square matrix with correlations below threshold set to 0,
        making the fraction of matrix elements that are non-zero equal
        to cost.  In addition, the main diagonal and upper triangle are
        set to 0.

    threshold: float
        Correlations below this value have been set to 0 in
        thresholded_corr_mat.

    Notes
    -----
    If not all correlations are unique, it is possible that there will
    be no way to achieve the cost without, e.g., arbitrarily removing
    one of two identical correlations while keeping the other.  Instead
    of making such an arbitrary choice, this function retains all
    identical correlations equal to or greater than threshold, even if
    this means cost is not exactly achieved.

    """
    thresholded_mat = np.tril(corr_mat, -1)
    n_nonzero = cost * tot_edges
    elements = thresholded_mat.ravel()
    threshold = elements[elements.argsort()[-n_nonzero]]
    thresholded_mat[thresholded_mat < threshold] = 0
    return thresholded_mat, threshold


def network_ind(ntwk_type,n_nodes):
    """Reads in a network type, number of nodes total and returns the indices of that network"""

    net_core ="dACC L_aIfO R_aIfO L_aPFC R_aPFC L_aThal R_aThal".split()
    net_fp = """L_frontcx  R_frontcx    L_IPL  R_IPL    L_IPS  R_IPS  L_PFC
    R_PFC L_precuneus    R_precuneus  midcing""".split()
    net_motor = """L_motor  R_motor L_preSMA R_preSMA SMA""".split()
    net_aal = " "
    
    subnets = { 'g': net_core,
                'b': net_fp,
                'y': net_motor,
                }
    ALL_LABELS = net_core+net_fp +net_motor

    if ntwk_type=='core':
        roi_ind=range(0,7)
        subnets = { 'g': net_core}
        ALL_LABELS = net_core
    elif ntwk_type=='FP':
        roi_ind=range(7,18)
        subnets = {'b': net_fp}
        ALL_LABELS = net_fp
    elif ntwk_type=='all':
        roi_ind=range(0,n_nodes)
        subnets = { 'g': net_core,
            'b': net_fp }#,
            #'y': net_motor,
            #}
        ALL_LABELS = net_core+net_fp# +net_motor
    elif ntwk_type=='aal':
        roi_ind=range(0,n_nodes)
        subnets = {'k': net_aal}
        ALL_LABELS = net_aal
    else:
        print('do not recognize network type')
    return roi_ind,subnets,ALL_LABELS


#-----------------------------------------------------------------------------
# Numpy utilities - Note: these have been sent into numpy itself, so eventually
# we'll be able to get rid of them here.
#-----------------------------------------------------------------------------

def fill_diagonal(a,val):
    """Fill the main diagonal of the given array of any dimensionality.

    For an array with ndim > 2, the diagonal is the list of locations with
    indices a[i,i,...,i], all identical.

    This function modifies the input array in-place, it does not return a
    value.

    This functionality can be obtained via diag_indices(), but internally this
    version uses a much faster implementation that never constructs the indices
    and uses simple slicing.

    Parameters
    ----------
    a : array, at least 2-dimensional.
      Array whose diagonal is to be filled, it gets modified in-place.

    val : scalar
      Value to be written on the diagonal, its type must be compatible with
      that of the array a.

    Examples
    --------
    >>> a = np.zeros((3,3),int)
    >>> fill_diagonal(a,5)
    >>> a
    array([[5, 0, 0],
           [0, 5, 0],
           [0, 0, 5]])

    The same function can operate on a 4-d array:
    >>> a = np.zeros((3,3,3,3),int)
    >>> fill_diagonal(a,4)

    We only show a few blocks for clarity:
    >>> a[0,0]
    array([[4, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])
    >>> a[1,1]
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 0]])
    >>> a[2,2]
    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 4]])

    See also
    --------
    - numpy.diag_indices: indices to access diagonals given shape information.
    - numpy.diag_indices_from: indices to access diagonals given an array.
    """
    return np.fill_diagonal(a,val)


def diag_indices(n,ndim=2):
    """Return the indices to access the main diagonal of an array.

    This returns a tuple of indices that can be used to access the main
    diagonal of an array with ndim (>=2) dimensions and shape (n,n,...,n).  For
    ndim=2 this is the usual diagonal, for ndim>2 this is the set of indices
    to access A[i,i,...,i] for i=[0..n-1].

    Parameters
    ----------
    n : int
      The size, along each dimension, of the arrays for which the returned
      indices can be used.

    ndim : int, optional
      The number of dimensions 

    Examples
    --------
    Create a set of indices to access the diagonal of a (4,4) array:
    >>> di = diag_indices(4)

    >>> a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    >>> a
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12],
           [13, 14, 15, 16]])
    >>> a[di] = 100
    >>> a
    array([[100,   2,   3,   4],
           [  5, 100,   7,   8],
           [  9,  10, 100,  12],
           [ 13,  14,  15, 100]])

    Now, we create indices to manipulate a 3-d array:
    >>> d3 = diag_indices(2,3)

    And use it to set the diagonal of a zeros array to 1:
    >>> a = np.zeros((2,2,2),int)
    >>> a[d3] = 1
    >>> a
    array([[[1, 0],
            [0, 0]],
    <BLANKLINE>
           [[0, 0],
            [0, 1]]])

    See also
    --------
    - numpy.diag_indices_from: create the indices based on the shape of an existing
    array. 
    """
    return np.diag_indices(n, ndim=ndim)


def diag_indices_from(arr):
    """Return the indices to access the main diagonal of an n-dimensional array.

    See diag_indices() for full details.

    Parameters
    ----------
    arr : array, at least 2-d
    """
    return np.diag_indices_from(arr)


def mask_indices(n,mask_func,k=0):
    """Return the indices to access (n,n) arrays, given a masking function.

    Assume mask_func() is a function that, for a square array a of size (n,n)
    with a possible offset argument k, when called as mask_func(a,k) returns a
    new array with zeros in certain locations (functions like triu() or tril()
    do precisely this).  Then this function returns the indices where the
    non-zero values would be located.

    Parameters
    ----------
    n : int
      The returned indices will be valid to access arrays of shape (n,n).

    mask_func : callable
      A function whose api is similar to that of numpy.tri{u,l}.  That is,
      mask_func(x,k) returns a boolean array, shaped like x.  k is an optional
      argument to the function.

    k : scalar
      An optional argument which is passed through to mask_func().  Functions
      like tri{u,l} take a second argument that is interpreted as an offset.

    Returns
    -------
    indices : an n-tuple of index arrays.
      The indices corresponding to the locations where mask_func(ones((n,n)),k)
      is True.

    Examples
    --------
    These are the indices that would allow you to access the upper triangular
    part of any 3x3 array:
    >>> iu = mask_indices(3,np.triu)

    For example, if `a` is a 3x3 array:
    >>> a = np.arange(9).reshape(3,3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    Then:
    >>> a[iu]
    array([0, 1, 2, 4, 5, 8])

    An offset can be passed also to the masking function.  This gets us the
    indices starting on the first diagonal right of the main one:
    >>> iu1 = mask_indices(3,np.triu,1)

    with which we now extract only three elements:
    >>> a[iu1]
    array([1, 2, 5])
    """ 
    m = np.ones((n,n),int)
    a = mask_func(m,k)
    return np.where(a != 0)


def tril_indices(n,k=0):
    """Return the indices for the lower-triangle of an (n,n) array.

    Parameters
    ----------
    n : int
      Sets the size of the arrays for which the returned indices will be valid.

    k : int, optional
      Diagonal offset (see tril() for details).

    Examples
    --------
    Commpute two different sets of indices to access 4x4 arrays, one for the
    lower triangular part starting at the main diagonal, and one starting two
    diagonals further right:
    
    >>> il1 = tril_indices(4)
    >>> il2 = tril_indices(4,2)

    Here is how they can be used with a sample array:
    >>> a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    >>> a
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12],
           [13, 14, 15, 16]])

    Both for indexing:
    >>> a[il1]
    array([ 1,  5,  6,  9, 10, 11, 13, 14, 15, 16])

    And for assigning values:
    >>> a[il1] = -1
    >>> a
    array([[-1,  2,  3,  4],
           [-1, -1,  7,  8],
           [-1, -1, -1, 12],
           [-1, -1, -1, -1]])

    These cover almost the whole array (two diagonals right of the main one):
    >>> a[il2] = -10 
    >>> a
    array([[-10, -10, -10,   4],
           [-10, -10, -10, -10],
           [-10, -10, -10, -10],
           [-10, -10, -10, -10]])

    See also
    --------
    - triu_indices : similar function, for upper-triangular.
    - mask_indices : generic function accepting an arbitrary mask function.
    """
    return np.tril_indices(n,k)   #mask_indices(n,np.tril,k)


def tril_indices_from(arr,k=0):
    """Return the indices for the lower-triangle of an (n,n) array.

    See tril_indices() for full details.
    
    Parameters
    ----------
    n : int
      Sets the size of the arrays for which the returned indices will be valid.

    k : int, optional
      Diagonal offset (see tril() for details).

    """
    return np.tril_indices_from(arr, k)
    
def triu_indices(n,k=0):
    """Return the indices for the upper-triangle of an (n,n) array.

    Parameters
    ----------
    n : int
      Sets the size of the arrays for which the returned indices will be valid.

    k : int, optional
      Diagonal offset (see triu() for details).

    Examples
    --------
    Commpute two different sets of indices to access 4x4 arrays, one for the
    upper triangular part starting at the main diagonal, and one starting two
    diagonals further right:

    >>> iu1 = triu_indices(4)
    >>> iu2 = triu_indices(4,2)

    Here is how they can be used with a sample array:
    >>> a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    >>> a
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12],
           [13, 14, 15, 16]])

    Both for indexing:
    >>> a[iu1]
    array([ 1,  2,  3,  4,  6,  7,  8, 11, 12, 16])

    And for assigning values:       
    >>> a[iu1] = -1
    >>> a
    array([[-1, -1, -1, -1],
           [ 5, -1, -1, -1],
           [ 9, 10, -1, -1],
           [13, 14, 15, -1]])

    These cover almost the whole array (two diagonals right of the main one):
    >>> a[iu2] = -10
    >>> a
    array([[ -1,  -1, -10, -10],
           [  5,  -1,  -1, -10],
           [  9,  10,  -1,  -1],
           [ 13,  14,  15,  -1]])

    See also
    --------
    - tril_indices : similar function, for lower-triangular.
    - mask_indices : generic function accepting an arbitrary mask function.
    """
    return np.triu_indices(n,k) #mask_indices(n,np.triu,k)


def triu_indices_from(arr,k=0):
    """Return the indices for the lower-triangle of an (n,n) array.

    See triu_indices() for full details.
    
    Parameters
    ----------
    n : int
      Sets the size of the arrays for which the returned indices will be valid.

    k : int, optional
      Diagonal offset (see triu() for details).

    """
    return np.tri_indices_from(arr, k)

def structured_rand_arr(size, sample_func=np.random.random,
                        ltfac=None, utfac=None, fill_diag=None):
    """Make a structured random 2-d array of shape (size,size).

    If no optional arguments are given, a symmetric array is returned.

    Parameters
    ----------
    size : int
      Determines the shape of the output array: (size,size).

    sample_func : function, optional.
      Must be a function which when called with a 2-tuple of ints, returns a
      2-d array of that shape.  By default, np.random.random is used, but any
      other sampling function can be used as long as it matches this API.

    utfac : float, optional
      Multiplicative factor for the upper triangular part of the matrix.
      
    ltfac : float, optional
      Multiplicative factor for the lower triangular part of the matrix.

    fill_diag : float, optional  
      If given, use this value to fill in the diagonal.  Otherwise the diagonal
      will contain random elements.

    Examples
    --------
    >>> np.random.seed(0)  # for doctesting
    >>> np.set_printoptions(precision=4)  # for doctesting
    >>> structured_rand_arr(4)
    array([[ 0.5488,  0.7152,  0.6028,  0.5449],
           [ 0.7152,  0.6459,  0.4376,  0.8918],
           [ 0.6028,  0.4376,  0.7917,  0.5289],
           [ 0.5449,  0.8918,  0.5289,  0.0871]])
    >>> structured_rand_arr(4,ltfac=-10,utfac=10,fill_diag=0.5)
    array([[ 0.5   ,  8.3262,  7.7816,  8.7001],
           [-8.3262,  0.5   ,  4.6148,  7.8053],
           [-7.7816, -4.6148,  0.5   ,  9.4467],
           [-8.7001, -7.8053, -9.4467,  0.5   ]])
    """
    # Make a random array from the given sampling function
    rmat = sample_func((size,size))
    # And the empty one we'll then fill in to return
    out = np.empty_like(rmat)
    # Extract indices for upper-triangle, lower-triangle and diagonal
    uidx = triu_indices(size,1)
    lidx = tril_indices(size,-1)
    didx = diag_indices(size)
    # Extract each part from the original and copy it to the output, possibly
    # applying multiplicative factors.  We check the factors instead of
    # defaulting to 1.0 to avoid unnecessary floating point multiplications
    # which could be noticeable for very large sizes.
    if utfac:
        out[uidx] = utfac * rmat[uidx]
    else:
        out[uidx] = rmat[uidx]
    if ltfac:
        out[lidx] = ltfac * rmat.T[lidx]
    else:
        out[lidx] = rmat.T[lidx]
    # If fill_diag was provided, use it; otherwise take the values in the
    # diagonal from the original random array.
    if fill_diag is not None:
        out[didx] = fill_diag
    else:
        out[didx] = rmat[didx]
        
    return out


def symm_rand_arr(size,sample_func=np.random.random,fill_diag=None):
    """Make a symmetric random 2-d array of shape (size,size).

    Parameters
    ----------
    size : int
      Size of the output array.

    sample_func : function, optional.
      Must be a function which when called with a 2-tuple of ints, returns a
      2-d array of that shape.  By default, np.random.random is used, but any
      other sampling function can be used as long as it matches this API.

    fill_diag : float, optional
      If given, use this value to fill in the diagonal.

    Examples
    --------
    >>> np.random.seed(0)  # for doctesting
    >>> np.set_printoptions(precision=4)  # for doctesting
    >>> symm_rand_arr(4)
    array([[ 0.5488,  0.7152,  0.6028,  0.5449],
           [ 0.7152,  0.6459,  0.4376,  0.8918],
           [ 0.6028,  0.4376,  0.7917,  0.5289],
           [ 0.5449,  0.8918,  0.5289,  0.0871]])
    >>> symm_rand_arr(4,fill_diag=4)
    array([[ 4.    ,  0.8326,  0.7782,  0.87  ],
           [ 0.8326,  4.    ,  0.4615,  0.7805],
           [ 0.7782,  0.4615,  4.    ,  0.9447],
           [ 0.87  ,  0.7805,  0.9447,  4.    ]])
      """
    return structured_rand_arr(size,sample_func,fill_diag=fill_diag)


def antisymm_rand_arr(size,sample_func=np.random.random):
    """Make an anti-symmetric random 2-d array of shape (size,size).

    Parameters
    ----------

    n : int
      Size of the output array.

    sample_func : function, optional.
      Must be a function which when called with a 2-tuple of ints, returns a
      2-d array of that shape.  By default, np.random.random is used, but any
      other sampling function can be used as long as matches this API.

    Examples
    --------
    >>> np.random.seed(0)  # for doctesting
    >>> np.set_printoptions(precision=4)  # for doctesting
    >>> antisymm_rand_arr(4)
    array([[ 0.    ,  0.7152,  0.6028,  0.5449],
           [-0.7152,  0.    ,  0.4376,  0.8918],
           [-0.6028, -0.4376,  0.    ,  0.5289],
           [-0.5449, -0.8918, -0.5289,  0.    ]])
      """
    return structured_rand_arr(size,sample_func,ltfac=-1.0,fill_diag=0)


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

def array_to_string(part):
    """The purpose of this function is to convert an array of numbers into
    a list of strings. Mainly for use with the plot_partition function that
    requires a dict of strings for node labels.

    """

    out_part=dict.fromkeys(part)
    
    for m in part.iterkeys():
        out_part[m]=str(part[m])
    
    return out_part

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
        

def assert_no_empty_modules(part):
    """Asserts that a partition contains no empty moudles.

    This function raises a ValueError exception if the input partition has an
    empty module.

    Parameters
    ----------
    part : dict
      A dict describing a graph partition.
    """
    for label, mod in part.iteritems():
        if len(mod)==0:
            raise ValueError("Module %s in partition is empty" % label)
