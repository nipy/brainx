"""Generic utilities that may be needed by the other modules.
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import numpy as np

import networkx as nx

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------
def format_matrix(data,s,b,lk,co,nouptri = False):
    """ Function which formats matrix for a particular subject and particular block (thresholds, upper-tris it) so that we can make a graph object out of it

    Parameters:
    -----------
    data = full data array
    s = subject
    b = block
    lk = lookup table for study
    co = cost value to threshold at
"""

    cmat = data[b,s]
    th = cost2thresh(co,s,b,lk,[]) #get the right threshold
    
    #cmat = replace_diag(cmat) #replace diagonals with zero
    cmat = thresholded_arr(cmat,th,fill_val=0)
    if not nouptri:
        cmat = np.triu(cmat,1)

    return cmat


def cost_size(nnodes):
    tot_edges = .5*nnodes*(nnodes-1)

    costs =  np.array(range(tot_edges+1),dtype=float)/tot_edges
    edges_short = tot_edges/2
    return costs,tot_edges,edges_short
    

def store_metrics(b, s, co, metd, arr):
    """Store a set of metrics into a structured array"""

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


def threshold_arr(cmat,threshold=0.0,threshold2=None):
    """Threshold values from the input matrix.

    Parameters
    ----------
    cmat : array
    
    threshold : float, optional.
      First threshold.
      
    threshold2 : float, optional.
      Second threshold.

    Returns
    -------
    indices, values: a tuple with ndim+1
    
    Examples
    --------
    >>> a = np.linspace(0,0.8,7)
    >>> a
    array([ 0.    ,  0.1333,  0.2667,  0.4   ,  0.5333,  0.6667,  0.8   ])
    >>> threshold_arr(a,0.3)
    (array([3, 4, 5, 6]), array([ 0.4   ,  0.5333,  0.6667,  0.8   ]))

    With two thresholds:
    >>> threshold_arr(a,0.3,0.6)
    (array([0, 1, 2, 5, 6]), array([ 0.    ,  0.1333,  0.2667,  0.6667,  0.8   ]))

    """
    # Select thresholds
    if threshold2 is None:
        th_low = -np.inf
        th_hi  = threshold
    else:
        th_low = threshold
        th_hi  = threshold2

    # Mask out the values we are actually going to use
    idx = np.where( (cmat < th_low) | (cmat > th_hi) )
    vals = cmat[idx]
    
    return idx + (vals,)


def thresholded_arr(arr,threshold=0.0,threshold2=None,fill_val=np.nan):
    """Threshold values from the input matrix and return a new matrix.

    Parameters
    ----------
    arr : array
    
    threshold : float
      First threshold.
      
    threshold2 : float, optional.
      Second threshold.

    Returns
    -------
    An array shaped like the input, with the values outside the threshold
    replaced with fill_val.
    
    Examples
    --------
    """
    a2 = np.empty_like(arr)
    a2.fill(fill_val)
    mth = threshold_arr(arr,threshold,threshold2)
    idx,vals = mth[:-1], mth[-1]
    a2[idx] = vals
    
    return a2


def normalize(arr,mode='direct',folding_edges=None):
    """Normalize an array to [0,1] range.

    By default, this simply rescales the input array to [0,1].  But it has a
    special 'folding' mode that allows for the normalization of an array with
    negative and positive values by mapping the negative values to their
    flipped sign

    Parameters
    ----------
    arr : 1d array
    
    mode : string, one of ['direct','folding']

    folding_edges : (float,float)
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
    array([ 1.    ,  0.6667,  0.3333,  0.    ,  0.    ,  0.2667,  0.5333,  0.8   ])
    """
    if mode == 'direct':
        return rescale_arr(arr,0,1)
    else:
        fa, fb = folding_edges
        amin, amax = arr.min(), arr.max()
        ra,rb = float(fa-amin),float(amax-fb) # in case inputs are ints
        if ra<0 or rb<0:
            raise ValueError("folding edges must be within array range")
        greater = arr>= fb
        upper_idx = greater.nonzero()
        lower_idx = (~greater).nonzero()
        # Two folding scenarios, we map the thresholds to zero but the upper
        # ranges must retain comparability.
        if ra > rb:
            lower = 1.0 - rescale_arr(arr[lower_idx],0,1.0)
            upper = rescale_arr(arr[upper_idx],0,float(rb)/ra)
        else:
            upper = rescale_arr(arr[upper_idx],0,1)
            # The lower range is trickier: we need to rescale it and then flip
            # it, so the edge goes to 0.
            resc_a = float(ra)/rb
            lower = rescale_arr(arr[lower_idx],0,resc_a)
            lower = resc_a - lower
        # Now, make output array
        out = np.empty_like(arr)
        out[lower_idx] = lower
        out[upper_idx] = upper
        return out


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


def cost2thresh(cost,sub,bl,lk,last):
    """A definition for loading the lookup table and finding the threshold associated with a particular cost for a particular subject in a particular block
    
    inputs:
    cost: cost value for which we need the associated threshold
    sub: subject number
    bl: block number
    lk: lookup table (block x subject x cost
    last: last threshold value

    output:
    th: threshold value for this cost"""

    #print cost,sub,bl
    
    ind=np.where(lk[bl][sub][1]==cost)
    th=lk[bl][sub][0][ind]
    
    if len(th)>1:
        th=th[0] #if there are multiple thresholds, go down to the lower cost ####Is this right?!!!####
        print 'multiple thresh'
    elif len(th)<1:
        th=last #if there is no associated thresh value because of repeats, just use the previous one
        print 'use previous thresh'
    else:
        th=th[0]
      
    #print th    
    return th


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
        print 'do not recognize network type'
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
    - diag_indices: indices to access diagonals given shape information.
    - diag_indices_from: indices to access diagonals given an array.
    """
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    if a.ndim == 2:
        # Explicit, fast formula for the common case.  For 2-d arrays, we
        # accept rectangular ones.
        step = a.shape[1] + 1
    else:
        # For more than d=2, the strided formula is only valid for arrays with
        # all dimensions equal, so we check first.
        if not np.alltrue(np.diff(a.shape)==0):
            raise ValueError("All dimensions of input must be of equal length")
        step = np.cumprod((1,)+a.shape[:-1]).sum()

    # Write the value out into the diagonal.
    a.flat[::step] = val


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
    - diag_indices_from: create the indices based on the shape of an existing
    array. 
    """
    idx = np.arange(n)
    return (idx,)*ndim


def diag_indices_from(arr):
    """Return the indices to access the main diagonal of an n-dimensional array.

    See diag_indices() for full details.

    Parameters
    ----------
    arr : array, at least 2-d
    """
    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")
    # For more than d=2, the strided formula is only valid for arrays with
    # all dimensions equal, so we check first.
    if not np.alltrue(np.diff(a.shape)==0):
        raise ValueError("All dimensions of input must be of equal length")

    return diag_indices(a.shape[0],a.ndim)

    
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
    return mask_indices(n,np.tril,k)


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
    if not arr.ndim==2 and arr.shape[0] == arr.shape[1]:
        raise ValueError("input array must be 2-d and square")
    return tril_indices(arr.shape[0],k)

    
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
    return mask_indices(n,np.triu,k)


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
    if not arr.ndim==2 and arr.shape[0] == arr.shape[1]:
        raise ValueError("input array must be 2-d and square")
    return triu_indices(arr.shape[0],k)


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
        
