"""Plotting utilities for networks"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Third-party
import numpy as np

import matplotlib
from matplotlib import cm
from matplotlib.colors import colorConverter
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrow
from matplotlib import pyplot as plt, mpl

import networkx as nx

# From this project
import util
import metrics

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------
def draw_matrix(mat,th1=None,th2=None,clim=None,cmap=None):
    """Draw a matrix, optionally thresholding it.
    """
    if th1 is not None:
        m2 = util.thresholded_arr(mat,th1,th2)
    else:
        m2 = mat
    ax = plt.matshow(m2,cmap=cmap)
    if clim is not None:
        ax.set_clim(*clim)
    plt.colorbar()
    return ax


def draw_arrows(G,pos,edgelist=None,ax=None,edge_color='k',alpha=1.0,
                width=1):
    """Draw arrows on a set of edges"""

    
    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = G.edges()

    if not edgelist or len(edgelist)==0: # no edges!
        return

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]],pos[e[1]]) for e in edgelist])

    arrow_colors = ( colorConverter.to_rgba('k', alpha), )
    a_pos = []

    # Radius of the nodes in world coordinates
    radius = 0.5
    head_length = 0.31
    overhang = 0.1

    #ipvars('edge_pos')  # dbg

    for src,dst in edge_pos:
        dd = dst-src
        nd = np.linalg.norm(dd)
        if nd==0: # source and target at same position
            continue

        s = 1.0-radius/nd
        dd *= s
        x1,y1 = src
        dx,dy = dd
        ax.arrow(x1,y1,dx,dy,lw=width,width=width,head_length=head_length,
                 fc=edge_color,ec='none',alpha=alpha,overhang=overhang)


def draw_graph(G,
               labels=None,
               node_colors=None,
               node_shapes=None,
               node_scale=1.0,
               edge_style='solid',
               edge_cmap=None,
               colorbar=False,
               vrange=None,
               layout=nx.circular_layout,
               title=None,
               font_family='sans-serif',
               font_size=9,
               stretch_factor=1.0,
               edge_alpha=True):
    """Draw a weighted graph with options to visualize link weights.

    The resulting diagram uses the rank of each node as its size, and the
    weight of each link (after discarding thresholded values, see below) as the
    link opacity.

    It maps edge weight to color as well as line opacity and thickness,
    allowing the color part to be hardcoded over a value range (to permit valid
    cross-figure comparisons for different graphs, so the same color
    corresponds to the same link weight even if each graph has a different
    range of weights).  The nodes sizes are proportional to their degree,
    computed as the sum of the weights of all their links.  The layout defaults
    to circular, but any nx layout function can be passed in, as well as a
    statically precomputed layout.
    
    Parameters
    ----------
    G : weighted graph
      The values must be of the form (v1,v2), with all v2 in [0,1].  v1 are
      used for colors, v2 for thickness/opacity.

    labels : list or dict, optional.
      An indexable object that maps nodes to strings.  If not given, the
      string form of each node is used as a label.  If False, no labels are
      drawn.
      
    node_colors : list or dict, optional.
      An indexable object that maps nodes to valid matplotlib color specs.  See
      matplotlib's plot() function for details.

    node_shapes : list or dict, optional.
      An indexable object that maps nodes to valid matplotlib shape specs.  See
      matplotlib's scatter() function for details.  If not given, circles are
      used.

    node_scale : float, optional
      A scale factor to globally stretch or shrink all nodes symbols by.

    edge_style : string, optional
      Line style for the edges, defaults to 'solid'.

    edge_cmap : matplotlib colormap, optional.
      A callable that returns valid color specs, like matplotlib colormaps.
      If not given, edges are colored black.

    colorbar : bool
      If true, automatically add a colorbar showing the mapping of graph weight
      values to colors.

    vrange : pair of floats
      If given, this indicates the total range of values that the weights can
      in principle occupy, and is used to set the lower/upper range of the
      colormap.  This allows you to set the range of multiple different figures
      to the same values, even if each individual graph has range variations,
      so that visual color comparisons across figures are valid.
      
    layout : function or layout dict, optional
      A NetworkX-like layout function or the result of a precomputed layout for
      the given graph.  NetworkX produces layouts as dicts keyed by nodes and
      with (x,y) pairs of coordinates as values, any function that produces
      this kind of output is acceptable.  Defaults to nx.circular_layout.

    title : string, optional.
      If given, title to put on the main plot.

    font_family : string, optional.
      Font family used for the node labels and title.

    font_size : int, optional.
      Font size used for the node labels and title.

    stretch_factor : float, optional
      A global scaling factor to make the graph larger (or smaller if <1).
      This can be used to separate the nodes if they start overlapping.

    edge_alpha: bool, optional
      Whether to weight the transparency of each edge by a factor equivalent to
      its relative weight

    Returns
    -------
    fig
      The matplotlib figure object with the plot.
    """
    # A few hardcoded constants, though their effect can always be controlled
    # via user-settable parameters.
    figsize = [6,6]
    # For the size of the node symbols
    node_size_base = 1000
    node_min_size = 200
    default_node_shape = 'o'
    # Default colors if none given
    default_node_color = 'r'
    default_edge_color = 'k'
    # Max edge width
    max_width = 13
    font_family = 'sans-serif'

    # We'll use the nodes a lot, let's make a numpy array of them
    nodes = np.array(sorted(G.nodes()))
    nnod = len(nodes)

    # Build a 'weighted degree' array obtained by adding the (absolute value)
    # of the weights for all edges pointing to each node:
    amat = nx.adj_matrix(G).A  # get a normal array out of it
    degarr = abs(amat).sum(0)  # weights are sums across rows

    # Map the degree to the 0-1 range so we can use it for sizing the nodes.
    try:
        odegree = util.rescale_arr(degarr,0,1)
        # Make an array of node sizes based on node degree
        node_sizes  = odegree * node_size_base + node_min_size
    except ZeroDivisionError:
        # All nodes same size
        node_sizes = np.empty(nnod,float)
        node_sizes.fill(0.5 * node_size_base + node_min_size)

    # Adjust node size list.  We square the scale factor because in mpl, node
    # sizes represent area, not linear size, but it's more intuitive for the
    # user to think of linear factors (the overall figure scale factor is also
    # linear). 
    node_sizes *= node_scale**2
    
    # Set default node properties
    if node_colors is None:
        node_colors = [default_node_color]*nnod

    if node_shapes is None:
        node_shapes = [default_node_shape]*nnod

    # Set default edge colormap
    if edge_cmap is None:
        # Make an object with the colormap API, that maps all input values to
        # the default color (with proper alhpa)
        edge_cmap = ( lambda val, alpha:
                      colorConverter.to_rgba(default_edge_color,alpha) )

    # if vrange is None, we set the color range from the values, else the user
    # can specify it

    # e[2] is edge value: edges_iter returns (i,j,data)
    #gvals = np.array([ e[2]['weight'] for e in G.edges(data=True) ]) #CG -no
    #longer has weight as a key?
    gvals = np.array([ e[2] for e in G.edges(data=True)])
    gvmin, gvmax = gvals.min(), gvals.max()
    #gvrange = gvmax-gvmin
    gvrange = gvmax['weight'] - gvmin['weight']
    if vrange is None:
        vrange = gvmin,gvmax
    # Now, construct the normalization for the colormap
    cnorm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1])

    # Create the actual plot where the graph will be displayed
    figsize = np.array(figsize,float)
    figsize *= stretch_factor
    
    fig = plt.figure(figsize=figsize)
    
    # If a colorbar is required, make a set of axes for both the main graph and
    # the colorbar, otherwise let nx do its thing
    if colorbar:
        # Make axes for both the graph and the colorbar
        left0, width0, sep = 0.01, 0.73, 0.03
        left, bottom, width, height = left0+width0+sep, 0.05, 0.03, 0.9
        ax_graph = fig.add_axes([left0,bottom, width0, height])
        ax_cbar = fig.add_axes([left,bottom, width, height])
        # Set the current axes to be the graph ones for nx to draw into
        fig.sca(ax_graph)
    
    # Compute positions for all nodes - nx has several algorithms
    if callable(layout):
        pos = layout(G)
    else:
        # The user can also provide a precomputed layout
        pos = layout

    # Draw nodes
    for nod in nodes:
        nx.draw_networkx_nodes(G,pos,nodelist=[nod],
                               node_color=node_colors[nod],
                               node_shape=node_shapes[nod],
                               node_size=node_sizes[nod])
    #CG: commented out above.... not working?
    #nx.draw_networkx_nodes(G,pos,
    #                       node_color=node_colors,
    #                       node_shape=node_shapes,
    #                       node_size=node_sizes)
        
    # Draw edges
    if not isinstance(G,nx.DiGraph):
        # Undirected graph, simple lines for edges
        # We need the size of the value range to properly scale colors
        #vsize = vrange[1] - vrange[0] #CG commented out
        vsize = vrange[1]['weight'] - vrange[0]['weight'] #CG
        gvals_normalized = G.metadata['vals_norm']
        for (u,v,y) in G.edges(data=True):
            # The graph value is the weight, and the normalized values are in
            # [0,1], used for thickness/transparency
            alpha = gvals_normalized[u,v]
            # Scale the color choice to the specified vrange, so that 
            #ecol = (y['weight']-vrange[0])/vsize #CG commented out
            ecol = (y['weight'] - vrange[0]['weight'])/vsize #CG
            #print 'u,v:',u,v,'y:',y,'ecol:',ecol  # dbg
            
            if edge_alpha:
                fade = alpha
            else:
                fade=1.0

            edge_color = [ tuple(edge_cmap(ecol,fade)) ]

                
            draw_networkx_edges(G, pos, edgelist=[(u,v)],
                                width=alpha*max_width,
                                edge_color=edge_color,
                                style=edge_style)
    else:
        # Directed graph, use arrows.
        # XXX - this is currently broken.
        raise NotImplementedError("arrow drawing currently broken")
    
        ## for (u,v,x) in G.edges(data=True):
        ##     y,w = x
        ##     draw_arrows(G,pos,edgelist=[(u,v)],
        ##                 edge_color=[w],
        ##                 alpha=w,
        ##                 edge_cmap=edge_cmap,
        ##                 width=w*max_width)

    # Draw labels.  If not given, we use the string form of the nodes.  If
    # labels is False, no labels are drawn.
    if labels is None:
        labels = map(str,nodes)

    if labels:
        lab_idx = range(len(labels))
        labels_dict = dict(zip(lab_idx,labels))
        nx.draw_networkx_labels(G,pos,labels_dict,font_size=font_size,
                                font_family=font_family)

    if title:        
        plt.title(title,fontsize=font_size)

    # Turn off x and y axes labels in pylab
    plt.xticks([])
    plt.yticks([])

    # Add a colorbar if requested
    if colorbar:
        cb1 = mpl.colorbar.ColorbarBase(ax_cbar, cmap=edge_cmap, norm=cnorm)
    else:
        # With no colorbar, at least adjust the margins so there's less dead
        # border around the graph (the colorbar code automatically sets those
        # for us above)
        e = 0.08
        plt.subplots_adjust(e,e,1-e,1-e)
        
    # Always return the MPL figure object so the user can further manipulate it
    return fig


def pick_lesion_colors(lesions,subnets):
    ss = {}
    for col,net in subnets.items():
        ss[col] = set(net)

    les_colors = {}
    for lesion in lesions:
        for col,net in ss.items():
            if lesion in net:
                les_colors[lesion] = col
                break
        else:
            raise ValueError("lesion %s not in given subnets" % lesion)
    return les_colors


def lab2node(labels,labels_dict):
    return [labels_dict[ll] for ll in labels]


def draw_lesion_graph(G,bl,labels=None,subnets=None,lesion_nodes=None):
    """
    Parameters

    subnets : dict
     A colors,label list dict of subnetworks that covers the graph
    """
    if labels is None:
        labels = G.nodes()

    if subnets is None:
        subnets = dict(b=G.nodes())

    all_nodes = set(labels)

    lab_idx = range(len(labels))
    labels_dict = dict(zip(labels,lab_idx))
    idx2lab_dict = dict(zip(lab_idx,labels))

    # Check that all subnets cover the whole graph
    subnet_nodes = []
    for ss in subnets.values():
        subnet_nodes.extend(ss)
    subnet_nodes = set(subnet_nodes)
    assert subnet_nodes == all_nodes

    # Check that the optional lesion list is contained in all the nodes
    if lesion_nodes is None:
        lesion_nodes = set()
    else:
        lesion_nodes = set(lesion_nodes)

    assert lesion_nodes.issubset(all_nodes),\
          "lesion nodes:%s not a subset of nodes" % lesion_nodes

    # Make a table that maps lesion nodes to colors
    lesion_colors = pick_lesion_colors(lesion_nodes,subnets)

    # Compute positions for all nodes - nx has several algorithms
    pos = nx.circular_layout(G)

    # Create the actual plot where the graph will be displayed
    #fig = plt.figure()

    #plt.subplot(1,12,bl+1)
    
    good_nodes = all_nodes - lesion_nodes
    # Draw nodes
    for node_color,nodes in subnets.items():
        nodelabels = set(nodes) - lesion_nodes
        nodelist = lab2node(nodelabels,labels_dict)
        nx.draw_networkx_nodes(G,pos,nodelist=nodelist,
                              node_color=node_color,node_size=700,
                              node_shape='o')

    for nod in lesion_nodes:
        nx.draw_networkx_nodes(G,pos,nodelist=[labels_dict[nod]],
                              node_color=lesion_colors[nod],node_size=700,
                              node_shape='s')

    # Draw edges
    draw_networkx_edges(G,pos)

    # Draw labels
    nx.draw_networkx_labels(G,pos,idx2lab_dict)


### Patched version for networx draw_networkx_edges, sent to Aric.
def draw_networkx_edges(G, pos,
                        edgelist=None,
                        width=1.0,
                        edge_color='k',
                        style='solid',
                        alpha=None,
                        edge_cmap=None,
                        edge_vmin=None,
                        edge_vmax=None, 
                        ax=None,
                        arrows=True,
                        **kwds):
    """Draw the edges of the graph G

    This draws only the edges of the graph G.

    pos is a dictionary keyed by vertex with a two-tuple
    of x-y positions as the value.
    See networkx.layout for functions that compute node positions.

    edgelist is an optional list of the edges in G to be drawn.
    If provided, only the edges in edgelist will be drawn. 

    edgecolor can be a list of matplotlib color letters such as 'k' or
    'b' that lists the color of each edge; the list must be ordered in
    the same way as the edge list. Alternatively, this list can contain
    numbers and those number are mapped to a color scale using the color
    map edge_cmap.  Finally, it can also be a list of (r,g,b) or (r,g,b,a)
    tuples, in which case these will be used directly to color the edges.  If
    the latter mode is used, you should not provide a value for alpha, as it
    would be applied globally to all lines.
    
    For directed graphs, "arrows" (actually just thicker stubs) are drawn
    at the head end.  Arrows can be turned off with keyword arrows=False.

    See draw_networkx for the list of other optional parameters.

    """
    try:
        import matplotlib
        import matplotlib.pylab as pylab
        import matplotlib.numerix as nmex
        import matplotlib.cbook as cb
        from matplotlib.colors import colorConverter,Colormap
        from matplotlib.collections import LineCollection
        import matplotlib.numerix.mlab as mlab
    except ImportError:
        raise ImportError, "Matplotlib required for draw()"
    except RuntimeError:
        pass # unable to open display

    if ax is None:
        ax=pylab.gca()

    if edgelist is None:
        edgelist=G.edges()

    if not edgelist or len(edgelist)==0: # no edges!
        return None

    # set edge positions
    edge_pos=nmex.asarray([(pos[e[0]],pos[e[1]]) for e in edgelist])
    
    if not cb.iterable(width):
        lw = (width,)
    else:
        lw = width

    if not cb.is_string_like(edge_color) \
           and cb.iterable(edge_color) \
           and len(edge_color)==len(edge_pos):
        if nmex.alltrue([cb.is_string_like(c) 
                         for c in edge_color]):
            # (should check ALL elements)
            # list of color letters such as ['k','r','k',...]
            edge_colors = tuple([colorConverter.to_rgba(c,alpha) 
                                 for c in edge_color])
        elif nmex.alltrue([not cb.is_string_like(c) 
                           for c in edge_color]):
            # If color specs are given as (rgb) or (rgba) tuples, we're OK
            if nmex.alltrue([cb.iterable(c) and len(c) in (3,4)
                             for c in edge_color]):
                edge_colors = tuple(edge_color)
                alpha=None
            else:
                # numbers (which are going to be mapped with a colormap)
                edge_colors = None
        else:
            raise ValueError('edge_color must consist of either color names or numbers')
    else:
        if len(edge_color)==1:
            edge_colors = ( colorConverter.to_rgba(edge_color, alpha), )
        else:
            raise ValueError('edge_color must be a single color or list of exactly m colors where m is the number or edges')
    edge_collection = LineCollection(edge_pos,
                                     colors       = edge_colors,
                                     linewidths   = lw,
                                     antialiaseds = (1,),
                                     linestyle    = style,     
                                     transOffset = ax.transData,             
                                     )

    # Note: there was a bug in mpl regarding the handling of alpha values for
    # each line in a LineCollection.  It was fixed in matplotlib in r7184 and
    # r7189 (June 6 2009).  We should then not set the alpha value globally,
    # since the user can instead provide per-edge alphas now.  Only set it
    # globally if provided as a scalar.
    if cb.is_numlike(alpha):
        edge_collection.set_alpha(alpha)

    # need 0.87.7 or greater for edge colormaps.  No checks done, this will
    # just not work with an older mpl
    if edge_colors is None:
        if edge_cmap is not None: assert(isinstance(edge_cmap, Colormap))
        edge_collection.set_array(nmex.asarray(edge_color))
        edge_collection.set_cmap(edge_cmap)
        if edge_vmin is not None or edge_vmax is not None:
            edge_collection.set_clim(edge_vmin, edge_vmax)
        else:
            edge_collection.autoscale()
        pylab.sci(edge_collection)

    arrow_collection=None

    if G.is_directed() and arrows:

        # a directed graph hack
        # draw thick line segments at head end of edge
        # waiting for someone else to implement arrows that will work 
        arrow_colors = ( colorConverter.to_rgba('k', alpha), )
        a_pos=[]
        p=1.0-0.25 # make head segment 25 percent of edge length
        for src,dst in edge_pos:
            x1,y1=src
            x2,y2=dst
            dx=x2-x1 # x offset
            dy=y2-y1 # y offset
            d=nmex.sqrt(float(dx**2+dy**2)) # length of edge
            if d==0: # source and target at same position
                continue
            if dx==0: # vertical edge
                xa=x2
                ya=dy*p+y1
            if dy==0: # horizontal edge
                ya=y2
                xa=dx*p+x1
            else:
                theta=nmex.arctan2(dy,dx)
                xa=p*d*nmex.cos(theta)+x1
                ya=p*d*nmex.sin(theta)+y1
                
            a_pos.append(((xa,ya),(x2,y2)))

        arrow_collection = LineCollection(a_pos,
                                colors       = arrow_colors,
                                linewidths   = [4*ww for ww in lw],
                                antialiaseds = (1,),
                                transOffset = ax.transData,             
                                )
        
    # update view        
    minx = mlab.amin(mlab.ravel(edge_pos[:,:,0]))
    maxx = mlab.amax(mlab.ravel(edge_pos[:,:,0]))
    miny = mlab.amin(mlab.ravel(edge_pos[:,:,1]))
    maxy = mlab.amax(mlab.ravel(edge_pos[:,:,1]))

    w = maxx-minx
    h = maxy-miny
    padx, pady = 0.05*w, 0.05*h
    corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)
    ax.update_datalim( corners)
    ax.autoscale_view()

    edge_collection.set_zorder(1) # edges go behind nodes            
    ax.add_collection(edge_collection)
    if arrow_collection:
        arrow_collection.set_zorder(1) # edges go behind nodes            
        ax.add_collection(arrow_collection)
        
    return edge_collection
