"""Directed graph plot example."""

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt, cm

import sys
sys.path.insert(0,'../..')

from brainx import util
from brainx import nxplot

reload(util)
reload(nxplot)

#-----------------------------------------------------------------------------
# main
#-----------------------------------------------------------------------------

size = 11
th2 = 0.6
th1 = -th2

# Split the node list in half, and use two colors for each group
split = size/2
nodes = range(size)
head,tail = nodes[:split],nodes[split:]
labels = ['%s%s' % (chr(i),chr(i+32)) for i in range(65,65+size)]
#labels = map(str,nodes)
colors = ['y' for _ in head] + ['r' for _ in tail]

mat = util.symm_rand_arr(size)
mat = 2*mat-1  # map values to [-1,1] range
util.fill_diagonal(mat,0)  # diag==0 so we don't get self-links

layout = nx.circular_layout

G = util.mat2graph(mat, threshold=th1,threshold2=th2)

pfig = nxplot.draw_graph(G,
                         labels=labels,
                         node_colors=colors,
                         layout = layout,
                         title = layout.func_name,
                         #edge_cmap=cm.PuOr
                         edge_cmap=cm.RdBu,
                         #edge_cmap=cm.jet,
                         colorbar=True,
                         )

if 0:
    layout_funcs = [ nx.circular_layout,
                     nx.fruchterman_reingold_layout,
                     nx.graphviz_layout,
                     nx.pydot_layout,
                     nx.pygraphviz_layout,
                     nx.random_layout,
                     nx.shell_layout,
                     nx.spectral_layout,
                     nx.spring_layout,
                     ]

    for layout in layout_funcs:
        pfig, G = nxplot.draw_graph(mat, threshold=th1,threshold2=th2,
                                    labels=labels,
                                    colors=colors,
                                    layout_function = layout,
                                    title = layout.func_name,
                                    edge_cmap=cm.PuOr
                                    #edge_cmap=cm.RdBu,
                                    #edge_cmap=cm.jet,
                                    )

#dmat = np.random.rand(size,size)
#nxplot.draw_graph(mat,dmat=dmat)
#print 'Mat:\n',mat

plt.show()
