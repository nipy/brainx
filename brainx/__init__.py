"""Top-level init file for brainx package.
"""

def patch_nx():
    """Temporary fix for NX's watts_strogatz routine, which has a bug in versions 1.1-1.3
    """

    import networkx as nx

    # Quick test to see if we get the broken version
    g = nx.watts_strogatz_graph(2, 0, 0)

    if g.number_of_nodes() != 2:
        # Buggy version detected.  Create a patched version and apply it to nx
        
        nx._watts_strogatz_graph_ori = nx.watts_strogatz_graph        

        def patched_ws(n, k, p, seed=None):
            if k<2:
                g = nx.Graph()
                g.add_nodes_from(range(n))
                return g
            else:
                return nx._watts_strogatz_graph_ori(n, k, p, seed)

        patched_ws.__doc__ = nx._watts_strogatz_graph_ori.__doc__

        # Applying monkeypatch now
        import warnings
        warnings.warn("Monkeypatching NetworkX's Watts-Strogatz routine")

        nx.watts_strogatz_graph = patched_ws
        

patch_nx()
