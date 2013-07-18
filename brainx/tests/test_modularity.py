"""Tests for modularity code.
"""

#-----------------------------------------------------------------------------
# Library imports
#-----------------------------------------------------------------------------

# Stdlib
import os
import copy
import math
import time

# Third party
import networkx as nx
import numpy as np
import nose.tools as nt
import numpy.testing as npt

# Our own
from brainx import modularity as mod
from brainx import util

# While debugging the library, reload everything
#map(reload,[mod,util])

#-----------------------------------------------------------------------------
# Local utility functions
#-----------------------------------------------------------------------------

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

    graph_partition = mod.GraphPartition(g,ppart)
    return graph_partition.modularity()


#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------
def test_graphpartition():
    """ test GraphPartition correctly handles graph whose
    nodes are strings"""
    graph = nx.Graph()
    graph.add_edge('a','b')
    graph.add_edge('c','d')
    index = {0:set([0,1]), 1:set([2,3])}
    gpart = mod.GraphPartition(graph, index)


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
                    g = mod.random_modular_graph(nnod, nmod, av_degree,
                                                 btwn_fraction)

                    # First, check the average degree.
                    av_degree_actual = np.mean(g.degree().values())
                    # Since we are generating random graphs, the actual average
                    # degree we get may be off from the reuqested one by a bit.
                    # We allow it to be off by up to 1.
                    #print 'av deg:',av_degree, av_degree_actual  # dbg
                    nt.assert_true (abs(av_degree-av_degree_actual)<1.25,
                          """av deg: %.2f  av deg actual: %.2f -
                          This is a stochastic test - repeat to confirm.""" %
                                          (av_degree, av_degree_actual))

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
                    mask = util.diag_stack(blocks)
                    mask[mask==0] = 2
                    mask = np.triu(mask,1)
                    btwn_real = np.sum(mat[mask == 2].flatten())
                    btwn_real_frac = btwn_real / nedg

                    #compare to what the actual values are
                    nt.assert_almost_equal(btwn_fraction,
                                           btwn_real_frac, 1,
                    "This is a stochastic test, repeat to confirm failure")


def test_modularity():
    """Test the values that go into the modularity calculation after randomly
    creating a graph"""
    # Given a partition with the correct labels of the input graph, verify that
    # the modularity returns 1

    # We need to measure the degree within/between modules
    nnods = 120, 240, 360
    nmods = 2, 3, 4
    av_degrees = 8, 10, 16

    for nnod in nnods:
        for nmod in nmods:
            for av_degree in av_degrees:
                g = mod.random_modular_graph(nnod, nmod, av_degree)
                #Compute the of nodes per module
                nnod_mod = nnod/nmod
                #Make a "correct" partition for the graph
                part = mod.perfect_partition(nmod,nnod_mod)
                #Make a graphpartition object
                graph_partition = mod.GraphPartition(g,part)
                #call modularity
                mod_meas = graph_partition.modularity()
                mod_true = 1.0 - 1.0/nmod
                npt.assert_almost_equal(mod_meas, mod_true, 2)


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

                g = mod.random_modular_graph(nnod, nmod, av_degree)

                #Make a "correct" partition for the graph
                part = mod.perfect_partition(nmod,nnod/nmod)

                #Make a random partition for the graph
                part_rand = dict()
                while len(part_rand) <= 1: #check if there is only one module
                    part_rand = mod.rand_partition(g)

                #List of modules in the partition
                r_mod=range(len(part))

                #Loop through pairs of modules
                for i in range(1): # DB: why is this necessary?
                    #select two modules to merge
                    mod_per = np.random.permutation(r_mod)
                    m1 = mod_per[0]; m2 = mod_per[1]

                    #make a graph partition object
                    graph_partition = mod.GraphPartition(g,part)

                    #index of nodes within the original module (before merge)
                    n1_init = list(graph_partition.index[m1])
                    n2_init = list(graph_partition.index[m2])
                    n_all_init = n1_init+n2_init

                    #calculate modularity before merging
                    mod_init = graph_partition.modularity()

                    #merge modules
                    merge_module,e1,a1,delta_energy_meas,type,m1,m2,m2 = \
                                  graph_partition.compute_module_merge(m1,m2)

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
                    npt.assert_almost_equal(delta_energy_meas,
                                                  delta_energy_true)
                    # Check that the list of nodes in the two original modules
                    # is equal to the list of nodes in the merged module
                    n_all_init.sort()
                    n_all.sort()
                    npt.assert_equal(n_all_init, n_all)

                    # Test that the keys are equivalent after merging modules
                    npt.assert_equal(r_mod[:-1],
                                           sorted(graph_part2.index.keys()))

                    # Test that the values in the mod_e and mod_a matrices for
                    # the merged module are correct.
                    npt.assert_equal(graph_part2.mod_e[min(m1,m2)],e1)
                    npt.assert_equal(graph_part2.mod_a[min(m1,m2)],a1)


def test_rename_keys():
    a = {0:0,1:1,2:2,4:4,5:5}
    mod.rename_keys(a, 3)
    npt.assert_equal(a, {0:0,1:1,2:2,3:4,4:5})

    a = {0:0,1:1,3:3,}
    mod.rename_keys(a, 2)
    npt.assert_equal(a, {0:0,1:1,2:3})

    # If called with the last key in dict, it should leave the input alone
    a = {0:0,1:1,2:2,3:3}
    mod.rename_keys(a, 3)
    npt.assert_equal(a, a)


def danon_benchmark():
    """This test comes from Danon et al 2005. It will create the line plot of
    Mututal Information vs. betweenness fraction to assess the performance of
    the simulated annealing algorithm."""
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
                        g = mod.random_modular_graph(nnod, nmod, av_degree,btwn_frac)
                        #Compute the # of nodes per module
                        nnod_mod = nnod/nmod
                        #Make a "correct" partition for the graph
                        ppart = mod.perfect_partition(nmod,nnod_mod)

                        graph_out, graph_dict =mod.simulated_annealing(g,
                        temperature = temperature,temp_scaling = temp_scaling,
                        tmin=tmin, extra_info = True)

                        #print "SA partition",graph_out.index
                        mi = mod.mutual_information(ppart,graph_out.index)
                        t2 = time.clock()
                        print 'Elapsed time: ', (float(t2-t1)/60), ' minutes'
                        print 'partition similarity: ',mi
                        mi_arr[ix,rep] = mi
                        ## plot_partition(g,graph_out.index,'mi: '+ str(mi),'danon_test_6mod'+str(btwn_frac)+'_graph.png')
                        x_mod.append(betweenness_to_modularity(g,ppart))


                    ## mi_arr_avg = np.mean(mi_arr,1)
                    ## plt.figure()
                    ## plt.plot(btwn_fracs,mi_arr_avg)
                    ## plt.xlabel('Betweenness fraction')
                    ## plt.ylabel('Mutual information')
                    ## plt.savefig('danon_test_6mod/danontest_btwn.png')

                    ## plt.figure()
                    ## plt.plot(x_mod,mi_arr_avg)
                    ## plt.xlabel('Modularity')
                    ## plt.ylabel('Mutual information')
                    ## plt.savefig('danon_test_6mod/danontest_mod.png')

    #plt.figure()
    #plt.plot(graph_dict['energy'], label = 'energy')
    #plt.plot(graph_dict['temperature'], label = 'temperature')
    #plt.xlabel('Iteration')

    return mi_arr


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
                    g = mod.random_modular_graph(nnod, nmod, av_degree,btwn_frac)
                    #Compute the # of nodes per module
                    nnod_mod = nnod/nmod
                    #Make a "correct" partition for the graph
                    ppart = mod.perfect_partition(nmod,nnod_mod)

                    graph_out, energy_array, rej_array, temp_array =mod.simulated_annealing(g,
                    temperature = temperature,temp_scaling = temp_scaling,
                    tmin=tmin, nochange_ratio_min = nochange_ratio_min)

                    print "perfect partition", ppart
                    print "SA partition",graph_out.index

                    t2 = time.clock()
                    print 'Elapsed time: ', float(t2-t1)/60, 'minutes'
                    print 'partition similarity: ',mod.mutual_information(ppart,graph_out.index)
                    return graph_out, g, energy_array, rej_array, ppart, temp_array


def test_mutual_information_simple():
    """MI computations with hand-validated values.
    """
    from math import log
    # Define two simple partitions that are off by one assignment
    a = {0:[0, 1], 1:[2, 3], 2:[4, 5]}
    b = {0:[0, 1], 1:[2, 3, 4], 2:[5]}
    N_true = np.array([ [2,0,0], [0,2,0], [0,1,1] ], dtype=float)
    N = mod.confusion_matrix(a, b)
    # test confusion matrix
    npt.assert_equal(N, N_true)
    # Now compute mi by hand
    num = -6*log(3)-4*log(2)
    den = -(3*log(2)+8*log(3)+log(6))
    mi_true = num/den
    mi = mod.mutual_information(a, b)
    npt.assert_almost_equal(mi, mi_true)
    # Let's now flip the labels and confirm that the computation is impervious
    # to module labels
    b2 = {2:[0, 1], 0:[2, 3, 4], 1:[5]}
    npt.assert_almost_equal(mod.mutual_information(b, b2), 1)
    npt.assert_almost_equal(mod.mutual_information(a, b2), mi)


def test_mutual_information_empty():
    """Validate that empty modules don't affect MI.
    """
    # Define two simple partitions that are off by one assignment
    a = {0:[0, 1], 1:[2, 3], 2:[4, 5]}
    b = {0:[0, 1], 1:[2, 3], 2:[4, 5], 3:[]}

    try:
        mod.mutual_information(a, b)
    except ValueError, e:
        nt.assert_equals(e.args[0], "Empty module in second partition.")

    try:
        mod.mutual_information(b, a)
    except ValueError, e:
        nt.assert_equals(e.args[0], "Empty module in first partition.")


def test_mutual_information():
    """ Test the function which returns the mutual information in two
    partitions

    XXX - This test is currently incomplete - it only checks the most basic
    case of MI(x, x)==1, but doesn't do any non-trivial checks.
    """

    # nnod_mod, av_degrees, nmods
    networks = [ [4, [2, 3], [2, 4, 6]],
                 [8, [4, 6], [4, 6, 8]],
                 [40, [20], [2]] ]

    for nnod_mod, av_degrees, nmods in networks:
        for nmod in nmods:
            nnod = nnod_mod*nmod
            for av_degree in av_degrees:
                #make a graph object
                g = mod.random_modular_graph(nnod, nmod, av_degree)

                #Compute the of nodes per module
                nnod_mod = nnod/nmod
                #Make a "correct" partition for the graph
                ppart = mod.perfect_partition(nmod,nnod_mod)

                #graph_out, mod_array =mod.simulated_annealing(g, temperature =
                #temperature,temp_scaling = temp_scaling, tmin=tmin)

                #test the perfect case for now: two of the same partition
                #returns 1
                mi_orig  = mod.mutual_information(ppart,ppart)
                npt.assert_equal(mi_orig,1)

                #move one node and test that mutual_information comes out
                #correctly
                graph_partition = mod.GraphPartition(g,ppart)
                graph_partition.node_update(0,0,1)

                mi = mod.mutual_information(ppart,graph_partition.index)
                npt.assert_array_less(mi, mi_orig)
                ## NOTE: CORRECTNESS NOT TESTED YET

                #merge modules and check that mutual information comes out
                #correctly/lower
                graph_partition2 = mod.GraphPartition(g,ppart)
                merged_module, e_new, a_new, d,t,m1,m2,x = graph_partition2.compute_module_merge(0,1)
                graph_partition2.apply_module_merge(m1,m2,merged_module,e_new,a_new)
                mi2 = mod.mutual_information(ppart,graph_partition2.index)
                npt.assert_array_less(mi2,mi_orig)
                ## NOTE: CORRECTNESS NOT TESTED YET

                #split modules and check that mutual information comes out
                #correclty/lower
                graph_partition3 = mod.GraphPartition(g,ppart)
                n1 = set(list(graph_partition3.index[0])[::2])
                n2 = set(list(graph_partition3.index[0])[1::2])

                (split_modules, e_new,
                 a_new, d, t, m,
                 n1,n2) = graph_partition3.compute_module_split(0,n1,n2)
                graph_partition3.apply_module_split(m, n1, n2, 
                                                    split_modules,
                                                    e_new, a_new)
                mi3 = mod.mutual_information(ppart, graph_partition3.index)
                npt.assert_array_less(mi3,mi_orig)
                ## NOTE: CORRECTNESS NOT TESTED YET


def test_random_mod():
    """ Test the GraphPartition operation that selects random modules 
    to merge and split
    XXX not working yet"""

    #nnod_mod, av_degrees, nmods
    networks = [ [4, [2, 3], [2, 4, 6]],
                 [8, [4, 6], [4, 6, 8]] ]

    n_iter = 100
    for nnod_mod, av_degrees, nmods in networks:
        for nmod in nmods:
            nnod = nnod_mod*nmod
            for av_degree in av_degrees:

                g = mod.random_modular_graph(nnod, nmod, av_degree)
                part = dict()
                while (len(part) <= 1) or (len(part) == nnod):
                    part = mod.rand_partition(g)

                graph_partition = mod.GraphPartition(g,part)

                for i in range(n_iter):
                    graph_partition.random_mod()

                    #check that the partition has > 1 modules
                    true = len(graph_partition)>1
                    npt.assert_equal(true,1)

                    #check that the partition has < nnod modules
                    true = len(graph_partition)<nnod
                    npt.assert_equal(true,1)



def test_random_nod():
    """ Test the GraphPartition operation that selects random nodes to move
    between modules """


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
                keep_list[i] = float(mod.decide_if_keeping(dE,temp))

            if dE <= 0:
                keep_correct = np.ones(iter)
                npt.assert_equal(keep_list,keep_correct)
            else:
                mean_keep = np.mean(keep_list)
                mean_correct = math.exp(-dE/temp)
                npt.assert_almost_equal(mean_keep,mean_correct, tolerance)

def test_sim_anneal_simple():
    """Very simple simulated_annealing test with a small network"""

    #
    nnod, nmod, av_degree, btwn_frac = 24, 3, 4, 0
    g = mod.random_modular_graph(nnod, nmod, av_degree, btwn_frac)

    #Compute the # of nodes per module
    ## nnod_mod = nnod/nmod
    #Make a "correct" partition for the graph
    ## ppart = mod.perfect_partition(nmod,nnod_mod)

    temperature = 10
    temp_scaling = 0.95
    tmin=1

    graph_out, graph_dict = mod.simulated_annealing(g,
               temperature = temperature, temp_scaling = temp_scaling,
               tmin=tmin, extra_info = True, debug=True)

    # Ensure that there are no empty modules
    util.assert_no_empty_modules(graph_out.index)

    ## mi = mod.mutual_information(ppart, graph_out.index)
    #nt.assert_equal(mi, 1)

def test_apply_module_split():
    """Test the GraphPartition operation that splits modules so that it returns
    a change in modularity that reflects the difference between the modularity
    of the new and old parititions.
    Also test that the module that was split now contains the correct nodes,
    the correct modularity update, the correct energy,and that no empty modules
    result from it."""

    # nnod_mod, av_degrees, nmods
    networks = [ [3, [2], [2, 3, 4]],
                 [4, [2, 3], [2, 4, 6]],
                 [8, [4, 6], [4, 6, 8]] ]

    for nnod_mod, av_degrees, nmods in networks:
        for nmod in nmods:
            nnod = nnod_mod*nmod
            for av_degree in av_degrees:

                g = mod.random_modular_graph(nnod, nmod, av_degree)

                # Make a "correct" partition for the graph
                ## part = mod.perfect_partition(nmod,nnod/nmod)

                # Make a random partition for the graph
                part_rand = mod.rand_partition(g, nnod/2)

                #List of modules in the partition that have two or more nodes
                r_mod = []
                for m, nodes in part_rand.iteritems():
                    if len(nodes)>2:
                        r_mod.append(m)

                # Note: The above can be written in this more compact, if
                # slightly less intuitively clear, list comprehension:
                # r_mod = [ m for m, nodes in part_rand.iteritems() if
                #          len(nodes)>2 ]

                #Module that we are splitting
                for m in r_mod:

                    graph_partition = mod.GraphPartition(g,part_rand)

                    #index of nodes within the original module (before split)
                    n_init = list(graph_partition.index[m])

                    #calculate modularity before splitting
                    mod_init = graph_partition.modularity()

                    # assign nodes to two groups
                    n1_orig,n2_orig = graph_partition.determine_node_split(m)

                    # make sure neither of these is empty
                    nt.assert_true(len(n1_orig)>= 1)
                    nt.assert_true(len(n2_orig)>= 1)

                    #make sure that there are no common nodes between the two
                    node_intersection = set.intersection(n1_orig,n2_orig)
                    nt.assert_equal(node_intersection,set([]))

                    #make sure that sum of the two node sets equals the
                    #original set
                    node_union = set.union(n1_orig,n2_orig)
                    npt.assert_equal(np.sort(list(node_union)),np.sort(n_init))

                    # split modules
                    split_modules,e1,a1,delta_energy_meas,type,m,n1,n2 = \
                               graph_partition.compute_module_split(m,n1_orig,n2_orig)

                    #note: n1 and n2 are output from this function (as well as
                    #inputs) because the function is called from within another
                    #(rand_mod) def but then output to the simulated script, so
                    #the node split needs to be passed along.

                    #as a simple confirmation, can make sure they match
                    npt.assert_equal(n1_orig,n1)
                    npt.assert_equal(n2_orig,n2)

                    #split_moduels should be a dictionary with two modules
                    #(0,1) that contain the node sets n1 and n2 respectively.
                    #test this.
                    npt.assert_equal(split_modules[0],n1)
                    npt.assert_equal(split_modules[1],n2)

                    #make a new graph partition equal to the old one and apply
                    #the module split to it (graph_part2)
                    graph_part2 = copy.deepcopy(graph_partition)
                    graph_part2.apply_module_split(m,n1,n2,split_modules,e1,a1)

                    #make a third graph partition using only the original graph
                    #and the partition from graph_part2
                    graph_part3 = mod.GraphPartition(g,graph_part2.index)

                    #index of nodes within the modules after splitting
                    n1_new = list(graph_part2.index[m])
                    n2_new = list(graph_part2.index[len(graph_part2)-1])
                    n_all = n1_new + n2_new

                    # recalculate modularity after splitting
                    mod_new = graph_part2.modularity()
                    mod_new_3 = graph_part3.modularity()

                    # difference between new and old modularity
                    delta_energy_true = -(mod_new - mod_init)

                    # Test that the measured change in energy by splitting a
                    # module is equal to the function output from module_split
                    npt.assert_almost_equal(delta_energy_meas,
                                                  delta_energy_true)

                    # Test that the nodes in the split modules are equal to the
                    # original nodes of the module
                    nt.assert_equal(sorted(list(n1)), sorted(n1_new))
                    nt.assert_equal(sorted(list(n2)), sorted(n2_new))

                    n_init.sort()
                    n_all.sort()
                    # Test that the initial list of nodes in the module are
                    # equal to the nodes in m1 and m2 (split modules)
                    npt.assert_equal(n_init,n_all)

                    # Test that the computed modularity found when
                    # apply_module_split is used is equal to the modularity you
                    # would find if using that partition and that graph
                    npt.assert_almost_equal(mod_new,mod_new_3)

                    # Check that there are no empty modules in the final
                    # partition
                    for m in graph_part2.index:
                        nt.assert_true(len(graph_part2.index[m]) > 0)


def test_apply_node_move():
    """Test the GraphPartition operation that moves a single node so that it
    returns a change in modularity that reflects the difference between the
    modularity of the new and old parititions"""

    # nnod_mod, av_degrees, nmods
    #networks = [ [3, [2], [2, 3, 4]],
    #             [4, [2, 3], [2, 4, 6]],
    #             [8, [4, 6], [4, 6, 8]] ]
    networks = [ [4, [2, 3], [2, 4, 6]],
                 [8, [4, 6], [4, 6, 8]] ]


    for nnod_mod, av_degrees, nmods in networks:
        for nmod in nmods:
            nnod = nnod_mod*nmod
            for av_degree in av_degrees:
                print nnod_mod,nmod,av_degree
                g = mod.random_modular_graph(nnod, nmod, av_degree)

                #Make a "correct" partition for the graph
                #part = mod.perfect_partition(nmod,nnod/nmod)

                #Make a random partition for the graph
                part_rand = dict()
                while len(part_rand) <= 1: #check if there is only one module
                    part_rand = mod.rand_partition(g)

                #List of modules in the partition
                r_mod=range(len(part_rand))

                #Make a graph_partition object
                graph_partition = mod.GraphPartition(g,part_rand)

                #select two modules to change node assignments
                mod_per = np.random.permutation(r_mod)
                m1 = mod_per[0]; m2 = mod_per[1]
                while len(graph_partition.index[m1]) <= 1:
                    mod_per = np.random.permutation(r_mod)
                    m1 = mod_per[0]
                    m2 = mod_per[1]

                #pick a random node to move between modules m1 and m2
                node_list=list(graph_partition.index[m1])
                nod_per = np.random.permutation(node_list)
                n = nod_per[0]

                #list of nodes within the original modules (before node move)
                ## n1_init = list(nod_per) #list(graph_partition.index[m1])
                ## n2_init = list(graph_partition.index[m2])
                ## n1_new = copy.deepcopy(n1_init)
                ## n2_new = copy.deepcopy(n2_init)

                # calculate modularity before node move
                mod_init = graph_partition.modularity()

                # move node from m1 to m2
                node_moved_mods,e1,a1,delta_energy_meas,n,m1,m2 = \
                              graph_partition.compute_node_update(n,m1,m2)

                graph_part2 = copy.deepcopy(graph_partition)
                m2_new = graph_part2.apply_node_update(n,m1,m2,node_moved_mods,e1,a1)

                #if the keys get renamed, the m1,m2 numbers are no longer the same

                #test that m2 now contains n
                nt.assert_true(n in graph_part2.index[m2_new])
                #if n not in graph_part2.index[m2_new]:
                #    1/0

                # recalculate modularity after splitting
                mod_new = graph_part2.modularity()

                # difference between new and old modularity
                delta_energy_true = -(mod_new - mod_init)
                #print delta_energy_meas,delta_energy_true

                # Test that the measured change in energy is equal to the true
                # change in energy calculated in the node_update function
                npt.assert_almost_equal(delta_energy_meas,delta_energy_true)


def test_adjust_partition():
    e = np.loadtxt(os.path.join(os.path.dirname(__file__), 'jazz.net'),
                   skiprows=3, dtype=int)[:, :2] - 1
    g = nx.Graph()
    g.add_edges_from(e)

    p0 = mod.newman_partition(g)
    p1 = mod.adjust_partition(g, p0, max_iter=6)

    npt.assert_(p0 > 0.38)
    npt.assert_(p1 > 0.42)


def test_empty_graphpartition():
    g = nx.Graph()
    g.add_node(1)
    npt.assert_raises(ValueError, mod.GraphPartition, g, {1: set(g.nodes())})


def test_badindex_graphpartition():
    """ when making a GraphPArtition, check index is valid"""
    ## index should be dict of sets
    e = np.loadtxt(os.path.join(os.path.dirname(__file__), 'jazz.net'),
                                skiprows=3, dtype=int)[:, :2] - 1
    g = nx.Graph()
    g.add_edges_from(e)
    index = {0: set(g.nodes()[:100]), 1: set(g.nodes()[100:])}
    gp = mod.GraphPartition(g, index)
    nt.assert_true(gp.index == index)
    npt.assert_raises(TypeError, mod.GraphPartition, g, {0: g.nodes()})
    npt.assert_raises(ValueError, mod.GraphPartition, g, 
                      {0:set(g.nodes()[:-1])})
    npt.assert_raises(TypeError, mod.GraphPartition, g, g.nodes())


if __name__ == "__main__":
    npt.run_module_suite()
