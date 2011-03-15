"""Tests for modularity code.
"""

#-----------------------------------------------------------------------------
# Library imports
#-----------------------------------------------------------------------------

# Stdlib
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
from decotest  import (as_unittest, ParametricTestCase, parametric)

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

@parametric
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
                    yield nt.assert_true (abs(av_degree-av_degree_actual) < 1, 
                           "av deg: %.2f  av deg actual: %.2f" %
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
                    yield nt.assert_almost_equal(btwn_fraction,
                                                 btwn_real_frac, 1 )

    
@parametric
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
                yield npt.assert_almost_equal(mod_meas, mod_true, 2)


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
                for i in range(1):
                    #select two modules to merge
                    mod_per = np.random.permutation(r_mod)
                    m1 = mod_per[0]; m2 = mod_per[1]
                    

                    #make a graph partition object
                    graph_partition = mod.GraphPartition(g,part)
                    
                    #index of nodes within the original module (before split)
                    n1_init = list(graph_partition.index[m1])
                    n2_init = list(graph_partition.index[m2])
                    n_all_init = n1_init+n2_init

                    #calculate modularity before splitting
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
                    yield npt.assert_almost_equal(delta_energy_meas,
                                                  delta_energy_true)
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
    of the new and old parititions. Also test that the module that was split
    now contains the correct nodes.""" 

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

                g = mod.random_modular_graph(nnod, nmod, av_degree)

                #Make a "correct" partition for the graph
                part = mod.perfect_partition(nmod,nnod/nmod)

                #Make a random partition for the graph
                part_rand = mod.rand_partition(g)

                #List of modules in the partition
                r_mod=range(len(part_rand))

                #Module that we are splitting
                for m in r_mod[::10]:

                    graph_partition = mod.GraphPartition(g,part_rand)

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

                    split_modules,e1,a1,delta_energy_meas,type,m,n1,n2 = \
                               graph_partition.compute_module_split(m,n1,n2)

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
    
                g = mod.random_modular_graph(nnod, nmod, av_degree)

                #Make a "correct" partition for the graph
                part = mod.perfect_partition(nmod,nnod/nmod)
                
                #Make a random partition for the graph
                part_rand = dict()
                while len(part_rand) <= 1: #check if there is only one module
                    part_rand = mod.rand_partition(g)

                #List of modules in the partition
                r_mod=range(len(part_rand))

                #select two modules to change node assignments
                mod_per = np.random.permutation(r_mod)
                m1 = mod_per[0]; m2 = mod_per[1]

                #Make a graph_partition object
                graph_partition = mod.GraphPartition(g,part_rand)
                
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

                node_moved_mods,e1,a1,delta_energy_meas,n,m1,m2 = \
                              graph_partition.compute_node_update(n,m1,m2)

                graph_part2 = copy.deepcopy(graph_partition)
                graph_part2.apply_node_update(n,m1,m2,node_moved_mods,e1,a1)
                # remove the first node from m1--because we defined n to equal
                # the first element of the randomized node_list 
                n1_new.pop(0)
                
                # append the node to m2
                n2_new.append(n)

                # recalculate modularity after splitting
                mod_new = graph_part2.modularity()
                
                # difference between new and old modularity
                delta_energy_true = -(mod_new - mod_init)
                print delta_energy_meas,delta_energy_true

                # Test that the measured change in energy is equal to the true
                # change in energy calculated in the node_update function
                yield npt.assert_almost_equal(delta_energy_meas,
                                              delta_energy_true)
                #yield npt.assert_equal(n1, n1_new)
                #yield npt.assert_equal(n2, n2_new)
                #yield npt.assert_equal(n_init.sort(),n_all.sort())

    
@parametric
def test_rename_keys():
    a = {0:0,1:1,2:2,4:4,5:5}
    mod.rename_keys(a, 3)
    yield npt.assert_equal(a, {0:0,1:1,2:2,3:4,4:5})

    a = {0:0,1:1,3:3,}
    mod.rename_keys(a, 2)
    yield npt.assert_equal(a, {0:0,1:1,2:3})

    a = {0:0,1:1,2:2,3:[]}
    mod.rename_keys(a, 3)
    yield npt.assert_equal(a, {0:0,1:1,2:2})


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
                g = mod.random_modular_graph(nnod, nmod, av_degree)

                #Compute the of nodes per module
                nnod_mod = nnod/nmod
                #Make a "correct" partition for the graph
                ppart = mod.perfect_partition(nmod,nnod_mod)

                #graph_out, mod_array =mod.simulated_annealing(g, temperature =
                #temperature,temp_scaling = temp_scaling, tmin=tmin)

                #test the perfect case for now: two of the same partition
                #returns 1
                mi  = mod.mutual_information(ppart,ppart)
                yield npt.assert_equal(mi,1)

                #move one node and test that mutual_information comes out
                #correctly
                graph_partition = mod.GraphPartition(g,ppart)
                graph_partition.node_update(0,0,1)

                mi2 = mod.mutual_information(ppart,graph_partition.index)
                mi2_correct = mi
                
                yield npt.assert_almost_equal(mi2,mi2_correct,tolerance)


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

                g = mod.random_modular_graph(nnod, nmod, av_degree)
                part = dict()
                while (len(part) <= 1) or (len(part) == nnod):
                    part = mod.rand_partition(g)

                graph_partition = mod.GraphPartition(g,part)

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
                keep_list[i] = float(mod.decide_if_keeping(dE,temp))

            if dE <= 0:
                keep_correct = np.ones(iter)
                yield npt.assert_equal(keep_list,keep_correct)
            else:
                mean_keep = np.mean(keep_list)
                mean_correct = math.exp(-dE/temp)
                yield npt.assert_almost_equal(mean_keep,mean_correct, tolerance)
