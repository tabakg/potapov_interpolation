from .. import Potapov
from .. import Roots
from .. import Time_Delay_Network
from .. import functions
from .. import Time_Sims_nonlin
from .. import Hamiltonian
from .. import phase_matching

import numpy.testing as testing
import numpy as np
import numpy.linalg as la
from scipy.integrate import ode
import scipy.constants as consts
import random

import matplotlib.pyplot as plt
import time

def plot_arr(arr,name='no_name'):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    cax = ax.matshow(np.asmatrix(arr), interpolation='nearest')
    fig.colorbar(cax)
    plt.savefig('plot of matrix ' + name + '.pdf')

def plot_voxels(solution_containing_voxels,base,i):
    voxels = np.zeros((1+pow(base,i+1),1+pow(base,i+1)))
    for (i1,i2) in solution_containing_voxels[i]:
        voxels[i1,i2] = 1
    plot_arr(voxels, name='voxels with resolution ' + str(i) )

def test_phase_matching_chi_2(plot=False,
                              eps = 0.006,
                              starting_i = 0,
                              max_i = 2,
                              base = 10,
                              pols = (1,1,-1)
                              ):

    k_of_omega1_omega2 = phase_matching.generate_k_func(pols)

    ranges = phase_matching.setup_ranges(max_i,base)
    solution_containing_voxels = phase_matching.voxel_solutions(ranges,k_of_omega1_omega2,
        max_i,base,starting_i,eps)

    if plot:
        for i in range(max_i+1):
            plot_voxels(solution_containing_voxels,base,i)

def test_get_freqs_from_ham():

    pols = (1,1,-1)

    eps = 0.006
    starting_i = 0
    max_i = 2
    base = 10

    min_value = 6.
    max_value = 20.

    ## The positive nu's to use.
    pos_nus_lst = np.random.uniform(min_value,max_value,20000)

    ## assign random polarizations
    polarizations = 2*np.random.randint(0,2,20000)-1

    ## Generate interacting triplets
    positive_omega_indices = phase_matching.make_positive_keys_chi2(pos_nus_lst,None,pols = pols)

    print len(positive_omega_indices)

    positive_omega_indices = [indices for indices in positive_omega_indices
        if all([pols[j] == polarizations[i] for j,i in enumerate(indices)])  ]

    print len(positive_omega_indices)

def test_Hamiltonian_callingmake_weight_keys():

    pols = (1,1,-1)

    eps = 0.006
    starting_i = 0
    max_i = 2
    base = 10

    min_value = 6.
    max_value = 20.

    ## The positive nu's to use.
    pos_nus_lst = np.random.uniform(min_value,max_value,20000)

    ham = Hamiltonian.Hamiltonian([],[],[])
    ham.omegas = [nu * 1e13 / (2*consts.pi) for nu in pos_nus_lst]

    ## assign random polarizations
    ham.polarizations = 2*np.random.randint(0,2,20000)-1

    ## make a nonlinearity of order 2 (make_weight_keys checks for this)
    ham.make_chi_nonlinearity(delay_indices=[0],start_nonlin=0,
                             length_nonlin=0.1,
                             chi_order=2)
    chi = ham.chi_nonlinearities[0]

    ## Use the make_weight_keys() with the selected ham and chi with
    ## the correct key_types
    weight_keys = Hamiltonian.Hamiltonian.make_weight_keys(ham, chi, key_types = 'search_voxels', pols = (1,1,-1) )

    print len(weight_keys)
    print weight_keys[100]


if __name__ == "__main__":
    test_phase_matching_chi_2(plot=True)
    test_get_freqs_from_ham()
