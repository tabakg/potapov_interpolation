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
                              eps = 0.00002,
                              starting_i = 0,
                              max_i = 4,
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


if __name__ == "__main__":
    test_phase_matching_chi_2(plot=True)
