import sympy as sp
import numpy as np
import scipy.constants
from sympy.utilities.autowrap import ufuncify
from scipy import interpolate
import scipy.constants as consts
import matplotlib.pyplot as plt
from functions import timeit
from functions import make_dict_values_to_lists_of_inputs
import itertools

@timeit
def setup_ranges(max_i,base,min_value = 6.,max_value = 11.):
    ranges= {}
    for i in range(max_i+1):
        ranges[i] = np.linspace(min_value,max_value,1+pow(base,i+1))
    return ranges

@timeit
def initial_voxels(ranges,k_of_nu1_nu2,max_i,base,starting_i,eps):
    solution_containing_voxels = {}
    eps_current = eps * pow(base,max_i-starting_i)
    solution_containing_voxels[starting_i] = {}

    for i1,om1 in enumerate(ranges[starting_i]):
        for i2,om2 in enumerate(ranges[starting_i]):
            err = k_of_nu1_nu2(om1,om2)
            if abs(err) < eps_current:
                solution_containing_voxels[starting_i][i1,i2] = err
    return solution_containing_voxels

@timeit
def add_high_res_voxels(ranges,k_of_nu1_nu2,max_i,base,starting_i,eps,solution_containing_voxels):
    for i in range(starting_i+1,max_i+1):
        eps_current = eps * pow(base,max_i-i)
        solution_containing_voxels[i] = {}
        for (i1,i2) in solution_containing_voxels[i-1]:
            step_size = int(base/2)
            max_length = pow(base,i+1)
            for i1_new in range(max(0,i1*base-step_size),min(max_length,i1*base+step_size+1)):
                for i2_new in range(max(0,i2*base-step_size),min(max_length,i2*base+step_size+1)):
                    err = k_of_nu1_nu2(ranges[i][i1_new],ranges[i][i2_new])
                    if abs(err) < eps_current:
                        solution_containing_voxels[i][i1_new,i2_new] = err

@timeit
def plot_voxels(solution_containing_voxels,i):
    voxels = np.zeros((1+pow(base,i+1),1+pow(base,i+1)))
    for (i1,i2) in solution_containing_voxels[i]:
        voxels[i1,i2] = 1
    plot_arr(voxels)

def voxel_solutions(ranges,k_of_nu1_nu2,max_i,base,starting_i,eps):
    solution_containing_voxels = initial_voxels(ranges,k_of_nu1_nu2,max_i,
                                                base,starting_i,eps)
    add_high_res_voxels(ranges,k_of_nu1_nu2,max_i,base,starting_i,eps,
                        solution_containing_voxels)
    return solution_containing_voxels

######### TODO: put the above methods in their own file.

def generate_k_func(pols=(1,1,-1),n_symb = None):

    lambd,nu,nu1,nu2,nu3,nu4 = sp.symbols(
        'lambda nu nu_1 nu_2 nu_3 nu_4')
    l2 = lambd **2

    if n_symb is None:
        def n_symb(pol=1):
            '''Valid for lambda between 0.5 and 5. (units are microns)'''
            s = 1.
            if pol == 1:
                s += 2.6734 * l2 / (l2 - 0.01764)
                s += 1.2290 * l2 / (l2 - 0.05914)
                s += 12.614 * l2 / (l2 - 474.6)
            else:
                s += 2.9804 * l2 / (l2 - 0.02047)
                s += 0.5981 * l2 / (l2 - 0.0666)
                s += 8.9543 * l2 / (l2 - 416.08)
            return sp.sqrt(s)

    def k_symb(symbol=nu,pol=1):
        '''k is accurate for nu inputs between 6-60 (units are 1e13 Hz).'''
        return ((n_symb(pol=pol) * symbol )
                    .subs(lambd,scipy.constants.c / (symbol*1e7)))

    expressions = [k_symb(nu1,pols[0]),
                   k_symb(nu2,pols[1]),
                   k_symb(nu3,pols[2])]
    dispersion_difference_function = sum(expressions)
    dispersion_difference_function = dispersion_difference_function.subs(
                                     nu3,-nu1-nu2)
    k_of_nu1_nu2 = ufuncify([nu1,nu2],
                                   dispersion_difference_function)
    return k_of_nu1_nu2

def make_positive_keys_chi2(pos_nus_lst,chi,eps=1e-4,
        starting_i = 0,max_i = 2,base = 10,pols = None):
    '''
    takes a Hamiltonian and a Chi_nonlin with chi_order = 2 and generates
    pairs of positive numbers (nu1,nu2) corresponding to solutions
    (nu1,nu2,-nu1-nu2) of the phase matching problem. Here,
    the nu1,nu2 are selected from ham.omegas (assumed to be positive).

    TODO: In the future, k_of_nu1_nu2 will be generated from
          chi_nonlin.refraction_index_func in the form of a function
          n_symb(pol=+1) -> sympy expression in lambd.

    TODO:
        if the error isn't being used, can use a set instead of a dict
        for solution_containing_voxels.
    '''
    if pols == None: ##some default values for polarization
        pols = (1,1,-1)

    ### In the future, k_of_nu1_nu2 will be generated from chi_nonlin.refraction_index_func
    k_of_nu1_nu2 = generate_k_func(pols)

    ## get the positive nus. Make a dict to the original index.
    min_value = min(pos_nus_lst)
    max_value = max(pos_nus_lst)

    ranges = setup_ranges(max_i,base,min_value = min_value,max_value = max_value)

    Delta = ranges[max_i][1] - ranges[max_i][0] ## spacing in grid used

    ## get index values
    values = [ int(round( (freq - min_value) / Delta)) for freq in pos_nus_lst]

    ## make a dict to remember which frequencies belong in which grid voxel.
    grid_indices_to_unrounded = make_dict_values_to_lists_of_inputs(values,pos_nus_lst)
    grid_indices_to_ham_index = make_dict_values_to_lists_of_inputs(values,range(len(pos_nus_lst)))

    solution_containing_voxels = voxel_solutions(ranges,k_of_nu1_nu2,
        max_i,base,starting_i,eps)

    ## Let's figure out which indices we can expect for nu3
    spacing = (max_value-min_value)/ pow(base,max_i+1)
    num_indices_from_zero = min_value / spacing  ## float, round up or down

    solutions_nu1_and_nu2 = solution_containing_voxels[max_i].keys()

    solution_indices = []
    for indices in solutions_nu1_and_nu2:
        for how_to_round_last_index in range(2):
            last_index = (sum(indices)
                          + int(num_indices_from_zero)
                          + how_to_round_last_index)
            if last_index < 0 or last_index >= len(ranges[max_i]):
                print "breaking!"
                break
            current_grid_indices = (indices[0],indices[1],last_index)
            if all([ind in grid_indices_to_ham_index for ind in current_grid_indices]):
                for it in itertools.product(*[grid_indices_to_ham_index[ind] for ind in current_grid_indices]):
                    solution_indices.append(it)

    return solution_indices

if __name__ == "__main__":

    pols = (1,1,-1)
    k_of_nu1_nu2 = generate_k_func(pols)

    eps = 0.006
    starting_i = 0
    max_i = 2
    base = 10

    min_value = 6.
    max_value = 20.
    ranges = setup_ranges(max_i,base,min_value,max_value)

    solution_containing_voxels = voxel_solutions(ranges,k_of_nu1_nu2,
        max_i,base,starting_i,eps)
