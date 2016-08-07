from .. import Potapov
from .. import Roots
from .. import Time_Delay_Network
from .. import functions
from .. import Time_Sims_nonlin
from .. import Hamiltonian
from .. import phase_matching_hash

import numpy.testing as testing
import numpy as np
import numpy.linalg as la
from scipy.integrate import ode
import scipy.constants as consts
import random

import matplotlib.pyplot as plt
import time

def test_make_positive_keys_chi3( eps = 2e-4,
                                  pols = (-1,-1,-1,-1),
                                  res = (1e-1,1e-1,1e-1),
                                  min_value = 10.,
                                  max_value = 55.
                                  ):

    pos_nus_lst = np.random.uniform(min_value,max_value,500)

    ## make a Hamilonian with a chi - 3.
    ham = Hamiltonian.Hamiltonian([],[],[])
    ham.make_chi_nonlinearity(delay_indices=[0],start_nonlin=0,
                             length_nonlin=0.1,
                             chi_order=3)
    chi = ham.chi_nonlinearities[0]

    matching_dict_hash = phase_matching_hash.make_positive_keys_chi3(pos_nus_lst, chi, eps=eps, pols = pols, res = res )

    print len(matching_dict_hash)
    return

def test_Hamiltonian_calling_make_weight_keys(eps = 2e-4,
                                              pols = (-1,-1,-1,-1),
                                              min_value = 10.,
                                              max_value = 55.):

    ## The positive nu's to use.
    pos_nus_lst = np.random.uniform(min_value,max_value,500)

    ham = Hamiltonian.Hamiltonian([],[],[])
    ham.omegas = [nu * 1e13 / (2*consts.pi) for nu in pos_nus_lst]

    ## assign random polarizations
    ham.polarizations = 2*np.random.randint(0,2,500)-1

    ## make a nonlinearity of order 2 (make_weight_keys checks for this)
    ham.make_chi_nonlinearity(delay_indices=[0],start_nonlin=0,
                             length_nonlin=0.1,
                             chi_order=3)
    chi = ham.chi_nonlinearities[0]

    ## Use the make_weight_keys() with the selected ham and chi with
    ## the correct key_types
    weight_keys = Hamiltonian.Hamiltonian.make_weight_keys(ham, chi, key_types = 'hash_method', pols = pols,  )

    L = len(weight_keys)
    print L
    if L > 0:
        print weight_keys[0]
    else:
        print "zero weight keys"
    return

if __name__ == "__main__":
    test_phase_matching_chi_3(plot=True)
    test_get_freqs_from_ham()
