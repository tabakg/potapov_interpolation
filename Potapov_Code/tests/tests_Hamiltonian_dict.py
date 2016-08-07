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

def make_lin_H_from_dict(ham,Omega,eps=1e-3):
    '''Make a linear Hamiltonian based on Omega.

    Args:
        Omega (complex-valued matrix):
            A matrix that describes the Hamiltonian of the system.

    Returns:
        Expression (sympy expression):
            A symbolic expression for the nonlinear Hamiltonian.

    '''
    self.make_dict_H_lin(Omega)
    H_lin_sp = sp.Float(0.)
    for i in range(self.m):
        for j in range(self.m):
            H_lin_sp += self.Dagger(self.a[i])*self.a[j] * self.Hamiltonian_dict_lin[(i,j),(+1,-1)]
    return H_lin_sp

def make_nonlin_H(ham,filtering_phase_weights=False,eps=1e-5):
    '''Old version of make_nonlin_H. To be used for testing.

    '''
    H_nonlin_sp = sp.Float(0.)
    for chi in ham.chi_nonlinearities:
        weight_keys = ham.make_weight_keys(chi)

        phase_matching_weights = ham.make_phase_matching_weights(
            weight_keys,chi,filtering_phase_weights,eps)

        for combination,pm_arr in phase_matching_weights:
            omegas_to_use = map(lambda i: ham.omegas[i],combination)
            omegas_with_sign = [omega * pm for omega,pm
                                in zip(omegas_to_use,pm_arr)]
            pols = map(lambda i: ham.polarizations[i],chi.delay_indices)
            chi_args = omegas_with_sign + pols
            H_nonlin_sp += ( ham.make_nonlin_term_sympy(combination,pm_arr) *
                chi.chi_function(*chi_args) *
                phase_matching_weights[combination,pm_arr] *
                np.prod([ham.E_field_weights[i] for i in combination]) )
    return H_nonlin_sp

def compare_dict_with_old_hams():
    '''
    Make the symbolic expression either directly or with a dictionary and
    confirm that the results are the same.
    '''

    ## Make a sample Time_Delay_Network, changing some parameters.
    X = Time_Delay_Network.Example3(r1 = 0.7, r3 = 0.7, max_linewidth=35.,max_freq=15.)

    ## run the Potapov procedure.
    ## Setting commensurate_roots to True will tell the program to identify
    ## the roots utilizing the periodic structure of the roots.
    X.run_Potapov(commensurate_roots = True)

    ## Get the roots, modes, and delays from the Time_Delay_Network.
    modes = X.spatial_modes
    roots = X.roots
    delays = X.delays

    ## Generated doubled-up ABCD matrices for the passive system.
    ## These matrices are not doubled up
    A,B,C,D = X.get_Potapov_ABCD(doubled=False)

    ## Generated doubled-up ABCD matrices for the passive system.
    ## These matrices not doubled up
    A_d,B_d,C_d,D_d = X.get_Potapov_ABCD(doubled=True)

    M = len(A)

    ## make an instance of Hamiltonian.
    ## The non-Hermitian part of A dictates the linear internal dynamics of the system
    ## the Hermitian part of A dictates the linear decay of the internal modes.

    ham = Hamiltonian.Hamiltonian(roots,modes,delays,Omega=-1j*A,nonlin_coeff = 0.)

    ## Add a chi nonlinearity to ham.
    ham.make_chi_nonlinearity(delay_indices=[0],start_nonlin=0,
                                 length_nonlin=0.1,
                                 chi_order=3)

    lin_H_sp_from_dict = make_lin_H_from_dict(ham,Omega)
    lin_H_sp = ham.make_lin_H(Omega)

    expr_lin = lin_H_sp_from_dict - lin_H_sp
    D = {el:1 for el in ham.a}
    D.update({ham.Dagger(el):1 for el in ham.a})
    expr_lin = expr_lin.subs(D)
    assert (abs(expr_lin) < eps)

    ham.make_dict_H_nonlin()
    nonlin_H_sp_from_dict = ham.Hamiltonian_dict_nonlin
    nonlin_H_sp = ham.make_nonlin_H()

    expr_nonlin = lin_H_sp_from_dict - lin_H_sp
    D = {el:1 for el in ham.a}
    D.update({ham.Dagger(el):1 for el in ham.a})
    expr_nonlin = expr_nonlin.subs(D)
    assert (abs(expr_nonlin) < eps)




if __name__ == "__main__":
    test_phase_matching_chi_2(plot=True)
