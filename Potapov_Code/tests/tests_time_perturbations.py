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

def test_altered_delay_pert(plot=False,eps=1e-5):
    r'''
    We will have a method to shift the delays in the network before the
    commensurate root analysis, which will be based on taking the average
    Delta_delays that result from the nonlinearities over the different
    frequencies. We test this here.

    It also tests the corresponding perturbation in the frequencies.

    We assume that the refraction_index_func and the input delays into
    the Time_Delay_Network have been adjusted so that refraction_index_func
    is close to zero in the desired frequency range.

    There are several effects of the delays being different for different
    modes. The most important one is an effective detuning for different
    modes (as well as decay). There are other effects as well. The effective
    mode volume will also change (this is taken into account in the
    Hamitonian class). However, this is not taken into account in the Potapov
    expansion because it becomes computationally difficult and the effect
    will be small. This could be done in principle. The time delays in the
    transfer function could be written as a function of frequency,
    :math:`T = T(\omega)`.
    The above function can be analytically continued to the complex plane.
    Then the transfer function would be expressed
    in terms of :math:`exp(-z T) = exp ( -z T (z))`.
    Once this is done, the complex root-finding procedure can be applied.
    The difficulty in using this approach is that the resulting functions no
    longer have a periodic structure that we could identify when the delays
    were commensurate.
    '''

    Ex = Time_Delay_Network.Example3( max_linewidth=15.,max_freq=500.)
    Ex.run_Potapov(commensurate_roots=True)
    modes = Ex.spatial_modes
    A,B,C,D = Ex.get_Potapov_ABCD(doubled=False)
    ham = Hamiltonian.Hamiltonian(Ex.roots,modes,Ex.delays,Omega=-1j*A,
                nonlin_coeff = 1.)

    ## This nonlinearity will depend on the frequency.
    chi_nonlin_test = Hamiltonian.Chi_nonlin(delay_indices=[0],start_nonlin=0,
                               length_nonlin=0.1*consts.c)
    chi_nonlin_test.refraction_index_func = lambda freq, pol: 1. + abs(freq / (5000*np.pi))
    ham.chi_nonlinearities.append(chi_nonlin_test)

    ## update delays, which are different becuase of the nonlinearity.
    ham.make_Delta_delays()
    #print ham.Delta_delays

    ## Perturb the roots to account for deviations in the index of refraction
    ## as a function of frequency.


    # print ham.roots
    perturb_func = Ex.get_frequency_pertub_func_z(use_ufuncify = True)
    ham.perturb_roots_z(perturb_func)
    # print ham.roots
    print len(ham.roots)
    # plt.plot(ham.omegas)
    if plot:
        plt.scatter(np.asarray(ham.roots).real,np.asarray(ham.roots).imag)
        plt.show()

    # TODO: make a function to perturb in several steps to avoid root-skipping.


def test_delay_perturbations(eps=1e-5):
    '''
    This funciton tests the parturbations for the delays for each frequency.

    It also tests the corresponding perturbation in the frequencies.
    '''

    Ex = Time_Delay_Network.Example3( max_linewidth=15.,max_freq=30.)
    Ex.run_Potapov(commensurate_roots=True)
    modes = Ex.spatial_modes
    M = len(Ex.roots)

    A,B,C,D = Ex.get_Potapov_ABCD(doubled=False)

    ham = Hamiltonian.Hamiltonian(Ex.roots,modes,Ex.delays,Omega=-1j*A,
                nonlin_coeff = 0.)

    ham.make_chi_nonlinearity(delay_indices=[0],start_nonlin=0,
                               length_nonlin=0.1*consts.c)
    ham.make_Delta_delays()
    #print ham.Delta_delays
    for row in ham.Delta_delays:
        for el in row:
            assert(el == 0)

    ## Now let's make a non-trivial nonlinearity.

    ## turn on the nonlin_coeff
    ham.nonlin_coeff = 1.

    ## set the index of refraction to be 2 for the nonlinearity
    ham.chi_nonlinearities[0].refraction_index_func = lambda *args: 2.

    ham.make_Delta_delays()
    # print ham.Delta_delays

    ## Next, generate the perturb_func and perturb the roots
    #print Ex.roots
    perturb_func = Ex.get_frequency_pertub_func_z(use_ufuncify = True)
    ham.perturb_roots_z(perturb_func)





if __name__ == "__main__":
    test_altered_delay_pert(plot=True)
