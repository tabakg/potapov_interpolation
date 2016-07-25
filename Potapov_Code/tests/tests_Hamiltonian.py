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

from sympy.physics.quantum import *
from sympy.physics.quantum.boson import *
from sympy.physics.quantum.operatorordering import *
from qnet.algebra.circuit_algebra import *

def test_Hamiltonian_with_doubled_equations(eps=1e-5):
    '''
    This method tests various methods in Hamiltonian and Time_Sims_nonlin.
    In particular, we compare the output from the classical equations of motion
    that results directly from the ABCD model versus the classical Hamiltonian
    equations of motion when we set the coefficient of the nonlinearity to zero.

    This method will NOT test the details of the nonlinear Hamiltonian.

    Args:
        eps[optional(float)]: how closely each point in time along the two
        tested trajectories should match.
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

    H = ham.make_H()

    ## Make the classical equation of motion from Hamilton's equations.
    eq_mot = ham.make_eq_motion()

    ## make a sample input function
    a_in = lambda t: np.asmatrix([1.]*np.shape(D_d)[-1]).T

    ## find f, the system evolution function from Hamilton's equations
    f = Time_Sims_nonlin.make_f(eq_mot,B_d,a_in)

    ## Generate the linear equations of motion from the original linear system matrices
    f_lin = Time_Sims_nonlin.make_f_lin(A_d,B_d,a_in)


    ## Simulate the systems (both linear and nonlinear).
    Y_lin = Time_Sims_nonlin.run_ODE(f_lin, a_in, C_d, D_d, 2*M, T = 15, dt = 0.01)
    Y_nonlin = Time_Sims_nonlin.run_ODE(f, a_in, C_d, D_d, 2*M, T = 15, dt = 0.01)

    for y_lin,y_nonlin in zip(Y_lin,Y_nonlin):
        assert abs(sum(y_lin - y_nonlin)) < eps


if __name__ == "__main__":
    test_Hamiltonian_with_doubled_equations()
