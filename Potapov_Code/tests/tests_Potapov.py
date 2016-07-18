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

def test_Potapov_1(eps=1e-7):
    '''
    Generate a finite_transfer_function from eigenvectors and eigenvalues.
    Then generate a Potapov product from the finite transfer function. These
    should be analytically equal. We test to see if they are close within some
    precision.
    '''
    vals = [1-1j,-1+1j, 2+2j]
    vecs = [ Potapov.normalize(np.matrix([-5.,4j])).T, Potapov.normalize(np.matrix([1j,3.]).T),
            Potapov.normalize(np.matrix([2j,7.]).T)]
    T = Potapov.finite_transfer_function(np.eye(2),vecs,vals)
    T_test = Potapov.get_Potapov(T,vals,vecs)

    points = [0.,10j,10.,-10j,10.+10j]

    assert all(np.amax(abs(T(z) - T_test(z))) < eps for z in points)


if __name__ == "__main__":
    test_Potapov_1()
