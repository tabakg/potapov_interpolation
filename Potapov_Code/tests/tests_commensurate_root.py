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

def test_commensurate_vecs_example_3():
    times = [time.clock()]
    X = Time_Delay_Network.Example3(tau1 = 0.1, tau2 = 0.2,tau3 = 0.1,tau4 = 0.2,)
    times.append(time.clock())
    X.make_commensurate_roots([(-60000,60000)])
    times.append(time.clock())
    X.make_commensurate_vecs()
    times.append(time.clock())
    times.append(time.clock())
    #print len(X.vecs)
    assert(len(X.roots) == len(X.vecs))
    times.append(time.clock())
    X.make_T_Testing()
    times.append(time.clock())
    T_testing = X.T_testing
    T = X.T
    print abs(T(-10j)-T_testing(-10j))
    print abs(T(-100j)-T_testing(-100j))
    print abs(T(-200j)-T_testing(-200j))

    print [times[i+1]-times[i] for i in range(len(times)-1)]



if __name__ == "__main__":
    test_make_T_denom_sym_separate_delays()
    test_commensurate_vecs_example_3()
