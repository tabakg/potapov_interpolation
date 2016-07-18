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

def test_example_1():
    Ex = Time_Delay_Network.Example1()
    Ex.run_Potapov()
    E = Ex.E
    roots = Ex.roots
    M1 = Ex.M1
    delays = Ex.delays
    modes = functions.spatial_modes(roots,M1,E,delays)
    assert( len(roots) == 3)

def test_example_2():
    Ex = Time_Delay_Network.Example2()
    Ex.run_Potapov()
    E = Ex.E
    roots = Ex.roots
    M1 = Ex.M1
    delays = Ex.delays
    modes = functions.spatial_modes(roots,M1,E,delays)
    assert( len(roots) == 7)

def test_example_3():
    Ex = Time_Delay_Network.Example3()
    Ex.run_Potapov()
    E = Ex.E
    roots = Ex.roots
    M1 = Ex.M1
    delays = Ex.delays
    modes = functions.spatial_modes(roots,M1,E,delays)
    assert( len(roots) == 11)

def test_example_4():
    Ex = Time_Delay_Network.Example4()
    Ex.run_Potapov()
    E = Ex.E
    roots = Ex.roots
    M1 = Ex.M1
    delays = Ex.delays
    modes = functions.spatial_modes(roots,M1,E,delays)
    assert( len(roots) == 8)

def test_commensurate_roots_example_3():
    X = Time_Delay_Network.Example3()
    X.make_commensurate_roots()
    assert(len(X.roots) == 0)
    X.make_commensurate_roots([(0,1000)])
    # assert(len(X.roots) == 91)
    # X.make_commensurate_roots([(0,10000)])
    # assert(len(X.roots) == 931)
    # X.make_commensurate_roots([(0,10000),(1e15,1e15 +10000)])
    # assert(len(X.roots) == 1891)




if __name__ == "__main__":
    test_example_1()
    test_example_2()
    test_example_3()
    test_example_4()
    test_commensurate_roots_example_3()
