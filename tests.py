import Potapov
import Roots
import Examples
import functions
import numpy as np
import numpy.testing as testing
import Time_Sims_nonlin
import Hamiltonian

import numpy as np
import numpy.linalg as la
from scipy.integrate import ode

import matplotlib.pyplot as plt

# def test_Potapov_1(eps=1e-7):
#     '''
#     Generate a finite_transfer_function from eigenvectors and eigenvalues.
#     Then generate a Potapov product from the finite transfer function. These
#     should be analytically equal. We test to see if they are close within some
#     precision.
#     '''
#     vals = [1-1j,-1+1j, 2+2j]
#     vecs = [ Potapov.normalize(np.matrix([-5.,4j])).T, Potapov.normalize(np.matrix([1j,3.]).T),
#             Potapov.normalize(np.matrix([2j,7.]).T)]
#     T = Potapov.finite_transfer_function(np.eye(2),vecs,vals)
#     T_test = Potapov.get_Potapov(T,vals)
#
#     points = [0.,10j,10.,-10j,10.+10j]
#
#     assert all(np.amax(abs(T(z) - T_test(z))) < eps for z in points)
#
# def two_sets_almost_equal(S1,S2,eps=1e-7):
#     '''
#     Tests if two iterables have the same elements up to some tolerance eps.
#
#     Args:
#         S1,S2 (lists): two lists
#         eps (optional[float]): precision for testing each elements
#
#     Returns:
#         True if the two sets are equal up to eps, false otherwise
#     '''
#     if len(S1) != len(S2):
#         return False
#
#     def almost_equal(el1,el2):
#         if abs(el1 - el2) < eps:
#             return True
#         else: return False
#
#     ran2 = range(len(S2))
#     for i in range(len(S1)):
#         found_match = False
#         for j in ran2:
#             if almost_equal(S1[i],S2[j]):
#                 found_match = True
#                 ran2.remove(j)
#                 break
#         if not found_match:
#             return False
#     return True
#
#
# def test_Roots_1():
#     '''
#     Make a square of length just under 5*pi. Find the roots of sine.
#     '''
#     N=5000
#     f = lambda z: np.sin(z)
#     fp = lambda z: np.cos(z)
#     x_cent = 0.
#     y_cent = 0.
#     width = 5.*np.pi-1e-5
#     height = 5.*np.pi-1e-5
#
#     roots = np.asarray(Roots.get_roots_rect(f,fp,x_cent,y_cent,width,height,N))
#     roots_inside_boundary = Roots.inside_boundary(roots,x_cent,y_cent,width,height)
#     two_sets_almost_equal(np.asarray(roots_inside_boundary)/np.pi,
#         [-4.,-3.,-2.,-1.,-0.,1.,2.,3.,4.] )
#
# def test_Roots_2():
#     '''
#     Make a square of length just over 5*pi. Find the roots of sine.
#     '''
#     N=5000
#     f = lambda z: np.sin(z)
#     fp = lambda z: np.cos(z)
#     x_cent = 0.
#     y_cent = 0.
#     width = 5.*np.pi+1e-5
#     height = 5.*np.pi+1e-5
#
#     roots = np.asarray(Roots.get_roots_rect(f,fp,x_cent,y_cent,width,height,N))
#     roots_inside_boundary = Roots.inside_boundary(roots,x_cent,y_cent,width,height)
#     two_sets_almost_equal(np.asarray(roots_inside_boundary)/np.pi,
#         [-5.,-4.,-3.,-2.,-1.,-0.,1.,2.,3.,4.,5.] )
#
# def test_example_1():
#     Ex = Examples.Example1()
#     Ex.run_Potapov()
#     E = Ex.E
#     roots = Ex.roots
#     M1 = Ex.M1
#     delays = Ex.delays
#     modes = functions.spatial_modes(roots,M1,E,delays)
#     assert( len(roots) == 3)
#
# def test_example_2():
#     Ex = Examples.Example2()
#     Ex.run_Potapov()
#     E = Ex.E
#     roots = Ex.roots
#     M1 = Ex.M1
#     delays = Ex.delays
#     modes = functions.spatial_modes(roots,M1,E,delays)
#     assert( len(roots) == 7)
#
# def test_example_3():
#     Ex = Examples.Example3()
#     Ex.run_Potapov()
#     E = Ex.E
#     roots = Ex.roots
#     M1 = Ex.M1
#     delays = Ex.delays
#     modes = functions.spatial_modes(roots,M1,E,delays)
#     assert( len(roots) == 11)
#
# def test_example_4():
#     Ex = Examples.Example4()
#     Ex.run_Potapov()
#     E = Ex.E
#     roots = Ex.roots
#     M1 = Ex.M1
#     delays = Ex.delays
#     modes = functions.spatial_modes(roots,M1,E,delays)
#     assert( len(roots) == 8)

def test_Hamiltonian(eps=1e-5):
    '''
    This method tests various methods in Hamiltonian and Time_Sims_nonlin.
    In particular, we compare the output from the classical equations of motion
    that results directly from the ABCD model versus the classical Hamiltonian
    equations of motion when we set the coefficient of the nonlinearity to zero.

    Args:
        eps[optional(float)]: how closely each point in time along the two
        tested trajectories should match.
    '''
    Ex = Examples.Example3(r1 = 0.9, r3 = 0.9, max_linewidth=15.,max_freq=10.)
    Ex.run_Potapov()
    modes = functions.spatial_modes(Ex.roots,Ex.M1,Ex.E,Ex.delays)
    M = len(Ex.roots)

    A,B,C,D = Potapov.get_Potapov_ABCD(Ex.roots,Ex.vecs,Ex.T,z=0.)
    A_d,C_d,D_d = map(Time_Sims_nonlin.double_up,(A,C,D))
    B_d = -Time_Sims_nonlin.double_up(C.H)

    ## make Hamiltonian with zero nonlin_coeff
    # ham = Hamiltonian.Hamiltonian(Ex.roots,modes,Ex.delays,
    #                           delay_indices = 0,start_nonlin = 0,length_nonlin = 0.1,
    #                           indices_of_refraction = 1.,
    #                           chi_order = 3,photons_annihilated = 2,nonlin_coeff=0.)
    ham = Hamiltonian.Hamiltonian(Ex.roots,modes,Ex.delays)

    ham.make_chi_nonlinearity(delay_indices=0,start_nonlin=0,
                               length_nonlin=0.1,indices_of_refraction=1.,
                               chi_order=3,chi_function=None,
                               )

    H = ham.make_H(-1j*A)
    eq_mot = ham.make_eq_motion()

    a_in = lambda t: np.asmatrix([1.]*np.shape(D_d)[-1]).T  ## make a sample input function

    ## find f for the linear and nonlinear systems
    f = Time_Sims_nonlin.make_f(eq_mot,B_d,a_in)
    f_lin = Time_Sims_nonlin.make_f_lin(A_d,B_d,a_in)

    Y_lin = Time_Sims_nonlin.run_ODE(f_lin, a_in, C_d, D_d, 2*M, T = 15, dt = 0.01)  ## select f here.
    Y_nonlin = Time_Sims_nonlin.run_ODE(f, a_in, C_d, D_d, 2*M, T = 15, dt = 0.01)  ## select f here.
    for y_lin,y_nonlin in zip(Y_lin,Y_nonlin):
        assert abs(sum(y_lin - y_nonlin)) < eps
