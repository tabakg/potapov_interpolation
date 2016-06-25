import Potapov
import Roots
import Time_Delay_Network
import functions
import numpy as np
import numpy.testing as testing
import Time_Sims_nonlin
import Hamiltonian

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

    ## TODO: make a function to perturb in several steps to avoid root-skipping.


# def test_delay_perturbations(eps=1e-5):
#     '''
#     This funciton tests the parturbations for the delays for each frequency.
#
#     It also tests the corresponding perturbation in the frequencies.
#     '''
#
#     Ex = Time_Delay_Network.Example3( max_linewidth=15.,max_freq=30.)
#     Ex.run_Potapov(commensurate_roots=True)
#     modes = Ex.spatial_modes
#     M = len(Ex.roots)
#
#     A,B,C,D = Ex.get_Potapov_ABCD(doubled=False)
#
#     ham = Hamiltonian.Hamiltonian(Ex.roots,modes,Ex.delays,Omega=-1j*A,
#                 nonlin_coeff = 0.)
#
#     ham.make_chi_nonlinearity(delay_indices=[0],start_nonlin=0,
#                                length_nonlin=0.1*consts.c)
#     ham.make_Delta_delays()
#     #print ham.Delta_delays
#     for row in ham.Delta_delays:
#         for el in row:
#             assert(el == 0)
#
#     ## Now let's make a non-trivial nonlinearity.
#
#     ## turn on the nonlin_coeff
#     ham.nonlin_coeff = 1.
#
#     ## set the index of refraction to be 2 for the nonlinearity
#     ham.chi_nonlinearities[0].refraction_index_func = lambda *args: 2.
#
#     ham.make_Delta_delays()
#     # print ham.Delta_delays
#
#     ## Next, generate the perturb_func and perturb the roots
#     #print Ex.roots
#     perturb_func = Ex.get_frequency_pertub_func_z(use_ufuncify = True)
#     ham.perturb_roots_z(perturb_func)


# def test_make_T_denom_sym_separate_delays():
#     X = Time_Delay_Network.Example3(tau1 = 0.1, tau2 = 0.2,tau3 = 0.1,tau4 = 0.2,)
#     X.get_symbolic_frequency_perturbation(simplify = False)  ## symbolic expr
#     X.make_commensurate_roots([(-100,100)])
#     M = len(X.delays)
#     perturb_func = X.get_frequency_pertub_func()
#     print perturb_func(X.roots[2],tuple(X.delays),(1e-3,)*M) ## value

# def test_commensurate_vecs_example_3():
#     times = [time.clock()]
#     X = Time_Delay_Network.Example3(tau1 = 0.1, tau2 = 0.2,tau3 = 0.1,tau4 = 0.2,)
#     times.append(time.clock())
#     X.make_commensurate_roots([(-60000,60000)])
#     times.append(time.clock())
#     X.make_commensurate_vecs()
#     times.append(time.clock())
#     times.append(time.clock())
#     #print len(X.vecs)
#     assert(len(X.roots) == len(X.vecs))
#     times.append(time.clock())
#     X.make_T_Testing()
#     times.append(time.clock())
#     T_testing = X.T_testing
#     T = X.T
#     print abs(T(-10j)-T_testing(-10j))
#     print abs(T(-100j)-T_testing(-100j))
#     print abs(T(-200j)-T_testing(-200j))

    #print [times[i+1]-times[i] for i in range(len(times)-1)]

# def test_example_3():
#     Ex = Time_Delay_Network.Example3()
#     Ex.run_Potapov()
#     E = Ex.E
#     roots = Ex.roots
#     M1 = Ex.M1
#     delays = Ex.delays
#     modes = functions.spatial_modes(roots,M1,E,delays)
#     print abs(Ex.T(-10j)-Ex.T_testing(-10j))
#     assert( len(roots) == 11)

#
# def test_Hamiltonian_with_doubled_equations(eps=1e-5):
#     '''
#     This method tests various methods in Hamiltonian and Time_Sims_nonlin.
#     In particular, we compare the output from the classical equations of motion
#     that results directly from the ABCD model versus the classical Hamiltonian
#     equations of motion when we set the coefficient of the nonlinearity to zero.
#
#     This method will NOT test the details of the nonlinear Hamiltonian.
#
#     Args:
#         eps[optional(float)]: how closely each point in time along the two
#         tested trajectories should match.
#     '''
#     Ex = Time_Delay_Network.Example3(r1 = 0.9, r3 = 0.9, max_linewidth=15.,max_freq=20.)
#     Ex.run_Potapov()
#     modes = Ex.spatial_modes
#     M = len(Ex.roots)
#
#     A,B,C,D = Ex.get_Potapov_ABCD(doubled=False)
#     A_d,B_d,C_d,D_d = Ex.get_Potapov_ABCD(doubled=True)
#
#     ham = Hamiltonian.Hamiltonian(Ex.roots,modes,Ex.delays,Omega=-1j*A,
#                 nonlin_coeff = 0.)
#
#     ham.make_chi_nonlinearity(delay_indices=0,start_nonlin=0,
#                                length_nonlin=0.1*consts.c)
#
#     ham.make_H()
#     eq_mot = ham.make_eq_motion()
#     a_in = lambda t: np.asmatrix([1.]*np.shape(D_d)[-1]).T  ## make a sample input function
#
#     ## find f for the linear and nonlinear systems
#     f = Time_Sims_nonlin.make_f(eq_mot,B_d,a_in)
#     f_lin = Time_Sims_nonlin.make_f_lin(A_d,B_d,a_in)
#
#     Y_lin = Time_Sims_nonlin.run_ODE(f_lin, a_in, C_d, D_d, 2*M, T = 15, dt = 0.01)  ## select f here.
#     Y_nonlin = Time_Sims_nonlin.run_ODE(f, a_in, C_d, D_d, 2*M, T = 15, dt = 0.01)  ## select f here.
#     for y_lin,y_nonlin in zip(Y_lin,Y_nonlin):
#         assert abs(sum(y_lin - y_nonlin)) < eps
#
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
#     T_test = Potapov.get_Potapov(T,vals,vecs)
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
#     Ex = Time_Delay_Network.Example1()
#     Ex.run_Potapov()
#     E = Ex.E
#     roots = Ex.roots
#     M1 = Ex.M1
#     delays = Ex.delays
#     modes = functions.spatial_modes(roots,M1,E,delays)
#     assert( len(roots) == 3)
#
# def test_example_2():
#     Ex = Time_Delay_Network.Example2()
#     Ex.run_Potapov()
#     E = Ex.E
#     roots = Ex.roots
#     M1 = Ex.M1
#     delays = Ex.delays
#     modes = functions.spatial_modes(roots,M1,E,delays)
#     assert( len(roots) == 7)
#
# def test_example_3():
#     Ex = Time_Delay_Network.Example3()
#     Ex.run_Potapov()
#     E = Ex.E
#     roots = Ex.roots
#     M1 = Ex.M1
#     delays = Ex.delays
#     modes = functions.spatial_modes(roots,M1,E,delays)
#     assert( len(roots) == 11)
#
# def test_example_4():
#     Ex = Time_Delay_Network.Example4()
#     Ex.run_Potapov()
#     E = Ex.E
#     roots = Ex.roots
#     M1 = Ex.M1
#     delays = Ex.delays
#     modes = functions.spatial_modes(roots,M1,E,delays)
#     assert( len(roots) == 8)
#
# def test_commensurate_roots_example_3():
#     X = Time_Delay_Network.Example3()
#     X.make_commensurate_roots()
#     assert(len(X.roots) == 0)
#     X.make_commensurate_roots([(0,1000)])
#     # assert(len(X.roots) == 91)
#     # X.make_commensurate_roots([(0,10000)])
#     # assert(len(X.roots) == 931)
#     # X.make_commensurate_roots([(0,10000),(1e15,1e15 +10000)])
#     # assert(len(X.roots) == 1891)




if __name__ == "__main__":
    test_altered_delay_pert(plot=True)
    #test_make_T_denom_sym_separate_delays()
    #test_Hamiltonian_with_doubled_equations()
    #test_example_3()
