# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 2015

@author: gil
@title: Time_Sims_Nonlin.py
"""

import Roots
import Potapov
import Time_Delay_Network
import functions
import Hamiltonian

import numpy as np
import numpy.linalg as la
import sympy as sp
import itertools

import matplotlib.pyplot as plt
from scipy.integrate import ode

from scipy.integrate import quad


def make_f(eq_mot,B,a_in):
    r'''Equations of motion, including possibly nonlinear internal dynamics.

    Args:
        eq_mot (function): The equations of motion, which map
            :math:`(t,a) \to v`. Here \math:`t` is a scalar corresponding to time,
            :math:`a` is an array of inputs correpsonding to the internal degrees
            of freedom, and :math:`v` is a complex-valued column matrix describing
            the gradient.

        B (matrix): The matrix multiplying the inputs to the system.

        a_in (function): The inputs to the system.

    Returns:
        (function):
        A function that maps :math:`(t,a) \to f'(t,a)`, where t is a scalar
        (time), and a is an array representing the state of the system.

    '''
    return lambda t,a: np.asarray(eq_mot(t,a)+B*a_in(t)).T[0]

def make_f_lin(A,B,a_in):
    r'''Linear equations of motion

    Args:
        A (matrix): The matrix for the linear equations of motion:
            :math:`\frac{d}{dt}\begin{pmatrix} a \\ a^+ \end{pmatrix} = A \begin{pmatrix} a \\ a^+ \end{pmatrix}+ B \breve a_{in} (t).`

        B (matrix): The matrix multiplying the inputs to the system.

        a_in (function): The inputs to the system :math:`\breve a`.

    Returns:
        (function);
        A function that maps :math:`(t,a) \to f'(t,a)`, where t is a scalar
        (time), and a is an array representing the state of the system.

    '''
    return lambda t,a: np.asarray(A*np.asmatrix(a).T+B*a_in(t)).T[0]

def run_ODE(f, a_in, C, D, num_of_variables, T = 10, dt = 0.01, y0 = None):
    '''Run the ODE for the given set of equations and record the outputs.

    Args:
        f (function): Evolution of the system.

        a_in (function): inputs as a function of time.

        C,D (matrices): matrices to use to obtain output from system state and
            input.

        num_of_variables (int): number of variables the system has.

        T (optional[positive float]): length of simulation.

        dt (optional[float]): time step used by the simulation.

    Returns:
        (array): 
        An array Y of outputs.

    '''
    if y0 is None:
        y0 = np.asmatrix([0.]*num_of_variables).T

    r = ode(f).set_integrator('zvode', method='bdf')
    r.set_initial_value(y0, 0.)
    Y = []
    while r.successful() and r.t < T:
        Y.append(C*r.y+D*a_in(r.t))
        r.integrate(r.t+dt)
    return Y

if __name__ == "__main__" and False:

    Ex = Time_Delay_Network.Example3(r1 = 0.9, r3 = 0.9, max_linewidth=15.,max_freq=25.)
    Ex.run_Potapov()
    modes = functions.spatial_modes(Ex.roots,Ex.M1,Ex.E)

    M = len(Ex.roots)

    A,B,C,D = Ex.get_Potapov_ABCD(doubled=False)
    A_d,B_d,C_d,D_d = Ex.get_Potapov_ABCD(doubled=True)

    ham = Hamiltonian.Hamiltonian(Ex.roots,modes,Ex.delays,Omega=-1j*A)

    ham.make_chi_nonlinearity(delay_indices=0,start_nonlin=0,
                               length_nonlin=0.1,indices_of_refraction=1.,
                               chi_order=3,chi_function=None,)

    ham.make_H()
    eq_mot = ham.make_eq_motion()
    a_in = lambda t: np.asmatrix([1.]*np.shape(D_d)[-1]).T  ## make a sample input function

    ## find f for the linear and nonlinear systems
    f = Time_Sims_nonlin.make_f(eq_mot,B_d,a_in)
    f_lin = Time_Sims_nonlin.make_f_lin(A_d,B_d,a_in)

    Y_lin = Time_Sims_nonlin.run_ODE(f_lin, a_in, C_d, D_d, 2*M, T = 15, dt = 0.01)  ## select f here.
    Y_nonlin = Time_Sims_nonlin.run_ODE(f, a_in, C_d, D_d, 2*M, T = 15, dt = 0.01)  ## select f here.

    for i in range(2):
        plt.plot([abs(y)[i][0,0] for y in Y ])
        plt.savefig('sample_graph'+str(i)+'.pdf',format='pdf')
