# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 2015

@author: gil
@title: Time_Sims_Nonlin.py
"""

import Roots
import Potapov
import Examples
import functions
import Hamiltonian

import numpy as np
import numpy.linalg as la
import sympy as sp
import itertools

import matplotlib.pyplot as plt
from scipy.integrate import ode

def make_f(eq_mot,B,a_in):
    '''
    Equations of motion, including possibly nonlinear internal dynamics.
    Args:
        eq_mot: The equations of motion, which take an array and return a
        matrix column.
        B: The matrix multiplying the inputs to the system.
        a_in: The inputs to the system

    Returns:
        A function that maps (t,a) -> f'(t,a), where t is a scalar (time), and
        a is an array representing the state of the system.
    '''
    return lambda t,a: np.asarray(eq_mot(a)+B*a_in(t)).T[0]

def make_f_lin(A,B,a_in):
    '''
    Linear equations of motion
    Args:
        A: The matrix for the linear equations of motion:
            d(a,a^H)/dt = A(a,a^H) + B * a_in (t)
        B: The matrix multiplying the inputs to the system.
        a_in: The inputs to the system
    Returns:
        A function that maps (t,a) -> f'(t,a), where t is a scalar (time), and
        a is an array representing the state of the system.
    '''
    return lambda t,a: np.asarray(A*np.asmatrix(a).T+B*a_in(t)).T[0]

def run_ODE(f, a_in, C, D, num_of_variables, T = 100, dt = 0.01):
    '''
    Run the ODE for the given set of equations and record the outputs.

    Args:
        f (function): Evolution of the system
        a_in (function): inputs as a function of time
        C,D: matrices to use to obtain output from system state and input
        num_of_variables (int): number of variables the system has
        T (optional[positive float]): length of simulation
        dt (optional[float]): time step used by the simulation
    Returns:
        An array Y of outputs
    '''
    r = ode(f).set_integrator('zvode', method='bdf')
    y0 = np.asmatrix([0.]*num_of_variables).T
    r.set_initial_value(y0, 0.)
    Y = []
    while r.successful() and r.t < T:
        Y.append(C*r.y+D*a_in(r.t))
        r.integrate(r.t+dt)
    return Y

def double_up(M1,M2=None):
    '''
    Takes a given matrix M1 and an optional matrix M2 and generates a
    doubled-up matrix to use for simulations when the doubled-up notation
    is needed. i.e.

    (M1,M2) -> (M1          M2)
               (conj(M2)    conj(M1))
    (M1,None) -> (M1          0)
                 (0    conj(M1))
    Args:
        M1: matrix to double-up
        M2: optional second matrix to double-up
    Returns:
        The doubled-up matrix.
    '''
    if M2 == None:
        M2 = np.zeros_like(M1)
    top = np.hstack([M1,M2])
    bottom = np.hstack([np.conj(M2),np.conj(M1)])
    return np.vstack([top,bottom])

if __name__ == "__main__":

    Ex = Examples.Example3(r1 = 0.9, r3 = 0.9, max_linewidth=15.,max_freq=25.)
    Ex.run_Potapov()
    modes = functions.spatial_modes(Ex.roots,Ex.M1,Ex.E)

    M = len(Ex.roots)

    A,B,C,D = Potapov.get_Potapov_ABCD(Ex.roots,Ex.vecs,Ex.T,z=0.)
    A_d,C_d,D_d = map(double_up,(A,C,D))
    B_d = -double_up(C.H)

    ham = Hamiltonian.Hamiltonian(Ex.roots,modes,Ex.delays,0,0,0.1,1000.,3,2)
    H = ham.make_H(-1j*A_d)
    eq_mot = ham.make_eq_motion()

    a_in = lambda t: np.asmatrix([1e-6]*np.shape(D_d)[-1]).T  ## make a sample input function

    f = make_f(eq_mot,B_d,a_in)
    f_lin = make_f_lin(A_d,B_d,a_in)

    Y = run_ODE(f, a_in, C_d, D_d, 2*M, T = 100, dt = 0.01)  ## select f here.

    for i in range(2):
        plt.plot([abs(y)[i][0,0] for y in Y ])
        plt.savefig('sample_graph'+str(i)+'.pdf',format='pdf')
