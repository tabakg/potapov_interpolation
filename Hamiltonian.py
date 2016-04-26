# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 2015

@author: gil
@title: Hamiltonian.py
"""

import Roots
import Potapov
import Examples
import functions

import numpy as np
import numpy.linalg as la
import sympy as sp
import scipy.constants as consts
import itertools

from sympy.physics.quantum import *
from sympy.physics.quantum.boson import *
from sympy.physics.quantum.operatorordering import *

class Chi_nonlin():
    '''Class to store the information in a particular nonlinear chi element.

    Attributes:
        delay_indices (list of indices): indices of delays to use.

        start_nonlin (positive float or list of positive floats): location of
        nonlinear crystal with respect to each edge.

        length_nonlin (float): length of the nonlinear element.

        chi_order (optional [int]): order of nonlinearity

        chi_function (optional [function]): strength of nonlinearity.
        first chi_order args are frequencies, next first chi_order args are
        frequencies, next chi_order args are indices of polarization.

    '''
    def __init__(self,delay_indices,start_nonlin,length_nonlin,indices_of_refraction,
                 chi_order=3,chi_function=None):
        self.delay_indices = delay_indices
        self.start_nonlin = start_nonlin
        self.length_nonlin = length_nonlin
        self.chi_order = chi_order
        self.indices_of_refraction = indices_of_refraction

        if chi_function == None:
            def chi_func(a,b,c,d,i,j,k):
                return 1.  #if abs(a+b+c+d) <= 2. else 0.
            self.chi_function = chi_func
        else:
            self.chi_function = chi_function

class Hamiltonian():
    '''A class to create a sympy expression for the Hamiltonian of a network.

    Attributes:
        roots (list of complex numbers): the poles of the transfer function.

        omegas (list of floats): the natural frequencies of the modes.

        modes (list of complex-valued column matrices): modes of the network.

        delays (list of floats): the delays in the network.

        nonlin_coeff (optional [float]): overall scaling for the nonlinearities.

        polarizations (optional [list]): the polarizations of the respective
        modes. These should match the arguments in Chi_nonlin.chi_func.

        cross_sectional_area (float): area of beams, used to determines the
        scaling for the various modes.

        chi_nonlinearities (lst): a list of Chi_nonlin instances.

    '''
    def __init__(self,roots,modes,delays,
        nonlin_coeff = 1.,polarizations = None,
        cross_sectional_area = 1e-10,
        chi_nonlinearities = [],
        ):
        self.roots = roots
        self.omegas = [root.imag / (2.*consts.pi) for root in self.roots]
        self.modes = modes
        self.m = len(roots)
        self.delays = delays
        self.normalize_modes()
        self.cross_sectional_area = cross_sectional_area
        if polarizations == None:
            self.polarizations = [1.]*self.m
        else:
            self.polarizations = polarizations
        self.volumes = self.mode_volumes()
        self.E_field_weights = self.make_E_field_weights()
        self.chi_nonlinearities = chi_nonlinearities
        self.a = [BosonOp('a_'+str(i)) for i in range(self.m)]
        self.H = 0.
        self.nonlin_coeff = nonlin_coeff

    def make_chi_nonlinearity(self,delay_indices,start_nonlin,
                               length_nonlin,indices_of_refraction,
                               chi_order=3,chi_function=None):
        '''Add an instance of Chi_nonlin to Hamiltonian.

        Args:
            delay_indices (int OR list/tuple of ints): the index representing the
            delay line along which the nonlinearity lies. If given a list/tuple
            then the nonlinearity interacts the N different modes.

            start_nonlin (float OR list/tuple of floats): the beginning of the
            nonlinearity. If a list/tuple then each nonlinearity begins at a
            different time along its corresponding delay line.

            length_nonlin (float): duration of the nonlinearity in terms of length.
            indices_of_refraction (float/int or list/tuple of float/int): the
            indices of refraction corresponding to the various modes. If float
            or int then all are the same.

            chi_order (optional [int]): order of the chi nonlinearity.

            chi_function (function): a function of 2*chi_order+1 parameters that
            returns the strenght of the interaction for given frequency
            combinations and polarizations. The first chi_order+1 parameters
            correspond to frequencies combined the the next chi_order parameters
            correspond to the various polarizations.
        '''

        chi_nonlinearity = Chi_nonlin(delay_indices,start_nonlin,
                                   length_nonlin,indices_of_refraction,
                                   chi_order=chi_order,chi_function=chi_function)
        self.chi_nonlinearities.append(chi_nonlinearity)

    def normalize_modes(self,):
        ''' Normalize the modes of Hamiltonian.

        '''
        for mode in self.modes:
            mode /= functions._norm_of_mode(mode,self.delays)

    def mode_volumes(self,):
        '''Find the effective volume of each mode to normalize the field.

        Returns:
            A list of the effective lengths of the various modes.

        '''

        volumes = []
        for mode in self.modes:
            for i,delay in enumerate(self.delays):
                volumes.append( delay * abs(mode[i,0]**2) *
                                self.cross_sectional_area )
        return volumes

    def make_nonlin_term_sympy(self,combination,pm_arr):
        '''Make symbolic nonlinear term using sympy.

        Example:
            >>> combination = [1,2,3]; pm_arr = [-1,1,1]
            >>> print Hamiltonian.make_nonlin_term_sympy(combination,pm_arr)
                a_1*Dagger(a_2)*Dagger(a_3)

        Args:
            combination (tuple/list of integers): indices of which terms to
            include pm_arr (tuple/list of +1 and -1): creation and
            annihilation indicators for the respective terms in combination.
        Returns:
            symbolic expression for the combination of creation and annihilation
            operators.

        '''
        r = 1
        for index,sign in zip(combination,pm_arr):
            if sign == 1:
                r*= Dagger(self.a[index])
            else:
                r *= self.a[index]
        return r

    def phase_weight(self,combination,pm_arr,chi):
        '''The weight to give to each nonlinear term characterized by the given
        combination and pm_arr.

        Args:
            combination (list/tuple of integers): which modes/roots to pick
            pm_arr (list of +1 and -1): creation and annihilation of modes
            chi (Chi_nonlin): the chi nonlinearity for which to compute
            the phase coefficient.
        Returns:
            The weight to add to the Hamiltonian

        '''
        omegas_to_use = np.array([self.omegas[i] for i in combination])
        modes_to_use = [self.modes[i] for i in combination]
        return functions.make_nonlinear_interaction(
                    omegas_to_use, modes_to_use, self.delays, chi.delay_indices,
                    chi.start_nonlin, chi.length_nonlin, pm_arr,
                    chi.indices_of_refraction)

    def make_phase_matching_weights(self,chi):
        '''Make a dict to store the weights for the selected components and the
        creation/annihilation information.

        Args:
            chi (Chi_nonlin): the chi nonlinearity for which to compute
            the phase coefficient.
        Returns:
            A dictionary of weights. Each key is a tuple consisting of two
            components: the first is a tuple of the indices of modes and the
            second is a tuple of +1 and -1.

        '''
        ## TODO: add a priori check to restrict exponential growth on the number
        ## of nonlienar coefficients
        list_of_pm_arr = list(itertools.product([-1, 1], repeat=3))

        weights = {}
        for pm_arr in list_of_pm_arr:
            field_combinations = itertools.combinations_with_replacement(
                range(self.m), chi.chi_order+1)
            for combination in field_combinations:
                weights[tuple(combination),tuple(pm_arr)] = self.phase_weight(
                    combination,pm_arr,chi)
        return weights

    def E_field_weight(self,mode_index):
        '''Make the weights for each field component E_i(n) = [weight] (a+a^+)

        Args:
            mode_index (int): The index of the mode.
        Returns:
            The weight in the equation above. It has form:
            sqrt[\hbar * \omega(n) / 2 V_eff(n) \epsilon].

        '''
        omega = self.omegas[mode_index]
        eps0 = consts.epsilon_0
        hbar = consts.hbar
        return np.sqrt(hbar * abs(omega) / (2 * eps0 * self.volumes[mode_index]) )

    def make_E_field_weights(self,):
        '''
        Returns:
            A dictionary from mode index to the E-field weight.

        '''
        weights = {}
        for mode_index in range(self.m):
            weights[mode_index] = self.E_field_weight(mode_index)
        return weights


    def make_nonlin_H_from_chi(self,chi,eps=1e-5):
        '''Make a nonlinear Hamiltonian based on nonlinear interaction terms

        Args:
            chi (Chi_nonlin): nonlinearity to use
            eps (optional[float]): Cutoff for the significance of a particular term.
        Returns:
            A symbolic expression for the nonlinear Hamiltonian.

        TODO: Further restrict terms iterated over to make the RWA (i.e.
            frequency-match terms).
        '''
        H_nonlin_sp = 0.
        for chi in self.chi_nonlinearities:
            phase_matching_weights = self.make_phase_matching_weights(chi)
            significant_phase_matching_weights = {k:v for k,v
                in phase_matching_weights.iteritems() if abs(v) > eps}
            for combination,pm_arr in significant_phase_matching_weights:
                omegas_to_use = map(lambda i: self.omegas[i],combination)
                omegas_with_sign = [omega * pm for omega,pm
                                    in zip(omegas_to_use,pm_arr)]
                pols = map(lambda i: self.polarizations[i],combination)
                chi_args = omegas_with_sign + pols
                H_nonlin_sp += ( self.make_nonlin_term_sympy(combination,pm_arr) *
                    chi.chi_function(*chi_args) *
                    significant_phase_matching_weights[combination,pm_arr] *
                    np.prod([self.E_field_weights[i] for i in combination]) )
            return H_nonlin_sp

    def make_lin_H(self,Omega):
        '''Make a linear Hamiltonian based on Omega.

        Args:
            Omega (complex-valued matrix) describes the Hamiltonian of the system.

        Returns:
            A symbolic expression for the nonlinear Hamiltonian.

        '''
        H_lin_sp = 0.
        for i in range(self.m):
            for j in range(self.m):
                H_lin_sp += Dagger(self.a[i])*self.a[j]*Omega[i,j]
        return H_lin_sp

    def make_H(self,Omega,eps=1e-5):
        '''Make a Hamiltonian combining the linear and nonlinear parts.

        Args:
            Omega (complex-valued matrix) describes the Hamiltonian of the system.
            Omega = -1j*A        <--- full dynamics (not necessarily Hermitian)
            Omega = (A-A.H)/(2j) <--- closed dynamics only (Hermitian part of above)
            eps (optional[float]): Cutoff for the significance of a particular term.

        Returns:
            A symbolic expression for the full Hamiltonian.

        '''
        H_nonlin = self.make_nonlin_H_from_chi(eps)
        H_lin = self.make_lin_H(Omega)
        self.H = H_lin + H_nonlin * self.nonlin_coeff
        return self.H

    def make_eq_motion(self,):
        '''Input is a tuple or list, output is a matrix vector.
        This generates Hamilton's equations of motion for a and a^H.
        These equations are CLASSICAL equations of motion. This means
        we replace the operators with c-numbers. The order of the operators
        will yield different results, so we assume the Hamiltonian is already
        in the desired order (e.g. normally ordered).

        Returns:
            A function that yields the Hamiltonian equations of motion based on
            the Hamiltonian given.
            The equations of motion take an array as an input and return a column
            vector as an output.

        '''

        ## c-numbers
        b = [sp.symbols('b'+str(i)) for i in range(self.m)]
        b_H = [sp.symbols('b_H'+str(i)) for i in range(self.m)]

        ## Hamiltonian is not always Hermitian. We use its complex conjugate.
        H_H = Dagger(self.H)

        def subs_c_number(expression,i):
            return expression.subs(self.a[i],b[i]).subs(Dagger(self.a[i]),b_H[i])

        H_c_numbers = self.H
        H_H_c_numbers = H_H
        for i in range(self.m):
            H_c_numbers = subs_c_number(H_c_numbers,i)
            H_H_c_numbers = subs_c_number(H_H_c_numbers,i)


        ## classical equations of motion
        diff_ls = ([1j*sp.diff(H_c_numbers,var) for var in b_H] +
                   [-1j*sp.diff(H_H_c_numbers,var) for var in b])
        fs = [sp.lambdify( tuple( b+b_H ),expression) for expression in diff_ls ]
        return lambda arr: (np.asmatrix([ f(* arr ) for f in fs])).T
