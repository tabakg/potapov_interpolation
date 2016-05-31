# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 2015

@author: gil
@title: Hamiltonian.py
"""

import Roots
import Potapov
import Time_Delay_Network
import functions

import numpy as np
import numpy.linalg as la
import sympy as sp
import scipy.constants as consts
from scipy.optimize import minimize

import itertools
import copy

#from sympy.printing.theanocode import theano_function

from sympy.physics.quantum import *
from sympy.physics.quantum.boson import *
from sympy.physics.quantum.operatorordering import *
from qnet.algebra.circuit_algebra import *


class Chi_nonlin():
    r'''Class to store the information in a particular nonlinear chi element.

    Attributes:
        delay_indices (list of indices):
            indices of delays to use.

        start_nonlin (positive float or list of positive floats):
            location of
            nonlinear crystal with respect to each edge.

        length_nonlin (float):
            length of the nonlinear element.

        refraction_index_func (function):
            the indices of refraction as a
            function of the netural frequency :math:`/omega`.

        chi_order (optional [int]):
            order of nonlinearity.

        chi_function (optional [function]):
            strength of nonlinearity.
            first (chi_order+1) args are frequencies,
            next (chi_order+1) args are indices of polarization.

    '''
    def __init__(self,delay_indices,start_nonlin,length_nonlin,
            refraction_index_func = lambda *args: 1.,
            chi_order=3,chi_function = lambda *args: 1.):
        self.delay_indices = delay_indices
        self.start_nonlin = start_nonlin
        self.length_nonlin = length_nonlin
        self.chi_order = chi_order
        self.refraction_index_func = refraction_index_func
        self.chi_function = chi_function

class Hamiltonian():
    '''A class to create a sympy expression for the Hamiltonian of a network.

    Attributes:
        roots (list of complex numbers):
            the poles of the transfer function.

        omegas (list of floats):
            The natural frequencies of the modes.

        modes (list of complex-valued column matrices):
            Modes of the network.

        delays (list of floats):
            The delays in the network.

        Omega (optional [matrix]):
            Quadratic Hamiltonian term for linear
            dynamics.

        nonlin_coeff (optional [float]):
            Overall scaling for the nonlinearities.

        polarizations (optional [list]):
            The polarizations of the respective
            modes. These should match the arguments in Chi_nonlin.chi_func.

        cross_sectional_area (float):
            Area of beams, used to determines the
            scaling for the various modes.

        chi_nonlinearities (list):
            A list of Chi_nonlin instances.

    TODO: Return L operator for QNET.

    '''
    def __init__(self,roots,modes,delays,
        Omega = None,
        nonlin_coeff = 1.,polarizations = None,
        cross_sectional_area = 1e-10,
        chi_nonlinearities = [],
        using_qnet_symbols = False,
        ):
        self.roots = roots
        self._update_omegas()
        self.modes = modes
        self.m = len(roots)
        self.delays = delays

        self.cross_sectional_area = cross_sectional_area

        self.Delta_delays = np.zeros((self.m,len(self.delays)))
        self.volumes = self.mode_volumes()
        self.normalize_modes()

        if Omega is None:
            self.Omega = np.asmatrix(np.zeros((m,m)))
        else:
            self.Omega = Omega
        if polarizations is None:
            self.polarizations = [1.]*len(self.delays)
        else:
            self.polarizations = polarizations
        self.E_field_weights = self.make_E_field_weights()
        self.chi_nonlinearities = chi_nonlinearities
        self.using_qnet_symbols = using_qnet_symbols
        if self.using_qnet_symbols:
            self.a = [Destroy(i) for i in range(self.m)]
        else:
            #self.a = [sp.symbols('a_'+str(i)) for i in range(self.m)]
            self.a = [BosonOp('a_'+str(i)) for i in range(self.m)]
        self.t = sp.symbols('t')
        self.H = 0.
        self.nonlin_coeff = nonlin_coeff

    def _update_omegas(self,):
        self.omegas = map(lambda z: z.imag / (2.*consts.pi), self.roots)
        return

    def Dagger(self, symbol):
        if self.using_qnet_symbols:
            return symbol.dag()
        else:
            return Dagger(symbol)

    # def adjusted_delays(self):
    #     '''
    #
    #     '''
    #     N = len(self.delays)
    #     M = len(self.Delta_delays)
    #     delay_adjustment = [0.]*N
    #     for i in range(M):
    #         for j in range(N):
    #             delay_adjustment[j] += self.Delta_delays[i,j] / M
    #     return map(sum, zip(delay_adjustment, self.delays))

    def make_Delta_delays(self,):
        '''
        Each different frequency will experience a different shift in delay
        lengths due to all nonlinearities present.
        We will store those shifts as a list of lists in the class.
        This list is called Delta_delays.
        The ith list will be the shifts in all the original delays for the ith
        root (i.e. frequency).

        Returns:
            None.
        '''
        self.Delta_delays = np.zeros((self.m,len(self.delays)))
        for chi in self.chi_nonlinearities:
            for i,omega in enumerate(self.omegas):
                for delay_index in chi.delay_indices:
                    pol = self.polarizations[delay_index]
                    index_of_refraction = chi.refraction_index_func(omega,pol)
                    self.Delta_delays[i,delay_index] = (
                        (index_of_refraction - 1.)* chi.length_nonlin/consts.c)
                #     print i,delay_index,'delta delay is', self.Delta_delays[i,delay_index]
                # print "Delta delays are", self.Delta_delays
        return

    def perturb_roots_z(self,perturb_func,eps = 1e-12):
        '''
        One approach to perturbing the roots is to use Newton's method.
        This is done here using a function perturb_func that corresponds to
        :math:`-f(z) / f'(z)` when the time delays are held fixed.
        The function perturb_func is generated in
        get_frequency_pertub_func_z.

        Args:
            perturb_func (function):
                The Newton's method function.

            eps (optional [float]):
                Desired precision for convergence.
        '''
        max_count = 10
        for j in range(max_count):
            self.make_Delta_delays() #delays depend on omegas
            old_roots = copy.copy(self.roots)
            for i,root in enumerate(self.roots):
                pert = perturb_func(root,map(sum,zip(self.delays,self.Delta_delays[i])))
                self.roots[i] += pert
            self._update_omegas()
            if all([abs(new-old) < eps for new,old in zip(self.roots,old_roots)]):
                print "root adjustment converged!"
                break
        else:
            "root adjustment aborted."

    def minimize_roots_z(self,func,dfunc,eps = 1e-12):
        '''
        One approach to perturb the roots is to use a function in :math:`x,y`
        that becomes minimized at a zero. This is done here.

        The result is an update to roots, omegas, and Delta_delays.

        Args:
            func, dfuncs (functions):
                Functions in x,y. The first becomes
                minimized at a zero and the second is the gradient in x,y.
                These functions are generated in
                Time_Delay_Network.get_minimizing_function_z.

            eps (optional [float]):
                Desired precision for convergence.

        '''
        max_count = 1
        for j in range(max_count):
            self.make_Delta_delays()
            old_roots = copy.copy(self.roots)
            for i,root in enumerate(self.roots):
                fun_z = lambda x,y: func(x,y,*map(sum,zip(self.delays,self.Delta_delays[i])))
                fun_z_2 = lambda arr: fun_z(*arr)
                dfun_z = lambda x,y: dfunc(x,y,*map(sum,zip(self.delays,self.Delta_delays[i])))
                dfun_z_2 = lambda arr: dfun_z(*arr)
                x0 = np.asarray([root.real,root.imag])
                minimized = minimize(fun_z_2,x0,jac = dfun_z_2).x
                self.roots[i] = minimized[0] + minimized[1] * 1j
            #print self.roots
            self._update_omegas()
            if all([abs(new-old) < eps for new,old in zip(self.roots,old_roots)]):
                print "root adjustment converged!"
                break
        else:
            "root adjustment aborted."

    # # def perturb_roots_T_and_z(self,perturb_func,eps = 1e-15):
    #     r'''
    #     For each root, use the corresponding perturbations in the delays
    #     to generate the perturbation of the root.
    #
    #     Args:
    #         perturb_func (function): A function whose input is a tuple of the
    #         form (z,Ts,Ts_Delta), where z is a complex number, while each
    #         Ts and Ts_Delta are lists of floats.
    #
    #         Each network pole `z^*` with the corresponding perturbed
    #         delays will satisfy `perturb_func(z^*,Ts,Ts_Delta) = 0`.
    #     '''
    #     # print 'roots before'
    #     # print self.roots
    #
    #     max_count = 200
    #     for j in range(max_count):
    #         old_roots = copy.copy(self.roots)
    #         for i,root in enumerate(self.roots):
    #             #updated_delays = map(sum,zip(self.delays,self.Delta_delays[i]))
    #             #self.make_Delta_delays()
    #             #updated_delta_delays = map(sum,zip(self.delays,self.Delta_delays[i]))
    #             pert = perturb_func(root,self.delays,self.Delta_delays[i])
    #             # print pert
    #             self.roots[i] += pert
    #         self._update_omegas()
    #         if all([abs(new-old) < eps for new,old in zip(self.roots,old_roots)]):
    #             break
    #
    #     # print 'roots after:'
    #     # print self.roots
    #     return

    def make_chi_nonlinearity(self,delay_indices,start_nonlin,
                               length_nonlin,refraction_index_func = lambda *args: 1.,
                               chi_order=3,chi_function = lambda *args: 1):
        r'''Add an instance of Chi_nonlin to Hamiltonian.

        Args:
            delay_indices (int OR list/tuple of ints):
                The index representing the
                delay line along which the nonlinearity lies. If given a list/tuple
                then the nonlinearity interacts the N different modes.

            start_nonlin (float OR list/tuple of floats):
                The beginning of the
                nonlinearity. If a list/tuple then each nonlinearity begins at a
                different time along its corresponding delay line.

            length_nonlin (float):
                Duration of the nonlinearity in terms of
                length. (Units in length)

            refraction_index_func (function):
                The indices of refraction as a
                function of the netural frequency :math:`/omega`.

            chi_order (optional [int]):
                Order of the chi nonlinearity.

            chi_function (function):
                A function of 2*chi_order+2 parameters that
                returns the strenght of the interaction for given frequency
                combinations and polarizations. The first chi_order+1 parameters
                correspond to frequencies combined the the next chi_order+1 parameters
                correspond to the various polarizations.

            TODO: check units everywhere, including f versus \omega = f / 2 pi.
        '''

        chi_nonlinearity = Chi_nonlin(delay_indices,start_nonlin,
                                   length_nonlin,refraction_index_func=refraction_index_func,
                                   chi_order=chi_order,chi_function=chi_function)
        self.chi_nonlinearities.append(chi_nonlinearity)

    def normalize_modes(self,):
        ''' Normalize the modes of Hamiltonian.

        '''
        for i,mode in enumerate(self.modes):
            mode /= functions._norm_of_mode(mode,map(sum, zip(self.delays,self.Delta_delays[i])))

    def mode_volumes(self,):
        '''Find the effective volume of each mode to normalize the field.

        Returns:
            A list of the effective lengths of the various modes.

        '''

        volumes = []
        for i,mode in enumerate(self.modes):
            for j,delay in enumerate(map(sum, zip(self.delays,self.Delta_delays[i]))):
                volumes.append( delay * abs(mode[j,0]**2) *
                                self.cross_sectional_area )
        return volumes

    def make_nonlin_term_sympy(self,combination,pm_arr):
        '''Make symbolic nonlinear term using sympy.

        Example:
            >>> combination = [1,2,3]; pm_arr = [-1,1,1]
            >>> print Hamiltonian.make_nonlin_term_sympy(combination,pm_arr)
                a_1*Dagger(a_2)*Dagger(a_3)

        Args:
            combination (tuple/list of integers):
                indices of which terms to include.

            pm_arr (tuple/list of +1 and -1):
                Creation and
                annihilation indicators for the respective terms in combination.

        Returns:
            (sympy expression):
                symbolic expression for the combination of creation and annihilation
                operators.

        '''
        r = 1
        for index,sign in zip(combination,pm_arr):
            if sign == 1:
                r*= self.Dagger(self.a[index])
            else:
                r *= self.a[index]
        return r

    def phase_weight(self,combination,pm_arr,chi):
        '''The weight to give to each nonlinear term characterized by the given
        combination and pm_arr.

        Args:
            combination (list/tuple of integers):
                Which modes/roots to pick

            pm_arr (list of +1 and -1):
                creation and annihilation of modes

            chi (Chi_nonlin):
                The chi nonlinearity for which to compute
                the phase coefficient.

        Returns:
            The weight to add to the Hamiltonian.

        '''
        omegas_to_use = np.array([self.omegas[i] for i in combination])
        modes_to_use = [self.modes[i] for i in combination]
        polarizations_to_use = [self.polarizations[i] for i in chi.delay_indices]
        indices_of_refraction = map(chi.refraction_index_func,
            zip(omegas_to_use,polarizations_to_use) )
        return functions.make_nonlinear_interaction(
                    omegas_to_use, modes_to_use, self.delays, chi.delay_indices,
                    chi.start_nonlin, chi.length_nonlin, pm_arr,
                    indices_of_refraction)

    def make_phase_matching_weights(self,weight_keys,chi,
        filtering_phase_weights = False ,eps = 1e-5):
        '''Make a dict to store the weights for the selected components and the
        creation/annihilation information.

        Args:
            weight_keys (list of tuples):
                Keys for weights to consider.
                Each key is a tuple consisting of two
                components: the first is a tuple of the indices of modes and the
                second is a tuple of +1 and -1.

            filtering_phase_weights (optional[Boolean]):
                Whether or not to
                filter the phase_matching_weights by the size of their values. The
                cutoff for their absolute value is given by eps.

            eps (optional [float]):
                Cutoff for filtering of weights.

        Returns:
            Weights (dict):
                A dictionary of weights with values corresponding to the
                phase matching coefficients.

        '''
        weights = {}
        for comb,pm_arr in weight_keys:
            weights[comb,pm_arr] = self.phase_weight(comb,pm_arr,chi)
        if filtering_phase_weights:
            weights = {k:v for k,v in weights.iteritems() if abs(v) > eps}
        return weights

    def E_field_weight(self,mode_index):
        r'''Make the weights for each field component :math:`E_i(n) = [\text{weight}] (a+a^\dagger)`.

        Args:
            mode_index (int):
                The index of the mode.

        Returns:
            Weight of E-field (float):
                It has form:
                :math:`[\hbar * \omega(n) / 2 V_{eff}(n) \epsilon]^{1/2}`.

        '''
        omega = self.omegas[mode_index]
        eps0 = consts.epsilon_0
        hbar = consts.hbar
        return np.sqrt(hbar * abs(omega) / (2 * eps0 * self.volumes[mode_index]) )

    def make_E_field_weights(self,):
        '''
        Returns:
            Weights (dict):
                A dictionary from mode index to the E-field weight.

        '''
        weights = {}
        for mode_index in range(self.m):
            weights[mode_index] = self.E_field_weight(mode_index)
        return weights

    def make_weight_keys(self,chi):
        r'''
        Make a list of keys for which various weights will be determined.
        Each key is a tuple consisting of two
        components: the first is a tuple of the indices of modes and the
        second is a tuple of +1 and -1.

        Args:
            chi (Chi_nonlin):
                the nonlinearity for which the weight will be
                found.

        Returns:
            Keys (list):
                A list of keys of the type described.

        '''
        weight_keys=[]
        list_of_pm_arr = list(itertools.product([-1, 1],
            repeat=chi.chi_order+1))
        field_combinations = itertools.combinations_with_replacement(
            range(self.m), chi.chi_order+1)  ##generator
        for combination in field_combinations:
            for pm_arr in list_of_pm_arr:
                weight_keys.append( (tuple(combination),tuple(pm_arr)) )
        return weight_keys

    def make_nonlin_H_from_chi(self,chi,filtering_phase_weights=False,eps=1e-5):
        '''Make a nonlinear Hamiltonian based on nonlinear interaction terms

        Args:
            chi (Chi_nonlin):
                Nonlinearity to use.

            eps (optional[float]):
                Cutoff for the significance of a particular term.

        Returns:
            Expression (sympy expression):
                A symbolic expression for the nonlinear Hamiltonian.

        TODO:  Make separate dictionaries for values of chi_function,
        for phase_matching_weights, and for producs of E_field_weights. filter
        the keys before generating terms.

        TODO: Make fast function for integrator; combine with make_f and make_f_lin

        '''
        H_nonlin_sp = sp.Float(0.)
        for chi in self.chi_nonlinearities:
            weight_keys = self.make_weight_keys(chi)

            phase_matching_weights = self.make_phase_matching_weights(
                weight_keys,chi,filtering_phase_weights,eps)

            for combination,pm_arr in phase_matching_weights:
                omegas_to_use = map(lambda i: self.omegas[i],combination)
                omegas_with_sign = [omega * pm for omega,pm
                                    in zip(omegas_to_use,pm_arr)]
                pols = map(lambda i: self.polarizations[i],chi.delay_indices)
                chi_args = omegas_with_sign + pols
                H_nonlin_sp += ( self.make_nonlin_term_sympy(combination,pm_arr) *
                    chi.chi_function(*chi_args) *
                    phase_matching_weights[combination,pm_arr] *
                    np.prod([self.E_field_weights[i] for i in combination]) )
        return H_nonlin_sp

    def make_lin_H(self,Omega):
        '''Make a linear Hamiltonian based on Omega.

        Args:
            Omega (complex-valued matrix):
                A matrix that describes the Hamiltonian of the system.

        Returns:
            Expression (sympy expression):
                A symbolic expression for the nonlinear Hamiltonian.

        '''
        H_lin_sp = sp.Float(0.)
        for i in range(self.m):
            for j in range(self.m):
                H_lin_sp += self.Dagger(self.a[i])*self.a[j]*Omega[i,j]
        return H_lin_sp

    def make_H(self,eps=1e-5):
        r'''Make a Hamiltonian combining the linear and nonlinear parts.

        The term -1j*A carries the effective linear Hamiltonian, including the
        decay term :math:`-\frac{i}{2} L^\dagger L`. However, this term does
        not include material effects including dielectric and nonlinear terms.
        It also does not include a term with contribution from the inputs.

        If one wishes to include terms due to coherent input, one can impose a
        linear Hamiltonian term consistent with the classical equations of
        motion. This yields the usual term :math:`i(a \alpha^* - a^\dagger \alpha)`.

        To obtain the form :math:`A = i \Omega - \frac{1}{2} C^\dagger C` with
        :math:`Omega` Hermitian, we notice :math:`A` can be split into Hermitian
        and anti-Hermitian parts. The anti-Hermitian part of A describes the
        closed dynamics only and the Hermitian part corresponds to the decay
        terms due to the coupling to the environment at the input/output ports.

        Args:
            Omega (complex-valued matrix):
                Describes the Hamiltonian of the system.

            eps (optional[float]):
                Cutoff for the significance of a particular term.


        Note:
            Omega = -1j*A        <--- full dynamics (not necessarily Hermitian)

            Omega = (A-A.H)/(2j) <--- closed dynamics only (Hermitian part of above)


        Returns:
            Expression (sympy expression):
                A symbolic expression for the full Hamiltonian.

        '''
        H_nonlin = self.make_nonlin_H_from_chi(eps)
        H_lin = self.make_lin_H(self.Omega)
        self.H = H_lin + H_nonlin * self.nonlin_coeff
        self.H = normal_order((self.H).expand())
        return self.H

    def move_to_rotating_frame(self, freqs = 0.,include_time_terms = True):
        r'''Moves the Hamiltonian to a rotating frame

        We apply a change of basis :math:`a_j \to a e^{- i \omega_j}` for
        each mode :math:`a_j`. This method modifies the symbolic Hamiltonian,
        so to use it the Hamiltonian sould already be constructed and stored.

        Args:
            freqs (optional [real number or list/tuple]):
                Frequency or list
                of frequencies to use to displace the Hamiltonian.

            include_time_terms (optional [boolean]):
                If this is set to true,
                we include the terms :math:`e^{- i \omega_j}` in the Hamiltonian
                resulting from a change of basis. This can be set to False if all
                such terms have already been eliminated (i.e. if the rotating wave
                approximation has been applied).

        TODO: replace the sine and cosine stuff with something nicer.
        Maybe utilize the _get_real_imag_func method in Time_Delay_Network.

        '''
        if type(freqs) in [float,long,int]:
            if freqs == 0.:
                return
            else:
                self.move_to_rotating_frame([freqs]*self.m)
        elif type(freqs) in [list,tuple]:
            for op,freq in zip(self.a,freqs):
                self.H -= freq * self.Dagger(op)* op
            self.H = (self.H).expand()
            if include_time_terms:
                for op,freq in zip(self.a,freqs):
                    self.H = (self.H).subs({
                        self.Dagger(op) : self.Dagger(op)*( sp.cos(freq * self.t)
                            + sp.I * sp.sin(freq * self.t) ),
                        op : op * ( sp.cos(freq * self.t)
                            - sp.I * sp.sin(freq * self.t) ),
                        })
                        ### Sympy has issues with the complex exponential...
                        # self.H = (self.H).subs({
                        #     self.Dagger(op) : self.Dagger(op)*sp.exp(sp.I * freq * self.t),
                        #     op : op*sp.exp(-sp.I * freq * self.t),
                        # })
                        ###
                self.H = (self.H).expand()
        else:
            print "freqs should be a real number or list of real numbers."
            return


    def make_eq_motion(self,):
        r'''Input is a tuple or list, output is a matrix vector.
        This generates Hamilton's equations of motion for a and a^H.
        These equations are CLASSICAL equations of motion. This means
        we replace the operators with c-numbers. The orde   r of the operators
        will yield different results, so we assume the Hamiltonian is already
        in the desired order (e.g. normally ordered).

        These equations of motion will not show effects of squeezing. To do
        this, we will need a full quantum picture.

        Returns:
            Equations of Motion (function):
                A function that yields the Hamiltonian equations of motion based on
                the Hamiltonian given. The equations of motion map
                :math:`(t,a) \to v`. where \math:`t` is a scalar corresponding to
                time, :math:`a` is an array of inputs correpsonding to the internal
                degrees of freedom, and :math:`v` is a complex-valued column matrix
                describing the gradient.

        '''
        if self.using_qnet_symbols:
            print "Warning: The Hamiltonian should be regular c-numbers!"
            print "Returning None"
            return None

        ## c-numbers
        b = [sp.symbols('b'+str(i)) for i in range(self.m)]
        b_conj = [sp.symbols('b_H'+str(i)) for i in range(self.m)]

        D_to_c_numbers = {self.a[i] : b[i] for i in range(self.m)}
        D_to_c_numbers.update({self.Dagger(self.a[i]) : b_conj[i] for
            i in range(self.m)})

        H_conj = self.Dagger(self.H)

        H_c_numbers = self.H.subs(D_to_c_numbers)
        H_conj_c_numbers = H_conj.subs(D_to_c_numbers)  ## don't have to copy


        ## classical equations of motion
        diff_ls = ([1j*sp.diff(H_c_numbers,var) for var in b_conj ] +
               [-1j*sp.diff(H_conj_c_numbers,var) for var in b])
        fs = ([sp.lambdify( (self.t, b + b_conj), expression)
            for expression in diff_ls ])
        return lambda t,arr: (np.asmatrix([ sp.N ( f(t,arr) )
            for f in fs], dtype = 'complex128' )).T

        #### In future implementations, we can use theano to make calculations faster
        #### right now however theano does not have good support for complex numbers.
        ###f = theano_function([self.t]+b+b_H, diff_ls)   ## f(t,b[0],b[1],...)
        ###F = lambda t,args: np.asarray(f(t,*args))                  ## F(t,(b[0],b[1],...))
        #return F
