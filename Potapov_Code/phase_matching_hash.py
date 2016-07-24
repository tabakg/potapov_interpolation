import sympy as sp
import numpy as np
import scipy.constants
from sympy.utilities.autowrap import ufuncify
from scipy import interpolate
import matplotlib.pyplot as plt

from functions import timeit
from functions import make_dict_values_to_lists_of_inputs


def generate_k_func_4wv(pols=(-1,-1,-1,-1), n_symb = None):
    '''
        Generates two functions that should sum to zero when the phase
        and frequency matching conditions are satisfied.

        Args:
            pols (optional[tuple]): list of polarizations for the four freqs.
            n_sym (optional[function]): index of refraction as a function of omega.

        Returns:
            diff_func_4wv_1 (function):
                a function of only two variables phi1 and phi2
            diff_func_4wv_2 (function):
                a function of only two variables phi1 and omega3
    '''

    ## from http://refractiveindex.info/?shelf=main&book=LiNbO3&page=Zelmon-o

    lambd,omega,omega1,omega2,omega3,omega4 = sp.symbols(
        'lambda omega omega_1 omega_2 omega_3 omega_4')
    l2 = lambd **2

    if n_symb is None:
        def n_symb(pol=1):
            s = 1.
            if pol == 1:
                s += 2.6734 * l2 / (l2 - 0.01764)
                s += 1.2290 * l2 / (l2 - 0.05914)
                s += 12.614 * l2 / (l2 - 474.6)
            else: # pol = -1
                s += 2.9804 * l2 / (l2 - 0.02047)
                s += 0.5981 * l2 / (l2 - 0.0666)
                s += 8.9543 * l2 / (l2 - 416.08)
            return sp.sqrt(s)

    def k_symb(symbol=omega,pol=1):
        '''
        k is accurate for omega inputs between 6-60.
        '''
        return ((n_symb(pol=pol) * symbol )
            .subs(lambd,scipy.constants.c / (symbol*1e7))) ## / scipy.constants.c

    phi1, phi2 = sp.symbols('phi_1 phi_2')
    ex1 = ( (k_symb(omega1,pol=pols[0])
            +k_symb(omega2,pol=pols[1]))
        .expand().subs({omega1:(phi1 + phi2)/2, omega2: (phi1-phi2)/2}) )
    ex2 = -((k_symb(omega3,pol=pols[2])
            +k_symb(omega4,pol=pols[3]))
        .expand().subs(omega4,-phi1-omega3) )

    diff_func_4wv_1 = ufuncify([phi1,phi2], ex1)
    diff_func_4wv_2 = ufuncify([phi1,omega3], ex2)

    return diff_func_4wv_1, diff_func_4wv_2

def eps_multiply_digitize(y,eps):
    '''
    Divide input by epsilong and round.
    '''
    return  map(lambda el: int(el/eps), y)

@timeit
def make_matching_dict_hash(diff_func_4wv_1,diff_func_4wv_2,
    phi1_range,phi2_range,omega3_range,eps=2e-4):
    '''
    Make a dictionary mapping points close to solutions of a phase-matching condition
    to error values using hash tables.

    Args:
        phi1_range (numpy array):
            list of phi1 values
        phi2_range (numpy array):
            list of phi2 values
        omega3_range (numpy array):
            list of omega3 values
        eps (optional[float]):
            error allowed for phase-matching condition

    Returns:
        Dictionary (dict):
            dict mapping points to error values.

    '''
    phi2_indices = range(len(phi2_range))
    omega3_indices = range(len(omega3_range))
    matching_dict = {}
    for phi1_index,phi1 in enumerate(phi1_range):
        y1 = diff_func_4wv_1(phi1,phi2_range)
        y2 = diff_func_4wv_2(phi1,omega3_range)

        y1_rounded = eps_multiply_digitize(y1,eps)
        y1_rounded_up =  [ind + 1 for ind in y1_rounded]
        y2_rounded = eps_multiply_digitize(y2,eps)
        y2_rounded_up =  [ind + 1 for ind in y2_rounded]

        D1 = make_dict_values_to_lists_of_inputs(
            y1_rounded+y1_rounded_up,2*phi2_indices)
        D2 = make_dict_values_to_lists_of_inputs(
            y2_rounded+y2_rounded_up,2*omega3_indices)

        inter = set(D1.keys()) & set(D2.keys())

        for el in inter:
            for ind1 in D1[el]:
                for ind2 in D2[el]:
                    err = y1[ind1]-y2[ind2]
                    if abs(err) < eps:
                        matching_dict[phi1_index,ind1,ind2] = err

    return matching_dict

if __name__ == "__main__":

    eps = 2e-4

    phi1_min = 30.
    phi1_max = 34.

    phi2_min = -13
    phi2_max = -9

    omega3_min = -26.
    omega3_max = -16.

    phi1_range = np.arange(phi1_min,phi1_max,0.01)
    phi2_range = np.arange(phi2_min,phi2_max,0.01)
    omega3_range = np.arange(omega3_min,omega3_max,0.01)

    phi1_indices = range(len(phi1_range))
    phi2_indices = range(len(phi2_range))

    diff_func_4wv_1,diff_func_4wv_2 = generate_k_func_4wv(pols=(-1,-1,-1,-1))

    f_phi12_omega3 = lambda phi1,phi2,omega3 : (
        diff_func_4wv_1(phi1_range[phi1],phi2_range[phi2])
        - diff_func_4wv_2(phi1_range[phi1],omega3_range[omega3]) )

    matching_dict_hash = make_matching_dict_hash(
        diff_func_4wv_1,diff_func_4wv_2,
        phi1_range,phi2_range,omega3_range,eps=eps)

    print len(matching_dict_hash)
