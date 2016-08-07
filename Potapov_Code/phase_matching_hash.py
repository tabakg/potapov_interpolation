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
            n_sym (optional[function]): index of refraction as a function of nu.

        Returns:
            diff_func_4wv_1 (function):
                a function of only two variables phi1 and phi2
            diff_func_4wv_2 (function):
                a function of only two variables phi1 and nu3
    '''

    ## from http://refractiveindex.info/?shelf=main&book=LiNbO3&page=Zelmon-o

    lambd,nu,nu1,nu2,nu3,nu4 = sp.symbols(
        'lambda nu nu_1 nu_2 nu_3 nu_4')
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

    def k_symb(symbol=nu,pol=1):
        '''
        k is accurate for nu inputs between 6-60.
        '''
        return ((n_symb(pol=pol) * symbol )
            .subs(lambd,scipy.constants.c / (symbol*1e7))) ## / scipy.constants.c

    phi1, phi2 = sp.symbols('phi_1 phi_2')
    ex1 = ( (k_symb(nu1,pol=pols[0])
            +k_symb(nu2,pol=pols[1]))
        .expand().subs({nu1:(phi1 + phi2)/2, nu2: (phi1-phi2)/2}) )
    ex2 = -((k_symb(nu3,pol=pols[2])
            +k_symb(nu4,pol=pols[3]))
        .expand().subs(nu4,-phi1-nu3) )

    diff_func_4wv_1 = ufuncify([phi1,phi2], ex1)
    diff_func_4wv_2 = ufuncify([phi1,nu3], ex2)

    def diff_func_4wv_1_ranges_checked(phi1,phi2):
        nus = [phi1 + phi2, phi1 - phi2]
        if any([abs(nu) < 6. or abs(nu) > 60. for nu in nus]):
            return float('NaN')
        else:
            return diff_func_4wv_1(phi1,phi2)

    def diff_func_4wv_2_ranges_checked(phi1,nu3):
        if abs(phi1) < 6. or abs(phi1) > 60.:
            return float('NaN')
        else:
            return diff_func_4wv_2(phi1,nu3)

    return diff_func_4wv_1_ranges_checked, diff_func_4wv_2_ranges_checked

def eps_multiply_digitize(y,eps):
    '''
    Divide input by epsilong and round.
    '''
    return  map(lambda el: int(el/eps), y)

@timeit
def make_matching_dict_hash(diff_func_4wv_1,diff_func_4wv_2,
    phi1_range,phi2_range,nu3_range,eps=2e-4):
    '''
    Make a dictionary mapping points close to solutions of a phase-matching condition
    to error values using hash tables.

    Args:
        phi1_range (numpy array):
            list of phi1 values
        phi2_range (numpy array):
            list of phi2 values
        nu3_range (numpy array):
            list of nu3 values
        eps (optional[float]):
            error allowed for phase-matching condition

    Returns:
        Dictionary (dict):
            dict mapping points to error values.

    '''
    phi2_indices = range(len(phi2_range))
    nu3_indices = range(len(nu3_range))
    matching_dict = {}
    for phi1_index,phi1 in enumerate(phi1_range):
        y1 = diff_func_4wv_1(phi1,phi2_range)
        y2 = diff_func_4wv_2(phi1,nu3_range)

        y1_rounded = eps_multiply_digitize(y1,eps)
        y1_rounded_up =  [ind + 1 for ind in y1_rounded]
        y2_rounded = eps_multiply_digitize(y2,eps)
        y2_rounded_up =  [ind + 1 for ind in y2_rounded]

        D1 = make_dict_values_to_lists_of_inputs(
            y1_rounded+y1_rounded_up,2*phi2_indices)
        D2 = make_dict_values_to_lists_of_inputs(
            y2_rounded+y2_rounded_up,2*nu3_indices)

        inter = set(D1.keys()) & set(D2.keys())

        for el in inter:
            for ind1 in D1[el]:
                for ind2 in D2[el]:
                    err = y1[ind1]-y2[ind2]
                    if abs(err) < eps:
                        matching_dict[phi1_index,ind1,ind2] = err

    return matching_dict

def make_positive_keys_chi3(pos_nus_lst,chi,eps=2e-4,pols = None, res = (1e-2,1e-2,1e-2) ):
    '''
    TODO: In the future, diff_func_4wv_1,diff_func_4wv_2 will be generated from
          chi_nonlin.refraction_index_func in the form of a function
          n_symb(pol=+1) -> sympy expression in lambd.
    '''
    ## get the positive nus. Make a dict to the original index.
    min_nu_value = min(pos_nus_lst)
    max_nu_value = max(pos_nus_lst)

    ## phi1 = (nu1 + nu2) / 2, nu's are postiive
    phi1_min = min_nu_value
    phi1_max = max_nu_value

    ## phi2 = (nu1 - nu2) / 2, nu's are postiive
    phi2_min = (max_nu_value - min_nu_value) / 2
    phi2_max = (min_nu_value - max_nu_value) / 2

    ## nu_3 had no transformations
    nu3_min = min_nu_value
    nu3_max = max_nu_value

    phi1_range = np.arange(phi1_min,phi1_max,res[0])
    phi2_range = np.arange(phi2_min,phi2_max,res[1])
    nu3_range = np.arange(nu3_min,nu3_max,res[2])

    phi1_indices = range(len(phi1_range))
    phi2_indices = range(len(phi2_range))

    diff_func_4wv_1,diff_func_4wv_2 = generate_k_func_4wv(pols=pols)

    matching_dict_hash = make_matching_dict_hash(
        diff_func_4wv_1,diff_func_4wv_2,
        phi1_range,phi2_range,nu3_range,eps=eps)

    return matching_dict_hash

# def f_phi12_nu3(phi1,phi2,nu3):
#     nus = [phi1 + phi2, phi1 - phi2, nu3]
#     if any([abs(nu) < 6. or abs(nu) > 60. for nu in nus]):
#         return float('NaN')
#     return (diff_func_4wv_1(phi1_range[phi1],phi2_range[phi2])
#         - diff_func_4wv_2(phi1_range[phi1],nu3_range[nu3]) )

if __name__ == "__main__":

    eps = 2e-4

    phi1_min = 30.
    phi1_max = 34.

    phi2_min = -13
    phi2_max = -9

    nu3_min = -26.
    nu3_max = -16.

    phi1_range = np.arange(phi1_min,phi1_max,0.01)
    phi2_range = np.arange(phi2_min,phi2_max,0.01)
    nu3_range = np.arange(nu3_min,nu3_max,0.01)

    phi1_indices = range(len(phi1_range))
    phi2_indices = range(len(phi2_range))

    diff_func_4wv_1,diff_func_4wv_2 = generate_k_func_4wv(pols=(-1,-1,-1,-1))

    matching_dict_hash = make_matching_dict_hash(
        diff_func_4wv_1,diff_func_4wv_2,
        phi1_range,phi2_range,nu3_range,eps=eps)

    print len(matching_dict_hash)
