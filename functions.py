# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:59:30 2015

@author: Gil Tabak
@title: Potapov

Functions used by other files.

"""

import numpy as np
import numpy.linalg as la

def der(f,z,eps = 1e-5):
    '''
    Estimate the derivative of the function f at z

    Given:
        f (function): the function to use
        z (complex number): point to evaluate
        eps(optional[complex number]): number to perturb z to find derivative

    Returns:
        Derivative of f
    '''
    return (f(z+eps)-f(z-eps))/(2*eps)

def limit(f,z0,N=10,eps=1e-3):
    '''
    Takes possibly matrix-valued function f and its simple pole z0 and returns
    limit_{z \to val} f(z). Estimates the value based on N surrounding
    points at a distance eps.

    Args:
        f (function): the function for which the limit will be found.
        z0 (complex number): The value at which the limit is evaluated.
        N (int): number of points used in the estimate.
        eps (optional[float]): distance from z0 at which estimating points are
        placed.

    Returns:
         The estimated value of limit_{z \to val} f(z).

    '''
    t=np.linspace(0.,2.*np.pi*(N-1.)/N,num=N)
    c=np.exp(1j*t)*eps
    try:
        s=sum(f(z0 + c_el) for c_el in c)/float(N)
        return s
    except:
        print "Something went wrong in estimating the limit."
        return

def factorial(n):
    '''
    Find the factorial of n.
    Args:
        n (integer)
    Returns:
        factorial of n
    '''
    end = 1
    for k in xrange(1,n+1):
        end *= k
    return end

def pade_approx(n):
    '''
    Args:
        n (integer)
    Returns:
        Denominator of symmetric Pade approximation of e^{-s} of order n
    '''
    output = [0]*(n+1)
    for k in xrange(0,n+1):
        output[n-k] = float(factorial(2*n-k)) * factorial(n) / \
                      float((factorial(2*n) )* factorial(k) * factorial(n - k) )
    return output

def pade_roots(n):
    '''
    Extract roots of Pade polynomial.
    Args:
        n (integer)
    Returns:
        Roots of Pade polynomial.
    '''
    return np.roots(pade_approx(n))

def Q(z,n):
    '''
    Numerator of Pade pproximation of e^z

    args:
        n (integer): order of approximation
        z (complex number):

    Returns:
        Value of Numerator of Pade approximation.


    '''
    coeffs = pade_approx(n)
    sum = 0
    for i in xrange(0,n+1):
        sum += coeffs[i]*pow(z,n-i)
    return sum

def Pade(n,z):
    '''
    Pade pproximation of e^z

    args:
        n (integer): order of approximation
        z (complex number):

    Returns:
        Value of Pade approximation.

    '''
    return Q(z,n)/Q(-z,n)

def spatial_modes(roots,M1,E):
    '''
    Obtain the spetial mode profile at each node up to a constant.

    Args:
        roots: The eigenvalues of the system
        matrix M1: The connectivity matrix among internal nodes
        E (matrix-valued function): Time-delay matrix

    Returns:
        A list of spatial eigenvectors.
    '''
    spatial_vecs = []
    for i in xrange(len(roots)):
        evals,evecs = la.eig(M1*E(roots[i]))
        spatial_vecs.append(evecs[:,np.argmin(abs(1.-evals))])
    return spatial_vecs

def inner_product_of_two_modes(root1,root2,v1,v2,delays,eps=1e-7,
                                func=lambda z : z.imag):
    '''
    This function integrates two spatial modes against each other
    along the various delays of the system. Each delay follows a
    node in the system.

    The frequency is assumed to be the imaginary part of each root.

    Args:
        root1,root2 (complex number): the two roots
        v1,v2 (column matrices): the amplitude of each mode at the
        various nodes
        delays (list of floats): The duration of each delay following
        each node in the system
        eps (optional[float]): under tolerance eps, assume the two
        roots are identical. The analytic expression of the intensity
        changes in this case.
        func (optional[funciton]): used to transform the roots. Default
        value is set to lambda z: z.imag, meaning we take the frequency
        of each mode.

    Returns:
        The inner product of the two modes.
        Sanity check: if root1==root2 and v1==v2, returns real value.
    '''
    s = 0j
    for delay,e1,e2 in zip(delays,v1,v2):
        if abs(func(root1-root2)) < eps:
            s+=e1*np.conj(e2)*delay
        else:
            s += (e1*e2.H*1j*(np.exp(-1j*delay*func(root1-root2)) - 1. )
                        /func(root1-root2) )
    return s[0,0]

def make_normalized_inner_product_matrix(roots,modes,delays,eps=1e-12,
                                func=lambda z : z.imag):
    '''
    Given a list of roots and a list of vectors representing the
    electric field at each node of the corresponding nodes, compute
    the normalized matrix representing the inner products among the
    various modes.

    TODO: add weights for different delays to account for geometry.
    '''
    dim = len(roots)
    norms = [0]*dim
    for i,(root,v) in enumerate(zip(roots,modes)):
        norms[i] = inner_product_of_two_modes(root,root,v,v,delays,eps=eps,
                                        func=func)
    inner_prods = np.zeros((dim,dim),dtype='complex_')
    for i in range(dim):
        for j in range(dim):
            inner_prods[i,j] = ((inner_product_of_two_modes(roots[i],roots[j],
                                modes[i],modes[j],delays)) /
                                np.sqrt(norms[i]*norms[j]) )
    return inner_prods

def make_nonlinear_interaction(roots, modes, delays, delay_index,
                                start_nonlin,end_nonlin,plus_or_minus_arr,
                                indices_of_refraction = None,
                                eps=1e-12,func=lambda z : z.imag):
    '''
    This function takes several (say M) roots and their corresponding modes,
    as well as the (N) delay lengths of the network, and determines the term
    we need to add to the Hamiltonian corresponding to the resulting
    nonlinearity. We assume there is a crystal going from start_nonlin to
    end_nonlin. The plus_or_minus_arr is an array of length m of 1 or -1
    used to determined whether a mode corresponds to a creation (1, a^\dag)
    or annihilation (-1,a) operator. The corresponding electric field integrated
    will be E^\dag for 1 and E for -1.

    The k-vectors are computed from the following formula:
    k = omega / v_p = omega n(omega) / c.

    Below we assume c == 1.

    If the indices of refraction n(omega_i) are given, we use them to compute
    the phase-mismatch delta_k. Otherwise we assume they are all equal to 1.
    '''

    if start_nonlin < 0:
        raise Exception('start_nonlin must be greater than 0.')

    if end_nonlin > delays[delay_index]:
        raise Exception('end_nonlin must be less than the delay of index delay_index.')

    M = len(roots)
    if len(modes) != M:
        raise Exception('number of modes different than number of roots.')

    if indices_of_refraction == None:
        indices_of_refraction = [1.] * M

    def pick_conj(m,sign):
        if sign == 1:
            return m
        elif sign == -1:
            return np.conj(m)
        else:
            raise Exception('bad input value -- must be 1 or -1.')

    ms = [m_vec[delay_index,0] for m_vec in modes]
    delta_k = sum([n*func(root)*sign for n,root,sign
           in zip(indices_of_refraction,roots,plus_or_minus_arr)])
    const = np.prod([pick_conj(m,sign) for m,sign in zip(ms,plus_or_minus_arr)])

    if abs(delta_k) < eps: ## delta_k \approx 0
        return const * (end_nonlin - start_nonlin)
    else:
        return const * 1j * (np.exp(-1j*delta_k*end_nonlin) -
                             np.exp(-1j*delta_k*start_nonlin) ) / delta_k
