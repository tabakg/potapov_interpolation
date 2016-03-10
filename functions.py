# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:59:30 2015

@author: Gil Tabak
@title: Potapov

Functions used by other files.

"""

import numpy as np

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



## Pade stuff
def factorial(n):
    end = 1
    for k in xrange(1,n+1):
        end *= k
    return end

# denominator of symmetric Pade approximation of e^{-s} of order n
def pade_approx(n):
    output = [0]*(n+1)
    for k in xrange(0,n+1):
        output[n-k] = float(factorial(2*n-k)) * factorial(n) / \
                      float((factorial(2*n) )* factorial(k) * factorial(n - k) )
    return output

def pade_roots(n):
    return np.roots(pade_approx(n))

def Q(z,n):
    coeffs = pade_approx(n)
    sum = 0
    for i in xrange(0,n+1):
        sum += coeffs[i]*pow(z,n-i)
    return sum

## approximates e^z
def Pade(n,z):
    return Q(z,n)/Q(-z,n)
