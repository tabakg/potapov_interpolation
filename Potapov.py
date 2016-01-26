# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:59:30 2015

@author: Gil Tabak
@title: Potapov

The code in this file implements the procedure for finding Blaschke-Potapov
products to approximate given functions near poles.

Please see section 6.2 in our manuscript for details: http://arxiv.org/abs/1510.08942
(to be published in EPJ QT).

"""

import numpy as np
import sympy as sp
sp.init_printing()
r,tau,z,theta = sp.symbols('r tau z theta', complex=True)
import pprint
import matplotlib.pyplot as plt
import numpy.linalg as la


def limit_T(f,z0,N=10,eps=1e-4):
    '''
    Takes matrix-valued function T and a simple pole val and returns
    limit_{z \to val} T(z)(z-val).
    '''
    dim = f(0).shape[0]
    t=np.linspace(0.,2.*np.pi*(N-1.)/N,num=N)
    c=np.exp(1j*t)*eps
    s = np.zeros((dim,dim))
    for c_el in c:
        s  = s + f(z0 + c_el)
    return s / float(N)

def plot(L,dx,func,(i,j),*args):
    '''
    A nice function for plotting.
    '''
    x = np.linspace(-L,L,2.*L/dx)
    for arg in args:
        plt.plot(x,[func(arg(x_el*1j)[i,j]) for x_el in x ])

def Potapov_prod(z,poles,vecs,N):
    '''
    Takes a transfer function T(z) that outputs numpy matrices for imaginary
    z = i \omega and the desired poles that characterize the modes.
    Returns the Potapov product as a function approximating the original
    transfer function.
    '''
    R = np.asmatrix(np.eye(N))
    for pole_i,vec in zip(poles,vecs):
        Pi = vec*vec.H
        R = R*(np.eye(N) - Pi + Pi * ( z + pole_i.conjugate() )/( z - pole_i) )
    return R

def get_Potapov_vecs(T,poles):
    '''
    Given a transfer function T and some poles, compute the residues about the
    poles and generate the eigenvectors to use for constructing the projectors
    in the Blaschke-Potapov factorization.
    '''
    N = T(0).shape[0]
    found_vecs = []
    for pole in poles:
        L = la.inv(Potapov_prod(pole,poles,found_vecs,N))*\
            limit_T(lambda z: (z-pole)*T(z),pole)
        [eigvals,eigvecs] = la.eig(L)
        index = np.argmax(map(abs,eigvals))
        big_vec = np.asmatrix(eigvecs[:,index])
        found_vecs.append(big_vec)
    return found_vecs

def get_Potapov(T,poles):
    '''
    Given a transfer function T and some poles, generate the Blaschke-Potapov
    product to reconstruct or approximate T, assuming that T can be represented
    by the Blaschke-Potapov product with the given poles. Also match the values
    of the functions at zero.

    If T is a Blaschke-Potapov function and the the given poles are the only poles,
    then T will be reconstructed.

    In general, there is possibly an analytic term that is not captured by using
    a Blaschke-Potapov approximation.
    '''
    N = T(0).shape[0]
    found_vecs = get_Potapov_vecs(T,poles)
    return lambda z: T(0)*Potapov_prod(0,poles,found_vecs,N).H*\
        Potapov_prod(z,poles,found_vecs,N)

def prod(z,U,eigenvectors,eigenvalues):
    '''
    Return the Blaschke-Potapov product with the given eigenvalues and
    eigenvectors and constant unitary factor U evaluated at z.
    '''
    if eigenvectors==[] or eigenvalues == []:
        return U
    else:
        vec = eigenvectors[-1]
        val = eigenvalues[-1]
        N = U.shape[0]
        return prod(z,U,eigenvectors[:-1],eigenvalues[:-1])*\
            (np.eye(N) - vec*vec.H + vec*vec.H*(z+val.conjugate())/(z-val))

def finite_transfer_function(U,eigenvectors,eigenvalues):
    '''
    Give a rational Blaschke-Potapov product of z with the given
    eigenvalues and eigenvectors and constant unitary factor U.
    '''
    return lambda z: prod(z,U,eigenvectors,eigenvalues)

def normalize(vec):
    '''
    Normalize a vector.
    '''
    return vec / la.norm(vec)

def get_ABCD(val, vec):
    '''
    Make the ABCD model of a single Potapov factor given some eigenvalue
    and eigenvector.

    The ABCD model can be used to obtain the dynamics of a linear system.
    '''
    N = vec.shape[0]
    return [val*vec.H*vec, vec.H, vec*(val+val.conjugate()), np.eye(N)]

def get_Potapov_ABCD(poles,vecs):
    '''
    Combine the ABCD models for the different degrees of freedom.
    '''
    if min(len(poles),len(vecs)) < 1:
        print "Emptry list into get_Potapov_ABCD"
    elif min(len(poles),len(vecs)) == 1:
        return get_ABCD(poles[0],vecs[0])
    else:
        [A1,B1,C1,D1] = get_Potapov_ABCD(poles[1:], vecs[1:])
        [A2,B2,C2,D2] = get_ABCD(poles[0],vecs[0])

        O = np.zeros((A1.shape[0],A2.shape[1]))
        A_first_row_block =  np.hstack((A1,O))
        A_second_row_block = np.hstack((B2 * C1, A2))
        A = np.vstack((A_first_row_block,A_second_row_block))
        B = np.vstack(( B1, B2*D1))
        C = np.hstack(( D2*C1, C2))
        D = D2*D1
        return [A,B,C,D]
