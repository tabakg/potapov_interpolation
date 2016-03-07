# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:59:30 2015

@author: Gil Tabak
@title: Potapov

Functions used by other files.

"""

import numpy as np

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
