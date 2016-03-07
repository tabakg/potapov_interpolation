# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 20:15:35 2015

@author: gil
@title: Rootfinder
"""
import numpy as np
from itertools import chain
from scipy import integrate
import math
import cmath as cm
from functions import limit

def Muller(x1,x2,x3,f,tol = 1e-12,N=400):
    '''
    A method that works well for finding roots locally in the complex plane.
    Uses three points for initial guess, x1,x2,x3.

    Args:
        x1,x2,x3 (complex numbers): initial points for the algorithm
        f (function): complex valued function for which to find roots
        tol (optional[float]): tolerance
        N(optional[int]): maximum number of iterations

    Returns:
        estimated root of the function f.
    '''
    n = 0
    x = x3

    if x1 == x2:
        print "Muller needs x1 and x2 different!!!"
        print "x1 = x2 = ", x1
        return x3

    if x2 == x3:
        print "Muller needs x2 and x3 different!!!"
        print "x2 = x3 = ", x2
        return x3

    if x1 == x3:
        print "Muller needs x1 and x3 different!!!"
        print "x1 = x3 = ", x1
        return x3


    while n < N and abs(f(x3))>tol:
        n+=1
        q = (x3 - x2) / (x2 - x1)
        A = q * f(x3) - q*(1.+q)*f(x2)+q**2.*f(x1)
        B = (2.*q+1.)*f(x3)-(1.+q)**2.*f(x2)+q**2.*f(x1)
        C = (1.+q)*f(x3)

        D1 = B+cm.sqrt(B**2-4.*A*C)
        D2 = B-cm.sqrt(B**2-4.*A*C)
        if abs(D1) > abs(D2):
            D = D1
        elif D1 == D2 == 0:
            if abs(f(x3))< tol:
                return x3
            else:
                print "Desired tolerance not reached and Muller denominator diverges.",
                "Please try different parameters in Muller for better results."
                return x3
        else: D = D2

        x = x3 - (x3-x2)*2.*C / D

        x1 = x2
        x2 = x3
        x3 = x
        #print x

    return x

def residues(f_frac,roots):
    '''
    Finds the resides of the give roots of f_frac

    Args:
        f_frac (function):
    '''
    return [limit(lambda z: (z-root)*f_frac(z),root) for root in roots]

## Functions that evaluate the f_frac after some roots and their residues are subtracted.
## The safe version checks for large values and division by zero.

def new_f_frac(f_frac,z0,residues,roots,val=None):
    if val == None:
        val = f_frac(z0)
    for res,root in zip(residues,roots):
        val -= res/(z0-root)
    return val

def new_f_frac_safe(f_frac,z0,residues,roots,max_ok,val=None):
    try:
        if val == None:
            val = f_frac(z0)
        if abs(val) < max_ok:
            return new_f_frac(f_frac,z0,residues,roots,val)
        else:
            #print "val too high. taking limit"
            return limit(lambda z: new_f_frac(f_frac,z,residues,roots),z0)
    except ZeroDivisionError:
        #print "div by zero in new_f_frac_safe. Using limit"
        return limit(lambda z: new_f_frac(f_frac,z,residues,roots),z0)

##given the values y_smooth, locations c, and the number to go up to,
##find the roots using the polynomial trick.

def find_roots(y_smooth,c,num_roots_to_find):
    p=[0]  ##placeholder
    for i in xrange(1,num_roots_to_find+1):
        p.append(integrate.trapz([el*z**i for el,z in zip(y_smooth,c)],c) )
    e = [1.]
    for k in xrange(1,num_roots_to_find+1):
        s = 0.
        for i in xrange(1,k+1):
            s += (-1.)**(i-1)*e[k-i]*p[i]
        e.append(s / k)
    coeff = [e[k]*(-1.)**(2.*num_roots_to_find-k)  for k in xrange(0,num_roots_to_find+1)]
    return np.roots(coeff)

def combine(eps=1e-5,*args):
    lst = list(chain(*args))
    return purge(lst, eps)

def purge(lst,eps=1e-5):
    if len(lst) > 0:
        for el in lst[:-1]:
            if abs(el-lst[-1]) < eps:
                return purge(lst[:-1],eps)
        return purge(lst[:-1],eps) + [lst[-1]]
    else: return []

## make a linespace method for complex numbers
def linspace(c1,c2,num=50):
    x1 = c1.real
    y1 = c1.imag
    x2 = c2.real*(num-1.)/num+x1*(1.)/num
    y2 = c2.imag*(num-1.)/num+y1*(1.)/num
    return [real+imag*1j for real,imag in zip(np.linspace(x1,x2,num=num),np.linspace(y1,y2,num=num)) ]

## make a rectangle centered at x_cent,y_cent. Find points along this rectangle.
## I use the convention that width/height make up half the dimensions of the rectangle.
def get_boundary(x_cent,y_cent,width,height,N):
    c1 = x_cent-width+(y_cent-height)*1j
    c2 = x_cent+width+(y_cent-height)*1j
    c3 = x_cent+width+(y_cent+height)*1j
    c4 = x_cent-width+(y_cent+height)*1j
    return  linspace(c1,c2,num=N)+\
            linspace(c2,c3,num=N)+\
            linspace(c3,c4,num=N)+\
            linspace(c4,c1,num=N)

## takes roots and the specification of a rectangular region
## returns the roots in the interior (or boundary) of the region.

def inside_boundary(roots_near_boundary,x_cent,y_cent,width,height):
    output =[]
    for root in roots_near_boundary:
        X = root.real
        Y = root.imag
        if x_cent - width <= X <= x_cent + width and \
           y_cent - height <= Y <= y_cent + height:
                output.append(root)
    return output

def get_max(y,outlier_coeff):
    q75, q50, q25 = np.percentile(y, [75 , 50, 25])
    IQR = q75-q25
    return outlier_coeff*(q50+IQR)

## TODO: handle division by zero case.
def find_outliers(y,max_ok):
    in_outlier = False
    outliers = [] ##list of outliers
    outlier = []  ##each outlier is characterized by a list of consecutive points (i,y[i]) such that y[i] >max_ok
    for i, el in enumerate(y):
        if abs(el) > max_ok:
            in_outlier = True
        else: in_outlier = False
        if in_outlier:
            outlier.append((i,el))
        if outlier != [] and not(in_outlier):
            outliers.append(outlier)
        if not(in_outlier):
            outlier=[]
    return outliers

## Just an experiment... maybe this would work better?
##

def find_maxes(y):
    maxes = []
    for i in xrange(-2,len(y)-2):
        if y[i] < y[i+1] > y[i+2]:
            maxes.append(  [(i,y[i]),(i+1,y[1+i]),(i+2,y[i+2] )]   )
    return maxes


### get_roots method with RECTANGULAR boundary and polynomial thing used.


###   I assume f is analytic with some zeros, and the zeros are simple
###   In the future we can extend to other situations.
###

## We take a function f, its derivative fp, a radius $R$ and a number of points PER BOUNDARY N.
## Optimization to do later: pass down evaluated points in recursion.
def get_roots_rect(f,fp,x_cent,y_cent,width,height,N=10,outlier_coeff=1,max_steps=5,known_roots=[]):

    c = get_boundary(x_cent,y_cent,width,height,N) ##TODO: reuse boundary points when passing into a smaller rectangle
    f_frac = lambda z: fp(z)/(2j*np.pi*f(z))
    y = [f_frac(z) for z in c]

    #print max(y)

    max_ok =  get_max(y,outlier_coeff)

    #outliers = find_outliers(y,max_ok)
    outliers = find_maxes(y)
    #print len(outliers)

    roots_near_boundary = []
    for outlier in outliers:
        first = outlier[0]
        last = outlier[-1]
        r = Muller(c[first[0]-1], c[last[0]+1], (c[first[0]]+c[last[0]])/2, f)
        roots_near_boundary.append(r)
    #print len(roots_near_boundary)

    subtracted_roots = purge(roots_near_boundary+known_roots)

    #print len(subtracted_roots)

    ## let's keep subtracting roots that are in the interior or at least close the the boundary
    ## It would make it a bit inefficient to keep roots from from the boundary.
    subtracted_roots = (inside_boundary(subtracted_roots,x_cent,y_cent,width+2.,height+2.))

    #print len(subtracted_roots)


    subtracted_residues = residues(f_frac,subtracted_roots)
    y_smooth = [new_f_frac_safe(f_frac,z_el,subtracted_residues,subtracted_roots,max_ok,y_el) for y_el,z_el in zip(y,c)]
    I0 = integrate.trapz(y_smooth, c)

    ## Next, divide the situation into two cases. If there's a few roots, find them.
    if I0 < 10:
        num_roots_interior = int(round(abs(I0)))
        if num_roots_interior == 0:
            return inside_boundary(subtracted_roots,x_cent,y_cent,width,height)
        if abs(num_roots_interior-I0)>0.005:
            print "Warning!! Number of roots may be imprecise for this N. Increase N for greater precision."
        print "Approx number of roots in current rect = ", abs(I0)
        rough_roots = find_roots(y_smooth,c,num_roots_interior)
        Muller_all = np.vectorize(Muller)
        interior_roots = purge(Muller_all(rough_roots-1e-5,rough_roots+1e-5,rough_roots,f).tolist())   ##TODO best way to pick points

        #inside_boundary_roots = inside_boundary(roots_near_boundary,x_cent,y_cent,width,height)
        combined_roots = purge(roots_near_boundary + interior_roots)
    else:
        combined_roots = purge(roots_near_boundary)
    ## if some interior roots are missed (or if there were many roots), we subdivide and search
    if I0>=10 or len(combined_roots) < num_roots_interior and max_steps != 0:
        x_list = [x_cent - width / 2.,x_cent - width / 2.,x_cent + width / 2.,x_cent + width / 2.]
        y_list = [y_cent - height / 2.,y_cent + height / 2.,y_cent - height / 2.,y_cent + height / 2.]
        for x,y in zip(x_list,y_list):
            roots_from_subrectangle  = get_roots_rect(f,fp,x,y,width/2.,height/2.,N,outlier_coeff,\
                                             max_steps=max_steps-1,known_roots=combined_roots)
            combined_roots = purge(combined_roots + roots_from_subrectangle)
    elif max_steps == 0:
        print "max_steps exceeded. Some interior roots have not been located."

    return inside_boundary(combined_roots,x_cent,y_cent,width,height)
