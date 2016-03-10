# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:37:37 2015

@author: gil
@title: examples
"""
import Roots
import Potapov
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import mpmath as mp
from functions import der
from functions import Pade

def plot_all(L,dx,labels,colors,lw,name,*args):
    '''
    A method to plot the absolute value and phase of each component for a list
    of matrix-valued functions in the complex plane along an axis.

    Args:
        L (float): plot from 0 to L
        dx (float):  distance between points
        labels (list of str): labels to use
        colors (list of srt): indicators of color for different curves
        lw (float): line width to use
        name (str): name of the file to save
        * args (a list of functions): A list of functions to plot

    Returns:
        None
    '''
    delta = np.pi
    x = np.linspace(0,L,2.*L/dx)
    [rows,cols] = args[0](0).shape
    plt.figure(figsize=(18,12))
    for k,func in enumerate([np.abs,np.angle]):
        for i in xrange(rows):
            for j in xrange(cols):
                plt.subplot(2*rows,cols,1+j+(2*i+k)*cols)
                plt.tight_layout()
                s = "norm of component " \
                    if k == 0 else "phase of component "
                s += "["+str(i)+","+str(j)+"]"
                plt.tick_params(labelsize=20)
                plt.title(s, fontsize=24)
                plt.xlabel('frequency (rad)', fontsize=20)


                for l,arg in enumerate(args):
                    y = np.asarray([func(arg(x_el*1j)[i,j]) for x_el in x ])
                    jumps = np.r_[0, np.where(np.abs(np.diff(y)) > delta)[0] + 1, y.size]
                    for m in range(jumps.size-1):
                        start, end = jumps[m:m + 2]
                        if m == 0:
                            plt.plot(x[start:end], y[start:end], colors[l],
                            label = labels[l], lw = lw)
                        else:
                            plt.plot(x[start:end], y[start:end], colors[l], lw = lw)
                    if k == 0:  ## plotting abs
                        plt.axis([0,L,0,1])
                    else:       ## ploting angle
                        plt.axis([0,L,-np.pi,np.pi])
    art = []
    lgd = plt.legend( loc=9, bbox_to_anchor=(0.5, -0.1),shadow=True, fancybox=True, fontsize=18)
    art.append(lgd)
    plt.savefig(name,additional_artists=art,
    bbox_inches="tight")
    return None

def example_helper(T,roots):
    '''
    Helper method for examples

    Args:
        T (function): The original
        roots (list of complex numbers): A list of roots found

    Returns:
        [T,T_testing,roots,vecs]
        T (function): The original function
        T_testing (function): The approximated function
        roots (list of complex numbers): A list of roots found
        vecs (list of matrices): A list of the corresponding eigenvectors
    '''
    T_testing = Potapov.get_Potapov(T,roots)
    vecs = Potapov.get_Potapov_vecs(T,roots)
    return [T,T_testing,roots,vecs]

def example1(max_freq = 30., max_linewidth = 1.):
    '''
    single input, single output with a single delay

    Args:
        max_freq (float): maximum frequency to look at in the complex plane
        max_linewidth (float): maximum linewidth to look at in the complex plane

    Returns:
        [T,T_testing,roots,vecs]
        T (function): The original function
        T_testing (function): The approximated function
        roots (list of complex numbers): A list of roots found
        vecs (list of matrices): A list of the corresponding eigenvectors
    '''
    tau = 0.3
    r = 0.8
    T = lambda z: np.matrix([(np.exp(-z*tau) - r)/(1.-r* np.exp(-z*tau))])
    T_denom = lambda z: (1.-r* np.exp(-z*tau))
    Tp_denom = lambda z: der(T_denom,z)
    roots = Roots.get_roots_rect(T_denom,Tp_denom,0,0,max_linewidth,max_freq,N=1000)
    return example_helper(T,roots)

def example2(max_freq = 10., max_linewidth = 0.5):
    '''
    two inputs, two outputs with a delay (i.e. Fabry-Perot)

    Args:
        max_freq (float): maximum frequency to look at in the complex plane
        max_linewidth (float): maximum linewidth to look at in the complex plane

    Returns:
        [T,T_testing,roots,vecs]
        T (function): The original function
        T_testing (function): The approximated function
        roots (list of complex numbers): A list of roots found
        vecs (list of matrices): A list of the corresponding eigenvectors
    '''
    r = 0.9
    tau = 1.
    N = 2
    e = lambda z: np.exp(-z*tau)
    T_denom = lambda z: (1.-r**2* e(z)**2)
    T = lambda z: -r*np.eye(N) + ((1.-r**2.)/T_denom(z)) * \
    np.matrix([[r*e(z)**2,e(z)],[e(z),r*e(z)**2]])
    Tp_denom = lambda z: der(T_denom,z)
    roots = Roots.get_roots_rect(T_denom,Tp_denom,0,0,max_linewidth,max_freq,N=1000)
    return example_helper(T,roots)

def example3(max_freq = 60., max_linewidth = 1.):
    '''
    Two inputs and two outputs, with four delays and third mirror
    This corresponds to figures 7 and 8 in our paper.

    Args:
        max_freq (float): maximum frequency to look at in the complex plane
        max_linewidth (float): maximum linewidth to look at in the complex plane

    Returns:
        [T,T_testing,roots,vecs]
        T (function): The original function
        T_testing (function): The approximated function
        roots (list of complex numbers): A list of roots found
        vecs (list of matrices): A list of the corresponding eigenvectors
    '''
    tau1 = 0.1
    tau2 = 0.23
    tau3 = 0.1
    tau4 = 0.17
    r1 = 0.9
    r2 = 0.4
    r3 = 0.8

    t1 = np.sqrt(1-r1**2)
    t2 = np.sqrt(1-r2**2)
    t3 = np.sqrt(1-r3**2)

    N = 4

    M1 = np.matrix([[0,-r1,0,0],
                    [-r2,0,t2,0],
                   [0,0,0,-r3],
                   [t2,0,r2,0]])

    M2 = np.matrix([[t1,0],
                    [0,0],
                    [0,t3],
                    [0,0]])

    M3 = np.matrix([[0,t1,0,0],
                    [0,0,0,t3]])

    M4 = np.matrix([[r1,0],
                    [0,r3]])

    E = lambda z: np.matrix([[np.exp(-tau1*z),0,0,0],
                         [0,np.exp(-tau2*z),0,0],
                         [0,0,np.exp(-tau3*z),0],
                         [0,0,0,np.exp(-tau4*z)]])

    T_denom = lambda z: la.det(np.eye(N) - M1*E(z))
    Tp_denom = lambda z: der(T_denom,z)

    T = lambda z: M3*E(z)*la.inv(np.eye(N) - M1*E(z))*M2+M4
    roots  = Roots.get_roots_rect(T_denom,Tp_denom,0,0,max_linewidth,max_freq,N=5000)
    return example_helper(T,roots)

def example4(max_freq = 50., max_linewidth = 3.):
    '''
    Two inputs and two outputs, with free delay (i.e. not in a loop).
    This corresponds to figures 9 and 10 in our paper.

    Args:
        max_freq (float): maximum frequency to look at in the complex plane
        max_linewidth (float): maximum linewidth to look at in the complex plane

    Returns:
        [T,T_testing,roots,vecs]
        T (function): The original function
        T_testing (function): The approximated function
        roots (list of complex numbers): A list of roots found
        vecs (list of matrices): A list of the corresponding eigenvectors
    '''
    tau1 = 0.1
    tau2 = 0.039
    tau3 = 0.11
    tau4 = 0.08
    r = 0.9
    t = np.sqrt(1-r**2)
    M = 4

    M1 = np.matrix([[0,0,-r,0],
                    [r,0,0,0],
                   [0,r,0,t],
                   [t,0,0,0]])

    M2 = np.matrix([[t,0],
                    [0,t],
                    [0,0],
                    [0,-r]])

    M3 = np.matrix([[0,0,t,0],
                    [0,t,0,-r]])

    M4 = np.matrix([[r,0],
                    [0,0]])

    E = lambda z: np.matrix([[np.exp(tau1*z),0,0,0],
                         [0,np.exp(tau2*z),0,0],
                         [0,0,np.exp(tau3*z),0],
                         [0,0,0,np.exp(tau4*z)]])

    T_denom = lambda z: la.det(np.eye(M) - M1*E(z))
    Tp_denom = lambda z: der(T_denom,z)
    T = lambda z: M3*E(z)*la.inv(np.eye(M) - M1*E(z))*M2+M4
    roots = Roots.get_roots_rect(T_denom,Tp_denom,0,0,max_linewidth,max_freq,N=2000)
    return example_helper(T,roots)

def example5(max_freq = 50., max_linewidth = 3.):
    '''
    Modified example 4, with analytic term.

    Args:
        max_freq (float): maximum frequency to look at in the complex plane
        max_linewidth (float): maximum linewidth to look at in the complex plane

    Returns:
        [T,T_testing,roots,vecs]
        T (function): The original function
        T_testing (function): The approximated function
        roots (list of complex numbers): A list of roots found
        vecs (list of matrices): A list of the corresponding eigenvectors
    '''
    tau1 = 0.1
    tau2 = 0.039
    tau3 = 0.11
    tau4 = 0.08
    r = 0.9
    t = np.sqrt(1-r**2)
    M = 4

    M1 = np.matrix([[0,0,-r,0],
                    [r,0,0,0],
                   [0,r,0,t],
                   [t,0,0,0]])

    M2 = np.matrix([[t,0],
                    [0,t],
                    [0,0],
                    [0,-r]])

    M3 = np.matrix([[0,0,t,0],
                    [0,t,0,-r]])

    M4 = np.matrix([[r,0],
                    [0,0]])

    E = lambda z: np.matrix([[np.exp((tau1+tau4)*z),0,0,0],
                         [0,np.exp((tau2-tau4)*z),0,0],
                         [0,0,np.exp(tau3*z),0],
                         [0,0,0,1.]])

    T_denom = lambda z: la.det(np.eye(M) - M1*E(z))
    Tp_denom = lambda z: der(T_denom,z)
    T = lambda z: M3*E(z)*la.inv(np.eye(M) - M1*E(z))*M2+M4
    roots = Roots.get_roots_rect(T_denom,Tp_denom,0,0,max_linewidth,max_freq,N=1000)
    return example_helper(T,roots)


#Like example 3, but with Pade

## Figure 14 in the paper

## don't need to feed in anything, just run 'as is'
def example6_pade():
    '''
    This example is the same as example 3, but we return a Pade approximation
    instead of a Potapov approximation. Instead of returnings roots, etc., we
    return a different kind of function (see below)

    Returns:
        A matrix-valued function T(z,n). n is the order of the approximation
        and z is the location of the function to be evaluated.
    '''
    tau1 = 0.1
    tau2 = 0.23
    tau3 = 0.1
    tau4 = 0.17
    r1 = 0.9
    r2 = 0.4
    r3 = 0.8

    t1 = np.sqrt(1-r1**2)
    t2 = np.sqrt(1-r2**2)
    t3 = np.sqrt(1-r3**2)

    N = 4

    M1 = np.matrix([[0,-r1,0,0],
                    [-r2,0,t2,0],
                   [0,0,0,-r3],
                   [t2,0,r2,0]])

    M2 = np.matrix([[t1,0],
                    [0,0],
                    [0,t3],
                    [0,0]])

    M3 = np.matrix([[0,t1,0,0],
                    [0,0,0,t3]])

    M4 = np.matrix([[r1,0],
                    [0,r3]])

    def E(z,n):
        taus = [tau1,tau2,tau3,tau4]
        tau_tot = sum(taus)
        ns = [np.int(np.round(n*t)) for t in taus]
        while (sum(ns) < n):
            j = np.argmax([abs(t/tau_tot - float(i)/n) for t,i in zip(taus,ns)])
            ns[j] +=1
        while (sum(ns) > n):
            j = np.argmax([abs(t/tau_tot - float(i)/n) for t,i in zip(taus,ns)])
            ns[j] -=1

        return np.matrix([[Pade(ns[0],-z*tau1),0,0,0],
                         [0,Pade(ns[1],-z*tau2),0,0],
                         [0,0,Pade(ns[2],-tau3*z),0],
                         [0,0,0,Pade(ns[3],-tau4*z)]])

    T = lambda z,n: M3*E(z,n)*la.inv(np.eye(N) - M1*E(z,n))*M2+M4
    return T

def plot3D(f,points = 2000):
    '''
    Make a color and hue plot in the complex plane for a given function

    Givens:
        f (function): to plot
        points(optional[int]): number of points to use per dimension

    '''
    fig = mp.cplot(f, [-10,10], [-10, 10], points = points)
    fig.savefig("complex_plane_plot.pdf")
    return

if __name__ == "__main__":

    ################
    ## Plot for complex-valued functions with and without time delay
    ## This part uses mpmath
    ################

    #tau = 1.
    #r = 0.8

    #T,T_1,roots1,vecs1 = example1(max_freq = 500)

    #a = 1
    #f = lambda z: (mp.exp(-z*tau) - r)/(1.-r* mp.exp(-z*tau))*mp.exp(-z*tau)
    #f2 = lambda z: (mp.exp(-z*tau) - r)/(1.-r* mp.exp(-z*tau))
    #plot3D(f,points = 5000)
    #plot3D(f2,points = 500000)

    ################
    ## Input/output plot for example 1
    ################

    T,T_1,roots1,vecs1 = example1(max_freq = 500)

    print T(0.)
    print T_1(0.)

    print T(-30+3j)
    print T_1(-30+3j)

    L = 100.
    dx = 0.3

    T,T_1,roots1,vecs2 = example1(max_freq = 30)
    T,T_2,roots2,vecs2 = example1(max_freq = 50)
    T,T_3,roots3,vecs3 = example1(max_freq = 80)
    T,T_4,roots4,vecs4 = example1(max_freq = 100)

    r_lst = [roots1,roots2,roots3,roots4]
    labels = ['Original T'] + ['Potapov T of Order '+str(len(r)) for r in r_lst]
    colors = ['b','r--','y--','m--','c--']

    plot_all(L,dx,labels,colors,0.5,'example_tmp.pdf',T,T_1,T_2,T_3,T_4)

    ################
    ## Input/output plot for example 3
    ################

    # L = 100.
    # dx = 0.3
    #
    # T,T_1,roots1,vecs1 = example3(max_freq = 30)
    # T,T_2,roots2,vecs2 = example3(max_freq = 50)
    # T,T_3,roots3,vecs3 = example3(max_freq = 80)
    # T,T_4,roots4,vecs4 = example3(max_freq = 100, )
    #
    # r_lst = [roots1,roots2,roots3,roots4]
    # labels = ['Original T'] + ['Potapov T of Order '+str(len(r)) for r in r_lst]
    # colors = ['b','r--','y--','m--','c--']
    #
    # plot_all(L,dx,labels,colors,0.5,'figure_8_v3.pdf',T,T_1,T_2,T_3,T_4)

    ###

    ###

    ############

    #example4(max_freq = 10,plot=True)

    ################
    ## Input/output plot for example 4
    ################

    # L = 100.
    # dx = 0.05

    # T,T_1,roots1,vecs1 = example4(max_freq = 30)  ## left out below
    # T,T_2,roots2,vecs2 = example4(max_freq = 50)
    # T,T_3,roots3,vecs3 = example4(max_freq = 80)
    # T,T_4,roots4,vecs4 = example4(max_freq = 100)
    # T,T_5,roots5,vecs5 = example4(max_freq = 125)
    #
    # T_correct,T_1_correct,roots1_correct,vecs1_correct = example5(max_freq = 30)
    #
    # r_lst = [roots2,roots3,roots4,roots5]
    # labels = ['Original T','T with feedforward removed'] + \
    #           ['Potapov T of Order ' +str(len(r)) for r in r_lst]
    # colors = ['b','black','r--','y--','m--','c--']
    #
    # plot_all(L,dx,labels,colors, 0.5,'figure_10_v3.pdf',
    #                 T,T_correct,T_1,T_2,T_3,T_4)

    #################
    ### Input/output plot for example 5
    #################

    # L = 100.
    # dx = 0.05
    #
    # T,T_1,roots1,vecs1 = example5(max_freq = 30)
    # T,T_2,roots2,vecs2 = example5(max_freq = 50)
    # T,T_3,roots3,vecs3 = example5(max_freq = 65)
    # T,T_4,roots4,vecs4 = example5(max_freq = 80)
    #
    # r_lst = [roots1,roots2,roots3,roots4]
    # labels = ['Original T'] + ['Potapov T of Order ' + str(len(r)) for r in r_lst]
    # colors = ['b','r--','y--','m--','c--']
    #
    # plot_all(L,dx,labels,colors,0.5,'example_tmp2.pdf',T,T_1,T_2,T_3,T_4)

    #################
    ### Input/output plot for example 6
    #################

    # T_orig,T_1,roots1,vecs1 = example3(max_freq = 4)
    # T = example6_pade()
    #
    # L = 100.
    # dx = 0.05
    # ns = [9,15,19]
    #
    # args = [T_orig]+[lambda z: T(z,9),lambda z: T(z,15),lambda z: T(z,19)]
    # lw = 0.5
    # name = "figure_14_v3.pdf"
    #
    # labels = ['Original T'] + ['Pade T of order ' + str(n) for n in ns]
    # colors = ['b','k--','r--','y--']
    #
    # plot_all(L,dx,labels,colors,0.5,'figure_14_v3.pdf',*args)

    ###########
    ## make scatter plot for the roots and poles of example 3
    ###########

    #import matplotlib.patches as patches
    #
    #
    #T,T_4,roots4,vecs4 = example3(max_freq = 400)
    #fig = plt.figure(figsize=(3,10))
    #
    #ax2 = fig.add_subplot(111)
    #ax2.add_patch(
    #    patches.Rectangle(
    #        (0.3,-0),
    #        0.5,
    #        100,
    #        fill=False      # remove background
    #    )
    #)
    #
    #ax2 = fig.add_subplot(111)
    #ax2.add_patch(
    #    patches.Rectangle(
    #        (0.3,-0),
    #        0.5,
    #        200,
    #        fill=False      # remove background
    #    )
    #)
    #fig.suptitle('Zero-polt scatter plot', fontsize=20)
    #
    #
    #plt.xlim(-1.,1.)
    #plt.ylim(-400,400)
    #
    #plt.xlabel('linewidth', fontsize=18)
    #plt.ylabel('frequency', fontsize=16)
    #plt.scatter(map(lambda z: z.real, roots4),map(lambda z: z.imag, roots4))
    #poles = map(lambda z: -z, roots4)
    #
    #plt.scatter(map(lambda z: z.real, poles),map(lambda z: z.imag, poles),c="red")
    #
    #plt.show()


    ##########
    ## make scatter plot for the roots and poles of example 4
    ##########
    #
    #print "making scatter plot for example 4"
    #
    #import matplotlib.patches as patches
    #
    #
    #T,T_4,roots4,vecs4 = example4(max_freq = 400)
    #scl = 1
    #fig = plt.figure(figsize=(6*scl,10*scl))
    #
    #
    #ax2 = fig.add_subplot(111)
    #ax2.add_patch(
    #    patches.Rectangle(
    #        (-2.9,-50),
    #        5.8,
    #        100,
    #        linestyle = 'dashed',
    #        color = 'red',
    #        fill=False      # remove background
    #    )
    #)
    #
    #ax2.add_patch(
    #    patches.Rectangle(
    #        (-2.95,-100),
    #        5.9,
    #        200,
    #        linestyle = 'dashed',
    #        color = 'blue',
    #        fill=False      # remove background
    #    )
    #)
    #
    #ax2.add_patch(
    #    patches.Rectangle(
    #        (-3,-150),
    #        6,
    #        300,
    #        linestyle = 'dashed',
    #        color = 'green',
    #        fill=False      # remove background
    #    )
    #)
    #
    #fig.suptitle('Pole-zero plot in the s-plane', fontsize=20)
    #
    #
    #plt.xlim(-4.,4.)
    #plt.ylim(-200,200)
    #
    #plt.axhline(0, color='black')
    #plt.axvline(0, color='black')
    #
    #plt.xlabel('Re(s)', fontsize=18)
    #plt.ylabel('Im(s)', fontsize=16)
    #xs = plt.scatter(map(lambda z: z.real, roots4),map(lambda z: z.imag, roots4),
    #                 s=80, facecolors='none', edgecolors='black',label='zero')
    #poles = map(lambda z: -z, roots4)
    #
    #os = plt.scatter(map(lambda z: z.real, poles),map(lambda z: z.imag, poles),
    #                 s=80,c="black",marker='x',label='pole')
    #
    #plt.rcParams['legend.scatterpoints'] = 1
    #
    #plt.legend(  handles=[xs,os])
    #
    #plt.savefig('eg4_scatter.pdf')
    #
    #plt.show()
