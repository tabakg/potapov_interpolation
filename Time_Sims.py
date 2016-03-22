# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:52:32 2015

@author: gil
@title: Time_Sims
"""

#example for using ODE library

from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
import Examples
import Potapov

def time_sim(Example, omega = 0., t1=150, dt=0.05, freq=None,
                port_in = 0, port_out = [0,1], kind='FP',
             ):
    '''
    takes an example and simulates it up to t1 increments of dt.
    freq indicates the maximum frequency where we look for modes
    omega indicates the frequency of driving. omega = 0 is DC.
    port_in and port_out are where the system is driven.
    '''
    E = Example(max_freq = freq) if freq != None else Example()
    E.run_Potapov()
    T,T_testing,poles,vecs = E.get_outputs()
    print "number of poles is ", len(poles)
    num = len(poles)
    [A,B,C,D] = Potapov.get_Potapov_ABCD(poles,vecs)

    y0 = np.matrix([[0]]*A.shape[1])
    t0 = 0

    force_func = lambda t: np.cos(omega*t)

    r = ode(f).set_integrator('zvode', method='bdf')
    r.set_initial_value(y0, t0).set_f_params(A,B,force_func,port_in)

    Y = [C*y0+D*force_func(t0)]

    while r.successful() and r.t < t1:
        r.integrate(r.t+dt)
        #print r.t, r.y
        u = force_func(r.t)
        Y.append(C*r.y+D*u)

    time = np.linspace(t0,t1,len(Y))
    plot_time(time,Y,port_out,port_in,num=num,kind=kind)
    return


def stack_func_port(force_func,forcing_port,t,max_size):
    u = np.vstack( (np.matrix([0]*forcing_port),force_func(t)) ) \
        if forcing_port > 0 else np.matrix([[force_func(t)]])
    u = np.vstack( (u,np.matrix([0]*(max_size-forcing_port-1)) )) \
        if (max_size-forcing_port-1) > 0 else u
    return u

def test_stacking():
    print "u ", stack_func_port(np.sin,1)

def f(t, y, A,B, force_func,forcing_port):
    u = stack_func_port(force_func,forcing_port,t,B.shape[1])
    return A*np.asmatrix(y).T+B*np.asmatrix(u)

def plot_time(time,y,port_out,port_in,num=0,kind='FP',format = 'pdf'):
    #plt.figure(1)
    plt.figure(figsize=(9,6))
    y_coords = [ [np.abs(y_el[i,port_in]) for y_el in y] for i in port_out]
    plt.xlabel('time',fontsize=24)
    plt.ylabel('Norm of Output',fontsize=24)
    plt.title('Time domain output with '+ str(num) \
    +(' Mode' if num == 1 else ' Modes'), fontsize=28 )
    [plt.plot(time,y_coords[i],label='Output port '+str(i)) for i in port_out]
    plt.tight_layout()
    plt.rcParams['legend.numpoints'] = 1
    plt.legend(loc = 5,fontsize=24)
    plt.tick_params(labelsize=20)
    plt.savefig(kind + str(num)+ '.' + format,format=format)
    return


if __name__ == "__main__":
    '''
    Run a single simulation of a Fabry-Perot cavity.

    freq is the maximum frequency to look for poles.
    Setting it to 1 only gets the pole at zero.
    '''
    ###########################################
    eg = Examples.Example2
    kind = 'FP'  ## Fabry-Perot
    time_sim(eg,port_out = [1],t1=50,dt=0.0005, freq = 1.)
    ###########################################

    '''
    Run several simulations of FP
    '''

    # eg = Examples.Example2
    # kind = 'FP'  ## Fabry-Perot
    #
    # for num in xrange(5,7):
    #     time_sim(eg,port_out = [0,1],t1=50,freq = 1.+np.pi*num,\
    #     kind=kind)
    # ###########################################

    '''
    Run a double - Fabry Perot (i.e. three mirrors)
    '''
    #eg = Examples.Example3
    #kind = 'DFP' ##'double'-Fabry-Perot
    #for num in xrange(105,106):
    #    time_sim(eg,port_out = 1,t1=10,dt=0.0001,freq = 1+np.pi*num,\
    #    kind=kind)
    ###########################################
