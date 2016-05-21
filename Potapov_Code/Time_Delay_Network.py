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
import sympy as sp
import matplotlib.pyplot as plt
#import mpmath as mp ## for complex-valued plots
from functions import double_up
from functions import der
from functions import Pade
from functions import spatial_modes
from functions import gcd_lst
import matplotlib.patches as patches

from decimal import Decimal

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
    return

class Time_Delay_Network():
    '''
    A class to contain the information of a passive linear network with time
    delays.

    Attributes:
        max_freq (optional [float]): maximum height in the complex plane

        max_linewidth (optional [float]): maximum width in the complex plane.

        N (optional [int]): number of points to use on the contour for finding
        the roots/poles of the network.

        center_freq (optional [float] ): how much to move the frame up or down
        the complex plane.

    '''
    def __init__(self,max_freq=30.,max_linewidth=1.,N=1000,center_freq = 0.):
        self.max_freq = max_freq
        self.max_linewidth = max_linewidth
        self.N = N
        self.Potapov_ran = False
        self.center_freq = center_freq
        return

    def _make_decimal_delays(self,):
        self.Decimal_delays = map(lambda x: Decimal(str(x)),self.delays)
        self.Decimal_gcd = self._find_commensurate(self.Decimal_delays)

    def make_roots(self):
        '''Generate the roots given the denominator of the transfer function.

        '''
        self.roots = Roots.get_roots_rect(self.T_denom,self.Tp_denom,
            -self.max_linewidth/2.,self.center_freq,
            self.max_linewidth/2.,self.max_freq,N=self.N)
        return

    def _find_commensurate(self,delays):
        '''
        Find the 'gcd' but for Decimal numbers.

        Args:
            delays(list of Demicals): numbers whose gcd will be found.

        Returns:
            Decimal gcd.
        '''
        mult = min([d.as_tuple().exponent for d in delays])
        power = 10**-mult
        delays = map(lambda x: x*power,delays)
        int_gcd = gcd_lst(delays)
        return int_gcd/power

    def _make_T_denom_sym(self,):
        r'''
        A method to prepare the symbolic expression T_denom_sym for further
        computations. This expression represents the denominator in terms of
        a symbol x, which represents the shortest time delay in the network.
        '''
        self._make_decimal_delays()
        self.x = sp.symbols('x')
        E_sym = sp.Matrix(np.zeros_like(self.M1))
        for i,delay in enumerate(self.Decimal_delays):
            E_sym[i,i] = self.x**int(delay / self.Decimal_gcd)
        M1_sym = sp.Matrix(self.M1)
        self.T_denom_sym = sp.apart((E_sym - M1_sym).det())
        ## I use apart above because sympy yields a function that is not
        ## completely reduced. Alternatively, can use *.as_numer_denom()
        ## and take the first component for the numerator. However, this could
        ## results in spurious roots if the denominator is nontrivial.
        return

    def get_symbolic_frequency_perturbation(self,simplify = False):
        r'''
        A method to prepare the symbolic expression T_denom_sym for further
        computations. This expression represents the denominator in terms of
        the various delays :math:`T_1,...,T_k` and the complex variable
        :math:`z`.

        This method treats the various delays as separate variables.

        Args:
            simplify (optiona[boolean]): simplify the output sympy expression.
        '''
        M = len(self.delays)
        self._make_decimal_delays()
        try:
            self.z,self.z_Delta,self.Ts,self.Ts_Delta
        except:
            self.z, self.z_Delta = sp.symbols('z dz')
            self.Ts = [sp.symbols('T_'+str(i)) for i in range(M)]
            self.Ts_Delta = [sp.symbols('dT_'+str(i)) for i in range(M)]

        xs = [sp.symbols('x_'+str(i)) for i in range(4) ]
        E_sym = sp.Matrix(np.zeros_like(self.M1))
        for i,delay in enumerate(self.Decimal_delays):
            E_sym[i,i] = xs[i]
        M1_sym = sp.Matrix(self.M1)
        num, den = (E_sym - M1_sym).det().as_numer_denom()
        D = {x: sp.exp(-self.z*T) for x,T in zip(xs,self.Ts)}
        exp_periodic = num.subs(D)
        T_expression = sum([exp_periodic.diff(T)*T_d
            for T,T_d in zip(self.Ts,self.Ts_Delta)])

        ## Next solve for the first-order perturbation.
        ## The commented-out line might be slow -- the code below does the same.
        #sol = sp.solve(T_expression + exp_periodic.diff(z)*z_Delta, z_Delta)[0]

        diff_z = exp_periodic.diff(self.z)
        T_temps = [sp.symbols('T_temp_'+str(i)) for i in range(M)]
        D_tmp = {self.z*T:T_t for T,T_t in zip(self.Ts,T_temps)}
        D_inv = {T_t:self.z*T for T,T_t in zip(self.Ts,T_temps)}
        D = {T:dT for T,dT in zip(self.Ts,self.Ts_Delta)}
        diff_z2 = diff_z.subs(D_tmp)
        diff_z3 = diff_z2.subs(D)
        diff_z4 = diff_z3.subs(D_inv)
        sol = -self.z*diff_z4 / diff_z

        return sp.simplify(sol) if simplify else sol

    def get_frequency_pertub_func(self,simplify = False):
        sym_freq_pert = self.get_symbolic_frequency_perturbation(simplify = simplify)
        return sp.lambdify( (self.z,self.Ts,self.Ts_Delta), sym_freq_pert)

    def _find_instances_in_range_good_initial_point(self,z,freq_range,T):
        '''
        Find numbers of the form :math:`z + Tni` where :math:`T` is the
        period and :math:`n` is an integer inside the given frequency range.
        Assumes the given z is in the desired frequency range.

        Args:
            z (complex number)
            freq_range (2-tuple): (minimum frequency, maximum frequency)

        Returns:
            list of numbers of the desired form.
        '''
        lst_in_range = [z]
        num_below = int((z.imag - freq_range[0])/T )
        num_above = int((freq_range[1] - z.imag)/T )
        above_range = (np.asarray(range(num_above))+1) * T
        below_range = (np.asarray(range(num_below))+1) * T
        lst_in_range += [z + 1j * disp for disp in above_range]
        lst_in_range += [z - 1j * disp for disp in below_range]
        return lst_in_range

    def _find_instances_in_range(self,z,freq_range,T):
        '''
        Find numbers of the form :math:`z + Tni` where :math:`T` is the
        period and :math:`n` is an integer inside the given frequency range.

        Args:
            z (complex number)
            freq_range (2-tuple): (minimum frequency, maximum frequency)

        Returns:
            list of numbers of the desired form. Empty list if none exist.
        '''
        if z.imag >= freq_range[0] and z.imag <= freq_range[1]:
            return self._find_instances_in_range_good_initial_point(z,freq_range,T)
        elif z.imag > freq_range[1]:
            min_dist = (int((z.imag - freq_range[1])/T)+1) * T
            max_dist = int((z.imag - freq_range[0]) / T) * T
            if min_dist > max_dist:
                return []
            else:
                return self._find_instances_in_range_good_initial_point(
                    z - 1j*min_dist,freq_range,T)
        else:
            min_dist = (int((freq_range[0] - z.imag)/T)+1) * T
            max_dist = int((freq_range[1] - z.imag)/T)  * T
            if min_dist > max_dist:
                return []
            else:
                return self._find_instances_in_range_good_initial_point(
                    z + 1j*min_dist,freq_range,T)

    def make_commensurate_roots(self,list_of_ranges = []):
        '''
        Assuming the delays are commensurate, obtain all the roots within the
        frequency ranges of interest. Sets self.roots a list of complex roots
        in the desired frequency ranges.

        Args:
            list_of_ranges (optional [list of 2-tuples]): list of frequency
            ranges of interest in the form:
            (minimum frequency, maximum frequency).
        '''

        self._make_T_denom_sym()

        poly = sp.Poly(self.T_denom_sym, self.x)
        poly_coeffs = poly.all_coeffs()
        roots = np.roots(poly_coeffs)
        zs = np.asarray(map(lambda r: np.log(r) / float(self.Decimal_gcd),
                        roots))

        T_gcd = 2.*np.pi / float(self.Decimal_gcd)

        self.map_root_to_commensurate_index = {}
        lst_to_return = []
        for freq_range in list_of_ranges:
            for i,r in enumerate(zs):
                prev_len = len(lst_to_return)
                new_roots = self._find_instances_in_range(r,freq_range,T_gcd)
                len_new_roots = len(new_roots)
                lst_to_return += new_roots
                for j in range(prev_len,prev_len + len_new_roots):
                    self.map_root_to_commensurate_index[j] = i
        self.roots = lst_to_return
        self.commensurate_roots = zs

    def make_commensurate_vecs(self,):
        self.commensurate_vecs = Potapov.get_Potapov_vecs(
            self.T,self.commensurate_roots)
        self.vecs = map(
            lambda i: self.commensurate_vecs[self.map_root_to_commensurate_index[i]],
            range(len(self.roots)) )
        return

    def make_T_Testing(self):
        '''Generate the approximating transfer function using the identified
        poles of the transfer function.

        '''
        self.T_testing = Potapov.get_Potapov(self.T,self.roots,self.vecs)
        return

    def make_vecs(self):
        '''Generate an ordered list of vectors representing the form of the
        Potapov factors.

        '''
        self.vecs = Potapov.get_Potapov_vecs(self.T,self.roots)
        return

    def make_spatial_modes(self,):
        '''Generate the spatial modes of the network.

        '''
        self.spatial_modes = spatial_modes(self.roots,self.M1,self.E,delays=self.delays)
        return

    def run_Potapov(self, commensurate_roots = False, filtering_roots = True):
        '''Run the entire Potapov procedure to find all important information.
        The generated roots, vecs, approximated transfer function T_Testing,
        and the spatial_modes are all stored in the class.

        Args:
            commensurate_roots (optional[boolean]): which root-finding method
            to use.

            filtering_roots (optional[boolean]): makes sure the poles of the
            transfer function all have negative real part. Drops ones that
            might not.
        '''
        self.Potapov_ran = True
        if commensurate_roots:
            self.make_commensurate_roots([(-self.max_freq,self.max_freq)])
            if filtering_roots:
                self.roots =  [r for r in self.roots if r.real <= 0]
            self.make_commensurate_vecs()
        else:
            self.make_roots()
            if filtering_roots:
                self.roots =  [r for r in self.roots if r.real <= 0]
            self.make_vecs()
        self.make_T_Testing()
        self.make_spatial_modes()
        return

    def get_outputs(self):
        '''Get some of the relevant outputs from the Potapov procedure.

        Returns:
            The original transfer function, the approximating generated
            transfer function, the identified poles of the transfer function,
            and the vectors representing the form of the Potapov factors.

        '''
        if self.Potapov_ran:
            return self.T,self.T_testing,self.roots,self.vecs
        else:
            raise Exception("Must run Potapov to get outputs!!!")
        return

    def get_Potapov_ABCD(self,z=0.,doubled=False):
        '''
        Find the ABCD matrices from the Time_Delay_Network.

        Args:
            z (optional [complex number]): location where to estimate D.

        Return:
            A,B,C,D matrices.

        '''
        A,B,C,D = Potapov.get_Potapov_ABCD(self.roots,self.vecs,self.T,z=z)
        if not doubled:
            return A,B,C,D
        else:
            A_d,C_d,D_d = map(double_up,(A,C,D))
            B_d = -double_up(C.H)
            return A_d,B_d,C_d,D_d

class Example1(Time_Delay_Network):
    '''
    Single input, single output with a single delay.
    '''
    def __init__(self, max_freq=30.,max_linewidth=1.,N=1000, center_freq = 0.,
            tau = 0.3,r = 0.8):
        Time_Delay_Network.__init__(self, max_freq,max_linewidth,N,center_freq)

        self.tau = tau
        self.delays = [tau]
        self.r = r
        self.M1=np.matrix([[r]])
        self.E = lambda z: np.exp(-z*self.tau)
        self.T = lambda z: np.matrix([(np.exp(-z*self.tau) - self.r)/
                                        (1.-self.r* np.exp(-z*self.tau))])
        self.T_denom = lambda z: (1.-self.r* np.exp(-z*self.tau))
        self.Tp_denom = lambda z: der(self.T_denom,z)

class Example2(Time_Delay_Network):
    '''
    Two inputs, two outputs with a delay (i.e. Fabry-Perot).
    '''
    def __init__(self, max_freq=10.,max_linewidth=10.,N=1000, center_freq = 0.,
                    r=0.9,tau = 1.):
        Time_Delay_Network.__init__(self, max_freq,max_linewidth,N,center_freq)
        self.r = r
        self.delays = [tau]
        e = lambda z: np.exp(-z*tau)
        dim = 2

        self.M1 = np.matrix([[0,r],[r,0]])
        self.E = lambda z: np.matrix([[e(z),0],[0,e(z)]])

        self.T_denom = lambda z: (1.-r**2* e(z)**2)
        self.T = lambda z: -r*np.eye(dim) + ((1.-r**2.)/self.T_denom(z)) * \
            np.matrix([[r*e(z)**2,e(z)],[e(z),r*e(z)**2]])
        self.Tp_denom = lambda z: der(self.T_denom,z)

class Example3(Time_Delay_Network):
    '''
    Two inputs and two outputs, with four delays and third mirror
    This corresponds to figures 7 and 8 in our paper.
    '''
    def __init__(self, max_freq=60.,max_linewidth=1.,N=5000, center_freq = 0.,
                r1=0.9,r2=0.4,r3=0.8,
                tau1 = 0.1, tau2 = 0.23,tau3 = 0.1,tau4 = 0.17,
                ):
        Time_Delay_Network.__init__(self, max_freq,max_linewidth,N,center_freq)


        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.delays =[tau1,tau2,tau3,tau4]

        t1 = np.sqrt(1-r1**2)
        t2 = np.sqrt(1-r2**2)
        t3 = np.sqrt(1-r3**2)

        dim = 4

        M1 = np.matrix([[0,-r1,0,0],
                        [-r2,0,t2,0],
                       [0,0,0,-r3],
                       [t2,0,r2,0]])
        self.M1 = M1

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
        self.E = E

        self.T_denom = lambda z: la.det(np.eye(dim) - M1*E(z))
        self.Tp_denom = lambda z: der(self.T_denom,z)
        self.T = lambda z: M3*E(z)*la.inv(np.eye(dim) - M1*E(z))*M2+M4

class Example4(Time_Delay_Network):
    '''
    Two inputs and two outputs, with free delay (i.e. not in a loop).
    This corresponds to figures 9 and 10 in our paper.
    '''
    def __init__(self, max_freq=100.,max_linewidth=3.,N=5000,center_freq = 0.):
        Time_Delay_Network.__init__(self, max_freq,max_linewidth,N,center_freq)

        tau1 = 0.1
        tau2 = 0.039
        tau3 = 0.11
        tau4 = 0.08
        self.delays = [tau1,tau2,tau3,tau4]
        r = 0.9
        t = np.sqrt(1-r**2)
        dim = 4

        M1 = np.matrix([[0,0,-r,0],
                        [r,0,0,0],
                       [0,r,0,t],
                       [t,0,0,0]])

        self.M1 = M1

        M2 = np.matrix([[t,0],
                        [0,t],
                        [0,0],
                        [0,-r]])

        M3 = np.matrix([[0,0,t,0],
                        [0,t,0,-r]])

        M4 = np.matrix([[r,0],
                        [0,0]])

        E = lambda z: np.matrix([[np.exp(-tau1*z),0,0,0],
                             [0,np.exp(-tau2*z),0,0],
                             [0,0,np.exp(-tau3*z),0],
                             [0,0,0,np.exp(-tau4*z)]])
        self.E = E

        self.T_denom = lambda z: la.det(np.eye(dim) - M1*E(z))
        self.Tp_denom = lambda z: der(self.T_denom,z)
        self.T = lambda z: M3*E(z)*la.inv(np.eye(dim) - M1*E(z))*M2+M4

class Example5(Time_Delay_Network):
    '''
    Modified example 4, with analytic term.
    '''
    def __init__(self, max_freq=50.,max_linewidth=3.,N=1000,center_freq = 0.,):
        Time_Delay_Network.__init__(self, max_freq ,max_linewidth,N,center_freq)
        tau1 = 0.1
        tau2 = 0.039
        tau3 = 0.11
        tau4 = 0.08
        self.delays = [tau1,tau2,tau3,tau4]

        r = 0.9
        t = np.sqrt(1-r**2)
        dim = 4

        M1 = np.matrix([[0,0,-r,0],
                        [r,0,0,0],
                       [0,r,0,t],
                       [t,0,0,0]])
        self.M1=M1

        M2 = np.matrix([[t,0],
                        [0,t],
                        [0,0],
                        [0,-r]])

        M3 = np.matrix([[0,0,t,0],
                        [0,t,0,-r]])

        M4 = np.matrix([[r,0],
                        [0,0]])

        E = lambda z: np.matrix([[np.exp(-(tau1+tau4)*z),0,0,0],
                             [0,np.exp(-(tau2-tau4)*z),0,0],
                             [0,0,np.exp(-tau3*z),0],
                             [0,0,0,1.]])
        self.E=E

        self.T_denom = lambda z: la.det(np.eye(dim) - M1*E(z))
        self.Tp_denom = lambda z: der(self.T_denom,z)
        self.T = lambda z: M3*E(z)*la.inv(np.eye(dim) - M1*E(z))*M2+M4

def example6_pade():
    '''
    This example is the same as example 3, but we return a Pade approximation
    instead of a Potapov approximation. Instead of returnings roots, etc., we
    return a different kind of function (see below).

    This is used for figure 14 of our paper.

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

    dim = 4

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

    T = lambda z,n: M3*E(z,n)*la.inv(np.eye(dim) - M1*E(z,n))*M2+M4
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
    print 'Running Examples.py'

    ################
    ## Plot for complex-valued functions with and without time delay
    ## This part uses mpmath
    ################
    # tau=1.; r=0.8
    # E = Example1(max_freq = 500., max_linewidth = 1.0,tau=tau,r=r)
    # E.run_Potapov()
    #
    # T,T_1,roots1,vecs1 = E.get_outputs()
    #
    # f = lambda z: (mp.exp(-z*tau) - r)/(1.-r* mp.exp(-z*tau))*mp.exp(-z*tau)
    # f2 = lambda z: (mp.exp(-z*tau) - r)/(1.-r* mp.exp(-z*tau))
    # plot3D(f,points = 5000)
    # plot3D(f2,points = 500000)

    ################
    ## Input/output plot for example 1
    ################

    # L = 100.
    # dx = 0.3
    # freqs = [30.,50.,80.,100.]
    # T_ls = []; roots_ls = []; vecs_ls = []
    #
    # for freq in freqs:
    #     E = Example1(max_freq = freq)
    #     E.run_Potapov()
    #     T,T_,roots,vecs = E.get_outputs()
    #     T_ls.append(T_)
    #     roots_ls.append(roots)
    #     vecs_ls.append(vecs)
    #
    # labels = ['Original T'] + ['Potapov T of Order '+str(len(r))
    #                             for r in roots_ls]
    # colors = ['b','r--','y--','m--','c--']
    #
    # plot_all(L,dx,labels,colors,0.5,'example_tmp.pdf',T,*T_ls)

    ################
    ## Input/output plot for example 3
    ################

    # L = 100.
    # dx = 0.3
    # freqs = [30.,50.,80.,100.]
    # T_ls = []; roots_ls = []; vecs_ls = []
    #
    # for freq in freqs:
    #     E = Example3(max_freq = freq)
    #     E.run_Potapov()
    #     T,T_,roots,vecs = E.get_outputs()
    #     T_ls.append(T_)
    #     roots_ls.append(roots)
    #     vecs_ls.append(vecs)
    #
    # labels = ['Original T'] + ['Potapov T of Order '+str(len(r))
    #                             for r in roots_ls]
    # colors = ['b','r--','y--','m--','c--']
    #
    # plot_all(L,dx,labels,colors,0.5,'figure_8_v3.pdf',T,*T_ls)


    ###########
    ## Testing example 3 as above, but now using commensurate roots.
    ###########

    L = 1000.
    dx = 0.5
    freqs = [300.,500.,800.,1000.]
    T_ls = []; roots_ls = []; vecs_ls = []

    for freq in freqs:
        E = Example3(max_freq = freq)
        E.run_Potapov(commensurate_roots=True)
        T,T_,roots,vecs = E.get_outputs()
        T_ls.append(T_)
        roots_ls.append(roots)
        vecs_ls.append(vecs)

    labels = ['Original T'] + ['Potapov T of Order '+str(len(r))
                                for r in roots_ls]
    colors = ['b','r--','y--','m--','c--']

    plot_all(L,dx,labels,colors,0.5,'figure_8_commensurate.pdf',T,*T_ls)


    ################
    ## Input/output plot for example 4
    ################

    # L = 100.
    # dx = 0.05
    # freqs = [50.,80.,100.,125.]
    # T_ls = []; roots_ls = []; vecs_ls = []
    #
    # for freq in freqs:
    #     E = Example4(max_freq = freq)
    #     E.run_Potapov()
    #     T,T_,roots,vecs = E.get_outputs()
    #     T_ls.append(T_)
    #     roots_ls.append(roots)
    #     vecs_ls.append(vecs)
    #
    # E5 = Example5(max_freq=30.)
    # E5.run_Potapov()
    # T_correct,T_1_correct,roots1_correct,vecs1_correct = E5.get_outputs()
    # T_ls = [T_correct] + T_ls
    #
    # labels = ['Original T','T with feedforward removed'] + \
    #           ['Potapov T of Order ' +str(len(r)) for r in roots_ls]
    # colors = ['b','black','r--','y--','m--','c--']
    #
    # plot_all(L,dx,labels,colors, 0.5,'figure_10_v4.pdf',
    #                 T,*T_ls)

    #################
    ### Input/output plot for example 5
    #################
    #
    # L = 100.
    # dx = 0.05
    # freqs = [30.,50.,65.,80.]
    # T_ls = []; roots_ls = []; vecs_ls = []
    #
    # for freq in freqs:
    #     E = Example5(max_freq = freq)
    #     E.run_Potapov()
    #     T,T_,roots,vecs = E.get_outputs()
    #     T_ls.append(T_)
    #     roots_ls.append(roots)
    #     vecs_ls.append(vecs)
    #
    # labels = ['Original T'] + ['Potapov T of Order '+str(len(r))
    #                             for r in roots_ls]
    # colors = ['b','r--','y--','m--','c--']
    #
    # plot_all(L,dx,labels,colors,0.5,'example_tmp2.pdf',T,*T_ls)


    #################
    ### Input/output plot for example 6
    #################

    # E = Example3()
    # T_orig,T_1,roots1,vecs1 = E.get_outputs()
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

    # E = Example3(max_freq = 400.)
    # E.run_Potapov()
    # T,T_3,roots3,vecs3 = E.get_outputs()
    # fig = plt.figure(figsize=(3,10))
    #
    # ax2 = fig.add_subplot(111)
    # ax2.add_patch(
    #    patches.Rectangle(
    #        (0.3,-0),
    #        0.5,
    #        100,
    #        fill=False      # remove background
    #    )
    # )
    #
    # ax2 = fig.add_subplot(111)
    # ax2.add_patch(
    #    patches.Rectangle(
    #        (0.3,-0),
    #        0.5,
    #        200,
    #        fill=False      # remove background
    #    )
    # )
    # fig.suptitle('Zero-polt scatter plot', fontsize=20)
    #
    #
    # plt.xlim(-1.,1.)
    # plt.ylim(-400,400)
    #
    # plt.xlabel('linewidth', fontsize=18)
    # plt.ylabel('frequency', fontsize=16)
    # plt.scatter(map(lambda z: z.real, roots3),map(lambda z: z.imag, roots3))
    # poles = map(lambda z: -z, roots3)
    #
    # plt.scatter(map(lambda z: z.real, poles),map(lambda z: z.imag, poles),c="red")
    #
    # plt.show()

    ##########
    ## make scatter plot for the roots and poles of example 4
    ##########
    #
    #print "making scatter plot for example 4"
    #
    #import matplotlib.patches as patches
    #
    #
    # E = Example4(max_freq = 400.)
    # E.run_Potapov()
    # T,T_4,roots4,vecs4 = E.get_outputs()
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
