import Potapov as P
import Roots as R
import Examples
import functions
import numpy as np
import numpy.testing as testing

def test_Potapov_1(eps=1e-7):
    '''
    Generate a finite_transfer_function from eigenvectors and eigenvalues.
    Then generate a Potapov product from the finite transfer function. These
    should be analytically equal. We test to see if they are close within some
    precision.
    '''
    vals = [1-1j,-1+1j, 2+2j]
    vecs = [ P.normalize(np.matrix([-5.,4j])).T, P.normalize(np.matrix([1j,3.]).T),
            P.normalize(np.matrix([2j,7.]).T)]
    T = P.finite_transfer_function(np.eye(2),vecs,vals)
    T_test = P.get_Potapov(T,vals)

    points = [0.,10j,10.,-10j,10.+10j]

    assert all(np.amax(abs(T(z) - T_test(z))) < eps for z in points)

def two_sets_almost_equal(S1,S2,eps=1e-7):
    '''
    Tests if two iterables have the same elements up to some tolerance eps.

    Args:
        S1,S2 (lists): two lists
        eps (optional[float]): precision for testing each elements

    Returns:
        True if the two sets are equal up to eps, false otherwise
    '''
    if len(S1) != len(S2):
        return False

    def almost_equal(el1,el2):
        if abs(el1 - el2) < eps:
            return True
        else: return False

    ran2 = range(len(S2))
    for i in range(len(S1)):
        found_match = False
        for j in ran2:
            if almost_equal(S1[i],S2[j]):
                found_match = True
                ran2.remove(j)
                break
        if not found_match:
            return False
    return True


def test_Roots_1():
    '''
    Make a square of length just under 5*pi. Find the roots of sine.
    '''
    N=5000
    f = lambda z: np.sin(z)
    fp = lambda z: np.cos(z)
    x_cent = 0.
    y_cent = 0.
    width = 5.*np.pi-1e-5
    height = 5.*np.pi-1e-5

    roots = np.asarray(R.get_roots_rect(f,fp,x_cent,y_cent,width,height,N))
    roots_inside_boundary = R.inside_boundary(roots,x_cent,y_cent,width,height)
    two_sets_almost_equal(np.asarray(roots_inside_boundary)/np.pi,
        [-4.,-3.,-2.,-1.,-0.,1.,2.,3.,4.] )

def test_Roots_2():
    '''
    Make a square of length just over 5*pi. Find the roots of sine.
    '''
    N=5000
    f = lambda z: np.sin(z)
    fp = lambda z: np.cos(z)
    x_cent = 0.
    y_cent = 0.
    width = 5.*np.pi+1e-5
    height = 5.*np.pi+1e-5

    roots = np.asarray(R.get_roots_rect(f,fp,x_cent,y_cent,width,height,N))
    roots_inside_boundary = R.inside_boundary(roots,x_cent,y_cent,width,height)
    two_sets_almost_equal(np.asarray(roots_inside_boundary)/np.pi,
        [-5.,-4.,-3.,-2.,-1.,-0.,1.,2.,3.,4.,5.] )

def test_example_1():
    Ex = Examples.Example1()
    Ex.run_Potapov()
    E = Ex.E
    roots = Ex.roots
    M1 = Ex.M1
    delays = Ex.delays
    modes = functions.spatial_modes(roots,M1,E)
    assert( len(roots) == 3)

def test_example_2():
    Ex = Examples.Example2()
    Ex.run_Potapov()
    E = Ex.E
    roots = Ex.roots
    M1 = Ex.M1
    delays = Ex.delays
    modes = functions.spatial_modes(roots,M1,E)
    assert( len(roots) == 7)

def test_example_3():
    Ex = Examples.Example3()
    Ex.run_Potapov()
    E = Ex.E
    roots = Ex.roots
    M1 = Ex.M1
    delays = Ex.delays
    modes = functions.spatial_modes(roots,M1,E)
    assert( len(roots) == 11)

def test_example_4():
    Ex = Examples.Example4()
    Ex.run_Potapov()
    E = Ex.E
    roots = Ex.roots
    M1 = Ex.M1
    delays = Ex.delays
    modes = functions.spatial_modes(roots,M1,E)
    assert( len(roots) == 8)
