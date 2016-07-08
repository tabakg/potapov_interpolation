Overview
--------

The purpose of this package is to characterize network of optical
components when time delays with feedback are present. The core of the
package identifies 'trapped' optical modes that result from feedback in
the system. These modes are found by identifying the roots/poles of the
transfer function containing only time delays and passive linear
components. Further analysis of the network yields a linear Hamiltonian
written in terms of the identified modes, linear operators coupling the
system modes to the environment, and an overall scattering matrix.
Nonlinear elements are added as additional Hamiltonian terms.

Our manuscript describing this method can be found on
http://arxiv.org/abs/1510.08942 or
http://www.epjquantumtechnology.com/content/3/1/3.

Installation
------------

Simply clone the repository, open a terminal window, type:

::

    git clone https://github.com/tabakg/potapov_interpolation
    cd potapov_interpolation
    python setup.py install

Files
-----

Time\_Delay\_Network.py
~~~~~~~~~~~~~~~~~~~~~~~

This module includes a class to contain the information of a passive
linear network with time delays. Several examples are included in this
module. Each example includes matrices that yield a transfer function.
This module contains methods to re-construct finite-dimensional
approximations of a transfer function of passive systems.

--``make_commensurate_roots()``: A method to determine the roots utilizing
the commensurate structure of the roots. The algorithm determines a polynomial
based on the gcd of the time delays to describe where poles of the transfer
function occur. The roots of the polynomial are found. The periodicity of the
exponential :math:`e^{-zT}` for the gcd delay is then used to find the roots
within desired frequency ranges.

--``make_roots()``: A method that find the poles of the transfer function within
a contour. This is a very general method in that the function needs only to be
meromorphic with poles of order 1. In particular this yields the poles even
when the delays are not commensurate.

--``run_Potapov()``: This function will run the Potapov procedure. The
instance of ``Time_Delay_Network`` will update its ``roots``, ``vecs``,
and ``T_testing``. These are respectively the poles of the transfer
function, a list of vectors (complex-valued column matrices) that
contain the information to reconstruct the matrix-valued projectors in
the Potapov representation, and an approximating transfer function that
has been generated using the Potapov interpolation procedure.
This method uses either ``make_roots()`` or ``make_commensurate_roots()``, which
the user can specify.

Potapov.py
~~~~~~~~~~

We implement the procedure for finding Blaschke-Potapov products to
approximate given functions near poles. Please see section 6.2 in our
manuscript for details. This procedure is used to generate the modes of
the passive linear network.

Given a rational matrix-valued function, we also construct the matrices
ABCD that give the state-space representation of the system (see
``get_Potapov_ABCD()``).

Roots.py
~~~~~~~~

A module for identifying the zeros of a complex-valued function.

functions.py
~~~~~~~~~~~~

Miscellaneous functions. Includes:

--``Pade()`` Generate a Pade approximation for delays that do NOT feed
back.

--``spatial_modes()`` Finding the spatial location of modes. This is
necessary to generate nonlinear terms. The required inputs to
``spatial_modes()`` are ``roots``,\ ``M1``,and ``E``. These are
respectively the poles of the transfer function, the directed
connectivity matrix of the internal nodes of a network, and a diagonal
matrix-valued function whose diagonal values correspond to the delays in
the Fourier domain.

--``make_nonlinear_interaction()`` Generate the weight of an interaction
term due to phase-matching.

Hamiltonian.py
~~~~~~~~~~~~~~

Includes a class ``Hamiltonian()`` to contain the information needed to
construct the Hamiltonian of the system, including nonlinear terms. This
class also includes a function ``make_eq_motion()`` to generate the
classical equations of motion from the nonlinear Hamiltonian.

Also includes a class ``Chi_nonlin()`` which contains the information
for a particular chi nonlinearity.

Time\_Sims.py
~~~~~~~~~~~~~

Integrate the dynamics of a passive in time using the ABCD matrices.

Time\_Sims\_nonlin.py
~~~~~~~~~~~~~~~~~~~~~

--``make_f_lin()`` generates outputs from ABCD model.

--``make_f()`` generates outputs from nonlinear Hamiltonian model.

--``run_ODE()`` integrates the equations of motion in time.

--``double_up()`` prepares a doubled-up system which can be used for
non-classical simulations.

Sample Usage -- Time Domain Simulation
--------------------------------------

See [Simple code example](Sample_Code_Usage.ipynb)
