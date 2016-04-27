# Overview
The purpose of this package is to characterize network of optical components
when time delays with feedback are present. The core of the package identifies
'trapped' optical modes that result from feedback in the system. These modes are
found by identifying the roots/poles of the transfer function containing only
time delays and passive linear components. Further analysis of the network
yields a linear Hamiltonian written in terms of the identified modes, linear
operators coupling the system modes to the environment, and an overall
scattering matrix. Nonlinear elements are added as additional Hamiltonian terms.

Our manuscript describing this method can be found on
http://arxiv.org/abs/1510.08942 or
http://www.epjquantumtechnology.com/content/3/1/3.

# Installation
Simply clone the repository, open a terminal window, type:

```
cd /path/to/my/repo
python setup.py install
```

# Files

## Time_Delay_Network.py
This module includes a class to contain the information of a passive linear
network with time delays.
Several examples are included in this module.
Each example includes matrices that yield a transfer function.
This module contains methods to re-construct finite-dimensional approximations
of a transfer function of passive systems.
--`run_Potapov()`: This function will run the Potapov procedure. The instance
of `Time_Delay_Network` will update its `roots`, `vecs`, and `T_testing`.
These are respectively the poles of the transfer function, a list of vectors
(complex-valued column matrices) that contain the information to reconstruct
the matrix-valued projectors in the Potapov representation, and an approximating
transfer function that has been generated using the Potapov interpolation
procedure.

## Potapov.py
We implement the procedure for finding Blaschke-Potapov
products to approximate given functions near poles.
Please see section 6.2 in
our manuscript for details.
This procedure is used to generate the modes of the passive linear network.

Given a rational matrix-valued function, we also construct the matrices ABCD
that give the state-space representation of the system (see `get_Potapov_ABCD()`).

## Roots.py
A module for identifying the zeros of a complex-valued function.

## functions.py
Miscellaneous functions. Includes:

--`Pade()` Generate a Pade approximation for delays that do NOT feed back.

--`spatial_modes()` Finding the spatial location of modes. This is necessary to
generate nonlinear terms.
The required inputs to `spatial_modes()` are `roots`,`M1`,and `E`. These are
respectively the poles of the transfer function, the directed connectivity
matrix of the internal nodes of a network, and a diagonal matrix-valued function
whose diagonal values correspond to the delays in the Fourier domain.

--`make_nonlinear_interaction()` Generate the weight of an interaction term due
to phase-matching.

## Hamiltonian.py
Includes a class `Hamiltonian()` to contain the information needed to construct the
Hamiltonian of the system, including nonlinear terms.
This class also includes a function `make_eq_motion()` to generate the classical
equations of motion from the nonlinear Hamiltonian.

Also includes a class `Chi_nonlin()` which contains the information for a particular
chi nonlinearity.

## Time_Sims.py

Integrate the dynamics of a passive in time using the ABCD matrices.

## Time_Sims_nonlin.py

--`make_f_lin()` generates outputs from ABCD model.

--`make_f()` generates outputs from nonlinear Hamiltonian model.

--`run_ODE()` integrates the equations of motion in time.

--`double_up()` prepares a doubled-up system which can be used for non-classical
simulations.

# Sample Usage

```
import Roots
import Potapov
import Time_Delay_Network
import Time_Sims_nonlin
import functions
import Hamiltonian

## Make a sample Time_Delay_Network, changing some parameters.
X = Time_Delay_Network.Example3(r1 = 0.7, r3 = 0.7, max_linewidth=35.)

## run the Potapov procedure.
X.run_Potapov()

## Extract the roots, modes, and delays
roots = X.roots
delays = X.delays
M1 = X.M1
E = X.E
modes = functions.spatial_modes(roots,M1,E)

```
