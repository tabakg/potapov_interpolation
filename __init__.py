import Roots
import Potapov
import Examples
import Time_Sims
import functions as f
import tests

import numpy as np
import numpy.linalg as la

Ex = Examples.Example4()
Ex.run_Potapov()
roots = Ex.roots
M1 = Ex.M1

print Ex.max_freq
print Ex.max_linewidth

print roots
