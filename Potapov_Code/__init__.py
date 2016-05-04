import Roots
import Potapov
import Time_Delay_Network
import Time_Sims
import functions
import tests

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def contour_plot(Mat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(abs(Mat), interpolation='nearest')
    fig.colorbar(cax)

if __name__ == "__main__" and False:

    Ex = Time_Delay_Network.Example3(r1 = 0.999, r3 = 0.999)
    Ex.run_Potapov()

    E = Ex.E
    roots = Ex.roots
    M1 = Ex.M1
    delays = Ex.delays
    modes = functions.spatial_modes(roots,M1,E)

    Mat = functions.make_normalized_inner_product_matrix(roots,modes,delays)

    contour_plot(Mat)
    

    print Ex.max_freq
    print Ex.max_linewidth

    print roots
