import itertools as it
import numpy as np
from matplotlib import pyplot as plt
from t2grids import *

# TODO: think about number of coefficients
# TODO: think about correlating coefficients
# TODO: think about adding mean or alpha as additional parameters

# Define important constants
N_TERMS = 10     # Number of terms in Fourier series to retain

WIDTH_H = 500   # Mean width in horizontal direction
ALPHA = 1.0     # Mean width of vertical direction relative to horizontal direction

COEF_SDS = 1

CAP_CENTRE = np.array([750, 750, -200])

A, B, C = 500, 500, 50

"""Returns the radius of the clay cap in a given direction"""
def get_radius(phi, theta, cs):

    # Compute the radius of the corresponding point on the ellipse
    re = np.sqrt(((np.sin(phi) * np.cos(theta) / A)**2 + \
                  (np.sin(phi) * np.sin(theta) / B)**2 + \
                    (np.cos(phi) / C)**2) ** -1)

    for n, m in it.product(range(N_TERMS), range(N_TERMS)):
        
        re += cs[n, m, 0] * np.cos(n * theta) * np.cos(m * phi) + \
              cs[n, m, 1] * np.cos(n * theta) * np.sin(m * phi) + \
              cs[n, m, 2] * np.sin(n * theta) * np.cos(m * phi) + \
              cs[n, m, 3] * np.sin(n * theta) * np.sin(m * phi)

    return re 

# Draw coefficients from Gaussian distribution
cs = np.random.normal(loc=0.0, scale=COEF_SDS, size=(N_TERMS, N_TERMS, 4))

# Read in mesh
geo = mulgrid("models/channel/gCH.dat").layermesh

cap = np.zeros((geo.num_cells, ))

for c in geo.cell:

    # Get angles between cell and centre of clay cap
    x, y, z = c.centre - CAP_CENTRE
    z += (100 / 500**2) * (x**2 + y**2) # Give cap a curved appearance

    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arccos(z / r)
    theta = np.arctan2(y, x)
    # if theta < 0: # TODO: check this fella
    #     theta += 2 * np.pi

    re = get_radius(phi, theta, cs)

    if r < re:
        cap[c.index] = 1.0

    # cap[c.index] = np.sqrt(x**2 + y**2 + z**2)

geo.slice_plot("y", value=cap)
geo.layer_plot(elevation=-250, value=cap)
