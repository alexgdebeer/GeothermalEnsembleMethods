import numpy as np
from scipy import special
from matplotlib import pyplot as plt

d = 1
tau = -1
nu = -1

b = (tau ** 2) * 2 * (np.pi ** 0.5) * \
    special.gamma(nu + 0.5 * d) / special.gamma(nu) * -1 # TODO: fix



# Draw a bunch of samples from a Gaussian
