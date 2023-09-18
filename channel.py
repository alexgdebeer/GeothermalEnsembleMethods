import numpy as np
from layermesh import mesh as lm

geo = lm.mesh("models/channel/gCH.h5")

cs_base = geo.layer[-1].cell
is_base = np.array([c.index for c in cs_base])
cxs = np.array([c.centre[0] for c in cs_base])
cys = np.array([c.centre[1] for c in cs_base])

rho = 100
nu = 100

dxs = cxs[:, np.newaxis] - cxs.T
dys = cys[:, np.newaxis] - cys.T
ds = dxs ** 2 + dys ** 2

sd = 50
mu = 100 + np.zeros((geo.num_columns, ))
cov = sd ** 2 * np.exp(-(1/(2*500**2)) * ds) + 1e-8 * np.eye(geo.num_columns)
rates = np.random.multivariate_normal(mu, cov)

upfls = np.zeros((geo.num_cells))
upfls[is_base] = rates

# Slope and intercept of underlying line
m = 0.3
c = 500

# Channel amplitude, period, width
a = 200
p = 1200
w = 200

def channel_bounds(x):
    ub = a * np.sin(2*np.pi*x / p) + m*x + c 
    return ub-w, ub 

for x, y, i in zip(cxs, cys, is_base):

    lb, ub = channel_bounds(x)
    if y < lb or y > ub:
        upfls[i]= 0.0

geo.layer_plot(value=upfls, colourmap="coolwarm")