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

# Channel amplitude, period, angle, intercept, width
a1 = np.random.uniform(100, 200)
a2 = np.random.uniform(500, 1200)
a3 = np.random.uniform(-np.pi/8, np.pi/8)
a4 = np.random.uniform(500, 1000)
a5 = np.random.uniform(75, 150)

def channel_bounds(x):
    ub = a1 * np.sin(2*np.pi*x / a2) + np.tan(a3)*x + a4 
    return ub-a5, ub+a5 

for x, y, i in zip(cxs, cys, is_base):

    lb, ub = channel_bounds(x)
    if y < lb or y > ub:
        upfls[i]= 0.0

geo.layer_plot(value=upfls, colourmap="coolwarm")