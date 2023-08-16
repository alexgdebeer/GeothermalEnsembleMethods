import numpy as np
from t2grids import *

np.random.seed(0)

xs = [100] * 15
ys = [100] * 15
zs = [25] * 10 + [50] * 5 + [100] * 5

origin = [0, 0, 0]

geo = mulgrid().rectangular(xs, ys, zs, atmos_type=1, origin=origin)

region_to_refine = [[450, 450], [1050, 1050]]

geo.refine(geo.columns_in_polygon(region_to_refine))

poly_inner = [[425, 425], [1075, 1075]]
poly_outer = [[250, 250], [1250, 1250]]
nodenames_inner = [n.name for n in geo.nodes_in_polygon(poly_inner)]
nodenames_outer = [n.name for n in geo.nodes_in_polygon(poly_outer)]

nodenames = [n for n in nodenames_outer if n not in nodenames_inner]

geo.optimize(nodenames=nodenames)

# Generate surface data
col_centres = [c.centre for c in geo.column.values()]
cxs = np.array([c[0] for c in col_centres])
cys = np.array([c[1] for c in col_centres])

# TODO: tidy
dxs = cxs[:, np.newaxis] - cxs.T
dys = cys[:, np.newaxis] - cys.T
ds = dxs ** 2 + dys ** 2

sd = 50
mu = -100 + np.zeros((geo.num_columns, ))
cov = sd ** 2 * np.exp(-(1/(2*500**2)) * ds) + 1e-8 * np.eye(geo.num_columns)

czs = np.random.multivariate_normal(mu, cov)
surf = np.array([[x, y, min(z, 0)] for x, y, z in zip(cxs, cys, czs)])
geo.fit_surface(surf)
#geo.layer_plot(layer=-500, variable=czs, colourmap="coolwarm")
#geo.slice_plot()

geo.write("models/channel/gCH.dat")
geo.layermesh.export("models/channel/gCH.msh")