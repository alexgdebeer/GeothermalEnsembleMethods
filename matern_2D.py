from layermesh import mesh as lm
import numpy as np
import pyvista as pv

from matern_fields import MaternField2D

MESH_NAME = "models/channel/gCH"

geo = lm.mesh(f"{MESH_NAME}.h5")
mesh = pv.PolyData([[*col.centre, 0.0] for col in geo.column]).delaunay_2d()
# mesh = pv.UnstructuredGrid("models/test/mesh_tri.msh").delaunay_2d().scale(10)

matern_field = MaternField2D(geo, mesh)

sigma = 1.0
lx = 150
ly = 150
lam = 1.42 * np.sqrt(lx * ly) # TODO: tune Robin parameter

W = np.random.normal(loc=0.0, scale=1.0, size=matern_field.n_points)
X = matern_field.generate_field(W, sigma, lx, ly, bcs="robin", lam=lam)

matern_field.plot(show_edges=True, scalars=X, cmap="turbo")
matern_field.layer_plot(values=X, colourmap="turbo")

# XS = np.zeros((mesh.n_points, 10_000))

# for i in range(10_000):

#     W = np.random.normal(loc=0.0, scale=1.0, size=mesh.n_points)
#     XS[:, i] = matern_field.generate_field(W, sigma, lx, ly)

#     if i % 100 == 0 and i != 0:
#         print(i)

# stds = np.std(XS, axis=1)