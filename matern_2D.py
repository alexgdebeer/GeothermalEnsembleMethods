from layermesh import mesh as lm
import numpy as np
import pyvista as pv

from matern_fields import BC, MaternField2D

MESH_NAME = "models/channel/gCH"

geo = lm.mesh(f"{MESH_NAME}.h5")
mesh = pv.PolyData([[*col.centre, 0.0] for col in geo.column]).delaunay_2d()
# mesh = pv.UnstructuredGrid("models/test/mesh_tri.msh").delaunay_2d().scale(10)

matern_field = MaternField2D(geo, mesh)

sigma = 1.0
lx = 200
ly = 200
lam = 1.42 * np.sqrt(lx * ly)

W = np.random.normal(loc=0.0, scale=1.0, size=matern_field.n_points)
X = matern_field.generate_field(W, sigma, lx, ly, bcs=BC.ROBIN, lam=lam)

matern_field.plot(values=X, show_edges=True, cmap="turbo")
matern_field.layer_plot(values=X, colourmap="turbo")