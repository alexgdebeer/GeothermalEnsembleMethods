from layermesh import mesh as lm
import numpy as np
import pyvista as pv

from matern_fields import BC, MaternField3D

MESH_NAME = "models/channel/gCH"

geo = lm.mesh(f"{MESH_NAME}.h5")
mesh = pv.UnstructuredGrid(f"{MESH_NAME}.vtu").triangulate()

matern_field = MaternField3D(geo, mesh)

sigma = 1.0
lx = 100
ly = 100
lz = 100
lam = 30 * np.sqrt(lx * ly * lz) # TODO: tune Robin parameter

W = np.random.normal(loc=0.0, scale=1.0, size=matern_field.n_points)
X = matern_field.generate_field(W, sigma, lx, ly, lz, bcs=BC.ROBIN, lam=lam)

matern_field.plot_points(values=X, show_edges=True, cmap="turbo")
matern_field.plot_cells(values=X, show_edges=True, cmap="turbo")
matern_field.plot_slice(values=X, colourmap="turbo")