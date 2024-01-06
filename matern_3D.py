import numpy as np
from src import grfs, models

MESH_NAME = "models/channel/gCH"

mesh = models.IrregularMesh(MESH_NAME)
matern_field = grfs.MaternField3D(mesh)

sigma = 1.0
lx = 1200
ly = 1200
lz = 300
lam = 1000 * np.cbrt(lx * ly * lz) # TODO: tune Robin parameter

W = np.random.normal(loc=0.0, scale=1.0, size=matern_field.n_points)
X = matern_field.generate_field(W, sigma, lx, ly, lz, bcs=grfs.BC.ROBIN, lam=lam)

matern_field.plot_points(values=X, show_edges=True, cmap="turbo")
matern_field.plot_cells(values=X, show_edges=True, cmap="turbo")
matern_field.plot_slice(values=X, colourmap="turbo")