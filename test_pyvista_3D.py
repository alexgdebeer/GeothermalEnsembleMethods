import numpy as np
import pyvista as pv

from matern_fields import MaternField3D
from layermesh import mesh as lm

MESH_NAME = "models/channel/gCH"

geo = lm.mesh(f"{MESH_NAME}.h5")
mesh = pv.UnstructuredGrid(f"{MESH_NAME}.vtu").triangulate()

matern_field = MaternField3D(geo, mesh)

sigma = 1.0
lx = 300
ly = 300
lz = 300
lam = 200.0 * np.sqrt(lx * ly * lz) # TODO: tune Robin parameter

W = np.random.normal(loc=0.0, scale=1.0, size=mesh.n_points)
X = matern_field.generate_field(W, sigma, lx, ly, lz, bcs="robin", lam=lam)

matern_field.plot(show_edges=True, scalars=X, cmap="turbo")
matern_field.slice_plot(X, colourmap="turbo")

# XS = np.zeros((mesh.n_points, 100))

# for i in range(100):
#     W = np.random.normal(loc=0.0, scale=1.0, size=mesh.n_points)
#     XS[:, i] = matern_field.generate_field(W, sigma, lx, ly, lz, bcs="robin", lam=lam)
#     print(i)

# std = np.std(XS, axis=1)