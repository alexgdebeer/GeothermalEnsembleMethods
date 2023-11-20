import numpy as np
from src.models import IrregularMesh
from src.grfs import MaternField2D, BC

MESH_NAME = "models/channel/gCH"

mesh = IrregularMesh(MESH_NAME) 
matern_field = MaternField2D(mesh)

sigma = 1.0
lx = 300
ly = 300
lam = 1.42 * np.sqrt(lx * ly)

W = np.random.normal(loc=0.0, scale=1.0, size=matern_field.n_points)
X = matern_field.generate_field(W, sigma, lx, ly, bcs=BC.ROBIN, lam=lam)

matern_field.plot(values=X, show_edges=True, cmap="turbo")
matern_field.layer_plot(values=X, colourmap="turbo")