import numpy as np
from GeothermalEnsembleMethods import grfs, models

MESH_NAME = "models/channel/gCH"

mesh = models.IrregularMesh(MESH_NAME) 
matern_field = grfs.MaternField2D(mesh)

sigma = 1.0
lx = 300
ly = 300
lam = 1.42 * np.sqrt(lx * ly)

W = np.random.normal(loc=0.0, scale=1.0, size=matern_field.n_points)
X = matern_field.generate_field(W, sigma, lx, ly, bcs=grfs.BC.ROBIN, lam=lam)

# matern_field.plot(values=X, show_edges=True, cmap="turbo")
# matern_field.layer_plot(values=X, colourmap="turbo")

a1 = 100
a2 = 1.0
a3 = 700
a4 = 600
a5 = 150

def in_channel(x, y):
    centre = a1 * np.sin(2 * np.pi * x / a3) + 0.3 * x + a4
    return centre - a5 < y < centre + a5


mu = 5
upflows = [X[c.column.index] for c in matern_field.m.cell]
upflows = [upflows[c.index] + mu if in_channel(*c.centre[:2]) else 0.0 for c in matern_field.m.cell]

matern_field.m.layer_plot(value=upflows, colourmap="coolwarm", linewidth=1.0)