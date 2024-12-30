import colorcet as cc
from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv

from GeothermalEnsembleMethods import MaternField2D, ModelType

np.random.seed(16)

class RegularMaternField2D(MaternField2D):

    def __init__(self, xs):

        self.dim = 2
        self.nu = 2 - self.dim / 2
        self.model_type = ModelType.MODEL3D
        self.inds = [0, 1]

        points = np.array([[x0, x1, 0.0] for x0 in xs for x1 in xs])
        self.fem_mesh = pv.PolyData(points).delaunay_2d()

        self.get_mesh_data()
        self.build_fem_matrices()

xs = np.linspace(0, 1000, 100)
nx = len(xs)
ncs = nx ** 2

m = RegularMaternField2D(xs)

W = np.random.normal(size=ncs)
sigma=1.0

fig, axes = plt.subplots(1, 4, figsize=(10, 2.75))

lengthscales = [200, 400, 600, 1000]

for i, l_i in enumerate(lengthscales):

    X_i = m.generate_field(W, sigma, l_i, l_i)
    X_i = np.reshape(X_i, (nx, nx))

    print(np.max(X_i))
    print(np.min(X_i))
        
    axes[i].pcolormesh(X_i, cmap="turbo", vmin=-2.1, vmax=2.5)
    axes[i].axis("off")
    axes[i].set_box_aspect(1)

plt.tight_layout()
plt.savefig("plots/lengthscales.pdf")