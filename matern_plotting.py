import colorcet as cc
from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv

from src.grfs import MaternField2D
from src.models import ModelType

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

xs = np.linspace(0, 1000, 80)
nx = len(xs)
ncs = nx ** 2

m = RegularMaternField2D(xs)

W = np.random.normal(size=ncs)
sigma=1.0

fig, axes = plt.subplots(1, 3, figsize=(6, 2))

lengthscales = [200, 600, 1000]

for l, ax in zip(lengthscales, axes.flat):

    X = m.generate_field(W, sigma, l, l)
    X = np.reshape(X, (nx, nx))

    ax.pcolormesh(X, cmap="turbo")

plt.show()