import cmocean
from matplotlib import pyplot as plt 
import numpy as np 
import pyvista as pv

from GeothermalEnsembleMethods.grfs import MaternField2D
from GeothermalEnsembleMethods.models import ModelType

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

def levels(p):
    if   p < -1.5: return -15.0
    elif p < -0.5: return -14.5
    elif p <  0.5: return -14.0
    elif p <  1.5: return -13.5
    else: return -13.0

def apply_level_sets(phi):
    perms = np.array([levels(p) for p in phi])
    return perms

xs = np.linspace(0, 1000, 80)
nx = len(xs)
ncs = nx ** 2

m = RegularMaternField2D(xs)

sigma = 1.0
lx = 1250
ly = 200

fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.2))

for i in range(3):

    W = np.random.normal(size=ncs)

    phi = m.generate_field(W, sigma, lx, ly)
    perms = apply_level_sets(phi)

    phi = np.reshape(phi, (nx, nx))
    perms = np.reshape(perms, (nx, nx))

    print(np.min(phi))
    print(np.max(phi))

    axes[0][i].pcolormesh(phi.T, cmap=cmocean.cm.thermal, vmin=-3.5, vmax=3.0, rasterized=True)
    axes[1][i].pcolormesh(perms.T, cmap=cmocean.cm.turbid.reversed(), vmin=-17, vmax=-13, rasterized=True)

    W = np.random.normal(size=ncs)

for ax in axes.flat: 
    ax.axis("off")
    ax.set_box_aspect(1)

plt.tight_layout()
plt.savefig("plots/level_set_func.pdf")