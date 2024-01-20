import colorcet as cc
from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv

from src.grfs import MaternField2D
from src.models import ModelType

np.random.seed(22) # 22 good

xs = np.linspace(0, 1000, 100)
nx = len(xs)
ncs = nx ** 2

n_levels = 3

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

def level_set(x):
    if x < -0.5:
        return -1
    if -0.5 < x < 0.5:
        return 0 
    else: 
        return 1

def generate_variable_field(base_field, variable_fields):
    
    variable_field = np.zeros(ncs)
    
    for i, f in enumerate(base_field): 
        if f == -1:
            variable_field[i] = 0.15 * variable_fields[0][i] - 1
        elif f == 0:
            variable_field[i] = 0.15 * variable_fields[1][i]
        else:
            variable_field[i] = 0.15 * variable_fields[2][i] + 1

    return variable_field

def generate_max_field(fields):

    max_field = np.zeros(ncs)

    # fields[-1] *= 0.3
    
    for i in range(ncs):
        if fields[0][i] < 1.0 and fields[1][i] < 0.0:
            max_field[i] = -1
        elif fields[1][i] < 1.0 and fields[0][i] > 0.0:
            max_field[i] = 1
        else: 
            max_field[i] = 0

    return max_field

field = RegularMaternField2D(xs)

ws = np.random.normal(size=(7, ncs))

sigma = 1.0

lx_ext = 1250
ly_ext = 250

lx_int = 50
ly_int = 50

fields_ext = [field.generate_field(w, sigma, lx_ext, ly_ext) for w in ws[:4]]
fields_int = [field.generate_field(w, sigma, lx_int, ly_int) for w in ws[4:]]

fields = fields_ext + fields_int

base_field = np.vectorize(level_set)(fields[0])
variable_field = generate_variable_field(base_field, fields[4:])
max_field = generate_max_field(fields[2:4])
variable_max_field = generate_variable_field(max_field, fields[4:])

base_field = np.reshape(base_field, (nx, nx))
variable_field = np.reshape(variable_field, (nx, nx))
max_field = np.reshape(max_field, (nx, nx))
variable_max_field = np.reshape(variable_max_field, (nx, nx))

field_list = [base_field, variable_field, max_field, variable_max_field]

cmap = "turbo"
vmin, vmax = np.min(field_list), np.max(field_list)

fig, axes = plt.subplots(1, 4, figsize=(10, 2.75))

axes[0].pcolormesh(base_field.T, cmap=cmap, vmin=vmin, vmax=vmax)
axes[1].pcolormesh(variable_field.T, cmap=cmap, vmin=vmin, vmax=vmax)
axes[2].pcolormesh(max_field.T, cmap=cmap, vmin=vmin, vmax=vmax)
axes[3].pcolormesh(variable_max_field.T, cmap=cmap, vmin=vmin, vmax=vmax)

for ax in axes.flat:
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

plt.tight_layout()
plt.savefig("plots/level_set_example.pdf")