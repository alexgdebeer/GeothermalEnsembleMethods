import numpy as np
import pyvista as pv
from layermesh import mesh as lm
from matern_fields import MaternField2D

MESH_PATH = "models/channel/gCH"

geo = lm.mesh(f"{MESH_PATH}.h5")
mesh = pv.PolyData([[*col.centre, 0.0] for col in geo.column]).delaunay_2d()
# mesh = pv.UnstructuredGrid("test/mesh_tri.msh").delaunay_2d().scale(10)
# mesh.plot(show_edges=True)

mesh["point_indices"] = np.arange(mesh.n_points, dtype=np.int32)

points = mesh.points[:, :2]
elements = mesh.regular_faces

boundary = mesh.extract_feature_edges(boundary_edges=True, 
                                      feature_edges=False, 
                                      non_manifold_edges=False, 
                                      manifold_edges=False)

boundary_points = boundary.cast_to_pointset()["point_indices"]
boundary_facets = boundary.lines.reshape(-1, 3)[:, 1:]
boundary_facets = np.array([boundary_points[f] for f in boundary_facets])

# p = pv.Plotter()
# p.add_mesh(mesh, show_edges=True)
# for line in boundary_facets:
#     p.add_points(mesh.points[line])
# p.show()

matern_field = MaternField2D()
matern_field.build_fem_matrices(points, elements, boundary_facets)

sigma = 1.0
lx = 150
ly = 150
lam = 1.42 * np.sqrt(lx * ly) # TODO: tune Robin parameter

W = np.random.normal(loc=0.0, scale=1.0, size=mesh.n_points)
X = matern_field.generate_field(W, sigma, lx, ly, bcs="robin")

p = pv.Plotter()
p.add_mesh(mesh, show_edges=True, scalars=X, cmap="turbo")
p.show()

# XS = np.zeros((mesh.n_points, 10_000))

# for i in range(10_000):

#     W = np.random.normal(loc=0.0, scale=1.0, size=mesh.n_points)
#     XS[:, i] = matern_field.generate_field(W, sigma, lx, ly)

#     if i % 100 == 0 and i != 0:
#         print(i)

# stds = np.std(XS, axis=1)

# p = pv.Plotter()
# p.add_mesh(mesh, show_edges=True, scalars=stds, cmap="turbo")
# p.show()

def plot_col_values(geo, values, **kwargs):
    """Plots a set of values that correspond to the columns of the mesh."""
    col_values = np.array([values[cell.column.index] for cell in geo.cell])
    geo.layer_plot(value=col_values, **kwargs)

plot_col_values(geo, X, colourmap="turbo")