import numpy as np
import pyvista as pv

from matern_fields import MaternField3D, generate_mesh_mapping_3D
from layermesh import mesh as lm

MESH_NAME = "models/channel/gCH"

geo = lm.mesh(f"{MESH_NAME}.h5")
mesh = pv.UnstructuredGrid(f"{MESH_NAME}.vtu").triangulate()

mesh["point_indices"] = np.arange(mesh.n_points, dtype=np.int32)

points = mesh.points
n_points = mesh.n_points
elements = mesh.cells_dict[10]

boundary = mesh.extract_geometry()
boundary_points = boundary.cast_to_pointset()["point_indices"]
boundary_facets = boundary.faces.reshape(-1, 4)[:, 1:]
boundary_facets = np.array([boundary_points[f] for f in boundary_facets])

matern_field = MaternField3D()

matern_field.build_fem_matrices(points, elements, boundary_facets)

sigma = 1.0
lx = 300
ly = 300
lz = 300
lam = 200.0 * np.sqrt(lx * ly * lz) # TODO: tune Robin parameter

W = np.random.normal(loc=0.0, scale=1.0, size=mesh.n_points)
X = matern_field.generate_field(W, sigma, lx, ly, lz, bcs="robin", lam=lam)

plotter = pv.Plotter()
plotter.add_mesh(mesh, show_edges=True, scalars=X, cmap="turbo")
plotter.show()

# XS = np.zeros((mesh.n_points, 100))

# for i in range(100):
#     W = np.random.normal(loc=0.0, scale=1.0, size=mesh.n_points)
#     XS[:, i] = matern_field.generate_field(W, sigma, lx, ly, lz, bcs="robin", lam=lam)
#     print(i)

# std = np.std(XS, axis=1)

H = generate_mesh_mapping_3D(mesh, geo)

# plotter = pv.Plotter()
# plotter.add_mesh(mesh, show_edges=True, scalars=X, cmap="turbo")
# plotter.show()
# plotter.save_graphic("3D_mesh.pdf", raster=False)

geo.slice_plot(value=H@X, colourmap="turbo")