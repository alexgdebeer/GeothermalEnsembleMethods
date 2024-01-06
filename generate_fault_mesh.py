import numpy as np
from layermesh import mesh as lm
from matplotlib import pyplot as plt

np.random.seed(0)
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

MESH_FOLDER = "models/fault"

def refine_geometry(geo, poly_to_refine):
    """Refines the columns of the model geometry within a given polygon, then 
    optimises the node placement in the region between another two polygons."""

    cols_to_refine = geo.find(poly_to_refine)
    geo.refine(cols_to_refine)

    triangles = geo.type_columns(3)
    geo.optimize(columns=triangles)

def generate_surface(geo, mu, sd, l):
    """Generates the top surface of the model geometry using a GRF with
    squared-exponential kernel."""

    col_centres = [c.centre for c in geo.column]

    cxs = np.array([c[0] for c in col_centres])
    cys = np.array([c[1] for c in col_centres])

    dxs = cxs[:, np.newaxis] - cxs.T
    dys = cys[:, np.newaxis] - cys.T
    ds = dxs ** 2 + dys ** 2

    mu *= np.ones((geo.num_columns, ))
    cov = sd**2 * np.exp(-(1/(2*l**2)) * ds) + 1e-4 * np.eye(geo.num_columns)

    czs = np.random.multivariate_normal(mu, cov)
    surf = np.array([[x, y, min(z, 0)] for x, y, z in zip(cxs, cys, czs)])

    return surf

def save(geo: lm.mesh, name):
    """Writes the model geometry to an h5 file (for use with Layermesh), a
    gmsh file (for use with Waiwera) and a vtu file (for use with PyVista)."""

    geo.write(f"{MESH_FOLDER}/{name}.h5")
    geo.export(f"{MESH_FOLDER}/{name}.msh", fmt="gmsh22")
    geo.export(f"{MESH_FOLDER}/{name}.vtu")

xs_crse = [375] * 16
xs_fine = [300] * 20

ys_crse = [375] * 16
ys_fine = [300] * 20

zs = [100] * 16 + [200] * 3 + [400] * 2

geo_crse = lm.mesh(rectangular=(xs_crse, ys_crse, zs))
geo_fine = lm.mesh(rectangular=(xs_fine, ys_fine, zs))

poly_to_refine = [(1500, 1500), (4500, 4500)]

mu_surf = -250
sd_surf = 100
l_surf = 2000

refine_geometry(geo_crse, poly_to_refine)
refine_geometry(geo_fine, poly_to_refine)

surf = generate_surface(geo_fine, mu_surf, sd_surf, l_surf)

geo_crse.fit_surface(surf)
geo_fine.fit_surface(surf)

geo_fine.slice_plot(title="Slice view")
geo_fine.layer_plot()

print(geo_crse.num_cells)
print(geo_fine.num_cells)

save(geo_crse, f"gFL{geo_crse.num_cells}")
save(geo_fine, f"gFL{geo_fine.num_cells}")