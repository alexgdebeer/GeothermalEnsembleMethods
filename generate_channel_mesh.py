import numpy as np
from layermesh import mesh as lm
from matplotlib import pyplot as plt

np.random.seed(0)
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

MESH_NAME = "models/channel/gCH"

def refine_geometry(geo, poly_to_refine):
    """Refines the columns of the model geometry within a given polygon, then 
    optimises the node placement in the region between another two polygons."""

    cols_to_refine = geo.find(poly_to_refine)
    geo.refine(cols_to_refine)

    triangles = geo.type_columns(3)
    geo.optimize(columns=triangles)

def fit_surface(geo, mu, sd, l, plot=False):
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
    geo.fit_surface(surf)

    if plot:
        geo.slice_plot(title="Slice view")
        geo.layer_plot(title="Top view", 
                       value=czs[[c.column.index for c in geo.cell]])

def save(geo: lm.mesh):
    """Writes the model geometry to an h5 file (for use with Layermesh), a
    gmsh file (for use with Waiwera) and a vtu file (for use with PyVista)."""

    geo.write(f"{MESH_NAME}.h5")
    geo.export(f"{MESH_NAME}.msh", fmt="gmsh22")
    geo.export(f"{MESH_NAME}.vtu")

xs = [100] * 15
ys = [100] * 15
zs = [25] * 16 + [50] * 4 + [100] * 4

geo = lm.mesh(rectangular=(xs, ys, zs))

poly_to_refine = [(450, 450), (1050, 1050)]

mu_surf = -60
sd_surf = 25
l_surf = 500

refine_geometry(geo, poly_to_refine)
fit_surface(geo, mu_surf, sd_surf, l_surf, plot=True)
save(geo)