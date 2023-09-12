import numpy as np
from t2grids import mulgrid
np.random.seed(0)

MESH_NAME = "models/channel/gCH"

def refine_geometry(geo, poly_to_refine, poly_inner, poly_outer):
    """Refines the columns of the model geometry within a given polygon, then 
    optimises the node placement in the region between another two polygons."""

    cols_to_refine = geo.columns_in_polygon(poly_to_refine)
    geo.refine(cols_to_refine)

    nodenames_inner = [n.name for n in geo.nodes_in_polygon(poly_inner)]
    nodenames_outer = [n.name for n in geo.nodes_in_polygon(poly_outer)]
    nodenames = [n for n in nodenames_outer if n not in nodenames_inner]

    geo.optimize(nodenames=nodenames)

def fit_surface(geo, mu, sd, l, plot=False):
    """Generates the top surface of the model geometry using a 
    squared-exponential kernel."""

    col_centres = [c.centre for c in geo.column.values()]

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
        geo.layer_plot(layer=-500, variable=czs, colourmap="coolwarm")
        geo.slice_plot()

def save(geo):
    """Writes the model geometry to an h5 file (for use with Layermesh), a
    gmsh file (for use with Waiwera) and a vtu file (for use with PyVista)."""

    geo.layermesh.write(f"{MESH_NAME}.h5")
    geo.layermesh.export(f"{MESH_NAME}.msh", fmt="gmsh22")
    geo.layermesh.export(f"{MESH_NAME}.vtu")

xs = [100] * 15
ys = [100] * 15
zs = [25] * 10 + [50] * 5 + [100] * 5

geo = mulgrid().rectangular(xs, ys, zs, atmos_type=1, origin=[0, 0, 0])

poly_to_refine = [[450, 450], [1050, 1050]]
poly_inner = [[425, 425], [1075, 1075]]
poly_outer = [[250, 250], [1250, 1250]]

mu_surf = -75
sd_surf = 30
l_surf = 500

refine_geometry(geo, poly_to_refine, poly_inner, poly_outer)
fit_surface(geo, mu_surf, sd_surf, l_surf, plot=True)
save(geo)