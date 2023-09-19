import itertools as it
from layermesh import mesh as lm
import numpy as np
import pyvista as pv
from scipy import stats

MESH_NAME = "models/channel/gCH"

def gaussian_to_uniform(x, lb, ub):
    return lb + stats.norm.cdf(x) * (ub - lb)

class ClayCap():

    def __init__(self, cell_centres, centre_bounds, 
                 width_h_bounds, width_v_bounds, 
                 dip_bounds, n_terms, coef_sds):
        
        self.cell_centres = cell_centres

        self.centre_bounds = centre_bounds
        self.dip_bounds = dip_bounds

        self.width_h_bounds = width_h_bounds 
        self.width_v_bounds = width_v_bounds

        self.n_terms = n_terms 
        self.coef_sds = coef_sds
        self.n_params = 6 + 4 * self.n_terms ** 2

    def cartesian_to_spherical(self, ds):
        rs = np.linalg.norm(ds, axis=1)
        phis = np.arccos(ds[:, 2] / rs)
        thetas = np.arctan2(ds[:, 1], ds[:, 0])
        return rs, phis, thetas
    
    def compute_cap_radii(self, phis, thetas, width_h, width_v, coefs):
        """Computes the radius of the clay cap in the direction of each cell, 
        by taking the radius of the (deformed) ellipse that forms the base
        of the cap, then adding the randomised Fourier series to it."""

        rs = np.sqrt(((np.sin(phis) * np.cos(thetas) / width_h)**2 + \
                      (np.sin(phis) * np.sin(thetas) / width_h)**2 + \
                      (np.cos(phis) / width_v)**2) ** -1)
        
        for n, m in it.product(range(self.n_terms), range(self.n_terms)):
        
            rs += coefs[n, m, 0] * np.cos(n * thetas) * np.cos(m * phis) + \
                  coefs[n, m, 1] * np.cos(n * thetas) * np.sin(m * phis) + \
                  coefs[n, m, 2] * np.sin(n * thetas) * np.cos(m * phis) + \
                  coefs[n, m, 3] * np.sin(n * thetas) * np.sin(m * phis)
        
        return rs
    
    def get_cap_params(self, params):
        """Given a set of unit normal variables, generates the corresponding 
        set of clay cap parameters."""

        centre = np.array([gaussian_to_uniform(params[i], *bnds)
                           for i, bnds in enumerate(self.centre_bounds)])
        width_h = gaussian_to_uniform(params[3], *self.width_h_bounds)
        width_v = gaussian_to_uniform(params[4], *self.width_v_bounds)
        dip = gaussian_to_uniform(params[5], *self.dip_bounds)

        coefs = self.coef_sds * params[6:]
        coefs = np.reshape(coefs, (self.n_terms, self.n_terms, 4))

        return centre, width_h, width_v, dip, coefs

    def get_cap_cells(self, params):
        """Returns an array of booleans that indicate whether each cell is 
        contained within the clay cap."""

        centre, width_h, width_v, dip, coefs = self.get_cap_params(params)

        ds = self.cell_centres - centre

        # TODO: should this be width_h, or mean(width_h)?
        ds[:, -1] += (dip / width_h**2) * (ds[:, 0]**2 + ds[:, 1]**2) 

        cell_radii, cell_phis, cell_thetas = self.cartesian_to_spherical(ds)

        cap_radii = self.compute_cap_radii(cell_phis, cell_thetas,
                                           width_h, width_v, coefs)
        
        return cell_radii < cap_radii

def plot(mesh, geo, cap):
    """Generates a 3D plot of the cells that make up the clay cap."""

    containing_cells = mesh.find_containing_cell([c.centre for c in geo.cell])

    cap_mesh = np.zeros((mesh.n_cells, ))
    cap_mesh[containing_cells] = cap[[c.index for c in geo.cell]]

    mesh.cell_data["cap_mesh"] = cap_mesh
    p = pv.Plotter()
    p.add_mesh(mesh.threshold([0.5, 1.5]), cmap="coolwarm")
    p.add_mesh(mesh.threshold([-0.5, 0.5]),  opacity=0.5, cmap="coolwarm")
    p.show()

geo = lm.mesh(f"{MESH_NAME}.h5")
mesh = pv.UnstructuredGrid(f"{MESH_NAME}.vtu")

cell_centres = np.array([c.centre for c in geo.cell])
centre_bounds = [(700, 800), (700, 800), (-300, -225)]
width_h_bounds = (425, 475)
width_v_bounds = (50, 75)
dip_bounds = (100, 200)

n_terms = 5
coef_sds = 5

clay_cap = ClayCap(cell_centres, centre_bounds, width_h_bounds, width_v_bounds, 
                   dip_bounds, n_terms, coef_sds)

params = np.random.normal(size=clay_cap.n_params)
cap = clay_cap.get_cap_cells(params)

plot(mesh, geo, cap)
# geo.slice_plot("y", value=cap, colourmap="coolwarm")
# geo.layer_plot(elevation=-250, value=cap, colourmap="coolwarm")
