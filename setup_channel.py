"""Setup script for 3D channel model."""

from itertools import product
import numpy as np
from scipy import stats
from GeothermalEnsembleMethods import grfs, likelihood, models

np.random.seed(1)

SECS_PER_WEEK = 60.0 ** 2 * 24.0 * 7.0
MESH_NAME = "models/channel/gCH"
MODEL_NAME = "models/channel/CH"

"""
Classes
"""

def gauss_to_unif(x, lb, ub):
    return lb + stats.norm.cdf(x) * (ub - lb)

class Channel():
    
    def __init__(self, cols, bounds):
        
        self.cols = cols 
        self.bounds = bounds

    def get_cols_in_channel(self, ps):
        """Returns the indices of the columns that are contained within the 
        channel specified by a given set of parameters."""
        
        def in_channel(x, y, a1, a2, a3, a4, a5):
            ub = a1 * np.sin(2*np.pi*x/a2) + np.tan(a3)*x + a4 
            return ub-a5 <= y <= ub+a5 

        ps = [gauss_to_unif(p, *self.bounds[i]) for i, p in enumerate(ps)]
        cols_in_channel = [c for c in self.cols if in_channel(*c.centre, *ps)]
        return cols_in_channel

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
        
        for n, m in product(range(self.n_terms), range(self.n_terms)):
        
            rs += coefs[n, m, 0] * np.cos(n * thetas) * np.cos(m * phis) + \
                  coefs[n, m, 1] * np.cos(n * thetas) * np.sin(m * phis) + \
                  coefs[n, m, 2] * np.sin(n * thetas) * np.cos(m * phis) + \
                  coefs[n, m, 3] * np.sin(n * thetas) * np.sin(m * phis)
        
        return rs
    
    def get_cap_params(self, params):
        """Given a set of unit normal variables, generates the corresponding 
        set of clay cap parameters."""

        centre = np.array([gauss_to_unif(params[i], *bnds)
                           for i, bnds in enumerate(self.centre_bounds)])
        width_h = gauss_to_unif(params[3], *self.width_h_bounds)
        width_v = gauss_to_unif(params[4], *self.width_v_bounds)
        dip = gauss_to_unif(params[5], *self.dip_bounds)

        coefs = self.coef_sds * params[6:]
        coefs = np.reshape(coefs, (self.n_terms, self.n_terms, 4))

        return centre, width_h, width_v, dip, coefs

    def get_cap_cells(self, params):
        """Returns an array of booleans that indicate whether each cell is 
        contained within the clay cap."""

        centre, width_h, width_v, dip, coefs = self.get_cap_params(params)

        ds = self.cell_centres - centre
        ds[:, -1] += (dip / width_h**2) * (ds[:, 0]**2 + ds[:, 1]**2) 

        cell_radii, cell_phis, cell_thetas = self.cartesian_to_spherical(ds)

        cap_radii = self.compute_cap_radii(cell_phis, cell_thetas,
                                           width_h, width_v, coefs)
        
        return cell_radii < cap_radii

class Prior():

    # TODO: add mean to Matern fields...

    def __init__(self, mesh, params_cap, params_channel, 
                 grf_cap, grf_ext, grf_upflows, level_width):

        self.mesh = mesh 

        self.params_cap = params_cap 
        self.params_channel = params_channel

        self.grf_cap = grf_cap 
        self.grf_ext = grf_ext
        self.grf_upflows = grf_upflows

        self.level_width = level_width

    def transform_perms():
        # Transform permeabilities using Matern objects
        # Apply level set
        # Determine which cells are in the clay cap
        pass 

    def sample(self, n=1):
        pass

"""
Model parameters
"""

mesh = models.IrregularMesh(MESH_NAME)

# TODO: specify feedzone locations
feedzone_locs = [(0.0, 0.0, 0.0)]
feedzone_qs = [0.0]
feedzones = [models.Feedzone(loc, q) 
             for (loc, q) in zip(feedzone_locs, feedzone_qs)]

"""
Clay cap
"""

cell_centres = np.array([c.centre for c in mesh.m.cell])
centre_bounds = [(700, 800), (700, 800), (-300, -225)]
width_h_bounds = (425, 475)
width_v_bounds = (50, 75)
dip_bounds = (100, 200)

n_terms = 5
coef_sds = 5

clay_cap = ClayCap(cell_centres, centre_bounds, 
                   width_h_bounds, width_v_bounds, 
                   dip_bounds, n_terms, coef_sds)

"""
Channel
"""

# Bounds for amplitude, period, angle, intercept, width
bounds = [(100, 200), (500, 1200), (-np.pi/8, np.pi/8), 
          (500, 1000), (75, 150)]

channel = Channel(mesh.m.column, bounds)

"""
Model functions
"""