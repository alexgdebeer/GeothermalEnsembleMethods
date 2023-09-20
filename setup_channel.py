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

class PermeabilityField():

    def __init__(self, mesh, mu, bounds):
        
        self.mesh = mesh
        self.mu = mu
        self.bounds = bounds

        self.grf = grfs.MaternField3D(self.mesh)

    def get_perms(self, ps):
        """Returns the set of permeabilities that correspond to a given set of 
        (whitened) parameters."""

        hyperps, W = ps[:4], ps[4:]
        hyperps = [gauss_to_unif(p, bnds)
                   for p, bnds in zip(hyperps, self.bounds)]
        
        X = self.grf.generate_field(W, *hyperps)
        return self.mu + X

class UpflowField():

    def __init__(self, mesh, mu, bounds):

        self.mesh = mesh 
        self.mu = mu 
        self.bounds = bounds 

        self.grf = grfs.MaternField2D(self.mesh)

    def get_upflows(self, ps):
        """Returns the set of upflows that correspond to a given set of 
        (whitened) parameters."""

        hyperps, W = ps[:3], ps[3:]
        hyperps = [gauss_to_unif(p, *bnds) 
                   for p, bnds in zip(hyperps, self.bounds)]
        
        X = self.grf.generate_field(W, *hyperps)
        return self.mu + X

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

    def __init__(self, cells, bounds, n_terms, coef_sds):
        
        self.cell_centres = [c.centre for c in cells]

        self.bounds = bounds 
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
    
    def get_cap_params(self, ps):
        """Given a set of unit normal variables, generates the corresponding 
        set of clay cap parameters."""

        geom = [gauss_to_unif(p, *self.bounds[i]) 
                   for i, p in enumerate(ps[:6])]

        coefs = np.reshape(self.coef_sds * ps[6:], 
                           (self.n_terms, self.n_terms, 4))

        return geom, coefs

    def get_cap_cells(self, params):
        """Returns an array of booleans that indicate whether each cell is 
        contained within the clay cap."""

        # Unpack parameters
        geom, coefs = self.get_cap_params(params)
        *centre, width_h, width_v, dip = geom

        ds = self.cell_centres - centre
        ds[:, -1] += (dip / width_h**2) * (ds[:, 0]**2 + ds[:, 1]**2) 

        cell_radii, cell_phis, cell_thetas = self.cartesian_to_spherical(ds)

        cap_radii = self.compute_cap_radii(cell_phis, cell_thetas,
                                           width_h, width_v, coefs)
        
        return cell_radii < cap_radii

class ChannelPrior():

    def __init__(self, mesh, cap, channel, 
                 grf_cap, grf_ext, grf_upflow, level_width):

        self.mesh = mesh 

        self.cap = cap
        self.channel = channel

        self.grf_cap = grf_cap 
        self.grf_ext = grf_ext
        self.grf_upflow = grf_upflow

        self.level_width = level_width

        # Will probably need an index for numbers of parameters...

    def transform_perms():
        # Transform lengthscales, standard deviations, etc
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

feedzone_locs = [(0.0, 0.0, 0.0)]
feedzone_qs = [0.0]
feedzones = [models.Feedzone(loc, q) 
             for (loc, q) in zip(feedzone_locs, feedzone_qs)]

"""
Clay cap
"""

# Bounds for centre of cap (x, y, z coordinates), width (horizontal and 
# vertical) and dip
bounds_geom_cap = [(700, 800), (700, 800), (-300, -225),
                   (425, 475), (50, 75), (100, 200)]
n_terms = 5
coef_sds = 5

# Mean of the (log-)permeabilities in the exterior / clay cap regions
mu_perm_ext = -14
mu_perm_cap = -16

# Bounds for marginal standard deviations and x, y, z lengthscales
bounds_perm_ext = [(0.25, 0.50), (1000, 1500), (1000, 1500), (200, 400)]
bounds_perm_cap = [(0.20, 0.30), (1000, 1500), (1000, 1500), (200, 400)]

# Generate the clay cap and exterior / clay cap permeability fields
clay_cap = ClayCap(mesh.m.cell, bounds_geom_cap, n_terms, coef_sds)
perm_field_ext = PermeabilityField(mesh, mu_perm_ext, bounds_perm_ext)
perm_field_cap = PermeabilityField(mesh, mu_perm_cap, bounds_perm_cap)

"""
Channel
"""

# Bounds for amplitude, period, angle, intercept, width
bounds_channel = [(100, 200), (500, 1200), (-np.pi/8, np.pi/8), (500, 1000), (75, 150)]

mu_upflow = -1 # TODO: figure out what this should be

# Bounds for marginal standard deviations and x, y lengthscales
bounds_upflow = [(-1, -1), (200, 400), (200, 400)] # TODO: figure out what sigma should be (depends on mesh too of course)

channel = Channel(mesh.m.column, bounds_channel)
upflow_field = UpflowField(mesh, mu_upflow, bounds_upflow)

"""
Prior
"""

level_width = 0.25 # Log(m^2), I think this is the same as previously

# TODO: parameters that aren't in here currently: mean of each random field 
# (might want to make the mean of the upflows decrease the further we are from 
# the centre of the model?), bounds of lengthscales and standard deviations 
# of each random field, 
# Could make a wrapper around the MaternXD classes that contains this 
# information?

prior = ChannelPrior(mesh, clay_cap, channel, 
                     perm_field_ext, perm_field_cap, upflow_field, level_width)

"""
Model functions
"""