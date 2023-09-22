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

    def __init__(self, mesh, grf, mu, bounds):
        
        self.mesh = mesh
        self.grf = grf
        self.mu = mu
        self.bounds = bounds

        self.n_pars = len(self.bounds) + self.grf.n_points

    def get_perms(self, pars):
        """Returns the set of permeabilities that correspond to a given
        set of (whitened) parameters."""

        hyperpars, W = pars[:4], pars[4:]
        hyperpars = [gauss_to_unif(par, *bnds)
                     for par, bnds in zip(hyperpars, self.bounds)]
        
        X = self.grf.generate_field(W, *hyperpars)
        return self.mu + self.grf.G @ X

class UpflowField():

    def __init__(self, mesh, grf, mu, bounds):

        self.mesh = mesh 
        self.grf = grf

        self.mu = mu 
        self.bounds = bounds 
        self.n_pars = len(self.bounds) + self.grf.n_points

    def get_upflows(self, pars):
        """Returns the set of upflows that correspond to a given set of 
        (whitened) parameters."""

        hyperpars, W = pars[:3], pars[3:]
        hyperpars = [gauss_to_unif(par, *bnds) 
                     for par, bnds in zip(hyperpars, self.bounds)]
        
        X = self.grf.generate_field(W, *hyperpars)
        upflows = np.zeros((self.mesh.m.num_cells, ))
        upflows[self.mesh.bottom_cell_inds] = self.mu + X
        return np.maximum(upflows, 0.0)

class Channel():
    
    def __init__(self, mesh, bounds):
        
        self.mesh = mesh
        self.bounds = bounds
        self.n_pars = len(self.bounds)

    def get_cells_in_channel(self, pars):
        """Returns the indices of the columns that are contained within 
        the channel specified by a given set of parameters."""
        
        def in_channel(x, y, a1, a2, a3, a4, a5):
            ub = a1 * np.sin(2*np.pi*x/a2) + np.tan(a3)*x + a4 
            return ub-a5 <= y <= ub+a5 

        pars = [gauss_to_unif(p, *self.bounds[i]) 
                for i, p in enumerate(pars)]
        
        cells_in_channel = [cell.index for cell in self.mesh.m.cell 
                            if in_channel(*cell.centre[:2], *pars)]
        return cells_in_channel

class ClayCap():

    def __init__(self, mesh, bounds, n_terms, coef_sds):
        
        self.cell_centres = mesh.cell_centres
        self.bounds = bounds 
        self.n_terms = n_terms 
        self.coef_sds = coef_sds
        self.n_pars = len(self.bounds) + 4 * self.n_terms ** 2

    def cart_to_sphere(self, coords):

        radii = np.linalg.norm(coords, axis=1)
        phis = np.arccos(coords[:, 2] / radii)
        thetas = np.arctan2(coords[:, 1], coords[:, 0])
        return radii, phis, thetas
    
    def compute_cap_radii(self, phis, thetas, width_h, width_v, coefs):
        """Computes the radius of the clay cap in the direction of each 
        cell, by taking the radius of the (deformed) ellipse that forms 
        the base of the cap, then adding the randomised Fourier series 
        to it."""

        radii = np.sqrt(((np.sin(phis) * np.cos(thetas) / width_h)**2 + \
                         (np.sin(phis) * np.sin(thetas) / width_h)**2 + \
                            (np.cos(phis) / width_v)**2) ** -1)
        
        for n, m in product(range(self.n_terms), range(self.n_terms)):
        
            radii += coefs[n, m, 0] * np.cos(n * thetas) * np.cos(m * phis) + \
                     coefs[n, m, 1] * np.cos(n * thetas) * np.sin(m * phis) + \
                     coefs[n, m, 2] * np.sin(n * thetas) * np.cos(m * phis) + \
                     coefs[n, m, 3] * np.sin(n * thetas) * np.sin(m * phis)
        
        return radii
    
    def get_cap_params(self, pars):
        """Given a set of unit normal variables, generates the 
        corresponding set of clay cap parameters."""

        geom = [gauss_to_unif(p, *self.bounds[i]) 
                for i, p in enumerate(pars[:6])]

        coefs = np.reshape(self.coef_sds * pars[6:], 
                           (self.n_terms, self.n_terms, 4))

        return geom, coefs

    def get_cells_in_cap(self, pars):
        """Returns an array of booleans that indicate whether each cell 
        is contained within the clay cap."""

        geom, coefs = self.get_cap_params(pars)
        *centre, width_h, width_v, dip = geom

        # Add some curvature to the cap
        ds = self.cell_centres - np.array(centre)
        ds[:, -1] += (dip / width_h**2) * (ds[:, 0]**2 + ds[:, 1]**2) 

        cell_radii, cell_phis, cell_thetas = self.cart_to_sphere(ds)
        cap_radii = self.compute_cap_radii(cell_phis, cell_thetas,
                                           width_h, width_v, coefs)
        return (cell_radii < cap_radii).nonzero()

class ChannelPrior():

    def __init__(self, mesh, cap, channel, 
                 grf_res, grf_cap, grf_flt, grf_upflow, level_width):

        self.mesh = mesh 

        self.cap = cap
        self.channel = channel
        self.grf_res = grf_res 
        self.grf_cap = grf_cap
        self.grf_flt = grf_flt
        self.grf_upflow = grf_upflow
        self.level_width = level_width

        self.param_counts = [cap.n_pars, channel.n_pars, 
                             grf_res.n_pars, grf_cap.n_pars, 
                             grf_flt.n_pars,
                             grf_upflow.n_pars]
        self.num_params = sum(self.param_counts)
        self.param_inds = np.cumsum(self.param_counts)

        self.inds = {
            "cap"       : np.arange(0, self.param_inds[0]),
            "channel"   : np.arange(*self.param_inds[0:2]),
            "grf_res"   : np.arange(*self.param_inds[1:3]),
            "grf_cap"   : np.arange(*self.param_inds[2:4]),
            "grf_flt"   : np.arange(*self.param_inds[3:5]),
            "grf_upflow": np.arange(*self.param_inds[4:6])
        }

    def transform(self, pars):

        cap_cells = self.cap.get_cells_in_cap(pars[self.inds["cap"]])
        channel_cells = self.channel.get_cells_in_channel(pars[self.inds["channel"]])

        perms_ext = self.grf_res.get_perms(pars[self.inds["grf_res"]])
        perms_cap = self.grf_cap.get_perms(pars[self.inds["grf_cap"]])
        perms_flt = self.grf_flt.get_perms(pars[self.inds["grf_flt"]])

        # TODO: apply level set

        upflows = self.grf_upflow.get_upflows(pars[self.inds["grf_upflow"]])

        # Filter the upflows so we just have the ones corresponding to the 
        # channel

        perms = np.copy(perms_ext)
        perms[channel_cells] = perms_flt[channel_cells]
        perms[cap_cells] = perms_cap[cap_cells]

        return perms, channel_cells, upflows

    def sample(self, n=1):
        ps = np.random.normal(size=(n, self.num_params))
        for p in ps:
            p = self.transform(p)
        return ps

"""
Model parameters
"""

mesh = models.IrregularMesh(MESH_NAME)
grf_2d = grfs.MaternField2D(mesh)
grf_3d = grfs.MaternField3D(mesh)

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
mu_perm_flt = -13

# Bounds for marginal standard deviations and x, y, z lengthscales
bounds_perm_res = [(0.25, 0.50), (1000, 1500), (1000, 1500), (200, 400)]
bounds_perm_cap = [(0.20, 0.30), (1000, 1500), (1000, 1500), (200, 400)]
bounds_perm_flt = [(0.20, 0.30), (1000, 1500), (1000, 1500), (200, 400)]

# Generate the clay cap and exterior / clay cap permeability fields
clay_cap = ClayCap(mesh, bounds_geom_cap, n_terms, coef_sds)
perm_field_res = PermeabilityField(mesh, grf_3d, mu_perm_ext, bounds_perm_res)
perm_field_cap = PermeabilityField(mesh, grf_3d, mu_perm_cap, bounds_perm_cap)
perm_field_flt = PermeabilityField(mesh, grf_3d, mu_perm_flt, bounds_perm_flt)

"""
Channel
"""

# Bounds for amplitude, period, angle, intercept, width
bounds_channel = [(100, 200), (500, 1200), (-np.pi/8, np.pi/8), (500, 1000), (75, 150)]

mu_upflow = -1 # TODO: figure out what this should be

# Bounds for marginal standard deviations and x, y lengthscales
bounds_upflow = [(-1, -1), (200, 400), (200, 400)] # TODO: figure out what sigma should be (depends on mesh too of course)

channel = Channel(mesh, bounds_channel)
upflow_field = UpflowField(mesh, grf_2d, mu_upflow, bounds_upflow)

"""
Prior
"""

level_width = 0.25 # Log(m^2), I think this is the same as previously

prior = ChannelPrior(mesh, clay_cap, channel, perm_field_res, 
                     perm_field_cap, perm_field_flt, upflow_field, 
                     level_width)

prior.sample(1)

"""
Model functions
"""