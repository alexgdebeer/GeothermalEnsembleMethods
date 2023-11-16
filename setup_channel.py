"""Setup script for 3D channel model."""

from itertools import product
import numpy as np
from scipy import stats
from GeothermalEnsembleMethods import consts, grfs, likelihood, models

np.random.seed(1)

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

        self.n_params = self.mesh.m.num_cells + 3

    def get_perms(self, ps):
        """Returns the set of permeabilities that correspond to a given 
        set of (whitened) parameters."""

        hyperps, W = ps[:4], ps[4:]
        hyperps = [gauss_to_unif(p, bnds)
                   for p, bnds in zip(hyperps, self.bounds)]
        
        X = self.grf.generate_field(W, *hyperps)
        return self.mu + self.grf.H @ X

    def apply_level_set(self, perms, level_width):

        min_level = np.floor(np.min(perms))
        max_level = np.ceil(np.max(perms)) + 1e-8 # HACK

        levels = np.arange(min_level, max_level, level_width)
        return np.array([levels[np.abs(levels - p).argmin()] for p in perms])

class UpflowField():

    def __init__(self, mesh, grf, mu, bounds):

        self.mesh = mesh 
        self.grf = grf

        self.mu = mu 
        self.bounds = bounds 

        self.n_params = 2 + self.mesh.m.num_columns

    def get_upflows(self, params):
        """Returns the set of upflows that correspond to a given set of 
        (whitened) parameters."""

        hyperparams, W = params[:3], params[3:]
        hyperparams = [gauss_to_unif(p, *bnds) 
                       for p, bnds in zip(hyperparams, self.bounds)]
        
        X = self.grf.generate_field(W, *hyperparams)
        return self.mu + self.grf.H @ X

class Channel():
    
    def __init__(self, mesh, bounds):
        
        self.mesh = mesh
        self.bounds = bounds

        self.n_params = 5

    def get_cells_in_channel(self, pars):
        """Returns the indices of the columns that are contained within 
        the channel specified by a given set of parameters."""
        
        def in_channel(x, y, a1, a2, a3, a4, a5):
            ub = a1 * np.sin(2*np.pi*x/a2) + np.tan(a3)*x + a4 
            return ub-a5 <= y <= ub+a5 

        pars = [gauss_to_unif(p, *self.bounds[i]) for i, p in enumerate(pars)]

        cols_in_channel = [col for col in self.mesh.column 
                           if in_channel(*col.centre, *pars)]
        
        cells = [cell.index for col in cols_in_channel for cell in col.cell]
        cols = [col.index for col in cols_in_channel]

        return cells, cols

class ClayCap():

    def __init__(self, mesh, bounds, n_terms, coef_sds):
        
        self.cell_centres = [cell.centre for cell in mesh.m.cell]
        self.col_centres = [col.centre for col in mesh.m.column]

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
        """Computes the radius of the clay cap in the direction of each 
        cell, by taking the radius of the (deformed) ellipse that forms 
        the base of the cap, then adding the randomised Fourier series 
        to it."""

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
        """Given a set of unit normal variables, generates the 
        corresponding set of clay cap parameters."""

        geom = [gauss_to_unif(p, *self.bounds[i]) 
                for i, p in enumerate(ps[:6])]

        coefs = np.reshape(self.coef_sds * ps[6:], 
                           (self.n_terms, self.n_terms, 4))

        return geom, coefs

    def get_cells_in_cap(self, params):
        """Returns an array of booleans that indicate whether each cell 
        is contained within the clay cap."""

        # Unpack parameters
        geom, coefs = self.get_cap_params(params)
        *centre, width_h, width_v, dip = geom

        ds = self.cell_centres - centre
        ds[:, -1] += (dip / width_h**2) * (ds[:, 0]**2 + ds[:, 1]**2) 

        cell_radii, cell_phis, cell_thetas = self.cartesian_to_spherical(ds)

        cap_radii = self.compute_cap_radii(cell_phis, cell_thetas,
                                           width_h, width_v, coefs)
        
        return (cell_radii < cap_radii).nonzero()
    
    def get_upflow_weightings(self, params):
        """Returns a list of values by which the upflow in each column 
        should be multiplied."""

        cx, cy, *_ = self.cap.get_cap_params(params[self.inds["cap"]])[0]

        s = 300 # TODO: make this into an input

        col_centres = [col.centre for col in self.mesh.m.column]
        col_weightings = np.array([np.exp(-(((c[0]-cx)/s)**2 + ((c[1]-cy)/s)**2)) / 2
                                   for c in col_centres]) 
        
        return col_weightings

class ChannelPrior():

    def __init__(self, mesh, cap, channel, 
                 grf_ext, grf_flt, grf_cap, 
                 grf_upflow, level_width):

        self.mesh = mesh 

        self.cap = cap
        self.channel = channel
        self.grf_ext = grf_ext 
        self.grf_flt = grf_flt
        self.grf_cap = grf_cap
        self.grf_upflow = grf_upflow
        self.level_width = level_width

        self.param_counts = [cap.n_params, channel.n_params, 
                             grf_ext.n_params, grf_flt.n_params, 
                             grf_cap.n_params, grf_upflow.n_params]
        
        self.num_params = sum(self.param_counts)
        self.param_inds = np.cumsum(self.param_counts)

        self.inds = {
            "cap"        : np.arange(*self.param_inds[0:2]),
            "channel"    : np.arange(*self.param_inds[1:3]),
            "grf_ext"    : np.arange(*self.param_inds[2:4]),
            "grf_flt"    : np.arange(*self.param_inds[3:5]),
            "grf_cap"    : np.arange(*self.param_inds[4:6]),
            "grf_upflow" : np.arange(*self.param_inds[5:7])
        }

    def transform(self, params):

        cap_cells = self.cap.get_cells_in_cap(params[self.inds["cap"]])
        flt_cells, flt_cols = self.channel.get_cells_in_channel(params[self.inds["channel"]])

        perms_ext = self.grf_ext.get_perms(params[self.inds["grf_ext"]])
        perms_flt = self.grf_flt.get_perms(params[self.inds["grf_flt"]])
        perms_cap = self.grf_cap.get_perms(params[self.inds["grf_cap"]])

        perms_ext = self.grf_ext.apply_level_set(perms_ext, self.level_width)
        perms_flt = self.grf_flt.apply_level_set(perms_flt, self.level_width)
        perms_cap = self.grf_cap.apply_level_set(perms_cap, self.level_width)

        perms = np.copy(perms_ext)
        perms[flt_cells] = perms_flt[flt_cells]
        perms[cap_cells] = perms_cap[cap_cells]

        upflow_weightings = self.cap.get_upflow_weightings(params[self.inds["cap"]])

        upflows = self.grf_upflow.get_upflows(params[self.inds["grf_upflow"]])
        upflows *= upflow_weightings
        upflows = upflows[flt_cols]

        return perms, upflows, flt_cols

    def sample(self, n=1):
        return np.random.normal(n, self.num_params)

"""
Model parameters
"""

mesh = models.IrregularMesh(MESH_NAME)
mesh.load_fem_mesh()

# TODO: add feedzone locations
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

# Mean of the (log-)permeabilities in the exterior, fault and clay cap
mu_perm_ext = -14
mu_perm_flt = -13
mu_perm_cap = -16

# Bounds for marginal standard deviations and x, y, z lengthscales
bounds_perm_ext = [(0.25, 0.50), (1000, 1500), (1000, 1500), (200, 400)]
bounds_perm_flt = [(0.20, 0.30), (1000, 1500), (1000, 1500), (200, 400)]
bounds_perm_cap = [(0.20, 0.30), (1000, 1500), (1000, 1500), (200, 400)]

grf_2d = grfs.MaternField2D(mesh)
grf_3d = grfs.MaternField3D(mesh)

# Generate the clay cap and permeability fields
clay_cap = ClayCap(mesh, bounds_geom_cap, n_terms, coef_sds)
perm_field_ext = PermeabilityField(mesh, grf_3d, mu_perm_ext, bounds_perm_ext)
perm_field_flt = PermeabilityField(mesh, grf_3d, mu_perm_flt, bounds_perm_flt)
perm_field_cap = PermeabilityField(mesh, grf_3d, mu_perm_cap, bounds_perm_cap)

"""
Channel
"""

# Bounds for amplitude, period, angle, intercept, width
bounds_channel = [(100, 200), (500, 1200), (-np.pi/8, np.pi/8), 
                  (500, 1000), (75, 150)]

mu_upflow = 0 # TODO: figure out what this should be

# Bounds for marginal standard deviations and x, y lengthscales
# TODO: figure out what sigma should be (depends on mesh too of course)
bounds_upflow = [(1e-8, 1e-8), (200, 400), (200, 400)]

channel = Channel(mesh, bounds_channel)
upflow_field = UpflowField(mesh, grf_2d, mu_upflow, bounds_upflow)

"""
Prior
"""

level_width = 0.25

prior = ChannelPrior(mesh, clay_cap, channel, perm_field_ext, 
                     perm_field_flt, perm_field_cap, 
                     upflow_field, level_width)

"""
Timestepping 
"""

dt = consts.SECS_PER_WEEK
tmax = 52 * consts.SECS_PER_WEEK

"""
Model functions
"""

def run_model(params):

    perms, upflows, upflow_cells = prior.transform(params)
    m = models.ChannelModel(path, mesh, perms, feedzones, upflows, dt, tmax)
    return m.run()