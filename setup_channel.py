"""Setup script for 3D channel model."""

from itertools import product
import numpy as np
from scipy import stats
from GeothermalEnsembleMethods import consts, grfs, likelihood, models

# np.random.seed(2)

MESH_NAME = "models/channel/gCH"
MODEL_NAME = "models/channel/CH"

"""
Classes
"""

# TODO: move somewhere else
def gauss_to_unif(x, lb, ub):
    return lb + stats.norm.cdf(x) * (ub - lb)

class PermeabilityField():

    def __init__(self, mesh, grf, mu, bounds):
        
        self.mesh = mesh
        self.grf = grf

        self.mu = mu
        self.bounds = bounds

        self.n_params = self.mesh.fem_mesh.n_points + len(self.bounds)

    def get_perms(self, ps):
        """Returns the set of permeabilities that correspond to a given 
        set of (whitened) parameters."""

        hyperps, W = ps[:4], ps[4:]
        hyperps = [gauss_to_unif(p, *bnds)
                   for p, bnds in zip(hyperps, self.bounds)]
        
        X = self.grf.generate_field(W, *hyperps)
        return self.mu + self.grf.G @ X

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

        self.n_hyperps = len(bounds)
        self.n_params = self.n_hyperps + self.mesh.m.num_columns

    def get_upflows(self, params):
        """Returns the set of upflows that correspond to a given set of 
        (whitened) parameters."""

        hyperparams, W = params[:self.n_hyperps], params[self.n_hyperps:]
        hyperparams = [gauss_to_unif(p, *bnds) 
                       for p, bnds in zip(hyperparams, self.bounds)]
        
        X = self.grf.generate_field(W, *hyperparams)
        return self.mu + X

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

        cols = [col for col in self.mesh.m.column if in_channel(*col.centre, *pars)]
        cells = [cell for col in cols for cell in col.cell]
        return cells, cols

class ClayCap():

    def __init__(self, mesh, bounds, n_terms, coef_sds):
        
        self.cell_centres = np.array([c.centre for c in mesh.m.cell])
        self.col_centres = np.array([c.centre for c in mesh.m.column])

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
    
    def get_upflow_weightings(self, params, mesh):
        """Returns a list of values by which the upflow in each column 
        should be multiplied."""

        cx, cy, *_ = self.get_cap_params(params)[0]

        s = 800 # TODO: make this into an input

        col_centres = [col.centre for col in mesh.m.column]
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

        self.param_counts = [0, cap.n_params, channel.n_params, 
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

        params = np.squeeze(params)

        cap_cell_inds = self.cap.get_cells_in_cap(params[self.inds["cap"]])

        perms_ext = self.grf_ext.get_perms(params[self.inds["grf_ext"]])
        perms_flt = self.grf_flt.get_perms(params[self.inds["grf_flt"]])
        perms_cap = self.grf_cap.get_perms(params[self.inds["grf_cap"]])

        perms_ext = self.grf_ext.apply_level_set(perms_ext, self.level_width)
        perms_flt = self.grf_flt.apply_level_set(perms_flt, self.level_width)
        perms_cap = self.grf_cap.apply_level_set(perms_cap, self.level_width)

        fault_cells, fault_cols = self.channel.get_cells_in_channel(params[self.inds["channel"]])
        fault_cell_inds = [c.index for c in fault_cells]
        fault_col_inds = [c.index for c in fault_cols]

        perms = np.copy(perms_ext)
        perms[fault_cell_inds] = perms_flt[fault_cell_inds]
        perms[cap_cell_inds] = perms_cap[cap_cell_inds]

        upflow_weightings = self.cap.get_upflow_weightings(params[self.inds["cap"]], self.mesh)

        upflow_rates = self.grf_upflow.get_upflows(params[self.inds["grf_upflow"]])
        upflow_rates *= upflow_weightings
        upflow_rates = upflow_rates[fault_col_inds]
        upflow_cells = [col.cell[-1] for col in fault_cols]

        upflows = [models.MassUpflow(c, max(r, 0.0)) 
                   for c, r in zip(upflow_cells, upflow_rates)]

        return perms, upflows

    def sample(self, n=1):
        return np.random.normal(size=(n, self.num_params))

"""
Model parameters
"""

mesh = models.IrregularMesh(MESH_NAME)

# TODO: add feedzone locations and specify production rates
feedzone_locs = [(500.0, 500.0, -500.0)]
feedzone_cells = [mesh.m.find(loc) for loc in feedzone_locs]
feedzone_rates = [0.0]
feedzones = [models.Feedzone(cell, rate) 
             for (cell, rate) in zip(feedzone_cells, feedzone_rates)]

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
bounds_perm_ext = [(0.4, 0.8), (1500, 2000), (1500, 2000), (200, 800)]
bounds_perm_flt = [(0.20, 0.30), (1500, 2000), (1500, 2000), (200, 800)]
bounds_perm_cap = [(0.20, 0.30), (1500, 2000), (1500, 2000), (200, 800)]

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
bounds_channel = [(-200, 200), (500, 1500), (-np.pi/8, np.pi/8), 
                  (650, 850), (75, 150)]

mu_upflow = 2.0e-6

# Bounds for marginal standard deviations and x, y lengthscales
bounds_upflow = [(0.5e-6, 1.0e-6), (200, 1000), (200, 1000)]

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

def plot_vals_on_mesh(mesh, vals):

    cell_centres = mesh.fem_mesh.cell_centers().points
    cell_vals = [vals[mesh.m.find(c, indices=True)] for c in cell_centres]
    
    mesh.fem_mesh["values"] = cell_vals
    slices = mesh.fem_mesh.slice_along_axis(n=5, axis="x")
    slices.plot(scalars="values", cmap="turbo")

def plot_upflows(mesh, upflows):

    values = np.zeros((mesh.m.num_cells, ))
    for upflow in upflows:
        values[upflow.cell.index] = upflow.rate

    mesh.m.layer_plot(value=values, colourmap="coolwarm")

def run_model(white_noise):

    perms, upflows = prior.transform(white_noise)

    plot_vals_on_mesh(mesh, perms)
    mesh.m.slice_plot(value=perms, colourmap="viridis")
    mesh.m.layer_plot(value=perms, colourmap="viridis")
    plot_upflows(mesh, upflows)

    m = models.ChannelModel(MODEL_NAME, mesh, perms, feedzones, upflows, dt, tmax)
    return m.run()

"""
Truth generation
"""

white_noise = prior.sample()
flag = run_model(white_noise)

print(flag)

import h5py

with h5py.File(f"{MODEL_NAME}_PR.h5", "r") as f:

    cell_inds = f["cell_index"][:, 0]
    src_inds = f["source_index"][:, 0]
    
    temps = f["cell_fields"]["fluid_temperature"][0][cell_inds]

plot_vals_on_mesh(mesh, temps)

# mesh.m.slice_plot(value=ts, colourmap="coolwarm")