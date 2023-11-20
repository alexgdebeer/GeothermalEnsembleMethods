"""Setup script for 2D vertical slice model."""

import numpy as np
from scipy import sparse, stats
from scipy.interpolate import RegularGridInterpolator

from src.consts import SECS_PER_WEEK
from src.grfs import Gaussian1D, Gaussian2D
from src.models import RegularMesh, Well, MassUpflow, Model2D, ExitFlag
from src import utils

np.random.seed(1)

DATA_FOLDER = "data/slice"
MODEL_FOLDER = "models/slice"

WHITE_PARAMS_PATH = f"{DATA_FOLDER}/white_params_true.txt"
PARAMS_PATH = f"{DATA_FOLDER}/params_true.txt"
OUTPUTS_PATH = f"{DATA_FOLDER}/full_outputs_true.txt"
REDUCED_OUTPUTS_PATH = f"{DATA_FOLDER}/reduced_outputs_true.txt"
OBSERVATIONS_PATH = f"{DATA_FOLDER}/observations.txt"

"""
Classes
"""

# TODO: add a num_params attribute to this if it doesn't have one already.
class SlicePrior():

    def __init__(self, mesh, d_depth_clay, 
                 d_perm_shal, d_perm_clay, d_perm_deep,
                 mass_rate_bounds, level_width):

        self.mesh = mesh

        self.d_depth_clay = d_depth_clay
        self.d_perm_shal = d_perm_shal
        self.d_perm_clay = d_perm_clay
        self.d_perm_deep = d_perm_deep

        self.generate_mu()
        self.generate_cov()

        self.chol_inv = np.linalg.cholesky(self.cov)
        self.chol = np.linalg.inv(self.chol_inv)

        self.mass_rate_bounds = mass_rate_bounds
        self.level_width = level_width

        self.is_depth_clay = np.array(range(self.d_depth_clay.nx))
        self.is_perm_shal = np.array(range(self.is_depth_clay[-1] + 1,
                                           self.is_depth_clay[-1] + 1 + \
                                            self.d_perm_shal.n_cells))
        self.is_perm_clay = np.array(range(self.is_perm_shal[-1] + 1,
                                           self.is_perm_shal[-1] + 1 + \
                                            self.d_perm_clay.n_cells))
        self.is_perm_deep = np.array(range(self.is_perm_clay[-1] + 1,
                                           self.is_perm_clay[-1] + 1 + \
                                            self.d_perm_deep.n_cells))
        
        self.num_params = -1 # TODO: add this properly

    def generate_mu(self):
        self.mu = np.hstack((self.d_depth_clay.mu, 
                             self.d_perm_shal.mu,
                             self.d_perm_clay.mu, 
                             self.d_perm_deep.mu, 
                             np.array([0.0])))

    def generate_cov(self):
        self.cov = sparse.block_diag((self.d_depth_clay.cov,
                                      self.d_perm_shal.cov,
                                      self.d_perm_clay.cov,
                                      self.d_perm_deep.cov,
                                      [1.0])).toarray()

    def apply_level_sets(self, perms):

        min_level = np.floor(np.min(perms))
        max_level = np.ceil(np.max(perms)) + 1e-8

        levels = np.arange(min_level, max_level, self.level_width)
        return np.array([levels[np.abs(levels - p).argmin()] for p in perms])

    def transform_mass_rate(self, mass_rate):
        return self.mass_rate_bounds[0] + \
            np.ptp(self.mass_rate_bounds) * stats.norm.cdf(mass_rate)

    def transform_perms(self, perms):

        perms = np.array(perms)

        clay_boundary = perms[self.is_depth_clay]
        perm_shal = perms[self.is_perm_shal]
        perm_clay = perms[self.is_perm_clay]
        perm_deep = perms[self.is_perm_deep]

        perms = np.copy(perm_shal)

        for i in range(self.d_perm_deep.n_cells):
            
            cell = self.d_perm_deep.cells[i]
            cx, cz = cell.centre[0], cell.centre[-1]
            x_ind = np.abs(self.d_depth_clay.xs - cx).argmin()

            if clay_boundary[x_ind] < cz:
                perms = np.append(perms, perm_clay[i])
            else: 
                perms = np.append(perms, perm_deep[i])

        return perms

    def sample(self, n=1):
        ws = np.random.normal(size=(len(self.mu), n))
        return self.mu[:, np.newaxis] + self.chol_inv @ ws

    def transform(self, ws):

        *perms, mass_rate = np.squeeze(ws)
        
        perms = self.transform_perms(perms)
        perms = 10 ** self.apply_level_sets(perms)
        mass_rate = self.transform_mass_rate(mass_rate)
        
        ps = np.append(perms, mass_rate)
        return ps

"""
Model parameters
"""

xmax, nx = 1500.0, 25
ymax, ny = 60.0, 1
zmax, nz = 1500.0, 25
tmax, nt = 104.0 * SECS_PER_WEEK, 24

dt = tmax / nt 
obs_time_inds = [0, 3, 6, 9, 12]
n_tobs = len(obs_time_inds)

n_blocks = nx * nz
n_wells = 5
n_temps_per_well = 6

mesh_name = f"{MODEL_FOLDER}/gSL{n_blocks}"
model_name = f"{MODEL_FOLDER}/SL{n_blocks}"

mesh = RegularMesh(mesh_name, xmax, ymax, zmax, nx, ny, nz)

well_xs = [200, 475, 750, 1025, 1300]
well_depths = [-1300] * n_wells
feedzone_depths = [-500] * n_wells
feedzone_rates = [-2.0] * n_wells

wells = [Well(x, 0.5*ymax, depth, mesh, fz_depth, fz_rate)
         for (x, depth, fz_depth, fz_rate) 
         in zip(well_xs, well_depths, feedzone_depths, feedzone_rates)]

upflow_loc = (0.5*xmax, 0.5*ymax, -zmax + 0.5*mesh.dz)
upflow_cell = mesh.m.find(upflow_loc)

"""
Constants and functions for extracting data
"""

n_temp_vals = n_blocks 
n_pressure_vals = n_wells * (nt + 1)
n_enthalpy_vals = n_wells * (nt + 1)

n_temp_obs =  n_wells * n_temps_per_well 
n_pressure_obs = len(obs_time_inds) * n_wells
n_enthalpy_obs = len(obs_time_inds) * n_wells

ns_obs_xs = np.array([x for x in well_xs for _ in range(n_temps_per_well)])
ns_obs_zs = np.array([-300, -500, -700, -900, -1100, -1300] * n_wells)

ts_val_is = list(range(n_temp_vals))
ps_val_is = list(range(ts_val_is[-1]+1, ts_val_is[-1]+1+n_pressure_vals))
es_val_is = list(range(ps_val_is[-1]+1, ps_val_is[-1]+1+n_enthalpy_vals))

ts_obs_is = list(range(n_temp_obs))
ps_obs_is = list(range(ts_obs_is[-1]+1, ts_obs_is[-1]+1+n_pressure_obs))
es_obs_is = list(range(ps_obs_is[-1]+1, ps_obs_is[-1]+1+n_enthalpy_obs))

def unpack_data_raw(fs):
    ts = np.reshape(fs[ts_val_is], (mesh.nx, mesh.nz))
    ps = np.reshape(fs[ps_val_is], (nt+1, n_wells))
    es = np.reshape(fs[es_val_is], (nt+1, n_wells))
    return ts, ps, es

def unpack_data_obs(gs):
    ts = np.reshape(gs[ts_obs_is], (n_temps_per_well, n_wells))
    ps = np.reshape(gs[ps_obs_is], (n_tobs, n_wells))
    es = np.reshape(gs[es_obs_is], (n_tobs, n_wells))
    return ts, ps, es

"""
Model functions
"""

def F(p_i):
    """Given a set of transformed parameters, forms and runs the 
    corresponding model, then returns the full model output."""

    *logks, upflow_rate = p_i
    upflows = [MassUpflow(upflow_cell, upflow_rate)]
    
    model = Model2D(model_name, mesh, logks, wells, upflows, dt, tmax)

    if (flag := model.run()) == ExitFlag.FAILURE: 
        return flag
    return model.get_pr_data()

def G(F_i):
    """Given a set of complete model outputs, returns the values 
    corresponding to the observations."""

    if type(F_i) == ExitFlag: 
        return F_i
    
    ts, ps, es = unpack_data_raw(F_i)

    # TODO: check this
    ts_interp = RegularGridInterpolator((mesh.xs, -mesh.zs), ts.T)
    ts = ts_interp(np.vstack((ns_obs_xs, -ns_obs_zs)).T).flatten()

    ps = ps[obs_time_inds, :].flatten()
    es = es[obs_time_inds, :].flatten()

    return np.concatenate((ts, ps, es))

"""
Prior
"""

mass_rate_bounds = (1.0e-1 / upflow_cell.volume, 2.0e-1 / upflow_cell.volume)
level_width = 0.25 # TODO: think about this

depth_shal = -60.0
cells_shal = [c for c in mesh.m.cell if c.centre[-1] > depth_shal]
cells_deep = [c for c in mesh.m.cell if c.centre[-1] <= depth_shal]

d_depth_clay = Gaussian1D(-350, 80, 500, mesh.xs)
d_perm_shal = Gaussian2D(-14, 0.25, 1500, 200, cells_shal)
d_perm_clay = Gaussian2D(-16, 0.25, 1500, 200, cells_deep)
d_perm_deep = Gaussian2D(-14, 0.50, 1500, 200, cells_deep)

prior = SlicePrior(mesh, d_depth_clay, 
                   d_perm_shal, d_perm_clay, d_perm_deep,
                   mass_rate_bounds, level_width)

"""
Truth
"""

try:
    w_t = np.genfromtxt(WHITE_PARAMS_PATH)
    p_t = np.genfromtxt(PARAMS_PATH)

except FileNotFoundError:
    utils.info("True parameters not found. Sampling a new set...")

    w_t = prior.sample()
    p_t = prior.transform(w_t)
    np.savetxt(WHITE_PARAMS_PATH, w_t)
    np.savetxt(PARAMS_PATH, p_t)

p_t = prior.transform(w_t)
*ks_t, upflow_rate_t = p_t
logks_t = np.reshape(np.log10(ks_t), (mesh.nx, mesh.nz))

try:
    F_t = np.genfromtxt(OUTPUTS_PATH)
    G_t = np.genfromtxt(REDUCED_OUTPUTS_PATH)

except FileNotFoundError:
    utils.info("True model outputs not found. Running model...")

    F_t = F(p_t)
    G_t = G(F_t)
    np.savetxt(OUTPUTS_PATH, F_t)

NF = len(F_t)
NG = len(G_t)

ts_t, ps_t, es_t = unpack_data_obs(G_t)

"""
Errors and observations
"""

max_t = ts_t.max()
max_p = ps_t.max()
max_e = es_t.max()

n_ts_obs = ts_t.size
n_ps_obs = ps_t.size
n_es_obs = es_t.size

cov_ts = (0.02 * max_t) ** 2 * np.eye(n_ts_obs)
cov_ps = (0.02 * max_p) ** 2 * np.eye(n_ps_obs)
cov_es = (0.02 * max_e) ** 2 * np.eye(n_es_obs)

C_e = sparse.block_diag((cov_ts, cov_ps, cov_es)).toarray()

try:
    y = np.genfromtxt(OBSERVATIONS_PATH)
except FileNotFoundError:
    utils.info("Observations not found. Generating...")
    y = np.random.multivariate_normal(G_t, C_e)

# TODO: add some functions for plotting the truth (true convective 
# plume, permeabilities, data, maybe vapour saturations...?)