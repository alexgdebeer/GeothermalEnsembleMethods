"""Setup script for 2D vertical slice model."""

import numpy as np
from scipy import sparse, stats
from scipy.interpolate import RegularGridInterpolator

from src.consts import SECS_PER_WEEK
from src.grfs import *
from src.models import *
from src import utils

np.random.seed(38)

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

class SlicePrior():

    def __init__(self, mesh, depth_shal, gp_boundary, 
                 grf_shal, grf_clay, grf_deep,
                 mass_rate_bounds):

        self.mesh = mesh

        self.depth_shal = depth_shal
        self.gp_boundary = gp_boundary
        
        self.grf_shal = grf_shal
        self.grf_clay = grf_clay
        self.grf_deep = grf_deep

        self.mass_rate_bounds = mass_rate_bounds

        self.param_counts = [0, gp_boundary.n_params, 
                             grf_shal.n_params, grf_clay.n_params, 
                             grf_deep.n_params, 1]

        self.n_params = sum(self.param_counts)
        self.param_inds = np.cumsum(self.param_counts)

        self.inds = {
            "boundary" : np.arange(*self.param_inds[0:2]),
            "grf_shal" : np.arange(*self.param_inds[1:3]),
            "grf_clay" : np.arange(*self.param_inds[2:4]),
            "grf_deep" : np.arange(*self.param_inds[3:5])
        }

    def combine_perms(self, boundary, perms_shal, perms_clay, perms_deep):

        perms = np.zeros((perms_shal.shape))
        for i, cell in enumerate(mesh.m.cell):

            cx, _, cz = cell.centre
            x_ind = np.abs(self.gp_boundary.xs - cx).argmin()

            if cz > self.depth_shal:
                perms[i] = perms_shal[i]
            elif cz > boundary[x_ind]:
                perms[i] = perms_clay[i]
            else: 
                perms[i] = perms_deep[i]

        return perms

    def transform_mass_rate(self, mass_rate):
        return self.mass_rate_bounds[0] + \
            np.ptp(self.mass_rate_bounds) * stats.norm.cdf(mass_rate)

    def transform(self, ws):

        ws = np.squeeze(ws)

        perms_shal = self.grf_shal.get_perms(ws[self.inds["grf_shal"]])
        perms_clay = self.grf_clay.get_perms(ws[self.inds["grf_clay"]])
        perms_deep = self.grf_deep.get_perms(ws[self.inds["grf_deep"]])

        perms_shal = self.grf_shal.level_set(perms_shal)
        perms_clay = self.grf_clay.level_set(perms_clay)
        perms_deep = self.grf_deep.level_set(perms_deep)

        boundary = self.gp_boundary.transform(ws[self.inds["boundary"]])
        perms = self.combine_perms(boundary, perms_shal, perms_clay, perms_deep)

        mass_rate = ws[-1]
        mass_rate = self.transform_mass_rate(mass_rate)
        
        ps = np.append(perms, mass_rate)
        return ps

    def sample(self, n=1):
        return np.random.normal(size=(self.n_params, n))

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

matern_field = MaternField2D(mesh, model_type=ModelType.MODEL2D)

mass_rate_bounds = (1.0e-1 / upflow_cell.volume, 2.0e-1 / upflow_cell.volume)

depth_shal = -60.0
cells_shal = [c for c in mesh.m.cell if c.centre[-1] > depth_shal]
cells_deep = [c for c in mesh.m.cell if c.centre[-1] <= depth_shal]

mu_boundary = -350
std_boundary = 80
l_boundary = 500
gp_boundary = Gaussian1D(mu_boundary, std_boundary, l_boundary, mesh.xs)

bounds_shal = [(0.5, 1.5), (1000, 2000), (300, 500)]
bounds_clay = [(1.0, 1.2), (1000, 2000), (300, 500)]
bounds_deep = [(1.0, 1.2), (1000, 2000), (300, 500)]

def levels_clay(p):
    if   p < -0.5: return -16.5
    elif p <  0.5: return -16.0
    else: return -15.5

def levels_exterior(p):
    if   p < -1.5: return -15.0
    elif p < -0.5: return -14.5
    elif p <  0.5: return -14.0
    elif p <  1.5: return -13.5
    else: return -13.0

grf_shal = PermeabilityField(mesh, matern_field, bounds_shal, 
                             levels_exterior, model_type=ModelType.MODEL2D)
grf_clay = PermeabilityField(mesh, matern_field, bounds_clay, 
                             levels_clay, model_type=ModelType.MODEL2D)
grf_deep = PermeabilityField(mesh, matern_field, bounds_deep, 
                             levels_exterior, model_type=ModelType.MODEL2D)

prior = SlicePrior(mesh, depth_shal, gp_boundary, 
                   grf_shal, grf_clay, grf_deep, mass_rate_bounds)

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

*ks_t, upflow_rate_t = p_t

mesh.m.slice_plot(value=ks_t, colourmap="viridis")

try:
    F_t = np.genfromtxt(OUTPUTS_PATH)
    G_t = np.genfromtxt(REDUCED_OUTPUTS_PATH)

except FileNotFoundError:
    utils.info("True model outputs not found. Running model...")

    F_t = F(p_t)
    G_t = G(F_t)
    np.savetxt(OUTPUTS_PATH, F_t)
    np.savetxt(REDUCED_OUTPUTS_PATH, G_t)

NF = len(F_t)
NG = len(G_t)

ts_t, ps_t, es_t = unpack_data_obs(G_t)

ns_temps = unpack_data_raw(F_t)[0]
mesh.m.slice_plot(value=ns_temps.flatten(), colourmap="coolwarm")

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
    np.savetxt(OBSERVATIONS_PATH, y)