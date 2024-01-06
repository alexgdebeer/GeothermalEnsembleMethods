"""Setup script for 2D vertical slice model."""

import numpy as np
from scipy import sparse

from src.consts import SECS_PER_WEEK
from src.data_handlers import *
from src.grfs import *
from src.models import *
from src.priors import SlicePrior

np.random.seed(24)

DATA_FOLDER = "data/slice"
MODEL_FOLDER = "models/slice"

W_TRUE_PATH = f"{DATA_FOLDER}/w_true.npy"
P_TRUE_PATH = f"{DATA_FOLDER}/p_true.npy"
F_TRUE_PATH = f"{DATA_FOLDER}/F_true.npy"
G_TRUE_PATH = f"{DATA_FOLDER}/G_true.npy"
OBS_PATH = f"{DATA_FOLDER}/obs.npy"
COV_PATH = f"{DATA_FOLDER}/C_e.npy"

READ_TRUTH = True

"""
Meshes
"""

xmax = 1500.0
ymax = 60.0
zmax = 1500.0

nx_fine, nz_fine = 35, 35
nx_crse, nz_crse = 25, 25

tmax, nt = 104.0 * SECS_PER_WEEK, 24
dt = tmax / nt

n_blocks_fine = nx_fine * nz_fine
n_blocks_crse = nx_crse * nz_crse

mesh_name_fine = f"{MODEL_FOLDER}/gSL{n_blocks_fine}"
mesh_name_crse = f"{MODEL_FOLDER}/gSL{n_blocks_crse}"
model_name_fine = f"{MODEL_FOLDER}/SL{n_blocks_fine}"
model_name_crse = f"{MODEL_FOLDER}/SL{n_blocks_crse}"

mesh_fine = SliceMesh(mesh_name_fine, xmax, ymax, zmax, nx_fine, nz_fine)
mesh_crse = SliceMesh(mesh_name_crse, xmax, ymax, zmax, nx_crse, nz_crse)

"""
Wells and feedzones
"""

well_xs = [200, 475, 750, 1025, 1300]
n_wells = len(well_xs)
well_depths = [-1300] * n_wells
feedzone_depths = [-500] * n_wells
feedzone_rates = [-2.0] * n_wells

wells_fine = [Well(x, 0.5*ymax, depth, mesh_fine, fz_depth, fz_rate)
              for (x, depth, fz_depth, fz_rate) 
              in zip(well_xs, well_depths, feedzone_depths, feedzone_rates)]

wells_crse = [Well(x, 0.5*ymax, depth, mesh_crse, fz_depth, fz_rate)
              for (x, depth, fz_depth, fz_rate) 
              in zip(well_xs, well_depths, feedzone_depths, feedzone_rates)]

upflow_loc_fine = (xmax/2, ymax/2, -zmax + mesh_fine.dz/2)
upflow_loc_crse = (xmax/2, ymax/2, -zmax + mesh_crse.dz/2)

upflow_cell_fine = mesh_fine.m.find(upflow_loc_fine)
upflow_cell_crse = mesh_crse.m.find(upflow_loc_crse)

"""
Observations
"""

temp_obs_zs = np.array([-300, -500, -700, -900, -1100, -1300])
temp_obs_cs = np.array([[x, z] for z in temp_obs_zs for x in well_xs])

prod_obs_ts = np.array([0, 13, 26, 39, 52]) * SECS_PER_WEEK

data_handler_crse = DataHandler2D(mesh_crse, wells_crse, temp_obs_cs, prod_obs_ts, tmax, nt)
data_handler_fine = DataHandler2D(mesh_fine, wells_crse, temp_obs_cs, prod_obs_ts, tmax, nt)

noise_level = 0.02

"""
Ensemble functions
"""

def generate_particle(p_i, num):
    name = f"{model_name_crse}_{num}"
    *logks, upflow_rate = p_i 
    upflows = [MassUpflow(upflow_cell_crse, upflow_rate)]
    model = Model2D(name, mesh_crse, logks, wells_crse, upflows, dt, tmax)
    return model

def get_result(particle: Model2D):
    F_i = particle.get_pr_data()
    G_i = data_handler_crse.get_obs(F_i)
    return F_i, G_i

"""
Prior
"""

depth_shal = -60.0

mu_boundary = -350
std_boundary = 80
l_boundary = 500

bounds_shal = [(0.5, 1.0), (1000, 2000), (200, 500)]
bounds_clay = [(0.5, 1.0), (1000, 2000), (200, 500)]
bounds_deep = [(0.75, 1.25), (1000, 2000), (200, 500)]

def levels_clay(p):
    """Level set mapping for clay cap."""
    if   p < -0.5: return -17.0
    elif p <  0.5: return -16.5
    else: return -16.0

def levels_exterior(p):
    """Level set mapping for shallow and deep (high-permeability) 
    regions."""
    if   p < -1.5: return -15.0
    elif p < -0.5: return -14.5
    elif p <  0.5: return -14.0
    elif p <  1.5: return -13.5
    else: return -13.0

def generate_prior(mesh, upflow_cell):

    grf = MaternField2D(mesh, model_type=ModelType.MODEL2D)

    mass_rate_bounds = (1.0e-1 / upflow_cell.column.area, 2.0e-1 / upflow_cell.column.area)

    gp_boundary = Gaussian1D(mu_boundary, std_boundary, l_boundary, mesh.xs)

    grf_shal = PermField(mesh, grf, bounds_shal, levels_exterior, model_type=ModelType.MODEL2D)
    grf_clay = PermField(mesh, grf, bounds_clay, levels_clay, model_type=ModelType.MODEL2D)
    grf_deep = PermField(mesh, grf, bounds_deep, levels_exterior, model_type=ModelType.MODEL2D)

    prior = SlicePrior(mesh, depth_shal, gp_boundary, grf_shal, grf_clay, grf_deep, mass_rate_bounds)
    
    return prior

prior = generate_prior(mesh_crse, upflow_cell_crse)

"""
Truth
"""

truth_dist = generate_prior(mesh_fine, upflow_cell_fine)

def generate_truth():
    """Generates the truth and observations using the fine model."""

    w_t = truth_dist.sample()
    p_t = truth_dist.transform(w_t)

    # TEMP: sanity check
    mesh_fine.m.slice_plot(value=p_t[:-1], colourmap="viridis")

    *logks_t, upflow_rate_t = p_t
    upflows = [MassUpflow(upflow_cell_fine, upflow_rate_t)]
    
    model = Model2D(model_name_fine, mesh_fine, logks_t, 
                    wells_fine, upflows, dt, tmax)

    model.run()
    F_t = model.get_pr_data()
    G_t = data_handler_fine.get_obs(F_t)

    np.save(W_TRUE_PATH, w_t)
    np.save(P_TRUE_PATH, p_t)
    np.save(F_TRUE_PATH, F_t)
    np.save(G_TRUE_PATH, G_t)

    return w_t, p_t, F_t, G_t

def read_truth():

    w_t = np.load(W_TRUE_PATH)
    p_t = np.load(P_TRUE_PATH)
    F_t = np.load(F_TRUE_PATH)
    G_t = np.load(G_TRUE_PATH)
    
    return w_t, p_t, F_t, G_t

def generate_data(G_t):

    temp_t, pres_t, enth_t = data_handler_fine.split_obs(G_t)

    cov_temp = (noise_level * np.max(temp_t)) ** 2 * np.eye(temp_t.size)
    cov_pres = (noise_level * np.max(pres_t)) ** 2 * np.eye(pres_t.size)
    cov_enth = (noise_level * np.max(enth_t)) ** 2 * np.eye(enth_t.size)

    C_e = sparse.block_diag((cov_temp, cov_pres, cov_enth)).toarray()
    y = np.random.multivariate_normal(G_t, C_e)

    np.save(OBS_PATH, y)
    np.save(COV_PATH, C_e)

    return y, C_e

def read_data():
    y = np.load(OBS_PATH)
    C_e = np.load(COV_PATH)
    return y, C_e

if READ_TRUTH:
    w_t, p_t, F_t, G_t = read_truth()
    y, C_e = read_data()

else:
    w_t, p_t, F_t, G_t = generate_truth()
    y, C_e = generate_data(G_t)

Np = n_blocks_crse + 1
NF = n_blocks_crse + 2 * n_wells * (nt + 1)
NG = len(y)