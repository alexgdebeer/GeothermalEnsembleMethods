import numpy as np

from src.consts import SECS_PER_WEEK
from src.data_handlers import DataHandler3D
from src.grfs import *
from src.models import *
from src.priors import FaultPrior

np.random.seed(256)

MODEL_FOLDER = "models/fault"
DATA_FOLDER = "data/fault"

TRUTH_FOLDER = f"{DATA_FOLDER}/truth"
MATERN_FOLDER = f"{DATA_FOLDER}/matern"

W_TRUE_PATH = f"{TRUTH_FOLDER}/w_true.npy"
P_TRUE_PATH = f"{TRUTH_FOLDER}/p_true.npy"
F_TRUE_PATH = f"{TRUTH_FOLDER}/F_true.npy"
G_TRUE_PATH = f"{TRUTH_FOLDER}/G_true.npy"
OBS_PATH = f"{TRUTH_FOLDER}/obs.npy"
COV_PATH = f"{TRUTH_FOLDER}/C_e.npy"

READ_TRUTH = True

MESH_PATH_CRSE = f"{MODEL_FOLDER}/gFL8788"
MESH_PATH_FINE = f"{MODEL_FOLDER}/gFL13383"

MODEL_PATH_CRSE = f"{MODEL_FOLDER}/FL8788"
MODEL_PATH_FINE = f"{MODEL_FOLDER}/FL13383"

MATERN_PATH_CRSE = f"{MATERN_FOLDER}/FL8788"
MATERN_PATH_FINE = f"{MATERN_FOLDER}/FL13383"

"""
Model parameters
"""

tmax, nt = 104.0 * SECS_PER_WEEK, 24
dt = tmax / nt

mesh_crse = IrregularMesh(MESH_PATH_CRSE)
mesh_fine = IrregularMesh(MESH_PATH_FINE)

well_xs = [1800, 3000, 4000, 2800, 2100, 3800,  3300, 1400, 4600]
well_ys = [3100, 3000,  3000, 1900, 3900, 2100, 4100, 1500, 4600]
n_wells = len(well_xs)
well_depths = [-2600] * n_wells
feedzone_depths = [-1200] * n_wells
feedzone_rates = [[[0.0, -0.25], [2 * tmax/4, -0.5]]] * n_wells

wells_crse = [Well(x, y, depth, mesh_crse, fz_depth, fz_rate)
              for (x, y, depth, fz_depth, fz_rate) 
              in zip(well_xs, well_ys, well_depths, feedzone_depths, feedzone_rates)]

wells_fine = [Well(x, y, depth, mesh_fine, fz_depth, fz_rate)
              for (x, y, depth, fz_depth, fz_rate) 
              in zip(well_xs, well_ys, well_depths, feedzone_depths, feedzone_rates)]

"""
Clay cap
"""

# Bounds for depth of clay cap, width (horizontal and vertical) and dip
bounds_geom_cap = [(-900, -775), (1400, 1600), (200, 300), (300, 600)]
n_terms = 5
coef_sds = 5

# Bounds for marginal standard deviations and x, y, z lengthscales
bounds_perm_ext = [(0.75, 1.5), (4000, 8000), (4000, 8000), (800, 3000)]
bounds_perm_flt = [(0.5, 1.0), (4000, 8000), (4000, 8000), (800, 3000)]
bounds_perm_cap = [(0.5, 1.0), (4000, 8000), (4000, 8000), (800, 3000)]

# Generate the Matern fields corresponding to each 
grf_2d_crse = MaternField2D(mesh_crse)
grf_2d_fine = MaternField2D(mesh_fine)

grf_3d_crse = MaternField3D(mesh_crse, folder=MATERN_PATH_CRSE)
grf_3d_fine = MaternField3D(mesh_fine, folder=MATERN_PATH_FINE)

def levels_ext(p):
    """Level set mapping for all non-fault and non-cap regions."""
    if   p < -1.5: return -15.5
    elif p < -0.5: return -15.0
    elif p <  0.5: return -14.5
    elif p <  1.5: return -14.0
    else: return -13.5

def levels_flt(p):
    """Level set mapping for fault."""
    if   p < -0.5: return -13.5
    elif p <  0.5: return -13.0
    else: return -12.5

def levels_cap(p):
    """Level set mapping for clay cap."""
    if   p < -0.5: return -17.0
    elif p <  0.5: return -16.5
    else: return -16.0

# Generate the clay cap and permeability fields
clay_cap_crse = ClayCap(mesh_crse, bounds_geom_cap, n_terms, coef_sds)
clay_cap_fine = ClayCap(mesh_fine, bounds_geom_cap, n_terms, coef_sds)

perm_field_ext_crse = PermField(mesh_crse, grf_3d_crse, bounds_perm_ext, levels_ext)
perm_field_flt_crse = PermField(mesh_crse, grf_3d_crse, bounds_perm_flt, levels_flt)
perm_field_cap_crse = PermField(mesh_crse, grf_3d_crse, bounds_perm_cap, levels_cap)
perm_field_ext_fine = PermField(mesh_fine, grf_3d_fine, bounds_perm_ext, levels_ext)
perm_field_flt_fine = PermField(mesh_fine, grf_3d_fine, bounds_perm_flt, levels_flt)
perm_field_cap_fine = PermField(mesh_fine, grf_3d_fine, bounds_perm_cap, levels_cap)

"""
Fault
"""

bounds_fault = [(1500, 4500), (1500, 4500)]

mu_upflow = 2.5e-4
bounds_upflow = [(0.1e-4, 0.5e-4), (500, 500), (500, 500)]

fault_crse = Fault(mesh_crse, bounds_fault)
fault_fine = Fault(mesh_fine, bounds_fault)

upflow_field_crse = UpflowField(mesh_crse, grf_2d_crse, mu_upflow, bounds_upflow)
upflow_field_fine = UpflowField(mesh_fine, grf_2d_fine, mu_upflow, bounds_upflow)

# Parameter associated with Gaussian kernel for upflows
ls_upflows = 1600

"""
Observations
"""

temp_obs_zs = [-800, -1100, -1400, -1700, -2000, -2300, -2600]
temp_obs_cs = np.array([[x, y, z] 
                        for z in temp_obs_zs 
                        for x, y in zip(well_xs, well_ys)])

prod_obs_ts = np.array([0, 13, 26, 39, 52]) * SECS_PER_WEEK

data_handler_crse = DataHandler3D(mesh_crse, wells_crse, temp_obs_cs, prod_obs_ts, tmax, nt)
data_handler_fine = DataHandler3D(mesh_fine, wells_fine, temp_obs_cs, prod_obs_ts, tmax, nt)

"""
Ensemble functions
"""

def generate_particle(p_i, num):
    name = f"{MODEL_PATH_CRSE}_{num}"
    logks_t, upflows_t = prior.split(p_i)
    model = Model3D(name, mesh_crse, logks_t, wells_crse, upflows_t, dt, tmax)
    return model

def get_result(particle: Model3D):
    F_i = particle.get_pr_data()
    G_i = data_handler_crse.get_obs(F_i)
    return F_i, G_i

"""
Prior
"""

prior = FaultPrior(
    mesh_crse, clay_cap_crse, fault_crse, 
    perm_field_ext_crse, perm_field_flt_crse, perm_field_cap_crse, 
    upflow_field_crse, ls_upflows)

"""
Truth generation
"""

noise_level = 0.05

truth_dist = FaultPrior(
    mesh_fine, clay_cap_fine, fault_fine, 
    perm_field_ext_fine, perm_field_flt_fine, perm_field_cap_fine, 
    upflow_field_fine, ls_upflows)

def generate_truth():

    w_t = truth_dist.sample()
    p_t = truth_dist.transform(w_t)

    logks_t, upflows_t = truth_dist.split(p_t)

    model_t = Model3D(
        MODEL_PATH_FINE, mesh_fine, 
        logks_t, wells_fine, upflows_t, dt, tmax
    )
    
    if model_t.run() != ExitFlag.SUCCESS:
        raise Exception("Truth failed to run.")

    F_t = model_t.get_pr_data()
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

Np = mesh_crse.m.num_cells + mesh_crse.m.num_columns
NF = mesh_crse.m.num_cells + 2 * n_wells * (nt+1)
NG = len(y)