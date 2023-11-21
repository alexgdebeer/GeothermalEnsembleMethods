"""Setup script for 2D vertical slice model."""

import numpy as np
from scipy import stats

from src.consts import SECS_PER_WEEK
from src.grfs import *
from src.models import *

np.random.seed(11) # 16, 38 not bad

DATA_FOLDER = "data/slice"
MODEL_FOLDER = "models/slice"

WS_TRUE_PATH = f"{DATA_FOLDER}/ws_true.txt"
PS_TRUE_PATH = f"{DATA_FOLDER}/ps_true.txt"
FS_TRUE_PATH = f"{DATA_FOLDER}/Fs_true.txt"
GS_TRUE_PATH = f"{DATA_FOLDER}/Gs_true.txt"
OBS_PATH = f"{DATA_FOLDER}/obs.txt"
COV_PATH = f"{DATA_FOLDER}/C_e.txt" # .npy?

"""
Prior
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
        for i, cell in enumerate(self.mesh.m.cell):

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
Meshes
"""

xmax = 1500.0
ymax = 60.0
zmax = 1500.0

nx_fine, nz_fine = 35, 35
nx_crse, nz_crse = 25, 25
ny = 1 # can probably get rid of this

tmax, nt = 104.0 * SECS_PER_WEEK, 24
dt = tmax / nt

n_blocks_fine = nx_fine * ny * nz_fine
n_blocks_crse = nx_crse * ny * nz_crse

mesh_name_fine = f"{MODEL_FOLDER}/gSL{n_blocks_fine}"
mesh_name_crse = f"{MODEL_FOLDER}/gSL{n_blocks_crse}"
model_name_fine = f"{MODEL_FOLDER}/SL{n_blocks_fine}"
model_name_crse = f"{MODEL_FOLDER}/SL{n_blocks_crse}"

mesh_fine = RegularMesh(mesh_name_fine, xmax, ymax, zmax, nx_fine, ny, nz_fine)
mesh_crse = RegularMesh(mesh_name_crse, xmax, ymax, zmax, nx_crse, ny, nz_crse)

"""
Wells and feedzones
"""

# TODO: make simpler
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
Observation times and locations
"""

# Natural state temperature observation locations
well_temp_depths = [-300, -500, -700, -900, -1100, -1300]
ns_temp_coords = np.array([[x, z] for x in well_xs for z in well_temp_depths])

# Pressure and enthalpy observation times
pr_obs_times = np.array([0, 13, 26, 39, 52]) * SECS_PER_WEEK

n_well_temp_depths = len(well_temp_depths)
n_pr_obs_times = len(pr_obs_times)

n_temp_obs = n_well_temp_depths * n_wells
n_pres_obs = n_pr_obs_times * n_wells
n_enth_obs = n_pr_obs_times * n_wells

"""
Constants and functions for extracting data
"""

# # Indices for extracting each state variable from the complete data
# temp_raw_inds_fine = np.arange(n_blocks_fine)
# pres_raw_inds_fine = np.arange(n_wells * (nt+1)) + temp_raw_inds_fine[-1] + 1
# enth_raw_inds_fine = np.arange(n_wells * (nt+1)) + pres_raw_inds_fine[-1] + 1

# temp_raw_inds_crse = np.arange(n_blocks_crse)
# pres_raw_inds_crse = np.arange(n_wells * (nt+1)) + temp_raw_inds_crse[-1] + 1
# enth_raw_inds_crse = np.arange(n_wells * (nt+1)) + pres_raw_inds_crse[-1] + 1

# # Indices for extracting each observations of each variable from the data

# def unpack_data_raw(Fs, mesh=mesh_crse, 
#                     temp_inds=temp_raw_inds_crse,
#                     pres_inds=pres_raw_inds_crse,
#                     enth_inds=enth_raw_inds_crse):
#     """Extracts the temperatures, pressures and enthalpies from a 
#     model run."""

#     temp = np.reshape(Fs[temp_inds], (mesh.nx, mesh.nz))
#     pres = np.reshape(Fs[pres_inds], (nt+1, n_wells))
#     enth = np.reshape(Fs[enth_inds], (nt+1, n_wells))
#     return temp, pres, enth

temp_obs_inds = np.arange(n_temp_obs)
pres_obs_inds = np.arange(n_pres_obs) + temp_obs_inds[-1] + 1
enth_obs_inds = np.arange(n_enth_obs) + pres_obs_inds[-1] + 1

def unpack_data_obs(Gs):
    """Extracts the temperatures, pressures and enthalpies from a 
    set of predictions."""

    temp = np.reshape(Gs[temp_obs_inds], (n_well_temp_depths, n_wells))
    pres = np.reshape(Gs[pres_obs_inds], (n_pr_obs_times, n_wells))
    enth = np.reshape(Gs[enth_obs_inds], (n_pr_obs_times, n_wells))
    return temp, pres, enth

"""
Model functions
"""

def G(p_i, mesh=mesh_crse, model_name=model_name_crse, 
      wells=wells_crse, upflow_cell=upflow_cell_crse):
    """Given a set of transformed parameters, forms and runs the 
    corresponding model, then returns the full model output and model
    predictions."""

    *logks, upflow_rate = p_i
    upflows = [MassUpflow(upflow_cell, upflow_rate)]
    
    model = Model2D(model_name, mesh, logks, wells, upflows, 
                    dt, tmax, ns_temp_coords, pr_obs_times)

    if (flag := model.run()) == ExitFlag.FAILURE: 
        return flag
    
    F_i, G_i = model.get_pr_data()
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

"""
Truth
"""

NOISE_LEVEL = 0.05 # Noise level as a percentage of the maximum of the raw data

def generate_data(G_t):

    temp_true, pres_true, enth_true = unpack_data_obs(G_t)

    cov_temp = (NOISE_LEVEL * np.max(temp_true)) ** 2 * np.eye(temp_true.size)
    cov_pres = (NOISE_LEVEL * np.max(pres_true)) ** 2 * np.eye(pres_true.size)
    cov_enth = (NOISE_LEVEL * np.max(enth_true)) ** 2 * np.eye(enth_true.size)

    C_e = sparse.block_diag((cov_temp, cov_pres, cov_enth)).toarray()
    y = np.random.multivariate_normal(G_t, C_e)
    return y, C_e

def generate_truth(mesh, model_name, wells, upflow_cell):
    """Generates the truth and observations using the fine model."""
    
    truth_dist = generate_prior(mesh, upflow_cell)

    ws_t = truth_dist.sample()
    ps_t = truth_dist.transform(ws_t)

    # TEMP: sanity check
    *ks_t, _ = ps_t
    mesh.m.slice_plot(value=ks_t, colourmap="viridis")

    Fs_t, Gs_t = G(ps_t, mesh, model_name, wells, upflow_cell)

    temp_true = Fs_t[:(mesh.nx * mesh.nz)]
    mesh.m.slice_plot(value=temp_true, colourmap="coolwarm")

    y, C_e = generate_data(Gs_t)

    np.savetxt(WS_TRUE_PATH, ws_t)
    np.savetxt(PS_TRUE_PATH, ps_t)
    np.savetxt(FS_TRUE_PATH, Fs_t)
    np.savetxt(GS_TRUE_PATH, Gs_t)
    np.savetxt(OBS_PATH, y)
    np.savetxt(COV_PATH, C_e)

    return ws_t, ps_t, Fs_t, Gs_t, y, C_e

def read_truth():

    ws_t = np.genfromtxt(WS_TRUE_PATH)
    ps_t = np.genfromtxt(PS_TRUE_PATH)
    Fs_t = np.genfromtxt(FS_TRUE_PATH)
    Gs_t = np.genfromtxt(GS_TRUE_PATH)
    y = np.genfromtxt(OBS_PATH)
    C_e = np.genfromtxt(COV_PATH)
    
    return ws_t, ps_t, Fs_t, Gs_t, y, C_e

ws_t, ps_t, Fs_t, Gs_t, y, C_e = generate_truth(
    mesh_fine, model_name_fine, wells_fine, upflow_cell_fine
)