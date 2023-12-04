"""Setup script for 2D vertical slice model."""

import numpy as np
from scipy import sparse, stats 
from scipy.interpolate import RegularGridInterpolator

from src.consts import SECS_PER_WEEK
from src.grfs import *
from src.models import *

np.random.seed(9) # 11 not bad

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

    def get_hyperparams(self, ws):
        hps_shal = self.grf_shal.get_hyperparams(ws[self.inds["grf_shal"]])
        hps_clay = self.grf_clay.get_hyperparams(ws[self.inds["grf_clay"]])
        hps_deep = self.grf_deep.get_hyperparams(ws[self.inds["grf_deep"]])
        return hps_shal, hps_clay, hps_deep

class DataHandler():

    def __init__(self, mesh: SliceMesh, temp_obs_xs, temp_obs_zs,
                 prod_obs_ts, tmax, nt):

        self.mesh = mesh

        self.n_wells = len(well_xs)
        self.tmax = tmax 
        self.ts = np.linspace(0, tmax, nt+1)
        self.nt = nt
        
        # Temperature observation coordinates
        self.temp_obs_xs = temp_obs_xs
        self.temp_obs_zs = temp_obs_zs
        self.temp_obs_cs = np.array([[x, z] for z in temp_obs_zs 
                                     for x in temp_obs_xs])
        
        # Production observation times
        self.prod_obs_ts = prod_obs_ts
        self.inds_prod_obs = np.searchsorted(self.ts, prod_obs_ts-EPS)

        self.n_prod_obs_ts = len(self.prod_obs_ts)

        self.n_temp_full = self.mesh.nx * self.mesh.nz 
        self.n_pres_full = (self.nt+1) * self.n_wells
        self.n_enth_full = (self.nt+1) * self.n_wells

        self.n_temp_obs = len(self.temp_obs_cs)
        self.n_pres_obs = self.n_prod_obs_ts * self.n_wells
        self.n_enth_obs = self.n_prod_obs_ts * self.n_wells

        self.generate_inds_full()
        self.generate_inds_obs()

    def generate_inds_full(self):
        """Generates indices used to extract temperatures, pressures 
        and enthalpies from a vector of complete data."""
        self.inds_full_temp = np.arange(self.n_temp_full)
        self.inds_full_pres = np.arange(self.n_pres_full) + 1 + self.inds_full_temp[-1]
        self.inds_full_enth = np.arange(self.n_enth_full) + 1 + self.inds_full_pres[-1]

    def generate_inds_obs(self):
        """Generates indices used to extract temperatures, pressures 
        and enthalpy observations from a vector of observations."""
        self.inds_obs_temp = np.arange(self.n_temp_obs)
        self.inds_obs_pres = np.arange(self.n_pres_obs) + 1 + self.inds_obs_temp[-1]
        self.inds_obs_enth = np.arange(self.n_enth_obs) + 1 + self.inds_obs_pres[-1]

    def get_full_temperatures(self, F_i):
        """Extracts the temperatures from a set of data."""
        temp = F_i[self.inds_full_temp]
        temp = np.reshape(temp, (self.mesh.nz, self.mesh.nx)) # TODO: check
        return temp
    
    def get_full_pressures(self, F_i):
        """Extracts the pressures from a complete set of data."""
        pres = F_i[self.inds_full_pres]
        return self.reshape_obs(pres)

    def get_full_enthalpies(self, F_i):
        """Extracts the enthalpies from a complete set of data."""
        enth = F_i[self.inds_full_enth]
        return self.reshape_obs(enth)
    
    def get_full_states(self, F_i):
        temp = self.get_full_temperatures(F_i)
        pres = self.get_full_pressures(F_i)
        enth = self.get_full_enthalpies(F_i)
        return temp, pres, enth 
    
    def get_obs_temperatures(self, temp_full):
        """Extracts the temperatures at each observation point from a 
        full set of temperatures."""
        mesh_coords = (self.mesh.xs, self.mesh.zs)
        interpolator = RegularGridInterpolator(mesh_coords, temp_full.T)
        temp_obs = interpolator(self.temp_obs_cs) # TODO: check
        return self.reshape_obs(temp_obs)
    
    def get_obs_pressures(self, pres_full):
        """Extracts the pressures at each observation location from a 
        full set of pressures."""
        return pres_full[self.inds_prod_obs, :]

    def get_obs_enthalpies(self, enth_full):
        """Extracts the enthalpies at each observation location from a
        full set of enthalpies."""
        return enth_full[self.inds_prod_obs, :]
    
    def get_obs_states(self, F_i):
        """Extracts the observations from a complete vector of model 
        output, and returns each type of observation individually."""
        temp_full, pres_full, enth_full = self.get_full_states(F_i)
        temp_obs = self.get_obs_temperatures(temp_full)
        pres_obs = self.get_obs_pressures(pres_full)
        enth_obs = self.get_obs_enthalpies(enth_full)
        return temp_obs, pres_obs, enth_obs
    
    def get_obs(self, F_i):
        """Extracts the observations from a complete vector of model
        output, and returns them as a vector."""
        temp_obs, pres_obs, enth_obs = self.get_obs_states(F_i)
        obs = np.concatenate((temp_obs.flatten(), 
                              pres_obs.flatten(), 
                              enth_obs.flatten()))
        return obs
    
    def reshape_obs(self, obs):
        """Reshapes observations 2D arrays such that each row contains
        the observations for a single well."""
        return np.reshape(obs, (-1, self.n_wells))

    def split_obs(self, G_i):
        """Splits a set of observations into temperatures, pressures 
        and enthalpies."""
        temp_obs = self.reshape_obs(G_i[self.inds_obs_temp])
        pres_obs = self.reshape_obs(G_i[self.inds_obs_pres])
        enth_obs = self.reshape_obs(G_i[self.inds_obs_enth])
        return temp_obs, pres_obs, enth_obs
    
    def downhole_temps(self, temps):
        """Generates the downhole temperatures for a given well.
        TODO: cut off at well depths?"""
        mesh_coords = (self.mesh.xs, self.mesh.zs)
        interpolator = RegularGridInterpolator(mesh_coords, temps.T)
        well_coords = np.array([[x, z] for z in self.mesh.zs for x in self.temp_obs_xs])
        temp_well = interpolator(well_coords)
        return self.reshape_obs(temp_well)

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

# TODO: this section can surely be made simpler

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
prod_obs_ts = np.array([0, 13, 26, 39, 52]) * SECS_PER_WEEK

data_handler_crse = DataHandler(mesh_crse, well_xs, temp_obs_zs, prod_obs_ts, tmax, nt)
data_handler_fine = DataHandler(mesh_fine, well_xs, temp_obs_zs, prod_obs_ts, tmax, nt)

noise_level = 0.02

"""
Model functions
"""

def F(p_i, mesh=mesh_crse, model_name=model_name_crse, 
      wells=wells_crse, upflow_cell=upflow_cell_crse, i=None):
    """Given a set of transformed parameters, forms and runs the 
    corresponding model, then returns the full model output and model
    predictions."""

    if i is not None:
        model_name += f"_{i}"

    *logks, upflow_rate = p_i
    upflows = [MassUpflow(upflow_cell, upflow_rate)]
    
    model = Model2D(model_name, mesh, logks, wells, upflows, dt, tmax)

    if (flag := model.run()) == ExitFlag.FAILURE: 
        return flag
    return model.get_pr_data()

def G(F_i, data_handler=data_handler_crse):
    return data_handler.get_obs(F_i)

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

def generate_truth(mesh: SliceMesh, model_name, wells, upflow_cell):
    """Generates the truth and observations using the fine model."""

    w_t = truth_dist.sample()
    p_t = truth_dist.transform(w_t)

    # TEMP: sanity check
    mesh.m.slice_plot(value=p_t[:-1], colourmap="viridis")

    F_t = F(p_t, mesh, model_name, wells, upflow_cell)
    G_t = G(F_t, data_handler_fine)

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
    w_t, p_t, F_t, G_t = generate_truth(mesh_fine, model_name_fine, 
                                        wells_fine, upflow_cell_fine)
    y, C_e = generate_data(G_t)