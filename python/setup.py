import numpy as np
import scipy.interpolate as interpolate
import scipy.sparse as sparse

import distributions as ds
import geo_models as gm

truth_from_file = True
params_path = "data/params_true.txt"
outputs_path = "data/outputs_true.txt"

"""
Model parameters
"""

SECS_PER_WEEK = 60.0 * 60.0 * 24.0 * 7.0

xmax, nx = 1500.0, 25
ymax, ny = 60.0, 1
zmax, nz = 1500.0, 25
tmax, nt = 104.0 * SECS_PER_WEEK, 24

dt = tmax / nt 
obs_time_inds = [0, 3, 6, 9, 12]
n_tobs = len(obs_time_inds)

n_blocks = nx * nz
n_wells = 5

mesh_name = f"models/gSL{n_blocks}"
model_name = f"models/SL{n_blocks}"

mesh = gm.Mesh(mesh_name, xmax, ymax, zmax, nx, ny, nz)
mesh.write_to_file()

feedzone_xs = [200, 475, 750, 1025, 1300]
feedzone_zs = [-500] * n_wells
feedzone_qs = [-2.0] * n_wells
feedzones = [gm.Feedzone((x, 0.5*ymax, z), q) 
         for (x, z, q) in zip(feedzone_xs, feedzone_zs, feedzone_qs)]

upflow_loc = (0.5*xmax, 0.5*ymax, -zmax + 0.5*mesh.dz)

"""
Constants and functions for extracting data
"""

n_temp_vals = n_blocks 
n_pressure_vals = n_wells * (nt + 1)
n_enthalpy_vals = n_wells * (nt + 1)

n_temp_obs =  6 * n_wells 
n_pressure_obs = len(obs_time_inds) * n_wells
n_enthalpy_obs = len(obs_time_inds) * n_wells

ns_obs_xs = np.array([x for x in feedzone_xs for _ in range(6)])
ns_obs_zs = np.array([-300, -500, -700, -900, -1100, -1300] * n_wells)

ts_val_is = list(range(n_temp_vals))
ps_val_is = list(range(ts_val_is[-1]+1, ts_val_is[-1]+1+n_pressure_vals))
es_val_is = list(range(ps_val_is[-1]+1, ps_val_is[-1]+1+n_enthalpy_vals))

ts_obs_is = list(range(n_temp_obs))
ps_obs_is = list(range(ts_obs_is[-1]+1, ts_obs_is[-1]+1+n_pressure_obs))
es_obs_is = list(range(ps_obs_is[-1]+1, ps_obs_is[-1]+1+n_enthalpy_obs))

def unpack_data_raw(fs):
    ts = np.reshape(fs[ts_val_is], (mesh.nx, mesh.nz))
    ps = np.reshape(fs[ps_val_is], (n_wells, nt+1))
    es = np.reshape(fs[es_val_is], (n_wells, nt+1))
    return ts, ps, es

def unpack_data_obs(gs):
    ts = np.reshape(gs[ts_obs_is], (n_wells, 6)) # TODO: don't hardcode 6
    ps = np.reshape(gs[ps_obs_is], (n_wells, n_tobs))
    es = np.reshape(gs[es_obs_is], (n_wells, n_tobs))
    return ts, ps, es

"""
Model functions
"""

def f(thetas):

    ks, q = prior.transform(thetas)
    upflows = [gm.MassUpflow(upflow_loc, q)]
    
    model = gm.Model(model_name, mesh, ks, feedzones, upflows, dt, tmax)
    flag = model.run()

    if flag != gm.ExitFlag.SUCCESS: return flag
    return model.get_pr_data()

def g(fs):

    if type(fs) == gm.ExitFlag: return fs
    ts, ps, es = unpack_data_raw(fs)

    interp = interpolate.RegularGridInterpolator((mesh.xs, -mesh.zs), ts)
    ts = interp(np.vstack((ns_obs_xs, -ns_obs_zs)).T).flatten()

    ps = ps[:, obs_time_inds].flatten()
    es = es[:, obs_time_inds].flatten()

    return np.concatenate((ts, ps, es))

"""
Prior
"""

mass_rate_bounds = (1.0e-1, 2.0e-1)
level_width = 0.25

depth_shal = -60.0
cells_shal = [c for c in mesh.m.cell if c.centre[-1] > depth_shal]
cells_deep = [c for c in mesh.m.cell if c.centre[-1] <= depth_shal]

gp_depth_clay = ds.Gaussian1D(-350, 80, 500, mesh.xs)
rf_perm_shal = ds.Gaussian2D(-14, 0.25, 1500, 200, cells_shal)
rf_perm_clay = ds.Gaussian2D(-16, 0.25, 1500, 200, cells_deep)
rf_perm_deep = ds.Gaussian2D(-14, 0.50, 1500, 200, cells_deep)

prior = ds.SlicePrior(mesh, gp_depth_clay, 
                      rf_perm_shal, rf_perm_clay, rf_perm_deep,
                      mass_rate_bounds, level_width)

"""
Truth
"""

if truth_from_file:
    thetas_t = np.genfromtxt(params_path)
else:
    thetas_t = prior.sample()

ks_t, q_t = prior.transform(thetas_t)

logks_t = np.reshape(np.log10(ks_t), (nx, nz))

if truth_from_file:
    f_t = np.genfromtxt(outputs_path)
else:
    f_t = f(thetas_t)

g_t = g(f_t)

Nf = len(f_t)
Ng = len(g_t)

ts_t, ps_t, es_t = unpack_data_obs(g_t)

"""
Observations / likelihood
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

# TODO: rescale everything to get an identity covariance, to make subsequent 
# TSVDs easier

cov_eta = sparse.block_diag((cov_ts, cov_ps, cov_es)).toarray()

ys = np.random.multivariate_normal(g_t, cov_eta)

likelihood = ds.Likelihood(ys, cov_eta)