import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from distributions import GaussianProcess, GaussianRF, Likelihood, SlicePrior
import geo_models as gm

np.random.seed(0)

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

mesh = gm.Mesh(f"gSL{n_blocks}", xmax, ymax, zmax, nx, ny, nz)
mesh.write_to_file()

feedzone_xs = [200, 475, 750, 1025, 1300]
feedzone_zs = [-500] * n_wells
feedzone_qs = [-2.0] * n_wells

wells = [gm.Well(gm.Feedzone((x, 0.5*ymax, z), q), obs_time_inds) # TODO: get rid of obs_time_inds here
         for (x, z, q) in zip(feedzone_xs, feedzone_zs, feedzone_qs)]

upflow_loc = (0.5*xmax, 0.5*ymax, -zmax + 0.5*mesh.dz)

# ----------------
# Constants and functions for extracting data
# ----------------

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
    ts = np.reshape(fs[ts_val_is], (mesh.nx, mesh.nz), order="F")
    ps = np.reshape(fs[ps_val_is], (nt+1, n_wells), order="F")
    es = np.reshape(fs[es_val_is], (nt+1, n_wells), order="F")
    return ts, ps, es

def unpack_data_obs(gs):
    ts = np.reshape(gs[ts_obs_is], (6, n_wells), order="F")
    ps = np.reshape(gs[ps_obs_is], (n_tobs, n_wells), order="F")
    es = np.reshape(gs[es_obs_is], (n_tobs, n_wells), order="F")
    return ts, ps, es

# ----------------
# Functions for running models
# ----------------

def f(thetas):

    ks, q = prior.transform(thetas)
    upflows = [gm.MassUpflow(upflow_loc, q)]
    
    model = gm.Model(f"SL{n_blocks}", mesh, ks, wells, upflows, dt, tmax)
    flag = model.run()

    if flag != gm.ExitFlag.SUCCESS: return flag
    return model.get_pr_data()

def g(fs):

    if type(fs) == gm.ExitFlag: return fs
    ts, ps, es = unpack_data_raw(fs)

    interp = RegularGridInterpolator((mesh.xs, -mesh.zs), ts)
    ts = interp(np.vstack((ns_obs_xs, -ns_obs_zs)).T).flatten(order="F")

    ps = ps[obs_time_inds, :].flatten(order="F")
    es = es[obs_time_inds, :].flatten(order="F")

    return np.concatenate((ts, ps, es))

# ----------------
# Definition of prior
# ----------------

mass_rate_bounds = (1.0e-1, 2.0e-1)
level_width = 0.25

depth_shal = -60.0
cells_shal = [c for c in mesh.m.cell if c.centre[-1] > depth_shal]
cells_deep = [c for c in mesh.m.cell if c.centre[-1] <= depth_shal]

gp_depth_clay = GaussianProcess(-350, 80, 500, mesh.xs)
rf_perm_shal = GaussianRF(-14, 0.25, 1500, 200, cells_shal)
rf_perm_clay = GaussianRF(-16, 0.25, 1500, 200, cells_deep)
rf_perm_deep = GaussianRF(-14, 0.50, 1500, 200, cells_deep)

prior = SlicePrior(mesh, gp_depth_clay, 
                   rf_perm_shal, rf_perm_clay, rf_perm_deep,
                   mass_rate_bounds, level_width)

# ----------------
# Truth generation
# ----------------

thetas_t = prior.sample()
ks_t, q_t = prior.transform(thetas_t)

logks_t = np.reshape(np.log10(ks_t), (nx, nz), order="F")

# mesh.m.slice_plot(value=np.log10(ks_t), colourmap="turbo")
# plt.pcolormesh(mesh.xs, mesh.zs, np.rot90(logks_t, k=3), cmap="turbo")
# plt.show()

fs_t = f(thetas_t)
gs_t = g(fs_t) # TODO: add noise

ts_t, ps_t, es_t = unpack_data_obs(gs_t)

max_t = ts_t.max()
max_p = ps_t.max()
max_e = es_t.max()

n_ts_obs = len(ts_t)
n_ps_obs = len(ps_t)
n_es_obs = len(es_t)

ny = len(gs_t)

cov = np.eye(ny) # TODO: fix

likelihood = Likelihood(gs_t, cov)