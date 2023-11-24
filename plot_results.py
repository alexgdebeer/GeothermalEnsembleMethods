"""Plots results from the slice model."""

import h5py
from matplotlib import pyplot as plt
import colorcet as cc

from setup_slice import *

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# To plot: 
# - Prior vs posterior ensemble members
# - Prior vs posterior predictions
# - Hyperparameters (permeability lengthscales and standard deviations, focus on clay cap and also lower section of geometry)

MIN_PERM = -17
MAX_PERM = -13.5

# Read in the observations
with h5py.File("data/slice/eki_dmc.h5", "r") as f:

    inds_succ = f["inds_succ"]

    ws = f["ws"][:][-1][:, inds_succ]
    ps = f["ps"][:][-1][:, inds_succ]
    Fs = f["Fs"][:][-1][:, inds_succ]
    Gs = f["Gs"][:][-1][:, inds_succ]

PLOT_MEAN = True
PLOT_STDS = False
PLOT_PREDICTIONS = False
PLOT_ENSEMBLE_MEMBERS = True
PLOT_UPFLOWS = False 
PLOT_HYPERPARAMS = True

WELL_TO_PLOT = 4

if PLOT_MEAN:

    # TODO: do properly
    mu_pri = prior.transform(np.zeros((prior.n_params, )))[:-1]
    mu_pri = np.reshape(mu_pri, (mesh_crse.nx, mesh_crse.nz))

    mu_post_w = np.mean(ws, axis=1)
    mu_post = prior.transform(mu_post_w)[:-1]
    mu_post = np.reshape(mu_post, (mesh_crse.nx, mesh_crse.nz))

    true_perms = np.reshape(p_t[:-1], (mesh_fine.nx, mesh_fine.nz))
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].pcolormesh(mesh_crse.xs, mesh_crse.zs, mu_pri, vmin=MIN_PERM, vmax=MAX_PERM, cmap=cc.cm.bgy)
    axes[1].pcolormesh(mesh_crse.xs, mesh_crse.zs, mu_post, vmin=MIN_PERM, vmax=MAX_PERM, cmap=cc.cm.bgy)
    axes[2].pcolormesh(mesh_fine.xs, mesh_fine.zs, true_perms, vmin=MIN_PERM, vmax=MAX_PERM, cmap=cc.cm.bgy)

    for ax in axes.flat:
        ax.set_box_aspect(1)

    axes[0].set_title("Prior Mean")
    axes[1].set_title("Posterior Mean")
    axes[2].set_title("Truth")

    plt.tight_layout()
    plt.show()

if PLOT_STDS:

    std_post = np.std(ps, axis=1)[:-1]
    std_post = np.reshape(std_post, (mesh_crse.nx, mesh_crse.nz))
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].pcolormesh(mesh_crse.xs, mesh_crse.zs, mu_pri, vmin=MIN_PERM, vmax=MAX_PERM)
    axes[1].pcolormesh(mesh_crse.xs, mesh_crse.zs, std_post, cmap=cc.cm.bgy)

    for ax in axes.flat:
        ax.set_box_aspect(1)

    axes[0].set_title("Prior")
    axes[1].set_title("Posterior")

    plt.tight_layout()
    plt.show()

if PLOT_PREDICTIONS:

    temp_t, pres_t, enth_t = data_handler_fine.get_full_states(F_t)
    temp_downhole_t = data_handler_fine.downhole_temps(temp_t)
    temp_obs, pres_obs, enth_obs = data_handler_fine.split_obs(y)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].plot(temp_downhole_t[:, WELL_TO_PLOT], mesh_fine.zs, c="k", zorder=2)
    axes[1].plot(data_handler_fine.ts/SECS_PER_WEEK, pres_t[:, WELL_TO_PLOT], c="k", zorder=2)
    axes[2].plot(data_handler_fine.ts/SECS_PER_WEEK, enth_t[:, WELL_TO_PLOT], c="k", zorder=2)

    axes[0].scatter(temp_obs[:, WELL_TO_PLOT], data_handler_crse.temp_obs_zs, c="k", s=10, zorder=3)
    axes[1].scatter(data_handler_crse.prod_obs_ts/SECS_PER_WEEK, pres_obs[:, WELL_TO_PLOT], c="k", s=10, zorder=3)
    axes[2].scatter(data_handler_crse.prod_obs_ts/SECS_PER_WEEK, enth_obs[:, WELL_TO_PLOT], c="k", s=10, zorder=3)

    for F_i in Fs.T:

        temp_i, pres_i, enth_i = data_handler_crse.get_full_states(F_i)
        temp_dh_i = data_handler_crse.downhole_temps(temp_i)
        
        axes[0].plot(temp_dh_i[:, WELL_TO_PLOT], mesh_crse.zs, c="skyblue", zorder=1, alpha=0.5)
        axes[1].plot(data_handler_crse.ts/SECS_PER_WEEK, pres_i[:, WELL_TO_PLOT], c="skyblue", zorder=1, alpha=0.5)
        axes[2].plot(data_handler_crse.ts/SECS_PER_WEEK, enth_i[:, WELL_TO_PLOT], c="skyblue", zorder=1, alpha=0.5)

    for ax in axes.flat:
        ax.set_box_aspect(1)

    axes[0].set_title("Temperatures")
    axes[1].set_title("Pressures")
    axes[2].set_title("Enthalpies")

    axes[0].set_ylabel("Elevation [m]")
    axes[1].set_ylabel("Pressure [MPa]")
    axes[2].set_ylabel("Enthalpy [KJ/kg]")

    axes[0].set_xlabel("Temperature [$^\circ$C]")
    axes[1].set_xlabel("Time [Months]")
    axes[2].set_xlabel("Time [Months]")

    plt.tight_layout()
    plt.show()

if PLOT_ENSEMBLE_MEMBERS:

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    for i, ax in enumerate(axes.flat):

        p_i = np.reshape(ps[:-1, i*10+1], (mesh_crse.nx, mesh_crse.nz))
        ax.pcolormesh(mesh_crse.xs, mesh_crse.zs, p_i, vmin=MIN_PERM, vmax=MAX_PERM, cmap=cc.cm.bgy)

        ax.set_box_aspect(1)

    plt.show()

if PLOT_UPFLOWS:

    plt.hist(ps[-1, :] * upflow_cell_crse.column.area, zorder=1)
    plt.axvline(p_t[-1] * upflow_cell_fine.column.area, zorder=2, c="k")

    plt.xlim((0.1, 0.2))

    plt.show()

if PLOT_HYPERPARAMS:

    hps = np.array([prior.get_hyperparams(w_i)[2] for w_i in ws.T])

    hps_t = truth_dist.get_hyperparams(w_t)[2]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0))
    axes[0].hist(hps[:, 0], label="Ensemble")
    axes[1].hist(hps[:, 1], label="Ensemble")
    axes[2].hist(hps[:, 2], label="Ensemble")

    axes[0].axvline(hps_t[0], label="Truth", color="k")
    axes[1].axvline(hps_t[1], label="Truth", color="k")
    axes[2].axvline(hps_t[2], label="Truth", color="k")

    axes[0].set_title("Standard Deviation")
    axes[1].set_title("Lengthscale ($x$)")
    axes[2].set_title("Lengthscale ($z$)")

    for i, ax in enumerate(axes):
        ax.set_xlim(bounds_deep[i])
        ax.legend()

    plt.legend()
    plt.tight_layout()
    plt.show()