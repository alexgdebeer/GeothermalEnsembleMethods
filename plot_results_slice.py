"""Plots results from the slice model."""

import colorcet
import h5py
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
from scipy.interpolate import NearestNDInterpolator

from plotting import *
from setup_slice import *

plt.rc("text", usetex=True)
plt.rc("font", family="serif")


# To plot: 
# - Prior vs posterior ensemble members
# - Prior vs posterior predictions
# - Hyperparameters (permeability lengthscales and standard deviations, focus on clay cap and also lower section of geometry)
# - Heatmap

TICK_SIZE = 8
LABEL_SIZE = 12

MIN_PERM = -17.0
MAX_PERM = -13.0

CMAP_PERM = "cet_bgy"
CMAP_PRED = matplotlib.colormaps.get_cmap("Pastel1").colors
CMAP_INTS = ListedColormap(["darkgray", "crimson"])
# CMAP_PRED = ["#00a0b0", "#00c49e", "#8be381"]
# COLOR_PRED = "lightskyblue"

RESULTS_FOLDER = "data/slice/results"

PLOT_MEAN_PRI = False
PLOT_MEAN_EKI = False
PLOT_STDS = False
PLOT_PREDICTIONS = False
PLOT_ENSEMBLE_MEMBERS = False
PLOT_UPFLOWS = False
PLOT_HEATMAPS = False
PLOT_HYPERPARAMS = False

WELL_TO_PLOT = 2

def read_data(fname):

    with h5py.File(fname, "r") as f:

        post_ind = f["post_ind"][0]

        inds_succ_pri = f[f"inds_succ_0"][:]
        inds_succ_post = f[f"inds_succ_{post_ind}"][:]

        results = {
            "ws_pri" : f[f"ws_0"][:, inds_succ_pri],
            "ps_pri" : f[f"ps_0"][:, inds_succ_pri],
            "Fs_pri" : f[f"Fs_0"][:, inds_succ_pri],
            "Gs_pri" : f[f"Gs_0"][:, inds_succ_pri],
            "ws_post" : f[f"ws_{post_ind}"][:, inds_succ_post],
            "ps_post" : f[f"ps_{post_ind}"][:, inds_succ_post],
            "Fs_post" : f[f"Fs_{post_ind}"][:, inds_succ_post],
            "Gs_post" : f[f"Gs_{post_ind}"][:, inds_succ_post],
        }

    return results

def get_mean(ws):
    """Returns the mean (in the transformed space) of an ensemble of 
    whitened particles."""
    mu_w = np.mean(ws, axis=1)
    mu = prior.transform(mu_w)[:-1]
    mu = np.reshape(mu, (mesh_crse.nx, mesh_crse.nz))
    return mu

def get_stds(ps):
    """Returns the standard deviations of each component of a set of 
    whitened particles."""
    stds = np.std(ps[:-1, :], axis=1)
    stds = np.reshape(stds, (mesh_crse.nx, mesh_crse.nz))
    return stds

def tufte_axis(ax, bnds_x, bnds_y, gap=0.1):
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_bounds(*bnds_x)
    ax.spines["left"].set_bounds(*bnds_y)

    dx = bnds_x[1] - bnds_x[0]
    dy = bnds_y[1] - bnds_y[0]

    ax.set_xlim(bnds_x[0] - gap*dx, bnds_x[1] + gap*dx)
    ax.set_ylim(bnds_y[0] - gap*dy, bnds_y[1] + gap*dy)

fnames = [
    f"{RESULTS_FOLDER}/eki_dmc.h5",
    f"{RESULTS_FOLDER}/eki_dmc_boot.h5",
    f"{RESULTS_FOLDER}/eki_dmc_inf.h5"
]

algnames = [
    "EKI",
    "EKI-BOOT",
    "EKI-INF"
]

results = {
    algname: read_data(fname) 
    for algname, fname in zip(algnames, fnames)
}

if PLOT_MEAN_PRI:

    mean_pri = prior.transform(np.zeros(prior.n_params))[:-1]
    mean_pri = np.reshape(mean_pri, (mesh_crse.nx, mesh_crse.nz))

    true_perms = np.reshape(p_t[:-1], (mesh_fine.nx, mesh_fine.nz))

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3))

    axes[0].pcolormesh(mesh_fine.xs, mesh_fine.zs, true_perms, vmin=MIN_PERM, vmax=MAX_PERM, cmap=CMAP_PERM)
    axes[1].pcolormesh(mesh_crse.xs, mesh_crse.zs, mean_pri, vmin=MIN_PERM, vmax=MAX_PERM, cmap=CMAP_PERM)

    for ax in axes.flat:
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].set_title("Truth", fontsize=LABEL_SIZE)
    axes[1].set_title("Prior", fontsize=LABEL_SIZE)

    plt.tight_layout()
    plt.savefig("plots/slice/means_pri.pdf")

if PLOT_MEAN_EKI:

    means_post = [
        get_mean(results["EKI"]["ws_post"]),
        get_mean(results["EKI-BOOT"]["ws_post"]),
        get_mean(results["EKI-INF"]["ws_post"])
    ]

    true_perms = np.reshape(p_t[:-1], (mesh_fine.nx, mesh_fine.nz))
    
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    axes[0].pcolormesh(mesh_crse.xs, mesh_crse.zs, means_post[0], vmin=MIN_PERM, vmax=MAX_PERM, cmap=CMAP_PERM)
    axes[1].pcolormesh(mesh_crse.xs, mesh_crse.zs, means_post[1], vmin=MIN_PERM, vmax=MAX_PERM, cmap=CMAP_PERM)
    axes[2].pcolormesh(mesh_crse.xs, mesh_crse.zs, means_post[2], vmin=MIN_PERM, vmax=MAX_PERM, cmap=CMAP_PERM)

    for ax in axes.flat:
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].set_title("EKI", fontsize=LABEL_SIZE)
    axes[1].set_title("EKI (Localisation)", fontsize=LABEL_SIZE)
    axes[2].set_title("EKI (Inflation)", fontsize=LABEL_SIZE)

    plt.tight_layout()
    plt.savefig("plots/slice/means_eki.pdf")

if PLOT_STDS:

    stds_pri = get_stds(results["EKI"]["ps_pri"])

    stds_post = [
        get_stds(results["EKI"]["ps_post"]),
        get_stds(results["EKI-BOOT"]["ps_post"]),
        get_stds(results["EKI-INF"]["ps_post"])
    ]

    for std in stds_post:
        print(np.mean(std))
    
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.45), layout="constrained")

    m0 = axes[0].pcolormesh(mesh_crse.xs, mesh_crse.zs, stds_pri, cmap=CMAP_PERM, vmin=0, vmax=1.5)
    m1 = axes[1].pcolormesh(mesh_crse.xs, mesh_crse.zs, stds_post[0], cmap=CMAP_PERM, vmin=0, vmax=1.5)
    m2 = axes[2].pcolormesh(mesh_crse.xs, mesh_crse.zs, stds_post[1], cmap=CMAP_PERM, vmin=0, vmax=1.5)
    m3 = axes[3].pcolormesh(mesh_crse.xs, mesh_crse.zs, stds_post[2], cmap=CMAP_PERM, vmin=0, vmax=1.5)

    cbar = fig.colorbar(m3, ax=axes[3])
    cbar.set_label("log$_{10}$(Permeability) [log$_{10}$(m$^2$)]", fontsize=LABEL_SIZE)

    for ax in axes.flat:
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].set_title("Prior", fontsize=LABEL_SIZE)
    axes[1].set_title("EKI")
    axes[2].set_title("EKI (Localisation)")
    axes[3].set_title("EKI (Inflation)")

    plt.savefig("plots/slice/stds_eki.pdf")

if PLOT_PREDICTIONS:

    # Define bounds for plotting
    temp_bounds_x = [0, 300]
    pres_bounds_x = [0, tmax / SECS_PER_WEEK]
    enth_bounds_x = [0, tmax / SECS_PER_WEEK]

    temp_bounds_y = [-1500, 0]
    pres_bounds_y = [4, 14]
    enth_bounds_y = [1100, 2100]

    # Format true states and observations
    temp_t, pres_t, enth_t = data_handler_fine.get_full_states(F_t)
    temp_downhole_t = data_handler_fine.downhole_temps(temp_t)
    temp_obs, pres_obs, enth_obs = data_handler_fine.split_obs(y)

    temp_downhole_t = temp_downhole_t[:, WELL_TO_PLOT]
    temp_obs = temp_obs[:, WELL_TO_PLOT]

    pres_t = pres_t[:, WELL_TO_PLOT]
    pres_obs = pres_obs[:, WELL_TO_PLOT]

    enth_t = enth_t[:, WELL_TO_PLOT]
    enth_obs = enth_obs[:, WELL_TO_PLOT]

    ts_fine = data_handler_fine.ts / SECS_PER_WEEK
    ts_obs_fine = data_handler_crse.prod_obs_ts / SECS_PER_WEEK

    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))

    # Plot temperatures
    for i, ax in enumerate(axes[0]):

        ax.plot(temp_downhole_t, mesh_fine.zs, c="k", ls="--", zorder=2)
        ax.scatter(temp_obs, temp_obs_zs, c="k", s=10, zorder=2)
        
        tufte_axis(ax, temp_bounds_x, temp_bounds_y)

        if i > 0: 
            ax.spines["left"].set_visible(False)
            ax.set_yticks([])
        else:
            ax.set_yticks([-1500, -750, 0])

    # Plot pressures
    for i, ax in enumerate(axes[1]):
        ax.plot(ts_fine, pres_t, c="k", ls="--", zorder=2)
        ax.scatter(ts_obs_fine, pres_obs, c="k", s=10, zorder=2)

        tufte_axis(ax, pres_bounds_x, pres_bounds_y)
        ax.set_xticks([0, 52, 104])

        if i > 0: 
            ax.spines["left"].set_visible(False)
            ax.set_yticks([])

    # Plot enthalpies
    for i, ax in enumerate(axes[2]):
        
        ax.plot(ts_fine, enth_t, c="k", ls="--", zorder=2)
        ax.scatter(ts_obs_fine, enth_obs, c="k", s=10, zorder=2)
        
        tufte_axis(ax, enth_bounds_x, enth_bounds_y)
        ax.set_xticks([0, 52, 104])

        if i > 0: 
            ax.spines["left"].set_visible(False)
            ax.set_yticks([])
        else: 
            ax.set_yticks([1100, 1600, 2100])

    for ax in axes.flat:
        ax.set_box_aspect(1)
        ax.tick_params(axis="both", which="both", labelsize=TICK_SIZE)

    axes[0][0].set_title("Prior", fontsize=LABEL_SIZE)
    axes[0][1].set_title("EKI", fontsize=LABEL_SIZE)
    axes[0][2].set_title("EKI (Localisation)", fontsize=LABEL_SIZE)
    axes[0][3].set_title("EKI (Inflation)", fontsize=LABEL_SIZE)

    axes[0][0].set_ylabel("Elevation [m]", fontsize=LABEL_SIZE)
    axes[1][0].set_ylabel("Pressure [MPa]", fontsize=LABEL_SIZE)
    axes[2][0].set_ylabel("Enthalpy [kJ/kg]", fontsize=LABEL_SIZE)
    
    axes[0][0].set_xlabel("Temperature [$^\circ$C]", fontsize=LABEL_SIZE)
    axes[1][0].set_xlabel("Time [Weeks]", fontsize=LABEL_SIZE)
    axes[2][0].set_xlabel("Time [Weeks]", fontsize=LABEL_SIZE)

    results_list = [
        results["EKI"]["Fs_pri"],
        results["EKI"]["Fs_post"],
        results["EKI-BOOT"]["Fs_post"],
        results["EKI-INF"]["Fs_post"]
    ]

    for i, Fs in enumerate(results_list):
        
        for F_i in Fs.T:

            temp_i, pres_i, enth_i = data_handler_crse.get_full_states(F_i)
            temp_dh_i = data_handler_crse.downhole_temps(temp_i)

            axes[0][i].plot(temp_dh_i[:, WELL_TO_PLOT], mesh_crse.zs, c=CMAP_PRED[0], zorder=1, alpha=0.5)
            axes[1][i].plot(data_handler_crse.ts/SECS_PER_WEEK, pres_i[:, WELL_TO_PLOT], c=CMAP_PRED[1], zorder=1, alpha=0.5)
            axes[2][i].plot(data_handler_crse.ts/SECS_PER_WEEK, enth_i[:, WELL_TO_PLOT], c=CMAP_PRED[3], zorder=1, alpha=0.5)

    plt.tight_layout(w_pad=-5)
    plt.savefig("plots/slice/predictions.pdf")

if PLOT_ENSEMBLE_MEMBERS:

    ps = results["EKI"]["ps_post"][:-1, :8]

    fig, axes = plt.subplots(2, 4, figsize=(8, 4.2))

    for i, ax in enumerate(axes.flat):

        p_i = np.reshape(ps.T[i], (mesh_crse.nx, mesh_crse.nz))
        ax.pcolormesh(mesh_crse.xs, mesh_crse.zs, p_i, vmin=MIN_PERM, vmax=MAX_PERM, cmap=CMAP_PERM)

        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("plots/slice/ensemble_members_eki.pdf")

if PLOT_UPFLOWS:

    upflow_t = p_t[-1] * upflow_cell_fine.column.area
    bnds_x = [0.1, 0.2]
    bnds_y = [0, 80]

    algnames = ["EKI", "EKI-BOOT", "EKI-INF"]

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    bins = np.linspace(0.1, 0.2, 11)

    for i, ax in enumerate(axes):

        if i == 0:
            upflows = results["EKI"]["ps_pri"][-1, :]
        else: 
            upflows = results[algnames[i-1]]["ps_post"][-1, :]

        upflows *= upflow_cell_crse.column.area

        ax.hist(upflows, zorder=1, color=CMAP_PRED.colors[13], density=True, bins=bins)
        ax.axvline(upflow_t, ymin=0.05, ymax=0.95, zorder=2, c="k", ls="--")

        tufte_axis(ax, bnds_x, bnds_y, gap=0.05)
        ax.set_box_aspect(1)
        ax.tick_params(axis="both", which="both", labelsize=TICK_SIZE)
        ax.set_xticks([0.1, 0.15, 0.2])

        if i != 0:
            ax.spines["left"].set_visible(False)
            ax.set_yticks([])

    axes[0].set_yticks([0, 40, 80])

    axes[0].set_title("Prior", fontsize=LABEL_SIZE)
    axes[1].set_title("EKI", fontsize=LABEL_SIZE)
    axes[2].set_title("EKI (Localisation)", fontsize=LABEL_SIZE)
    axes[3].set_title("EKI (Inflation)", fontsize=LABEL_SIZE)

    axes[0].set_ylabel("Density", fontsize=LABEL_SIZE)
    axes[1].set_xlabel("Upflow Rate [kg/s]", fontsize=LABEL_SIZE)

    plt.tight_layout()
    plt.savefig("plots/slice/upflow_rates.pdf")

if PLOT_HEATMAPS:

    perms_t = p_t[:mesh_fine.m.num_cells]

    centres_crse = [c.centre for c in mesh_crse.m.cell]
    centres_fine = [c.centre for c in mesh_fine.m.cell]

    interp = NearestNDInterpolator(centres_fine, perms_t)
    perms_interp = interp(centres_crse)

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.7))

    results_list = [
        results["EKI"]["ps_pri"],
        results["EKI"]["ps_post"],
        results["EKI-BOOT"]["ps_post"],
        results["EKI-INF"]["ps_post"]
    ]

    for i, ps in enumerate(results_list):

        perms_post = ps[:mesh_crse.m.num_cells, :]

        bnds = np.quantile(perms_post, [0.05, 0.95], axis=1)

        cells_in_interval = np.zeros((mesh_crse.m.num_cells))
        for j, perm in enumerate(perms_interp):
            if not (bnds[0][j] - 1e-4 <= perm <= bnds[1][j] + 1e-4):
                cells_in_interval[j] = 1.0

        newshape = (mesh_crse.nx, mesh_crse.nz)
        cells_in_interval = np.reshape(cells_in_interval, newshape)

        axes[i].pcolormesh(mesh_crse.xs, mesh_crse.zs, cells_in_interval, 
                           cmap=CMAP_INTS)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_box_aspect(1)

    axes[0].set_title("Prior", fontsize=LABEL_SIZE)
    axes[1].set_title("EKI", fontsize=LABEL_SIZE)
    axes[2].set_title("EKI (Localisation)", fontsize=LABEL_SIZE)
    axes[3].set_title("EKI (Inflation)", fontsize=LABEL_SIZE)

    plt.tight_layout()
    plt.savefig("plots/slice/intervals.pdf")

if PLOT_HYPERPARAMS:

    hps = np.array([prior.get_hyperparams(w_i)[2] for w_i in results["EKI-BOOT"]["ws_post"].T])

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