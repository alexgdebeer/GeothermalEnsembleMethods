import h5py
import itertools as it
import numpy as np

from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import matplotlib.pyplot as plt 
import seaborn as sns

from setup import *

plt.rc("text", usetex=True)
sns.set_style("whitegrid", {"font.family": "serif", "grid.linestyle": ""})

np.random.seed(1)

TICK_SIZE = 10
LABEL_SIZE = 12
TITLE_SIZE = 14

CMAP_BINARY = ListedColormap(["crimson", "lightgrey"])
CMAP_TEST = ListedColormap(["whitesmoke", "gainsboro", "silver"])

READ_DATA = True
PLOT_MESH = True
PLOT_SAMPLES = True
PLOT_TRUTH = True
PLOT_POSTERIORS = True
PLOT_INTERVALS = True
PLOT_PREDICTIONS = True
PLOT_UPFLOWS = True
PLOT_TRANSIENT_DATA = True
PLOT_ENSEMBLE = True
CALC_MAHALANOBIS_DISTS = False

LOGK_MIN = -16.5
LOGK_MAX = -13.0
SDK_MIN = 0.0
SDK_MAX = 1.4

GRID_XTICKS = [0, 500, 1000, 1500]
GRID_ZTICKS = [-1500, -1000, -500, 0]
GRID_XTICKLABS = [str(t) for t in GRID_XTICKS]
GRID_ZTICKLABS = [str(t) for t in GRID_ZTICKS]

UPFLOW_TICKS = [0.1, 0.15, 0.2]
UPFLOW_TICKLABS = [str(t) for t in UPFLOW_TICKS]

TIME_TICKS = [0, 12, 24]
TIME_TICKLABS = [str(t) for t in TIME_TICKS]

# Define x / y limits for pressures and enthalpies
P_LIMS = (1.8, 6.2)
E_LIMS = (900, 1600)

P_ZTICKS = [2, 4, 6]
P_ZTICKLABS = [str(t) for t in P_ZTICKS]

E_ZTICKS = [1000, 1250, 1500]
E_ZTICKLABS = [str(t) for t in E_ZTICKS]

TEMPS_T = np.reshape(f_t[:n_blocks], (nx, nz))

ALG_NAMES = [
    "Prior",
    "ES-MDA",
    "ES-MDA (Loc.)",
    "EnRML",
    "EnRML (Loc.)"
]

FPATHS = [
    "data/mda_100",
    "data/mda_100",
    "data/mda_100_loc_cycle",
    "data/enrml_100",
    "data/enrml_100_loc_cycle"
]

N_ALGS = len(ALG_NAMES)

# Define wells to plot pressures and enthalpies for
WELL_PS = 0
WELL_ES = 2

# Read in true pressures / enthalpies and observations
PS_T = unpack_data_raw(f_t)[1][:, WELL_PS]
ES_T = unpack_data_raw(f_t)[2][:, WELL_ES]

PS_OBS = np.reshape(ys[ps_obs_is], (n_tobs, n_wells))[:, WELL_PS]
ES_OBS = np.reshape(ys[es_obs_is], (n_tobs, n_wells))[:, WELL_ES]


def read_data(fpath, pri=False):

    i = 0 if pri else -1

    with h5py.File(f"{fpath}.h5", "r") as f:
        inds = f["inds"]
        ts = f["ts"][i][:, inds]
        fs = f["fs"][i][:, inds]

    # Extract permeabilities and mass upflows
    qs = np.array([prior.transform(t)[1] for t in ts.T])
    ks = np.array([prior.transform(t)[0] for t in ts.T]).T
    logks = np.log10(ks)

    # Generate prior mean
    mu_post = np.log10(prior.transform(np.mean(ts, axis=1))[0])
    mu_post = np.reshape(mu_post, (mesh.nx, mesh.nz))
    sd_post = np.reshape(np.std(logks, axis=1), (mesh.nx, mesh.nz))

    # Form array that indicates whether truth is contained in central
    # 95% of ensemble
    q_lower = np.quantile(logks, 0.025, axis=1)
    q_upper = np.quantile(logks, 0.975, axis=1)

    q_lower = np.reshape(q_lower, (mesh.nx, mesh.nz))
    q_upper = np.reshape(q_upper, (mesh.nx, mesh.nz))

    in_range = (q_lower <= logks_t) & (logks_t <= q_upper)

    # Extract pressures and enthalpies
    ps = np.array([unpack_data_raw(f)[1][:, WELL_PS] for f in fs.T]).T
    es = np.array([unpack_data_raw(f)[2][:, WELL_ES] for f in fs.T]).T

    return {"logks": logks, "mu_post": mu_post, 
            "sd_post": sd_post, "in_range": in_range,
            "qs": qs, "ps": ps, "es": es}


if READ_DATA:
    ALG_DATA = {n: read_data(p, n=="Prior") for n, p in zip(ALG_NAMES, FPATHS)}


if PLOT_MESH:

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    zs_bound_2 = prior.sample()[:mesh.nx].squeeze()
    zones = np.zeros((mesh.nx, mesh.nz))

    for i, j in it.product(range(mesh.nx), range(mesh.nz)):
        if mesh.zs[j] > -60.0:
            zones[j, i] = 1.0
        elif mesh.zs[j] > zs_bound_2[i]:
            zones[j, i] = 0.5
        else:
            zones[j, i] = 0.0

    # Define grid
    grid = np.zeros((mesh.nx, mesh.nz))
    grid = np.ma.masked_array(grid, grid == 0.0)

    # Define coordinates of boundaries
    xs_bound_1 = [0, 1500]
    zs_bound_1 = [-60, -60]

    zs_bound_2 = [zs_bound_2[0], *zs_bound_2, zs_bound_2[-1]]
    xs_bound_2 = [0, *mesh.xs, mesh.xmax]

    # Define each subdomain
    cs_zone_1 = np.array([
        [0, mesh.xmax, mesh.xmax, 0],
        [-60, -60, 0, 0]]).T

    cs_zone_2 = np.array([
        [*xs_bound_2, mesh.xmax, 0], 
        [*zs_bound_2, -60, -60]]).T
    
    cs_zone_3 = np.array([
        [*xs_bound_2, mesh.xmax, 0],
        [*zs_bound_2, -mesh.zmax, -mesh.zmax]]).T

    zone_1 = Polygon(cs_zone_1, facecolor="silver", zorder=0)
    zone_2 = Polygon(cs_zone_2, facecolor="gainsboro", zorder=0)
    zone_3 = Polygon(cs_zone_3, facecolor="whitesmoke", zorder=0)

    # Plot grid
    ax.pcolormesh(mesh.xs, mesh.zs, grid, 
                  cmap=CMAP_TEST, edgecolors="darkgrey")

    # Plot subdomains
    ax.add_patch(zone_1) 
    ax.add_patch(zone_2)
    ax.add_patch(zone_3)

    # Plot boundaries
    ax.plot(xs_bound_1, zs_bound_1, c="k", linewidth=1.5, linestyle=(0, (5, 1)), zorder=1)
    ax.plot(xs_bound_2, zs_bound_2, c="k", linewidth=1.5, zorder=1)

    for (i, fz) in enumerate(feedzones):
        x, z = fz.loc[0], fz.loc[-1]
        ax.plot([x, x], [0, -1300], linewidth=1.5, color="royalblue", zorder=2)
        ax.scatter([x], [z], color="royalblue", s=20)
        plt.text(x-110, 40, s=f"Well {i+1}", color="royalblue", fontsize=TICK_SIZE)

    ax.set_xlabel("$x$ [m]", fontsize=LABEL_SIZE)
    ax.set_ylabel("$z$ [m]", fontsize=LABEL_SIZE)
    ax.set_box_aspect(1)

    ax.set_xticks(GRID_XTICKS, labels=GRID_XTICKLABS)
    ax.set_yticks(GRID_ZTICKS, labels=GRID_ZTICKLABS)
    ax.tick_params(labelsize=TICK_SIZE)

    legend_elements = [
        Patch(facecolor="silver", edgecolor="darkgrey", label="$\Omega_{1}$"),
        Patch(facecolor="gainsboro", edgecolor="darkgrey", label="$\Omega_{2}$"),
        Patch(facecolor="whitesmoke", edgecolor="darkgrey", label="$\Omega_{3}$"),
        Line2D([0], [0], color="k", linestyle=(0, (5, 1)), label="$\omega_{1}(x)$"),
        Line2D([0], [0], color="k", label="$\omega_{2}(x)$"),
        Line2D([0], [0], color="royalblue", label="Well tracks"),
        Line2D([0], [0], color="royalblue", marker="o", markersize=5, linestyle="", label="Feedzones"),
    ]

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.55, 0.80), frameon=False, fontsize=TICK_SIZE)

    for s in ax.spines.values():
        s.set_edgecolor("darkgrey")

    plt.tight_layout()
    plt.savefig("plots/mesh.pdf")
    plt.clf()


if PLOT_SAMPLES:

    nrows = 2
    ncols = 5

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9.5, 4))

    prior_samples = prior.sample(ncols)

    for i in range(ncols):

        logks_u = prior.transform_perms(prior_samples[:-1, i])
        logks = prior.apply_level_sets(logks_u)

        logks_u = np.reshape(logks_u, (mesh.nx, mesh.nz))
        logks = np.reshape(logks, (mesh.nx, mesh.nz))

        axes[0, i].pcolormesh(mesh.xs, mesh.zs, logks_u,
                              cmap="turbo", vmin=LOGK_MIN, vmax=LOGK_MAX)
        axes[1, i].pcolormesh(mesh.xs, mesh.zs, logks,
                              cmap="turbo", vmin=LOGK_MIN, vmax=LOGK_MAX)

    for ax in axes.flat:
        ax.set_box_aspect(1)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("plots/prior_samples.pdf")
    plt.clf()


if PLOT_TRUTH:

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 2.5))

    m1 = axes[0].pcolormesh(mesh.xs, mesh.zs, logks_t, 
                            cmap="turbo", vmin=LOGK_MIN, vmax=LOGK_MAX)
    
    m2 = axes[1].pcolormesh(mesh.xs, mesh.zs, TEMPS_T, cmap="coolwarm")

    c1 = fig.colorbar(m1, ax=axes[0], shrink=0.9)
    c2 = fig.colorbar(m2, ax=axes[1], shrink=0.9)

    c1.set_label("log(Permeability) [log(m$^2$)]", fontsize=LABEL_SIZE)
    c2.set_label("Temperature [$^\circ$C]", fontsize=LABEL_SIZE)

    for ax in axes.flat:
        ax.set_box_aspect(1)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("plots/truth.pdf")
    plt.clf()


if PLOT_POSTERIORS:

    fig, axes = plt.subplots(nrows=2, ncols=N_ALGS, figsize=(11, 4.9), 
                             sharex=True, sharey=True)

    for j, alg in enumerate(ALG_DATA):

        axes[0][j].set_title(alg, fontsize=TITLE_SIZE)

        axes[0][j].pcolormesh(
            mesh.xs, mesh.zs, ALG_DATA[alg]["mu_post"],
            cmap="turbo", vmin=LOGK_MIN, vmax=LOGK_MAX)

        axes[1][j].pcolormesh(
            mesh.xs, mesh.zs, ALG_DATA[alg]["sd_post"],
            cmap="turbo", vmin=SDK_MIN, vmax=SDK_MAX)

    for ax in axes.flat:
        ax.axis("off")
        ax.set_box_aspect(1)

    plt.tight_layout(w_pad=1.0, h_pad=1.0)
    plt.savefig("plots/posteriors.pdf")
    plt.clf()


if PLOT_INTERVALS:

    fig, axes = plt.subplots(nrows=1, ncols=N_ALGS, figsize=(11, 2.85), 
                             sharex=True, sharey=True)

    for i, alg in enumerate(ALG_DATA):

        axes[i].set_title(alg, fontsize=TITLE_SIZE)

        axes[i].pcolormesh(
            mesh.xs, mesh.zs, ALG_DATA[alg]["in_range"],
            cmap=CMAP_BINARY)

    for ax in axes:
        ax.axis("off")
        ax.set_box_aspect(1)

    legend_elements = [
        Patch(facecolor="lightgrey", label="Contained in central\n95\% of ensemble"),
        Patch(facecolor="crimson", label="Not contained in central\n95\% of ensemble")
    ]

    fig.legend(handles=legend_elements, loc="center", bbox_to_anchor=(0.50, 0.08), 
               frameon=False, ncols=len(legend_elements), fontsize=TICK_SIZE)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("plots/intervals.pdf")
    plt.clf()



if PLOT_UPFLOWS:

    fig, axes = plt.subplots(nrows=1, ncols=N_ALGS, 
                             figsize=(11, 3.0), sharey=True)

    for i, alg in enumerate(ALG_DATA):

        sns.histplot(ALG_DATA[alg]["qs"], ax=axes[i], 
                     binwidth=0.01, stat="density", 
                     color="orange", edgecolor="darkorange", 
                     binrange=mass_rate_bounds)
        
        axes[i].axvline(q_t, color="k", linewidth=1.5)

        axes[i].set_title(alg, fontsize=TITLE_SIZE)
        axes[i].set_xlim(mass_rate_bounds)
        axes[i].set_box_aspect(1)

    axes[2].set_xlabel("Upflow Rate [kg/s]", fontsize=LABEL_SIZE)
    axes[0].set_ylabel("Density", fontsize=LABEL_SIZE)

    for ax in axes.flat:
        ax.set_xticks(UPFLOW_TICKS, labels=UPFLOW_TICKLABS)
        ax.tick_params(labelsize=TICK_SIZE)

    legend_elements = [
        Patch(facecolor="orange", edgecolor="darkorange", label="Ensemble"),
        Line2D([0], [0], c="k", label="Truth")
    ]

    fig.legend(handles=legend_elements, loc="center", bbox_to_anchor=(0.50, 0.05), 
               frameon=False, ncols=len(legend_elements), fontsize=TICK_SIZE)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("plots/upflows.pdf")
    plt.clf()


if PLOT_TRANSIENT_DATA:

    fig, axes = plt.subplots(nrows=2, ncols=N_ALGS, 
                             figsize=(11, 5.25), sharey="row")

    for j, alg in enumerate(ALG_DATA):

        axes[0][j].plot(ALG_DATA[alg]["ps"] / 1e+6, color="royalblue", 
                        zorder=1, linewidth=1.0)
        axes[1][j].plot(ALG_DATA[alg]["es"] / 1e+3, color="seagreen", 
                        zorder=1, linewidth=1.0)

        axes[0][j].plot(PS_T / 1e+6, color="k", linewidth=1.5, zorder=2)
        axes[1][j].plot(ES_T / 1e+3, color="k", linewidth=1.5, zorder=2)
        axes[0][j].scatter(obs_time_inds, PS_OBS / 1e+6, c="k", s=15, zorder=3)
        axes[1][j].scatter(obs_time_inds, ES_OBS / 1e+3, c="k", s=15, zorder=3)

        axes[0][j].axvline(12, color="k", linestyle=(0, (5, 1)), linewidth=1.5)
        axes[1][j].axvline(12, color="k", linestyle=(0, (5, 1)), linewidth=1.5)

        axes[0][j].set_yticks(P_ZTICKS, labels=P_ZTICKLABS)
        axes[1][j].set_yticks(E_ZTICKS, labels=E_ZTICKLABS)
        axes[0][j].set_ylim(P_LIMS)
        axes[1][j].set_ylim(E_LIMS)

        axes[0][j].set_title(alg, fontsize=TITLE_SIZE)

        axes[0][j].set_xticklabels([])
        axes[1][j].set_xticks(TIME_TICKS, labels=TIME_TICKLABS)

    for ax in axes.flat:
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=TICK_SIZE)

    axes[0][0].set_ylabel(f"Pressure [MPa]", fontsize=LABEL_SIZE)
    axes[1][0].set_ylabel(f"Enthalpy [kJ/kg]", fontsize=LABEL_SIZE)
    axes[1][2].set_xlabel("Time [Months]", fontsize=LABEL_SIZE)

    legend_elements = [
        Line2D([0], [0], c="royalblue", label="Ensemble pressures"),
        Line2D([0], [0], c="seagreen", label="Ensemble enthalpies"),
        Line2D([0], [0], c="k", label="Truth"),
        Line2D([0], [0], c="k", marker="o", ms=5, ls="", label="Observations"),
        Line2D([0], [0], c="k", ls=(0, (5, 1)), label="End of observation period")
    ]

    fig.legend(handles=legend_elements, loc="center", bbox_to_anchor=(0.5, 0.05), 
               frameon=False, ncols=len(legend_elements), fontsize=TICK_SIZE)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("plots/transient_data.pdf")
    plt.clf()


if PLOT_ENSEMBLE:

    fig, axes = plt.subplots(nrows=2, ncols=N_ALGS, figsize=(11, 4.75))

    for j, alg in enumerate(ALG_DATA):

        axes[0][j].set_title(alg, fontsize=TITLE_SIZE)

        for i in range(2):
            logks_i = np.reshape(ALG_DATA[alg]["logks"][:, i], (mesh.nx, mesh.nz))
            axes[i][j].pcolormesh(mesh.xs, mesh.zs, logks_i,
                                  cmap="turbo", vmin=LOGK_MIN, vmax=LOGK_MAX)

    for ax in axes.flat:
        ax.axis("off")
        ax.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig("plots/ensemble_members.pdf")
    plt.clf()


if CALC_MAHALANOBIS_DISTS:

    ts_t = np.concatenate((logks_t.flatten(), [q_t]))

    for alg in ALG_DATA:

        mu = np.concatenate((ALG_DATA[alg]["mu_post"].flatten(), [np.mean(ALG_DATA[alg]["qs"])]))

        ts_e = np.concatenate((
            ALG_DATA[alg]["logks"],
            ALG_DATA[alg]["qs"][np.newaxis, :]))

        cov_ts = np.cov(ts_e)
        cov_ts += 1e-10 * np.diag(np.diag(cov_ts))

        cov_inv = np.linalg.inv(cov_ts) # Unstable?
        m_d = np.sqrt((ts_t - mu).T @ cov_inv @ (ts_t - mu))

        print(f"{alg}: {m_d}")
