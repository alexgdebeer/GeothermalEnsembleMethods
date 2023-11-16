import h5py
import itertools as it
import numpy as np

from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import matplotlib.pyplot as plt 

from setup import *

import matplotlib as mpl

mpl.rcParams["text.usetex"] = True 
mpl.rcParams["text.latex.preamble"] = r"\usepackage[cm]{sfmath}"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = "cm"

np.random.seed(1)

TITLE_SIZE = 14
LABEL_SIZE = 12
LEGEND_SIZE = 12
TICK_SIZE = 10

CMAP_BINARY = ListedColormap(["crimson", "lightgrey"])
CMAP_TEST = ListedColormap(["whitesmoke", "gainsboro", "silver"])

READ_DATA = False
PLOT_MESH = False
PLOT_SAMPLES = False
PLOT_TRUTH = True
PLOT_DATA = False
PLOT_POSTERIORS = False
PLOT_INTERVALS = False
PLOT_PREDICTIONS = False
PLOT_UPFLOWS = False
PLOT_TRANSIENT_DATA = False
PLOT_ENSEMBLE = False

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

ELEV_TICKS = [0, -750, -1500]
ELEV_TICKLABS = [str(t) for t in ELEV_TICKS]

# Define x / y limits for pressures and enthalpies
P_LIMS = (1.8, 6.2)
E_LIMS = (900, 1600)
T_LIMS = (25, 270)

T_XTICKS = [100, 200]
T_XTICKLABS = [str(t) for t in T_XTICKS]

P_ZTICKS = [2, 4, 6]
P_ZTICKLABS = [str(t) for t in P_ZTICKS]

E_ZTICKS = [1000, 1250, 1500]
E_ZTICKLABS = [str(t) for t in E_ZTICKS]

TEMPS_T = np.reshape(f_t[:n_blocks], (nx, nz))

def get_well_temps(fs, well_x):

    well_xs = np.repeat([well_x], mesh.nz)

    ts, _, _ = unpack_data_raw(fs)
    ts_interp = interpolate.RegularGridInterpolator((mesh.xs, -mesh.zs), ts.T)
    ts = ts_interp(np.vstack((well_xs, -mesh.zs)).T).flatten()
    
    return ts

# Define wells to plot data for
WELL_TS = 3
WELL_PS = 0
WELL_ES = 2

TS_T = get_well_temps(f_t, feedzone_xs[WELL_TS])
PS_T = unpack_data_raw(f_t)[1][:, WELL_PS]
ES_T = unpack_data_raw(f_t)[2][:, WELL_ES]

TS_OBS = np.reshape(ys[ts_obs_is], (n_wells, n_temps_per_well))[WELL_TS, :]
PS_OBS = np.reshape(ys[ps_obs_is], (n_tobs, n_wells))[:, WELL_PS]
ES_OBS = np.reshape(ys[es_obs_is], (n_tobs, n_wells))[:, WELL_ES]

ALG_NAMES = [
    "Prior\nSamples",
    "Ensemble Randomised\nMaximum Likelihood",
    "Ensemble Randomised\nMaximum Likelihood (Loc.)",
    "Ensemble Kalman\nInversion",
    "Ensemble Kalman\nInversion (Loc.)"
]

FPATHS = [
    "data/mda_100",
    "data/mda_100",
    "data/mda_100_loc_cycle",
    "data/enrml_100",
    "data/enrml_100_loc_cycle"
]

N_ALGS = len(ALG_NAMES)

PLOT_FOLDER = "plots/slice"

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
    ALG_DATA = {n: read_data(p, "prior" in n.lower()) for n, p in zip(ALG_NAMES, FPATHS)}


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
    ax.plot(xs_bound_1, zs_bound_1, c="k", linewidth=1.5, zorder=1)
    ax.plot(xs_bound_2, zs_bound_2, c="k", linewidth=1.5, zorder=1)

    well_num = lambda i : f"Well {i+1}"

    for (i, fz) in enumerate(feedzones):
        x, z = fz.loc[0], fz.loc[-1]
        ax.plot([x, x], [0, -1300], linewidth=1.5, color="royalblue", zorder=2)
        ax.scatter([x], [z], color="royalblue", s=20)
        # plt.text(x-110, 40, s=well_num(i), color="royalblue", fontsize=TICK_SIZE)

    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.set_xlabel("$x$ [m]", fontsize=LABEL_SIZE)
    ax.set_ylabel("$z$ [m]", fontsize=LABEL_SIZE)
    ax.set_box_aspect(1)

    ax.set_xticks(GRID_XTICKS, labels=GRID_XTICKLABS)
    ax.set_yticks(GRID_ZTICKS, labels=GRID_ZTICKLABS)
    ax.tick_params(labelsize=TICK_SIZE)

    legend_elements = [
        Patch(facecolor="silver", edgecolor="darkgrey", label="Shallow region"),
        Patch(facecolor="gainsboro", edgecolor="darkgrey", label="Clay cap"),
        Patch(facecolor="whitesmoke", edgecolor="darkgrey", label="Deep region"),
        Line2D([0], [0], c="royalblue", label="Well tracks"),
        Line2D([0], [0], c="royalblue", marker="o", ms=5, ls="", label="Feedzones"),
    ]

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.55, 0.80), 
              frameon=False, fontsize=TICK_SIZE)

    for s in ax.spines.values():
        s.set_edgecolor("darkgrey")

    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/mesh.pdf")
    plt.clf()


if PLOT_SAMPLES:

    nrows = 2
    ncols = 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5, 4))

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
        ax.set_xticks([])
        ax.set_yticks([])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    axes[0, 0].set_ylabel("Level Set\nFunction", fontsize=TITLE_SIZE)
    axes[1, 0].set_ylabel("Transformed\nPermeabilities", fontsize=TITLE_SIZE)

    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/prior_samples.pdf")
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

    axes[0].set_title("Permeabilities", fontsize=TITLE_SIZE)
    axes[1].set_title("Natural State Temperatures", fontsize=TITLE_SIZE)

    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/truth.pdf")
    plt.clf()


if PLOT_DATA:

    fig, axes = plt.subplots(1, 3, figsize=(7, 3))

    axes[0].plot(TS_T, mesh.zs, color="k", linewidth=1.5, zorder=2, ls=(0, (5, 1)))
    axes[1].plot(PS_T / 1e+6, color="k", linewidth=1.5, zorder=2, ls=(0, (5, 1)))
    axes[2].plot(ES_T / 1e+3, color="k", linewidth=1.5, zorder=2, ls=(0, (5, 1)))

    axes[0].scatter(TS_OBS, ns_obs_zs[:n_temps_per_well], c="k", s=15, zorder=3)
    axes[1].scatter(obs_time_inds, PS_OBS / 1e+6, c="k", s=15, zorder=3)
    axes[2].scatter(obs_time_inds, ES_OBS / 1e+3, c="k", s=15, zorder=3)

    axes[0].set_title("Natural State\nTemperatures", fontsize=TITLE_SIZE)
    axes[1].set_title("Feedzone\nPressures", fontsize=TITLE_SIZE)
    axes[2].set_title("Feedzone\nEnthalpies", fontsize=TITLE_SIZE)

    axes[0].set_xlabel("Temperature [$^\circ$C]", fontsize=LABEL_SIZE)
    axes[1].set_xlabel("Time [Months]", fontsize=LABEL_SIZE)
    axes[2].set_xlabel("Time [Months]", fontsize=LABEL_SIZE)

    axes[0].set_ylabel("Elevation [m]", fontsize=LABEL_SIZE)
    axes[1].set_ylabel("Pressure [MPa]", fontsize=LABEL_SIZE)
    axes[2].set_ylabel("Enthalpy [kJ/kg]", fontsize=LABEL_SIZE)

    axes[0].set_yticks(ELEV_TICKS, labels=ELEV_TICKLABS)
    axes[1].set_xticks(TIME_TICKS, labels=TIME_TICKLABS)
    axes[2].set_xticks(TIME_TICKS, labels=TIME_TICKLABS)

    axes[0].set_xticks(T_XTICKS, labels=T_XTICKLABS)
    axes[1].set_yticks(P_ZTICKS, labels=P_ZTICKLABS)
    axes[2].set_yticks(E_ZTICKS, labels=E_ZTICKLABS)

    axes[0].set_xlim(T_LIMS)
    axes[1].set_ylim(P_LIMS)
    axes[2].set_ylim(E_LIMS)

    axes[1].axvline(12, color="darkgrey", linestyle=(0, (5, 1)), linewidth=1.5, zorder=0)
    axes[2].axvline(12, color="darkgrey", linestyle=(0, (5, 1)), linewidth=1.5, zorder=0)

    for ax in axes:

        ax.set_box_aspect(1)
        ax.tick_params(bottom=False, top=False, left=False, right=False, 
                       labelsize=TICK_SIZE)

    legend_elements = [
        Line2D([0], [0], c="k", ls=(0, (5, 1)), label="Truth"),
        Line2D([0], [0], c="k", marker="o", ms=5, ls="", label="Observations"),
        Line2D([0], [0], c="darkgrey", ls=(0, (5, 1)), label="End of Observation Period")
    ]

    fig.legend(handles=legend_elements, loc="center", bbox_to_anchor=(0.5, 0.05), 
               frameon=False, ncols=len(legend_elements), fontsize=LEGEND_SIZE)

    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/data.pdf")


if PLOT_POSTERIORS:

    fig, axes = plt.subplots(nrows=2, ncols=N_ALGS, figsize=(11, 5.0), 
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

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)

    axes[0][0].set_ylabel("Means", fontsize=TITLE_SIZE)
    axes[1][0].set_ylabel("Standard Deviations", fontsize=TITLE_SIZE)

    plt.tight_layout(w_pad=1.0, h_pad=1.0)
    plt.savefig(f"{PLOT_FOLDER}/posteriors.pdf")
    plt.clf()


if PLOT_INTERVALS:

    fig, axes = plt.subplots(nrows=1, ncols=N_ALGS, figsize=(11, 3.25), 
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
        Patch(facecolor="lightgrey", label="Contained in Central\n95\% of Ensemble"),
        Patch(facecolor="crimson", label="Not Contained in Central\n95\% of Ensemble")
    ]

    fig.legend(handles=legend_elements, loc="center", bbox_to_anchor=(0.50, 0.08), 
               frameon=False, ncols=len(legend_elements), fontsize=LEGEND_SIZE)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    plt.savefig(f"{PLOT_FOLDER}/intervals.pdf")
    plt.clf()


if PLOT_UPFLOWS:

    fig, axes = plt.subplots(nrows=1, ncols=N_ALGS, 
                             figsize=(11, 3.0), sharey=True)

    bins = np.linspace(*mass_rate_bounds, 11)

    for i, alg in enumerate(ALG_DATA):

        axes[i].hist(ALG_DATA[alg]["qs"], 
                     bins=bins, density=True, 
                     color="gold", edgecolor="darkorange")
        
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
        Patch(facecolor="gold", edgecolor="darkorange", label="Ensemble"),
        Line2D([0], [0], c="k", label="Truth")
    ]

    fig.legend(handles=legend_elements, loc="center", bbox_to_anchor=(0.50, 0.05), 
               frameon=False, ncols=len(legend_elements), fontsize=LEGEND_SIZE)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"{PLOT_FOLDER}/upflows.pdf")
    plt.clf()


if PLOT_TRANSIENT_DATA:

    fig, axes = plt.subplots(nrows=2, ncols=N_ALGS, 
                             figsize=(11, 5.55), sharey="row")

    for j, alg in enumerate(ALG_DATA):

        axes[0][j].plot(ALG_DATA[alg]["ps"] / 1e+6, color="lightskyblue", 
                        zorder=1, linewidth=1.0)
        axes[1][j].plot(ALG_DATA[alg]["es"] / 1e+3, color="mediumseagreen", 
                        zorder=1, linewidth=1.0)

        axes[0][j].plot(PS_T / 1e+6, color="k", linewidth=1.5, zorder=2, ls=(0, (5, 1)))
        axes[1][j].plot(ES_T / 1e+3, color="k", linewidth=1.5, zorder=2, ls=(0, (5, 1)))
        axes[0][j].scatter(obs_time_inds, PS_OBS / 1e+6, c="k", s=15, zorder=3)
        axes[1][j].scatter(obs_time_inds, ES_OBS / 1e+3, c="k", s=15, zorder=3)

        axes[0][j].axvline(12, color="darkgrey", linestyle=(0, (5, 1)), linewidth=1.5, zorder=0)
        axes[1][j].axvline(12, color="darkgrey", linestyle=(0, (5, 1)), linewidth=1.5, zorder=0)

        axes[0][j].set_yticks(P_ZTICKS, labels=P_ZTICKLABS)
        axes[1][j].set_yticks(E_ZTICKS, labels=E_ZTICKLABS)
        axes[0][j].set_ylim(P_LIMS)
        axes[1][j].set_ylim(E_LIMS)

        axes[0][j].set_title(alg, fontsize=TITLE_SIZE)

        axes[0][j].set_xticklabels([])
        axes[1][j].set_xticks(TIME_TICKS, labels=TIME_TICKLABS)

    for ax in axes.flat:

        ax.tick_params(bottom=False, top=False, left=False, right=False)
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=TICK_SIZE)

    axes[0][0].set_ylabel(f"Pressure [MPa]", fontsize=LABEL_SIZE)
    axes[1][0].set_ylabel(f"Enthalpy [kJ/kg]", fontsize=LABEL_SIZE)
    axes[1][2].set_xlabel("Time [Months]", fontsize=LABEL_SIZE)

    legend_elements = [
        Line2D([0], [0], c="lightskyblue", label="Ensemble Pressures"),
        Line2D([0], [0], c="mediumseagreen", label="Ensemble Enthalpies"),
        Line2D([0], [0], c="k", ls=(0, (5, 1)), label="Truth"),
        Line2D([0], [0], c="k", marker="o", ms=5, ls="", label="Observations"),
        Line2D([0], [0], c="darkgrey", ls=(0, (5, 1)), label="End of Observation Period")
    ]

    fig.legend(handles=legend_elements, loc="center", bbox_to_anchor=(0.5, 0.05), 
               frameon=False, ncols=len(legend_elements), fontsize=LEGEND_SIZE)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"{PLOT_FOLDER}/transient_data.pdf")
    plt.clf()


if PLOT_ENSEMBLE:

    fig, axes = plt.subplots(nrows=2, ncols=N_ALGS, figsize=(11, 5))

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
    plt.savefig(f"{PLOT_FOLDER}/ensemble_members.pdf")
    plt.clf()