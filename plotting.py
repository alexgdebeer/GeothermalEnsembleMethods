import h5py
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns

from setup import *

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
sns.set_style("whitegrid")
sns.set(font="Computer Modern")

np.random.seed(1)

LABEL_SIZE = 14
TITLE_SIZE = 20

plot_model_grid = True
plot_logks_prior = True
plot_truth = True
plot_ensemble_results = True

logkmin = -16.5
logkmax = -13.0
sdmin = 0.0
sdmax = 1.4

xticks = [0, 500, 1000, 1500]
zticks = [-1500, -1000, -500, 0]
xticklabels = [str(t) for t in xticks]
zticklabels = [str(t) for t in zticks]

temps_t = np.reshape(f_t[:n_blocks], (nx, nz))

def remove_frame_and_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_frame_on(False)

if plot_model_grid:

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.pcolormesh(mesh.xs, mesh.zs, 0.1 * np.ones((nx, nz)), 
                  cmap="Greys", vmin=0, vmax=1, edgecolors="silver")

    for (i, fz) in enumerate(feedzones):
        x, z = fz.loc[0], fz.loc[-1]
        ax.plot([x, x], [0, -1300], linewidth=1.5, color="k", zorder=1)
        ax.scatter([x], [z], zorder=2, color="k")
        ax.text(x+30, z-15, s=f"W{i}", fontsize=14)

    ax.set_xlabel("$x$ (m)", fontsize=24)
    ax.set_ylabel("$z$ (m)", fontsize=24)
    ax.set_aspect(1)
    ax.set_xticks(xticks, labels=xticklabels, fontsize=14)
    ax.set_yticks(zticks, labels=zticklabels, fontsize=14)

    for spine in ax.spines.values():
        spine.set_edgecolor("silver")

    plt.tight_layout()
    plt.savefig("plots/grid.pdf")
    plt.clf()

if plot_logks_prior:

    nrows = 2
    ncols = 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 6))

    prior_samples = prior.sample(4)

    for (i, ax) in enumerate(axes.flatten()):

        *ks, _ = prior.transform(prior_samples[:, i])
        logks = np.reshape(np.log10(ks), (nx, nz))

        ax.pcolormesh(mesh.xs, mesh.zs, logks,
                      cmap="turbo", vmin=logkmin, vmax=logkmax)
        ax.set_aspect(1)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("plots/prior_samples.pdf")
    plt.clf()

if plot_truth:

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

    m1 = axes[0].pcolormesh(mesh.xs, mesh.zs, logks_t, 
                            cmap="turbo", vmin=logkmin, vmax=logkmax)
    
    m2 = axes[1].pcolormesh(mesh.xs, mesh.zs, temps_t, cmap="coolwarm")

    c1 = fig.colorbar(m1, ax=axes[0])
    c2 = fig.colorbar(m2, ax=axes[1])

    c1.set_label("log(Permeability) [log(m$^2$)]", fontsize=14)
    c2.set_label("Temperature [$^\circ$C]", fontsize=14)

    axes[0].set_aspect(1)
    axes[1].set_aspect(1)

    axes[0].axis("off")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("plots/truth.pdf")
    plt.clf()

if plot_ensemble_results:

    well_ps = 0
    well_es = 2

    ps_t = unpack_data_raw(f_t)[1][:, well_ps]
    es_t = unpack_data_raw(f_t)[2][:, well_es]

    ps_obs = np.reshape(ys[ps_obs_is], (n_tobs, n_wells))[:, well_ps]
    es_obs = np.reshape(ys[es_obs_is], (n_tobs, n_wells))[:, well_es]

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

        mu_post = np.reshape(np.mean(logks, axis=1), (mesh.nx, mesh.nz))
        sd_post = np.reshape( np.std(logks, axis=1), (mesh.nx, mesh.nz))

        print(sd_post.max())

        q_lower = np.quantile(logks, 0.025, axis=1)
        q_upper = np.quantile(logks, 0.975, axis=1)
        q_lower = np.reshape(q_lower, (mesh.nx, mesh.nz))
        q_upper = np.reshape(q_upper, (mesh.nx, mesh.nz))

        in_range = (q_lower <= logks_t) & (logks_t <= q_upper)

        ps = np.array([unpack_data_raw(f)[1][:, well_ps] for f in fs.T]).T
        es = np.array([unpack_data_raw(f)[2][:, well_es] for f in fs.T]).T

        return {"mu_post": mu_post, "sd_post": sd_post, "in_range": in_range,
                "qs": qs, "ps": ps, "es": es}

    alg_names = [
        "Prior",
        "MDA",
        "MDA (Localised)",
        "EnRML",
        "EnRML (Localised)"
    ]

    fpaths = [
        "data/mda_25",
        "data/mda_25",
        "data/mda_25_loc_boot",
        "data/enrml_25",
        "data/enrml_25_loc_boot"
    ]

    fig, axes = plt.subplots(nrows=6, ncols=len(alg_names), figsize=(15, 18))

    for (i, (alg_name, fpath)) in enumerate(zip(alg_names, fpaths)):

        pri = alg_name == "Prior"
        alg_data = read_data(fpath, pri)

        mesh_mu = axes[0][i].pcolormesh(mesh.xs, mesh.zs, alg_data["mu_post"],
                                        cmap="turbo", vmin=logkmin, vmax=logkmax)

        mesh_sd = axes[1][i].pcolormesh(mesh.xs, mesh.zs, alg_data["sd_post"],
                                        cmap="turbo", vmin=sdmin, vmax=sdmax)
        
        axes[2][i].pcolormesh(mesh.xs, mesh.zs, alg_data["in_range"],
                              cmap="turbo")
        
        binwidth = 0.01 if alg_name == "Prior" else 0.002

        axes[0][i].set_xticks(xticks, xticklabels)
        axes[0][i].set_yticks(zticks, zticklabels)
        axes[1][i].set_xticks(xticks, xticklabels)
        axes[1][i].set_yticks(zticks, zticklabels)
        axes[2][i].set_xticks(xticks, xticklabels)
        axes[2][i].set_yticks(zticks, zticklabels)

        # Plot the mass flow rate distribution
        sns.histplot(alg_data["qs"], ax=axes[3][i], 
                     binwidth=binwidth, stat="density", 
                     color="darkorange", edgecolor="darkorange", 
                     alpha=0.6, binrange=mass_rate_bounds)
        
        axes[3][i].axvline(q_t, color="k", linestyle="--")
        axes[3][i].set_xlim(mass_rate_bounds)
        axes[3][i].set_ylabel(None)

        # Plot modelled pressures and enthalpies
        axes[4][i].plot(alg_data["ps"] / 1e+6, color="royalblue", zorder=1)
        axes[5][i].plot(alg_data["es"] / 1e+3, color="seagreen", zorder=1)

        # Plot true pressures and enthalpies
        axes[4][i].plot(ps_t / 1e+6, color="k", zorder=2)
        axes[5][i].plot(es_t / 1e+3, color="k", zorder=2)
        axes[4][i].scatter(obs_time_inds, ps_obs / 1e+6, color="k", zorder=3)
        axes[5][i].scatter(obs_time_inds, es_obs / 1e+3, color="k", zorder=3)

        axes[4][i].set_ylim((2.0, 6.0))
        axes[5][i].set_ylim((900, 1600))

        # Plot titles
        axes[0][i].set_title(alg_name, fontsize=TITLE_SIZE)

    # Plot x labels
    axes[0][2].set_xlabel("$x$ [m]", fontsize=LABEL_SIZE)
    axes[1][2].set_xlabel("$x$ [m]", fontsize=LABEL_SIZE)
    axes[2][2].set_xlabel("$x$ [m]", fontsize=LABEL_SIZE)
    axes[3][2].set_xlabel("Upflow Rate [kg/s]", fontsize=LABEL_SIZE)
    axes[4][2].set_xlabel("Time [Months]", fontsize=LABEL_SIZE)
    axes[5][2].set_xlabel("Time [Months]", fontsize=LABEL_SIZE)

    # Plot y labels
    axes[0][0].set_ylabel("$z$ [m]", fontsize=LABEL_SIZE)
    axes[1][0].set_ylabel("$z$ [m]", fontsize=LABEL_SIZE)
    axes[2][0].set_ylabel("$z$ [m]", fontsize=LABEL_SIZE)
    axes[3][0].set_ylabel("Density", fontsize=LABEL_SIZE)
    axes[4][0].set_ylabel(f"Pressure [MPa]", fontsize=LABEL_SIZE)
    axes[5][0].set_ylabel(f"Enthalpy [kJ/kg]", fontsize=LABEL_SIZE)

    for ax in axes.flatten():
        ax.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig("plots/ensemble_results.pdf")
    plt.clf()

# with h5py.File("data/enrml_25_bootstrap_loc.h5", "r") as f:
#     fs = f["fs"][-1][:, f["inds"]]

# for f_i in fs.T:
#     ts, ps, es = unpack_data_raw(f_i)
#     plt.plot(es[:,2], color="k")

# plt.show()