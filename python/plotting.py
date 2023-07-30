import h5py
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns

from setup import *

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
sns.axes_style("whitegrid")

np.random.seed(1)

plot_model_grid = False
plot_logks_prior = False
plot_truth = False
plot_ensemble_results = False

logkmin = -16.5
logkmax = -13.0

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

    with h5py.File("data/mda_test_2.h5", "r") as f:
        inds = f["inds"]
        ts = f["ts"][-1][:,inds]
        fs = f["fs"][-1][:,inds]

    ks_trans = np.array([prior.transform(t)[0] for t in ts.T]).T
    qs_trans = np.array([prior.transform(t)[1] for t in ts.T])
    logks_trans = np.log10(ks_trans)

    mu_post = np.reshape(np.mean(logks_trans, axis=1), (mesh.nx, mesh.nz))
    std_post = np.reshape(np.std(logks_trans, axis=1), (mesh.nx, mesh.nz))

    q_lower = np.quantile(logks_trans, 0.025, axis=1)
    q_upper = np.quantile(logks_trans, 0.975, axis=1)
    q_lower = np.reshape(q_lower, (mesh.nx, mesh.nz))
    q_upper = np.reshape(q_upper, (mesh.nx, mesh.nz))

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))

    plot_tr = axes[0].pcolormesh(mesh.xs, mesh.zs, logks_t, cmap="turbo", vmin=logkmin, vmax=logkmax)
    plot_mu = axes[1].pcolormesh(mesh.xs, mesh.zs, mu_post, cmap="turbo", vmin=logkmin, vmax=logkmax)

    in_range = (q_lower <= logks_t) & (logks_t <= q_upper)
    p3 = axes[2].pcolormesh(mesh.xs, mesh.zs, std_post, cmap="turbo")
    p4 = axes[3].pcolormesh(mesh.xs, mesh.zs, in_range, cmap="turbo")

    axes[4].hist(qs_trans, density=True, bins=10)
    axes[4].axvline(q_t, color="k", linestyle="--")
    axes[4].set_xlim(mass_rate_bounds)

    # sns.kdeplot(qs_trans, ax=axes[4], bw_adjust=0.25)

    axes[0].set_title("Truth", fontsize=16)
    axes[1].set_title("Mean", fontsize=16)
    axes[2].set_title("Standard deviations", fontsize=16)
    axes[3].set_title("In central 95\% of ensemble?", fontsize=16)
    axes[4].set_title("Upflow distribution", fontsize=16)

    axes[0].set_ylabel("ES-MDA")

    for ax in axes.flatten():
        ax.set_box_aspect(1)
        #remove_frame_and_ticks(ax)

    plt.tight_layout()
    plt.savefig("plots/ensemble_results.pdf")
    plt.clf()

# with h5py.File("data/mda_test.h5", "r") as f:
#     inds = f["inds"]
#     ts = f["ts"][:,inds,-1]
#     fs = f["fs"][:,inds,-1]

with h5py.File("data/mda_test_2.h5", "r") as f:
    fs = f["fs"][0][:, f["inds"]]

for f_i in fs.T:
    ts, ps, es = unpack_data_raw(f_i)
    plt.plot(ps[:,2])

ts, ps, es = unpack_data_raw(f_t)
plt.plot(ps[:,2])
plt.show()

# plt.plot(es_t.T)
# plt.show()