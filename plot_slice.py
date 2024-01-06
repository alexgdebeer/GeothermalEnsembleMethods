import colorcet as cc
from matplotlib import pyplot as plt

from setup_slice import *

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

perm_min = -17.0
perm_max = -13.0
perm_cmap = cc.cm.bgy
temp_cmap = "coolwarm"

TICK_SIZE = 8
LABEL_SIZE = 12

temp_full, pres_full, enth_full = data_handler_fine.get_full_states(F_t)
downhole_temps = data_handler_fine.downhole_temps(temp_full)

temp_obs, pres_obs, enth_obs = data_handler_fine.split_obs(y)

logperms = np.reshape(p_t[:-1], (mesh_fine.nx, mesh_fine.nz))

plot_truth = False
plot_prior_samples = False
plot_data = True

well_num = 1

if plot_truth:

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")

    perm_mesh = axes[0].pcolormesh(mesh_fine.xs, mesh_fine.zs, logperms, 
                                   vmin=perm_min, vmax=perm_max, cmap=perm_cmap)
    temp_mesh = axes[1].pcolormesh(mesh_fine.xs, mesh_fine.zs, temp_full, 
                                   vmax=340, cmap=temp_cmap)

    perm_cbar = fig.colorbar(perm_mesh, ax=axes[0])
    temp_cbar = fig.colorbar(temp_mesh, ax=axes[1])
    
    perm_cbar.set_label("log$_{10}$(Permeability) [log$_{10}$(m$^2$)]", fontsize=14)
    temp_cbar.set_label("Temperature [$^{\circ}$C]", fontsize=14)

    for ax in axes:
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig("plots/slice/truth.pdf")

if plot_prior_samples:

    fig, axes = plt.subplots(2, 4, figsize=(8, 4.2))

    for ax in axes.flat: 

        w_i = prior.sample().squeeze()
        ks_t = prior.transform(w_i)[:-1]
        ks_t = np.reshape(ks_t, (mesh_crse.nx, mesh_crse.nz))

        ax.pcolormesh(mesh_crse.xs, mesh_crse.zs, ks_t, 
                      vmin=perm_min, vmax=perm_max, cmap=perm_cmap)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig("plots/slice/prior_samples.pdf")

def tufte_axis(ax, bnds_x, bnds_y):
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_bounds(*bnds_x)
    ax.spines["left"].set_bounds(*bnds_y)

    dx = bnds_x[1] - bnds_x[0]
    dy = bnds_y[1] - bnds_y[0]

    ax.set_xlim(bnds_x[0] - 0.1*dx, bnds_x[1] + 0.1*dx)
    ax.set_ylim(bnds_y[0] - 0.1*dy, bnds_y[1] + 0.1*dy)

if plot_data:

    ts = data_handler_fine.ts / SECS_PER_WEEK
    ts_obs = data_handler_fine.prod_obs_ts / SECS_PER_WEEK

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    axes[0].plot(downhole_temps[:, well_num], mesh_fine.zs, c="k", ls="--")
    axes[0].scatter(temp_obs[:, well_num], temp_obs_zs, c="k", s=25)

    axes[1].plot(ts, pres_full[:, well_num], c="k", ls="--")
    axes[1].scatter(ts_obs, pres_obs[:, well_num], c="k", s=25)

    axes[2].plot(ts, enth_full[:, well_num], c="k", ls="--")
    axes[2].scatter(ts_obs, enth_obs[:, well_num], c="k", s=25)
    
    for ax in axes.flat:

        ax.tick_params(axis="both", which="both", labelsize=TICK_SIZE)
        ax.set_box_aspect(1)
    
    temp_bounds_x = [0, 300]
    pres_bounds_x = [0, tmax / SECS_PER_WEEK]
    enth_bounds_x = [0, tmax / SECS_PER_WEEK]

    temp_bounds_y = [-1500, 0]
    pres_bounds_y = [4, 14]
    enth_bounds_y = [1200, 1500]

    tufte_axis(axes[0], temp_bounds_x, temp_bounds_y)
    tufte_axis(axes[1], pres_bounds_x, pres_bounds_y)
    tufte_axis(axes[2], enth_bounds_x, enth_bounds_y)

    axes[0].set_ylabel("Elevation [m]", fontsize=LABEL_SIZE)
    axes[1].set_ylabel("Pressure [MPa]", fontsize=LABEL_SIZE)
    axes[2].set_ylabel("Enthalpy [kJ/kg]", fontsize=LABEL_SIZE)

    axes[0].set_xlabel("Temperature [$^\circ$C]", fontsize=LABEL_SIZE)
    axes[1].set_xlabel("Time [Weeks]", fontsize=LABEL_SIZE)
    axes[2].set_xlabel("Time [Weeks]", fontsize=LABEL_SIZE)

    axes[1].set_xticks([0, 52, 104])
    axes[2].set_xticks([0, 52, 104])

    axes[0].set_yticks([-1500, -750, 0])
    axes[2].set_yticks([1200, 1300, 1400, 1500])

    plt.tight_layout()
    plt.savefig("plots/slice/data.pdf")

# plot_logperms(logperms)
# plot_temps_full(temp_full)
# plot_pressures(pres_full, pres_obs)
# plot_enthalpies(enth_full, enth_obs)
# plot_downhole_temps(downhole_temps, temp_obs)