"""
Plots the quantities related to the true (fine) slice model.
"""

from matplotlib import pyplot as plt

from setup_slice import *

temp_full, pres_full, enth_full = data_handler_fine.get_full_states(F_t)
downhole_temps = data_handler_fine.downhole_temps(temp_full)

temp_obs, pres_obs, enth_obs = data_handler_fine.split_obs(y)

logperms = np.reshape(p_t[:-1], (mesh_fine.nx, mesh_fine.nz))

def plot_logperms(logperms):
    plt.pcolormesh(mesh_fine.xs, mesh_fine.zs, logperms, cmap="viridis")
    plt.colorbar()
    plt.show()

def plot_temps_full(temps):
    plt.pcolormesh(mesh_fine.xs, mesh_fine.zs, temps, cmap="coolwarm")
    plt.colorbar()
    plt.show()

def plot_pressures(true_pressures, obs):

    ts = data_handler_fine.ts / SECS_PER_WEEK
    ts_obs = data_handler_fine.prod_obs_ts / SECS_PER_WEEK

    for i in range(n_wells):
        plt.plot(ts, true_pressures[:, i])
        plt.scatter(ts_obs, obs[:, i])
    
    plt.show()

def plot_enthalpies(true_enthalpies, obs):

    ts = data_handler_fine.ts / SECS_PER_WEEK
    ts_obs = data_handler_fine.prod_obs_ts / SECS_PER_WEEK

    for i in range(n_wells):
        plt.plot(ts, true_enthalpies[:, i])
        plt.scatter(ts_obs, obs[:, i])
    
    plt.show()

def plot_downhole_temps(true_temps, obs):

    for i in range(n_wells):
        plt.plot(true_temps[:, i], mesh_fine.zs)
        plt.scatter(obs[:, i], data_handler_fine.temp_obs_zs)
    
    plt.show()

plot_logperms(logperms)
plot_temps_full(temp_full)
plot_pressures(pres_full, pres_obs)
plot_enthalpies(enth_full, enth_obs)
plot_downhole_temps(downhole_temps, temp_obs)