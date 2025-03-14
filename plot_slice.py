import h5py
from scipy.interpolate import NearestNDInterpolator

from plotting import *
from setup_slice import *

plt.style.use("plots/paper.mplstyle")

RESULTS_FOLDER = "data/slice/results"
PLOTS_FOLDER = "plots/slice"

RESULTS_FNAMES = [
    f"{RESULTS_FOLDER}/eki_dmc.h5",
    f"{RESULTS_FOLDER}/eki_dmc_boot.h5",
    f"{RESULTS_FOLDER}/eki_dmc_inf.h5"
]

ALGNAMES = ["EKI", "EKI-BOOT", "EKI-INF"]

COMPUTE_MISFIT = True

PLOT_MESH = False
PLOT_TRUTH = False
PLOT_DATA = False

PLOT_PRIOR_PARTICLES = False

PLOT_MEAN_PRI = False
PLOT_MEAN_EKI = False
PLOT_STDS = False
PLOT_POST_PARTICLES = False
PLOT_UPFLOWS = False
PLOT_PREDICTIONS = True
PLOT_INTERVALS = False

PLOT_CBARS = False

DATA_WELL = 1
WELL_TO_PLOT = 2

WELL_DEPTH = -1300

def read_data(fname):

    with h5py.File(fname, "r") as f:

        post_ind = f["post_ind"][0]
        # print(post_ind)

        # success_rate = np.mean([
        #     len(f[f"inds_succ_{i}"]) for i in range(post_ind+1)
        # ]) / 100
        # print(success_rate)

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

results = {
    algname: read_data(fname) 
    for algname, fname in zip(ALGNAMES, RESULTS_FNAMES)
}


if COMPUTE_MISFIT:
    
    L_e = np.sqrt(np.linalg.inv(C_e))
    misfits = np.mean((L_e @ (results["EKI"]["Gs_post"] - y[:, None])) ** 2, axis=0)
    
    print(np.max(misfits))
    print(np.mean(misfits))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(misfits, bins=np.linspace(0, 13, 53))
    ax.set_title("Synthetic 2D Model")
    ax.set_xlabel(r"$\frac{1}{d}||\bm{y}_{\mathrm{obs}} - \mathcal{G}(\bm{\theta})||_{\bm{C}_{\epsilon}}^{2}$")
    ax.set_ylabel("Count")
    ax.axvline(x=1.0, ymin=1/22, ymax=21/22, c="grey", ls="--")
    tufte_axis(ax, bnds_x=(0, 13), bnds_y=(0, 25), xticks=np.arange(14), gap=0.05)
    plt.savefig("plots/misfit_slice.pdf")
    
if PLOT_MESH:

    fname = f"{PLOTS_FOLDER}/mesh.pdf"
    plot_mesh_2d(mesh_crse, prior, wells_crse, fname)

if PLOT_DATA: 

    temp_t, pres_t, enth_t = data_handler_fine.get_full_states(F_t)
    zs, temp_t = data_handler_fine.downhole_temps(temp_t)
    temp_obs, pres_obs, enth_obs = data_handler_fine.split_obs(y)

    ts = data_handler_fine.ts / (52 * SECS_PER_WEEK)
    ts_obs = data_handler_crse.prod_obs_ts / (52 * SECS_PER_WEEK)
    zs_obs = temp_obs_zs

    temp_lims_x = (0, 300)
    temp_lims_y = (WELL_DEPTH, 0)
    pres_lims_x = (0, 2)
    pres_lims_y = (4, 14)
    enth_lims_x = (0, 2)
    enth_lims_y = (1000, 1600)

    data_end = 1

    fname = f"{PLOTS_FOLDER}/data.pdf"
    plot_data(temp_t[:, DATA_WELL], pres_t, enth_t, zs, ts, 
              temp_obs, pres_obs, enth_obs, zs_obs, ts_obs, 
              temp_lims_x, temp_lims_y, pres_lims_x, pres_lims_y, 
              enth_lims_x, enth_lims_y, data_end, 
              DATA_WELL, fname)

if PLOT_TRUTH:
    
    perm_t = np.reshape(p_t[:-1], (mesh_fine.nx, mesh_fine.nz))
    temp_t = data_handler_fine.get_full_temperatures(F_t)

    fname = f"{PLOTS_FOLDER}/truth.pdf"
    plot_truth_2d(mesh_fine, perm_t, temp_t, fname)

if PLOT_PRIOR_PARTICLES:
    
    perms = results["EKI"]["ps_pri"][:-1, :4]
    
    fname = f"{PLOTS_FOLDER}/particles_pri.pdf"
    plot_particles_2d(mesh_crse, perms, fname)

if PLOT_MEAN_PRI:

    perm_t = np.reshape(p_t[:-1], (mesh_fine.nx, mesh_fine.nz))
    mean_pri = prior.transform(np.zeros(prior.n_params))[:-1]
    mean_pri = np.reshape(mean_pri, (mesh_crse.nx, mesh_crse.nz))

    vals = [perm_t, mean_pri]
    meshes = [mesh_fine, mesh_crse]
    labels = ["Truth", "Prior Mean"]

    fname = f"{PLOTS_FOLDER}/means_pri.pdf"
    plot_grid_2d(vals, meshes, labels, fname)

if PLOT_MEAN_EKI:

    perm_t = np.reshape(p_t[:-1], (mesh_fine.nx, mesh_fine.nz))
    mean_pri = prior.transform(np.zeros(prior.n_params))[:-1]
    mean_pri = np.reshape(mean_pri, (mesh_crse.nx, mesh_crse.nz))
    mean_eki = get_mean(results["EKI"]["ws_post"])

    vals = [perm_t, mean_pri, mean_eki]
    
    meshes = [mesh_fine, mesh_crse, mesh_crse]
    labels = ["Truth", "Prior Mean", "EKI"]

    fname = f"{PLOTS_FOLDER}/means_eki.pdf"
    plot_grid_2d(vals, meshes, labels, fname)

if PLOT_STDS:

    vals = [
        get_stds(results["EKI"]["ps_pri"]),
        get_stds(results["EKI"]["ps_post"])
    ]
    
    meshes = [mesh_crse] * 2
    labels = ["Prior", "EKI"]
    
    fname = f"{PLOTS_FOLDER}/stds.pdf"
    plot_grid_2d(vals, meshes, labels, fname, 
                 vmin=MIN_STDS_2D, vmax=MAX_STDS_2D)

if PLOT_PREDICTIONS:

    Fs = [
        results["EKI"]["Fs_pri"], 
        results["EKI"]["Fs_post"]
    ]

    temp_t, pres_t, enth_t = data_handler_fine.get_full_states(F_t)
    zs, temp_t = data_handler_fine.downhole_temps(temp_t)
    temp_obs, pres_obs, enth_obs = data_handler_fine.split_obs(y)

    ts = data_handler_fine.ts / (52 * SECS_PER_WEEK)
    ts_obs = data_handler_crse.prod_obs_ts / (52 * SECS_PER_WEEK)
    zs_obs = temp_obs_zs

    temp_lims_x = (0, 340)
    temp_lims_y = (WELL_DEPTH, 0)
    pres_lims_x = (0, 2)
    pres_lims_y = (2, 14)
    enth_lims_x = (0, 2)
    enth_lims_y = (800, 2600)

    data_end = 1

    fname = f"{PLOTS_FOLDER}/predictions.pdf"
    plot_predictions(Fs, data_handler_crse, temp_t[:, WELL_TO_PLOT], 
                     pres_t[:, WELL_TO_PLOT], enth_t[:, WELL_TO_PLOT], 
                     ts, zs, temp_obs, pres_obs, enth_obs, ts_obs, zs_obs, 
                     temp_lims_x, temp_lims_y, pres_lims_x, pres_lims_y,
                     enth_lims_x, enth_lims_y, data_end, 
                     WELL_TO_PLOT, fname, dim=2)

if PLOT_POST_PARTICLES:

    perms = results["EKI"]["ps_post"][:-1, :4]
    
    fname = f"{PLOTS_FOLDER}/particles_post.pdf"
    plot_particles_2d(mesh_crse, perms, fname)

if PLOT_UPFLOWS:

    upflow_t = p_t[-1] * upflow_cell_fine.column.area
    # print(upflow_t)
    # print(p_t[-1])
    bnds_x = (0.1, 0.2)
    bnds_y = (0, 80)

    upflows = [
        results["EKI"]["ps_pri"][-1, :],
        results["EKI"]["ps_post"][-1, :]
    ]

    upflows = [u * upflow_cell_crse.column.area for u in upflows]

    fname = f"{PLOTS_FOLDER}/upflows.pdf"
    plot_upflows_2d(upflow_t, upflows, bnds_x, bnds_y, fname)

if PLOT_INTERVALS:

    def get_cells_in_interval(ps):

        perms_post = ps[:mesh_crse.m.num_cells, :]

        bnds = np.quantile(perms_post, [0.025, 0.975], axis=1)

        cells_in_interval = np.zeros((mesh_crse.m.num_cells))
        for j, perm in enumerate(perms_interp):
            if not (bnds[0][j] - 1e-4 <= perm <= bnds[1][j] + 1e-4):
                cells_in_interval[j] = 1.0

        return np.reshape(cells_in_interval, (mesh_crse.nx, mesh_crse.nz))

    perms_t = p_t[:mesh_fine.m.num_cells]

    centres_crse = [c.centre for c in mesh_crse.m.cell]
    centres_fine = [c.centre for c in mesh_fine.m.cell]

    interp = NearestNDInterpolator(centres_fine, perms_t)
    perms_interp = interp(centres_crse)

    results_list = [
        results["EKI"]["ps_pri"],
        results["EKI"]["ps_post"]
    ]

    vals = [get_cells_in_interval(ps) for ps in results_list]
    meshes = [mesh_crse] * 2
    labels = ["Prior", "EKI"]

    fname = f"{PLOTS_FOLDER}/intervals.pdf"
    plot_grid_2d(vals, meshes, labels, fname, 
                 vmin=0, vmax=1, cmap=CMAP_INTERVALS, cbar=False)