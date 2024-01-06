import h5py
from scipy.interpolate import NearestNDInterpolator

from plotting import *
from setup_fault import *

RESULTS_FOLDER = "data/fault/results"
PLOTS_FOLDER = "plots/fault"

RESULTS_FNAMES = [
    f"{RESULTS_FOLDER}/eki_dmc.h5",
    f"{RESULTS_FOLDER}/eki_dmc_boot.h5",
    f"{RESULTS_FOLDER}/eki_dmc_inf.h5"
]

ALGNAMES = ["EKI", "EKI-BOOT", "EKI-INF"]

PLOT_MESH = True
PLOT_TRUTH = False
PLOT_DATA = False

PLOT_PRIOR_PARTICLES = False
PLOT_PRIOR_FAULTS = False
PLOT_PRIOR_CAPS = False

PLOT_MEANS = False
PLOT_STDS = False
PLOT_POST_PARTICLES = False
PLOT_POST_FAULTS = False
PLOT_PREDICTIONS = False

PLOT_INTERVALS = False
PLOT_HYPERPARAMS = False

PLOT_CBARS = False

DATA_WELL = 3
WELL_TO_PLOT = 2

def read_results(algnames, fnames):

    results = {}

    for algname, fname in zip(algnames, fnames):

        with h5py.File(fname, "r") as f:

            post_ind = f["post_ind"][0]

            inds_succ_pri = f[f"inds_succ_0"][:]
            inds_succ_post = f[f"inds_succ_{post_ind}"][:]

            results[algname] = {
                "ws_pri" : f[f"ws_0"][:, inds_succ_pri],
                "ps_pri" : f[f"ps_0"][:, inds_succ_pri],
                "Fs_pri" : f[f"Fs_0"][:, inds_succ_pri],
                "Gs_pri" : f[f"Gs_0"][:, inds_succ_pri],
                "ws_post" : f[f"ws_{post_ind}"][:, inds_succ_post],
                "ps_post" : f[f"ps_{post_ind}"][:, inds_succ_post],
                "Fs_post" : f[f"Fs_{post_ind}"][:, inds_succ_post],
                "Gs_post" : f[f"Gs_{post_ind}"][:, inds_succ_post]
            }

    return results

fem_mesh_crse = pv.UnstructuredGrid(f"{mesh_crse.name}.vtu")
fem_mesh_fine = pv.UnstructuredGrid(f"{mesh_fine.name}.vtu")
results = read_results(ALGNAMES, RESULTS_FNAMES)

if PLOT_MESH:

    fname = f"{PLOTS_FOLDER}/mesh.png"
    plot_mesh_3d(fem_mesh_crse, wells_crse, feedzone_depths, fname)

    fname = f"{PLOTS_FOLDER}/mesh_coarse.pdf"
    plot_grid_layer_3d(mesh_crse, wells_crse, fname)

    fname = f"{PLOTS_FOLDER}/mesh_fine.pdf"
    plot_grid_layer_3d(mesh_fine, wells_fine, fname)

if PLOT_TRUTH:

    logks_t = p_t[:mesh_fine.m.num_cells]
    fname = f"{PLOTS_FOLDER}/perms_true.png"
    plot_slice(mesh_fine.m, fem_mesh_fine, logks_t, fname)

    temps_t = F_t[:mesh_fine.m.num_cells]
    fname = f"{PLOTS_FOLDER}/temps_true.png"
    plot_slice(mesh_fine.m, fem_mesh_fine, temps_t, fname, 
               cmap=CMAP_TEMP, vmin=MIN_TEMP_3D, vmax=MAX_TEMP_3D)

    upflows_t = p_t[-mesh_fine.m.num_columns:]
    fname = f"{PLOTS_FOLDER}/upflows_true.pdf"
    plot_fault_true_3d(mesh_fine.m, upflows_t, fname)

if PLOT_DATA:

    temp_t, pres_t, enth_t = data_handler_fine.get_full_states(F_t)
    zs, temp_t = data_handler_fine.downhole_temps(temp_t, DATA_WELL)
    temp_obs, pres_obs, enth_obs = data_handler_fine.split_obs(y)

    ts = data_handler_fine.ts / (52 * SECS_PER_WEEK)
    ts_obs = data_handler_crse.prod_obs_ts / (52 * SECS_PER_WEEK)
    zs_obs = np.array(temp_obs_zs)

    temp_lims_x = (0, 200)
    temp_lims_y = (-2600, -500)
    pres_lims_x = (0, 2)
    pres_lims_y = (4, 8)
    enth_lims_x = (0, 2)
    enth_lims_y = (500, 900)

    data_end = 1

    fname = f"{PLOTS_FOLDER}/data.pdf"
    plot_data(temp_t, pres_t, enth_t, zs, ts, 
              temp_obs, pres_obs, enth_obs, zs_obs, ts_obs, 
              temp_lims_x, temp_lims_y, pres_lims_x, pres_lims_y, 
              enth_lims_x, enth_lims_y, data_end, 
              DATA_WELL, fname)

if PLOT_PRIOR_PARTICLES:

    logks = results["EKI"]["ps_pri"][:mesh_crse.m.num_cells, :6].T
    fname = f"{PLOTS_FOLDER}/particles_pri.png"
    plot_slices_3d(mesh_crse.m, fem_mesh_crse, logks, fname)

if PLOT_PRIOR_FAULTS:

    upflows = results["EKI"]["ps_pri"][-mesh_crse.m.num_columns:, :8].T
    fname = f"{PLOTS_FOLDER}/upflows_pri.pdf"
    plot_faults_3d(mesh_crse.m, upflows, fname)

if PLOT_PRIOR_CAPS:

    ws_cap = results["EKI"]["ws_pri"][prior.inds["cap"], 10:13]
    cap_cell_inds = [prior.cap.get_cells_in_cap(w) for w in ws_cap.T]
    print(cap_cell_inds)
    fname = f"{PLOTS_FOLDER}/caps_pri.png"
    plot_caps_3d(mesh_crse.m, fem_mesh_crse, cap_cell_inds, fname)

if PLOT_MEANS:

    def get_mean(ws):
        mean_ws = np.mean(ws, axis=1)
        mean = prior.transform(mean_ws)[:mesh_crse.m.num_cells]
        return mean

    means = [
        p_t[:mesh_fine.m.num_cells],
        prior.transform(np.zeros(prior.n_params))[:mesh_crse.m.num_cells],
        get_mean(results["EKI"]["ws_post"]),
        get_mean(results["EKI-BOOT"]["ws_post"]),
        get_mean(results["EKI-INF"]["ws_post"])
    ]

    fname = f"{PLOTS_FOLDER}/means.png"
    plot_means_3d(mesh_fine.m, mesh_crse.m, 
                  fem_mesh_fine, fem_mesh_crse, means, fname)

if PLOT_STDS:
    
    def get_stds(ps):
        return np.std(ps, axis=1)[:mesh_crse.m.num_cells]
    
    stds = [
        get_stds(results["EKI"]["ps_pri"]),
        get_stds(results["EKI"]["ps_post"]),
        get_stds(results["EKI-BOOT"]["ps_post"]),
        get_stds(results["EKI-INF"]["ps_post"])
    ]

    fname = f"{PLOTS_FOLDER}/stds.png"
    plot_stds_3d(mesh_crse.m, fem_mesh_crse, stds, fname)

if PLOT_POST_PARTICLES:

    logks = results["EKI"]["ps_post"][:mesh_crse.m.num_cells, :6].T
    fname = f"{PLOTS_FOLDER}/particles_post.png"
    plot_slices_3d(mesh_crse.m, fem_mesh_crse, logks, fname)

if PLOT_POST_FAULTS:

    upflows = results["EKI"]["ps_post"][-mesh_crse.m.num_columns:, :8].T
    fname = f"{PLOTS_FOLDER}/upflows_post.pdf"
    plot_faults_3d(mesh_crse.m, upflows, fname)

if PLOT_PREDICTIONS:

    Fs = [
        results["EKI"]["Fs_pri"], 
        results["EKI"]["Fs_post"], 
        results["EKI-BOOT"]["Fs_post"], 
        results["EKI-INF"]["Fs_post"]
    ]

    temp_t, pres_t, enth_t = data_handler_fine.get_full_states(F_t)
    zs, temp_t = data_handler_fine.downhole_temps(temp_t, WELL_TO_PLOT)
    temp_obs, pres_obs, enth_obs = data_handler_fine.split_obs(y)

    ts = data_handler_fine.ts / (52 * SECS_PER_WEEK)
    ts_obs = data_handler_crse.prod_obs_ts / (52 * SECS_PER_WEEK)
    zs_obs = temp_obs_zs

    temp_lims_x = (0, 300)
    temp_lims_y = (-2600, -500)
    pres_lims_x = (0, 2)
    pres_lims_y = (1, 9)
    enth_lims_x = (0, 2)
    enth_lims_y = (200, 1300)

    data_end = 1

    fname = f"{PLOTS_FOLDER}/predictions.pdf"
    plot_predictions(Fs, data_handler_crse, 
                     temp_t, pres_t[:, WELL_TO_PLOT], enth_t[:, WELL_TO_PLOT], ts, zs, 
                     temp_obs, pres_obs, enth_obs, ts_obs, zs_obs, 
                     temp_lims_x, temp_lims_y, pres_lims_x, pres_lims_y,
                     enth_lims_x, enth_lims_y, data_end, 
                     WELL_TO_PLOT, fname)

if PLOT_INTERVALS:

    def get_interval(perms, perms_t):
        
        bnds = np.quantile(perms, [0.025, 0.975], axis=1)

        cells_in_interval = np.zeros((mesh_crse.m.num_cells))
        for i, (lb, ub, perm) in enumerate(zip(bnds[0], bnds[1], perms_t)):
            if not (lb-1e-4 <= perm <= ub+1e-4):
                cells_in_interval[i] = 1.0

        print(np.mean(cells_in_interval))
        return cells_in_interval

    centres_crse = [c.centre for c in mesh_crse.m.cell]
    centres_fine = [c.centre for c in mesh_fine.m.cell]

    perms_t = p_t[:mesh_fine.m.num_cells]
    interp = NearestNDInterpolator(centres_fine, perms_t)
    perms_interp = interp(centres_crse)

    perms_list = [
        results["EKI"]["ps_pri"][:mesh_crse.m.num_cells, :],
        results["EKI"]["ps_post"][:mesh_crse.m.num_cells, :],
        results["EKI-BOOT"]["ps_post"][:mesh_crse.m.num_cells, :],
        results["EKI-INF"]["ps_post"][:mesh_crse.m.num_cells, :]
    ]

    intervals = [get_interval(perms, perms_interp) 
                 for perms in perms_list]

    fname = f"{PLOTS_FOLDER}/intervals.png"
    plot_intervals_3d(mesh_crse.m, fem_mesh_crse, intervals, fname)

if PLOT_HYPERPARAMS:
    
    ind = 0 # External

    def get_hyperparams(ws):
        return np.array([prior.get_hyperparams(w_i)[ind] for w_i in ws.T])

    hps = [
        get_hyperparams(results["EKI"]["ws_pri"])[:, [0, 1, 3]],
        get_hyperparams(results["EKI"]["ws_post"])[:, [0, 1, 3]],
        get_hyperparams(results["EKI-BOOT"]["ws_post"])[:, [0, 1, 3]],
        get_hyperparams(results["EKI-INF"]["ws_post"])[:, [0, 1, 3]],
    ]

    hps_t = np.array(truth_dist.get_hyperparams(w_t)[ind])[[0, 1, 3]]

    std_lims_x = bounds_perm_ext[0]
    std_lims_y = (0, 4) 
    lenh_lims_x = bounds_perm_ext[1]
    lenh_lims_y = (0, 0.001)
    lenv_lims_x = bounds_perm_ext[3]
    lenv_lims_y = (0, 0.002)

    labels = ["Standard Deviation", "$x_{1}$ Lengthscale [m]", "$x_{3}$ Lengthscale [m]"]

    fname = f"{PLOTS_FOLDER}/hyperparams.pdf"
    plot_hyperparams(hps, hps_t, std_lims_x, std_lims_y, lenh_lims_x, 
                     lenh_lims_y, lenv_lims_x, lenv_lims_y, labels, fname) 

if PLOT_CBARS:

    temp_fname = f"{PLOTS_FOLDER}/cbar_temps.pdf"
    perm_fname = f"{PLOTS_FOLDER}/cbar_perms.pdf"
    stds_fname = f"{PLOTS_FOLDER}/cbar_stds.pdf"

    plot_colourbar(CMAP_TEMP, MIN_TEMP_3D, MAX_TEMP_3D, LABEL_TEMP, temp_fname)
    plot_colourbar(CMAP_PERM, MIN_PERM_3D, MAX_PERM_3D, LABEL_PERM, perm_fname)
    plot_colourbar(CMAP_PERM, MIN_STDS_3D, MAX_STDS_3D, LABEL_PERM, stds_fname)