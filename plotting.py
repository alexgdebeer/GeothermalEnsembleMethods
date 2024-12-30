from copy import deepcopy

import cmocean
from layermesh import mesh as lm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
import pyvista as pv

from GeothermalEnsembleMethods import DataHandler, Prior, Well


np.random.seed(1)

FULL_PAGE = 14.0
HALF_PAGE = 7.0

MIN_PERM_3D = -17.0
MAX_PERM_3D = -12.5

MIN_PERM_2D = -17.0
MAX_PERM_2D = -13.0

MIN_TEMP_2D = 30
MAX_TEMP_2D = 375

MIN_TEMP_3D = 20
MAX_TEMP_3D = 300

MIN_STDS_2D = 0.0
MAX_STDS_2D = 1.5

MIN_STDS_3D = 0.0
MAX_STDS_3D = 1.5

MIN_UPFL_3D = 0.0
MAX_UPFL_3D = 2.75e-4

CMAP_PERM = cmocean.cm.turbid.reversed()
CMAP_UPFL = cmocean.cm.thermal
CMAP_TEMP = cmocean.cm.balance
CMAP_INTERVALS = LinearSegmentedColormap.from_list(
    name="intervals", 
    colors=["silver", "red"]
)

COL_WELLS = "royalblue"
COL_GRID = "darkgrey"

COL_SHAL = "silver"
COL_CLAY = "gainsboro"
COL_DEEP = "whitesmoke"

COL_UPFL = "tab:red"

COL_TEMP = "tab:orange"
COL_PRES = "tab:blue"
COL_ENTH = "tab:green"

COL_STD = "tab:orange"
COL_LENH = "tab:blue"
COL_LENV = "tab:green"

COL_DATA_END = "darkgrey"

SLICE_HEIGHT = 1200

ALG_LABELS = ["Prior", "EKI", "EKI (Localisation)", "EKI (Inflation)"]

LABEL_ELEV = "Elevation [m]"
LABEL_TIME = "Time [Years]"

LABEL_PERM = r"$\log_{10}(k)$ [$\log_{10}$(m$^2$)]"
LABEL_UPFL_2D = r"Upflow [kg s$^{-1}$]"
LABEL_UPFL_3D = r"Mass Flux [kg s$^{-1}$ m$^{-2}$]"

LABEL_TEMP = r"Temperature [$^{\circ}$C]"
LABEL_PRES = r"Pressure [MPa]"
LABEL_ENTH = r"Enthalpy [kJ kg$^{-1}$]"

LABEL_STD = "Standard Deviation"
LABEL_LENH = "Horizontal Lengthscale [m]"
LABEL_LENV = "Vertical Lengthscale [m]"

LABEL_X1 = r"$x_{1}$ [km]"
LABEL_X2 = r"$x_{2}$ [km]"
LABEL_X3 = r"$x_{3}$ [km]"

CAMERA_POSITION = (13_000, 15_000, 6_000)


def tufte_axis(
    ax, 
    bnds_x: tuple, 
    bnds_y: tuple, 
    gap: int=0.1
) -> None:
    """Restyles an axis."""
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_bounds(*bnds_x)
    ax.spines["left"].set_bounds(*bnds_y)

    dx = bnds_x[1] - bnds_x[0]
    dy = bnds_y[1] - bnds_y[0]

    ax.set_xlim(bnds_x[0] - gap*dx, bnds_x[1] + gap*dx)
    ax.set_ylim(bnds_y[0] - gap*dy, bnds_y[1] + gap*dy)

    ax.set_xticks(bnds_x)
    ax.set_yticks(bnds_y)


def get_well_name(i: int):
    """Returns a string containing the name of a given well."""
    return r"\texttt{WELL " + f"{i+1}" + r"}"


def map_to_fem_mesh(mesh, fem_mesh, vals):
    """Maps a set of values from a layermesh mesh onto a PyVista mesh.
    """
    return [vals[mesh.find(c.center, indices=True)] for c in fem_mesh.cell]


def get_well_tubes(
    wells: list[Well],
    feedzone_depths: list[float]
) -> list:
    """Returns a set of tubes and well feedzones to be plotted using 
    PyVista.
    """
        
    lines = pv.MultiBlock()
    for well in wells:
        line = pv.Line(*well.coords)
        lines.append(line)
    
    bodies = (
        lines
        .combine()
        .extract_geometry()
        .clean()
        .split_bodies()
    )

    tubes = pv.MultiBlock()
    for body in bodies:
        tubes.append(body.extract_geometry().tube(radius=35))

    for i, well in enumerate(wells):
        centre = (well.x, well.y, feedzone_depths[i])
        feedzone = pv.SolidSphere(outer_radius=70, center=centre)
        tubes.append(feedzone)

    return tubes


def get_layer_polys(
    mesh: lm.mesh, 
    cmap: str
) -> PolyCollection:
    """Returns a collection of polygons correpsonding to the bottom
    layer of a mesh.
    """

    verts = [
        [n.pos for n in c.column.node] 
        for c in mesh.layer[-1].cell
    ]

    polys = PolyCollection(
        verts, 
        cmap=cmap, 
        linewidth=0.01, 
        edgecolor="face"
    )

    return polys


def plot_data(
    temp_t: np.ndarray, 
    pres_t: np.ndarray, 
    enth_t: np.ndarray, 
    zs: np.ndarray, 
    ts: np.ndarray, 
    temp_obs: np.ndarray, 
    pres_obs: np.ndarray, 
    enth_obs: np.ndarray, 
    zs_obs: np.ndarray, 
    ts_obs: np.ndarray, 
    temp_lims_x: tuple, 
    temp_lims_y: tuple, 
    pres_lims_x: tuple, 
    pres_lims_y: tuple,
    enth_lims_x: tuple, 
    enth_lims_y: tuple, 
    data_end: float, 
    well_to_plot: int, 
    fname: str
) -> None:
    """Plots the temperature, pressure and enthalpy data corresponding 
    to a given well, as well as the true values of each quantity.
    """
    
    figsize = (0.75*FULL_PAGE, 0.25*FULL_PAGE)
    _, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].plot(temp_t, zs, c="k", zorder=3)
    axes[1].plot(ts, pres_t[:, well_to_plot], c="k", zorder=3)
    axes[2].plot(ts, enth_t[:, well_to_plot], c="k", zorder=3)

    axes[0].scatter(temp_obs[:, well_to_plot], zs_obs, c="k", s=15, zorder=3)
    axes[1].scatter(ts_obs, pres_obs[:, well_to_plot], c="k", s=15, zorder=3)
    axes[2].scatter(ts_obs, enth_obs[:, well_to_plot], c="k", s=15, zorder=3)

    axes[1].axvline(data_end, c=COL_DATA_END, ls="--", ymin=1/12, ymax=11/12, zorder=1)
    axes[2].axvline(data_end, c=COL_DATA_END, ls="--", ymin=1/12, ymax=11/12, zorder=1)

    tufte_axis(axes[0], temp_lims_x, temp_lims_y)
    tufte_axis(axes[1], pres_lims_x, pres_lims_y)
    tufte_axis(axes[2], enth_lims_x, enth_lims_y)

    for ax in axes.flat:
        ax.set_box_aspect(1)

    axes[0].set_ylabel(LABEL_ELEV)
    axes[1].set_ylabel(LABEL_PRES)
    axes[2].set_ylabel(LABEL_ENTH)

    axes[0].set_xlabel(LABEL_TEMP)
    axes[1].set_xlabel(LABEL_TIME)
    axes[2].set_xlabel(LABEL_TIME)

    plt.savefig(fname)


def plot_predictions(
    Fs: list[np.ndarray], 
    data_handler: DataHandler, 
    temp_t: np.ndarray, 
    pres_t: np.ndarray, 
    enth_t: np.ndarray, 
    ts: np.ndarray, 
    zs: np.ndarray,
    temp_obs: np.ndarray, 
    pres_obs: np.ndarray, 
    enth_obs: np.ndarray, 
    ts_obs: np.ndarray, 
    zs_obs: np.ndarray,
    temp_lims_x: tuple, 
    temp_lims_y: tuple, 
    pres_lims_x: tuple, 
    pres_lims_y: tuple,
    enth_lims_x: tuple, 
    enth_lims_y: tuple, 
    data_end: float, 
    well_num: int, 
    fname: str, 
    dim: int=3
) -> None:
    """Plots the temperature, pressure and enthalpy predictions 
    corresponding to a given well, alongside the truth and the 
    observations.
    """

    num_cols = len(Fs)

    figsize = (FULL_PAGE*(num_cols/4.0), 1.6*HALF_PAGE)
    fig, axes = plt.subplots(3, num_cols, figsize=figsize, sharey="row")

    for j in range(num_cols):

        tufte_axis(axes[0][j], temp_lims_x, temp_lims_y)
        tufte_axis(axes[1][j], pres_lims_x, pres_lims_y)
        tufte_axis(axes[2][j], enth_lims_x, enth_lims_y)

        axes[0][j].plot(temp_t, zs, c="k", zorder=3)
        axes[1][j].plot(ts, pres_t, c="k", zorder=3)
        axes[2][j].plot(ts, enth_t, c="k", zorder=3)

        axes[0][j].scatter(temp_obs[:, well_num], zs_obs, c="k", s=10, zorder=3)
        axes[1][j].scatter(ts_obs, pres_obs[:, well_num], c="k", s=10, zorder=3)
        axes[2][j].scatter(ts_obs, enth_obs[:, well_num], c="k", s=10, zorder=3)

        axes[1][j].axvline(data_end, c=COL_DATA_END, ls="--", ymin=1/12, ymax=11/12, zorder=1)
        axes[2][j].axvline(data_end, c=COL_DATA_END, ls="--", ymin=1/12, ymax=11/12, zorder=1)

        for F_j in Fs[j].T:

            temp_j, pres_j, enth_j = data_handler.get_full_states(F_j)

            if dim == 3:
                zs_j, temp_j = data_handler.downhole_temps(temp_j, well_num)
            else: 
                zs_j, temp_j = data_handler.downhole_temps(temp_j)
                temp_j = temp_j[:, well_num]
            
            axes[0][j].plot(temp_j, zs_j, c=COL_TEMP, zorder=2, alpha=0.4)
            axes[1][j].plot(ts, pres_j[:, well_num], c=COL_PRES, zorder=2, alpha=0.4)
            axes[2][j].plot(ts, enth_j[:, well_num], c=COL_ENTH, zorder=2, alpha=0.4)

    for j in range(num_cols):
        axes[0][j].set_xlabel(LABEL_TEMP)
        axes[1][j].set_xlabel(LABEL_TIME)
        axes[2][j].set_xlabel(LABEL_TIME)
        axes[0][j].set_title(ALG_LABELS[j])

    for ax in axes.flat:
        ax.set_box_aspect(1)

    axes[0][0].set_ylabel(LABEL_ELEV)
    axes[1][0].set_ylabel(LABEL_PRES)
    axes[2][0].set_ylabel(LABEL_ENTH)

    fig.align_ylabels()
    plt.savefig(fname)


def plot_hyperparams(
    hps: list[np.ndarray], 
    hps_t: list[float], 
    std_lims_x: tuple, 
    std_lims_y: tuple, 
    lenh_lims_x: tuple, 
    lenh_lims_y: tuple, 
    lenv_lims_x: tuple, 
    lenv_lims_y: tuple, 
    labels: tuple, 
    fname: str
) -> None:
    """Generates plots of the hyperparameters of a Whittle-Matern field
    (i.e., standard deviation, a horizontal lengthscale and the vertical 
    lengthscale.)
    """

    figsize = (FULL_PAGE, 0.9 * FULL_PAGE)
    fig, axes = plt.subplots(3, 4, figsize=figsize, sharey="row")

    bins_std = np.linspace(std_lims_x[0], std_lims_x[1], 11)
    bins_lenh = np.linspace(lenh_lims_x[0], lenh_lims_x[1], 11)
    bins_lenv = np.linspace(lenv_lims_x[0], lenv_lims_x[1], 11)

    for j in range(4):

        tufte_axis(axes[0][j], std_lims_x, std_lims_y, gap=0.05)
        tufte_axis(axes[1][j], lenh_lims_x, lenh_lims_y, gap=0.05)
        tufte_axis(axes[2][j], lenv_lims_x, lenv_lims_y, gap=0.05)

        axes[0][j].hist(hps[j][:, 0], density=True, color=COL_STD, bins=bins_std, zorder=1)
        axes[1][j].hist(hps[j][:, 1], density=True, color=COL_LENH, bins=bins_lenh, zorder=1)
        axes[2][j].hist(hps[j][:, 2], density=True, color=COL_LENV, bins=bins_lenv, zorder=1)
        
        axes[0][j].axvline(hps_t[0], c="k", ymin=1/20, ymax=19/20, zorder=2)
        axes[1][j].axvline(hps_t[1], c="k", ymin=1/20, ymax=19/20, zorder=2)
        axes[2][j].axvline(hps_t[2], c="k", ymin=1/20, ymax=19/20, zorder=2)

    axes[0][0].set_title(ALG_LABELS[0])
    axes[0][1].set_title(ALG_LABELS[1])
    axes[0][2].set_title(ALG_LABELS[2])
    axes[0][3].set_title(ALG_LABELS[3])

    for i in range(3):
        axes[i][0].set_ylabel("Density")
        for j in range(4):
            axes[i][j].set_xlabel(labels[i])

    for ax in axes.flat:
        ax.set_box_aspect(1)

    fig.align_ylabels()
    plt.savefig(fname)


def plot_colourbar(
    cmap: str, 
    vmin: float, 
    vmax: float, 
    label: str, 
    fname: str, 
    power: bool=False
) -> None:
    """Plots a colourbar for use as part of another plot."""
    
    _, ax = plt.subplots(figsize=(HALF_PAGE, 0.4*HALF_PAGE))
    m = ax.pcolormesh(np.zeros((10, 10)), cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(m, ax=ax)
    cbar.set_label(label)
    if power:
        cbar.formatter.set_powerlimits((0, 0))
    plt.savefig(fname)


def plot_mesh_2d(
    mesh: lm.mesh, 
    prior: Prior, 
    wells: list[Well], 
    fname: str 
) -> None:
    """Plots the mesh of the vertical slice model."""

    _, ax = plt.subplots(figsize=(HALF_PAGE, 0.7*HALF_PAGE))

    ws = prior.sample().squeeze()
    zs_bound_2 = prior.gp_boundary.transform(ws[:mesh.nx])

    # Define grid
    grid = np.zeros((mesh.nx, mesh.nz))
    grid = np.ma.masked_array(grid, grid == 0.0)

    # Define coordinates of boundaries
    xs_bound_1 = [0, 1500]
    zs_bound_1 = [-60, -60]

    zs_bound_2 = [1.5 * zs_bound_2[0] - 0.5 *zs_bound_2[1], *zs_bound_2, 
                  1.5 * zs_bound_2[-1] - 0.5 * zs_bound_2[-2]]
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

    zone_1 = Polygon(cs_zone_1, facecolor=COL_SHAL, zorder=0)
    zone_2 = Polygon(cs_zone_2, facecolor=COL_CLAY, zorder=0)
    zone_3 = Polygon(cs_zone_3, facecolor=COL_DEEP, zorder=0)

    ax.pcolormesh(mesh.xs, mesh.zs, grid, edgecolors=COL_GRID, lw=1.5)

    ax.add_patch(zone_1) 
    ax.add_patch(zone_2)
    ax.add_patch(zone_3)

    ax.plot(xs_bound_1, zs_bound_1, c="k", linewidth=2.0, linestyle=(0, (5, 1)), zorder=1)
    ax.plot(xs_bound_2, zs_bound_2, c="k", linewidth=2.0, zorder=1)

    for (i, well) in enumerate(wells):
        x, z = well.x, well.feedzone_cell.centre[-1]
        ax.plot([x, x], [0, -1300], linewidth=2.0, color=COL_WELLS, zorder=2)
        ax.scatter([x], [z], color=COL_WELLS, s=35)
        plt.text(x-110, 40, s=get_well_name(i), color=COL_WELLS)

    ax.set_xlabel(LABEL_X1)
    ax.set_ylabel(LABEL_X3)
    ax.set_box_aspect(1)
    set_lims_2d(ax, remove_spines=False)
    ax.tick_params(length=0)

    legend_elements = [
        Patch(facecolor=COL_SHAL, edgecolor=COL_GRID, label=r"$\Omega_{\mathcal{S}}$"),
        Patch(facecolor=COL_CLAY, edgecolor=COL_GRID, label=r"$\Omega_{\mathcal{C}}$"),
        Patch(facecolor=COL_DEEP, edgecolor=COL_GRID, label=r"$\Omega_{\mathcal{D}}$"),
        Line2D([0], [0], c="k", ls=(0, (5, 1)), label=r"$\omega_{\mathcal{S}}$"),
        Line2D([0], [0], c="k", label=r"$\omega_{\mathcal{D}}$"),
        Line2D([0], [0], c=COL_WELLS, label="Well Tracks"),
        Line2D([0], [0], c=COL_WELLS, marker="o", ms=5, ls="", label="Feedzones"),
    ]

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.75), frameon=False)

    for s in ax.spines.values():
        s.set(edgecolor=COL_GRID, lw=1.5)

    plt.savefig(fname)


def set_lims_2d(ax, remove_spines=True):
    """Sets the limits of an axis object as the boundaries of domain of 
    the vertical slice model.
    """

    ax.set_box_aspect(1)
    ax.set_xlim([0, 1500])
    ax.set_ylim([-1500, 0])
    ax.set_xticks([0, 1500])
    ax.set_yticks([-1500, 0])
    ax.set_xticklabels([r"$0$", r"$1.5$"])
    ax.set_yticklabels([r"$-1.5$", r"$0$"])
    ax.tick_params(length=0)

    if remove_spines:
        for s in ax.spines:
            ax.spines[s].set_visible(False)


def plot_truth_2d(mesh, perm_t, temp_t, fname):

    size = (0.6*HALF_PAGE, 0.8*HALF_PAGE)
    fig, axes = plt.subplots(2, 1, figsize=size, sharex=True)

    perm_mesh = axes[0].pcolormesh(
        mesh.xs, mesh.zs, perm_t, 
        vmin=MIN_PERM_2D, vmax=MAX_PERM_2D, 
        cmap=CMAP_PERM, rasterized=True
    )

    temp_mesh = axes[1].pcolormesh(
        mesh.xs, mesh.zs, temp_t,
        vmin=MIN_TEMP_2D, vmax=MAX_TEMP_2D, 
        cmap=CMAP_TEMP, rasterized=True
    )

    perm_cbar = fig.colorbar(perm_mesh, ax=axes[0])
    temp_cbar = fig.colorbar(temp_mesh, ax=axes[1])

    perm_cbar.set_label(LABEL_PERM)
    temp_cbar.set_label(LABEL_TEMP)

    for ax in axes:
        ax.set_ylabel(LABEL_X3)
        set_lims_2d(ax)
    
    axes[1].set_xlabel(LABEL_X1)

    plt.savefig(fname)


def plot_particles_2d(mesh, vals, fname, vmin=MIN_PERM_2D, 
                      vmax=MAX_PERM_2D, cmap=CMAP_PERM):

    fig, axes = plt.subplots(2, 2, figsize=(HALF_PAGE, 0.825*HALF_PAGE), sharex=True, sharey=True,)

    for i, ax in enumerate(axes.flat):

        val = np.reshape(vals.T[i], (mesh.nx, mesh.nz))
        im = ax.pcolormesh(
            mesh.xs, mesh.zs, val, 
            vmin=vmin, vmax=vmax, 
            cmap=cmap, rasterized=True
        )
        
        set_lims_2d(ax)

    for ax in axes[-1]:
        ax.set_xlabel(LABEL_X1)
    for ax in axes.T[0]:
        ax.set_ylabel(LABEL_X3)

    cbar = fig.colorbar(im, ax=axes, shrink=0.5)
    cbar.set_label(LABEL_PERM)

    plt.savefig(fname)


def plot_upflows_2d(upflow_t, upflows, bnds_x, bnds_y, fname):

    n_subfigs = len(upflows)

    size = (n_subfigs/2*HALF_PAGE, 0.58*HALF_PAGE)
    _, axes = plt.subplots(1, n_subfigs, figsize=size, sharey=True)
    bins = np.linspace(bnds_x[0], bnds_x[1], 11)

    for i, ax in enumerate(axes):

        ax.hist(upflows[i], color=COL_UPFL, density=True, bins=bins, zorder=1)
        ax.axvline(upflow_t, c="k", ymin=1/20, ymax=19/20, zorder=2)

        tufte_axis(ax, bnds_x, bnds_y, gap=0.05)
        ax.set_box_aspect(1)
        ax.set_xlabel(LABEL_UPFL_2D)

    for i, ax in enumerate(axes.flat):
        ax.set_title(ALG_LABELS[i])
    axes[0].set_ylabel("Density")

    plt.savefig(fname)


def plot_grid_2d(vals, meshes, labels, fname, vmin=MIN_PERM_2D, 
                 vmax=MAX_PERM_2D, cmap=CMAP_PERM, cbar=True):

    n_vals = len(vals)

    width = n_vals*FULL_PAGE/4
    height = 0.235*FULL_PAGE if cbar else 0.25*FULL_PAGE
    height = 0.25*FULL_PAGE
    size = (width, height)
    fig, axes = plt.subplots(1, n_vals, figsize=size, sharey=True)

    for i, (v, m, l) in enumerate(zip(vals, meshes, labels)):
        
        mesh = axes[i].pcolormesh(
            m.xs, m.zs, v, 
            vmin=vmin, vmax=vmax, 
            cmap=cmap, rasterized=True
        )

        axes[i].set_title(l)

    for ax in axes.flat:
        set_lims_2d(ax)
        ax.set_xlabel(LABEL_X1)

    axes[0].set_ylabel(LABEL_X3)

    if cbar:
        cbar = fig.colorbar(mesh, ax=axes[-1])
        cbar.set_label(LABEL_PERM)
    

    plt.savefig(fname)


def plot_mesh_3d(fem_mesh: pv.UnstructuredGrid, wells, feedzone_depths, fname):

    tubes = get_well_tubes(wells, feedzone_depths)
    edges = fem_mesh.extract_geometry().extract_all_edges()

    p = pv.Plotter(off_screen=True, window_size=(2400, 2400))
    p.add_mesh(fem_mesh, color=COL_DEEP, opacity=0.2)
    p.add_mesh(tubes, color=COL_WELLS)
    p.add_mesh(edges, line_width=2, color=COL_GRID)

    p.camera.position = CAMERA_POSITION
    p.add_light(pv.Light(position=CAMERA_POSITION, intensity=0.2))
    p.screenshot(fname)


def plot_grid_layer_3d(mesh, wells, fname):

    _, ax = plt.subplots(figsize=(0.5 * FULL_PAGE, 0.5 * FULL_PAGE))

    elevs = [c.column.surface for c in mesh.m.cell]

    mesh.m.layer_plot(axes=ax, linewidth=0.75, value=elevs, colourmap="terrain", linecolour=COL_GRID)

    for i, well in enumerate(wells):
        ax.scatter(well.x, well.y, s=20, c=COL_WELLS)
        plt.text(well.x-360, well.y+160, s=get_well_name(i), color=COL_WELLS)
    
    ax.set_xlabel(LABEL_X1)
    ax.set_ylabel(LABEL_X2)
    
    for s in ax.spines.values():
        s.set_edgecolor(COL_GRID)
    
    ax.set_xlim(0, 6000)
    ax.set_ylim(0, 6000)
    ax.set_xticks([0, 3000, 6000])
    ax.set_yticks([0, 3000, 6000])
    ax.tick_params(length=0)
    ax.set_facecolor(COL_DEEP)
    
    plt.savefig(fname)


def plot_fault_true_3d(mesh, vals, fname):

    total_upflow = sum([c.area * u for u, c in zip(vals, mesh.column)])
    # print(total_upflow)

    _, ax = plt.subplots(figsize=(0.25*FULL_PAGE, 0.18*FULL_PAGE))

    polys = get_layer_polys(mesh, CMAP_UPFL)
    polys.set_array(vals)
    polys.set_clim(MIN_UPFL_3D, MAX_UPFL_3D)

    ax.add_collection(polys)
    cbar = plt.colorbar(polys, ax=ax)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.set_label(LABEL_UPFL_3D)

    ax.set_xlim(0, 6000)
    ax.set_ylim(0, 6000)
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(fname)


def set_layer_axes(ax):

    ax.set_xlim(0, 6000)
    ax.set_ylim(0, 6000)
    ax.set_box_aspect(1)
    ax.set_xticks([0, 3000, 6000])
    ax.set_yticks([0, 3000, 6000])
    ax.set_xticklabels([0, 3, 6])
    ax.set_yticklabels([0, 3, 6])
    ax.tick_params(length=0)


def plot_faults_3d(mesh: lm.mesh, upflows, fname):

    size = (HALF_PAGE, 0.825*HALF_PAGE)
    fig, axes = plt.subplots(2, 2, figsize=size, sharex=True, sharey=True)
    polys = get_layer_polys(mesh, CMAP_UPFL)
    polys.set_clim(0, 2.75e-4)

    for i, ax in enumerate(axes.flat):

        polys_i = deepcopy(polys)
        ax.add_collection(polys_i)
        polys_i.set_array(upflows[i])

        set_layer_axes(ax)
    
    for ax in axes[1]:
        ax.set_xlabel(LABEL_X1)

    for ax in axes.T[0]:
        ax.set_ylabel(LABEL_X2)

    cbar = fig.colorbar(polys_i, ax=axes, shrink=0.5)
    cbar.set_label(LABEL_UPFL_3D)
    cbar.formatter.set_powerlimits((0, 0))

    plt.savefig(fname)


def plot_means_3d(mesh_fine, mesh_crse, fem_mesh_fine, fem_mesh_crse, 
                  means, fname, cmap=CMAP_PERM,
                  vmin=MIN_PERM_3D, vmax=MAX_PERM_3D):
    
    n_means = len(means)
    window_size = (n_means * SLICE_HEIGHT, SLICE_HEIGHT)

    p = pv.Plotter(shape=(1, n_means), window_size=window_size, border=False, off_screen=True)
    p.add_light(pv.Light(position=CAMERA_POSITION, intensity=0.5))

    outline_fine = fem_mesh_fine.outline()
    outline_crse = fem_mesh_crse.outline()

    for i, mean in enumerate(means):

        if i == 0:
            mesh = mesh_fine
            fem_mesh = fem_mesh_fine
            outline = outline_fine
        else:
            mesh = mesh_crse 
            fem_mesh = fem_mesh_crse
            outline=outline_crse

        fem_mesh["vals"] = map_to_fem_mesh(mesh, fem_mesh, mean)
        slice = fem_mesh.clip(normal="x")

        p.subplot(0, i)
        p.add_mesh(outline, line_width=6, color=COL_GRID)
        p.add_mesh(slice, cmap=cmap, clim=(vmin, vmax), show_scalar_bar=False)
        p.camera.position = CAMERA_POSITION

    p.screenshot(fname)


def plot_stds_3d(mesh, fem_mesh, stds, fname):
    
    num_stds = len(stds)

    wsize = (num_stds * SLICE_HEIGHT, SLICE_HEIGHT)
    p = pv.Plotter(shape=(1, num_stds), window_size=wsize, border=False, off_screen=True)

    outline = fem_mesh.outline()
    p.add_light(pv.Light(position=CAMERA_POSITION, intensity=0.5))

    for i, std in enumerate(stds):

        print(np.mean(std))

        fem_mesh["vals"] = map_to_fem_mesh(mesh, fem_mesh, std)
        slice = fem_mesh.clip(normal="x")

        p.subplot(0, i)
        p.add_mesh(outline, line_width=6, color=COL_GRID)
        p.add_mesh(slice, cmap=CMAP_PERM, clim=(MIN_STDS_3D, MAX_STDS_3D), show_scalar_bar=False)
        p.camera.position = CAMERA_POSITION

    p.screenshot(fname)


def plot_intervals_3d(mesh, fem_mesh, intervals, fname):
    
    num_intervals = len(intervals)
    wsize = (num_intervals * SLICE_HEIGHT, SLICE_HEIGHT)
    p = pv.Plotter(shape=(1, num_intervals), window_size=wsize, border=False, off_screen=True)

    outline = fem_mesh.outline()
    p.add_light(pv.Light(position=CAMERA_POSITION, intensity=0.5))

    for i, interval in enumerate(intervals):

        fem_mesh["vals"] = map_to_fem_mesh(mesh, fem_mesh, interval)
        slice = fem_mesh.clip(normal="x")

        p.subplot(0, i)
        p.add_mesh(outline, line_width=6, color=COL_GRID)
        p.add_mesh(slice, cmap=CMAP_INTERVALS, show_scalar_bar=False)
        p.camera.position = CAMERA_POSITION

    p.screenshot(fname)


def plot_slice(mesh, fem_mesh, vals, fname, cmap=CMAP_PERM, vmin=MIN_PERM_3D, vmax=MAX_PERM_3D):
    
    p = pv.Plotter(window_size=(1200, 1200), off_screen=True)#, border=False)
    p.add_light(pv.Light(position=(4000, 3500, 1000), intensity=0.5))

    outline = fem_mesh.outline()
    fem_mesh["vals"] = map_to_fem_mesh(mesh, fem_mesh, vals)
    slice = fem_mesh.clip(normal="x")

    p.add_mesh(outline, line_width=6, color=COL_GRID)
    p.add_mesh(slice, cmap=cmap, clim=(vmin, vmax), show_scalar_bar=False)
    p.camera.position = CAMERA_POSITION
    
    p.screenshot(fname, transparent_background=True)


def plot_slices_3d(mesh, fem_mesh, vals, fname, cmap=CMAP_PERM, vmin=MIN_PERM_3D, vmax=MAX_PERM_3D):

    p = pv.Plotter(shape=(2, 2), window_size=(650, 650), border=False, off_screen=True)

    outline = fem_mesh.outline()
    p.add_light(pv.Light(position=(4000, 3500, 1000), intensity=0.5))

    for i in range(2):
        for j in range(2):

            fem_mesh["vals"] = map_to_fem_mesh(mesh, fem_mesh, vals[3*i+j])
            slice = fem_mesh.clip(normal="x")

            p.subplot(i, j)
            p.add_mesh(outline, line_width=6, color=COL_GRID)
            p.add_mesh(slice, cmap=cmap, clim=(vmin, vmax), show_scalar_bar=False)
            p.camera.position = CAMERA_POSITION
            
    p.screenshot(fname, scale=3)


def convert_inds(mesh, fem_mesh, inds):
    return [1 if mesh.find(c.center, indices=True) in inds else 0 
            for c in fem_mesh.cell]


def plot_caps_3d(mesh, fem_mesh, cap_cell_inds, fname):

    camera_position_top = (6000, 7000, 3000)
    camera_position_bottom = (6000, 7000, -4000)# (1000, 2000, 6000)

    caps = [convert_inds(mesh, fem_mesh, cap_cell_inds[i]) 
            for i in range(2)]

    p = pv.Plotter(shape=(2, 2), window_size=(2400, 2000), border=False, off_screen=True)

    light = pv.Light(position=(0, 0, -2000), intensity=0.35)
    p.add_light(light)

    for j in range(2):

        fem_mesh["cap"] = caps[j]

        cap = fem_mesh.threshold(0.5)
        cap_r = cap.rotate_x(0)

        outline = cap.extract_surface().extract_feature_edges()
        outline_r = cap_r.extract_surface().extract_feature_edges()

        p.subplot(0, j)
        p.add_mesh(cap, color="#b48d3e", lighting=True)
        p.add_mesh(outline, line_width=3, color="k")
        p.camera.position = camera_position_top

        p.subplot(1, j)
        p.add_mesh(cap_r, color="#b48d3e", lighting=True)
        p.add_mesh(outline_r, line_width=3, color="k")
        p.camera.position = camera_position_bottom

    p.screenshot(fname)
