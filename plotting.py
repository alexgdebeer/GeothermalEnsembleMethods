from copy import deepcopy

import colorcet
from layermesh import mesh as lm
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
import pyvista as pv

from src.consts import *

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

np.random.seed(1)

TITLE_SIZE = 16
LABEL_SIZE = 14
LEGEND_SIZE = 14
TICK_SIZE = 12

MIN_PERM_3D = -17.0
MAX_PERM_3D = -12.5

MIN_PERM_2D = -17.0
MAX_PERM_2D = -13.0

MIN_TEMP_2D = 30
MAX_TEMP_2D = 330

MIN_TEMP_3D = 20
MAX_TEMP_3D = 300

MIN_STDS_2D = 0.0
MAX_STDS_2D = 1.5

MIN_STDS_3D = 0.0
MAX_STDS_3D = 1.5

CMAP_PERM = "cet_bgy"
CMAP_TEMP = "coolwarm"
CMAP_UPFLOW = "cet_fire"
CMAP_INTERVALS = "coolwarm"

COL_WELLS = "royalblue"
COL_GRID = "darkgrey"

COL_SHAL = "silver"
COL_CLAY = "gainsboro"
COL_DEEP = "whitesmoke"

COL_UPFL = "mediumorchid"

COL_TEMP = "darkorange"
COL_PRES = "limegreen"
COL_ENTH = "deepskyblue"

COL_STD = "darkorange"
COL_LENH = "limegreen"
COL_LENV = "deepskyblue"

COL_DATA_END = "darkgrey"

FULL_WIDTH = 10.0
SLICE_HEIGHT = 1200

ALG_LABELS = ["Prior", "EKI", "EKI (Localisation)", "EKI (Inflation)"]

LABEL_ELEV = "Elevation [m]"
LABEL_TIME = "Time [Years]"

LABEL_PERM = "log$_{10}$(Permeability) [log$_{10}$(m$^2$)]"
LABEL_UPFL = "Upflow [kg/s]"

LABEL_TEMP = "Temperature [$^{\circ}$C]"
LABEL_PRES = "Pressure [MPa]"
LABEL_ENTH = "Enthalpy [kJ/kg]"

LABEL_STD = "Standard Deviation"
LABEL_LENH = "Horizontal Lengthscale [m]"
LABEL_LENV = "Vertical Lengthscale [m]"

LABEL_X1 = "$x_{1}$ [m]"
LABEL_X2 = "$x_{2}$ [m]"

CAMERA_POSITION = (13_000, 15_000, 6_000)

def tufte_axis(ax, bnds_x, bnds_y, gap=0.1):
    
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

def get_well_name(i):
    return r"\texttt{WELL " + f"{i+1}" + r"}"

def map_to_fem_mesh(mesh, fem_mesh, vals):
    return [vals[mesh.find(c.center, indices=True)] for c in fem_mesh.cell]

def get_well_tubes(wells, feedzone_depths):
        
    lines = pv.MultiBlock()
    for well in wells:
        line = pv.Line(*well.coords)
        lines.append(line)
    bodies = lines.combine().extract_geometry().clean().split_bodies()

    tubes = pv.MultiBlock()
    for body in bodies:
        tubes.append(body.extract_geometry().tube(radius=35))

    for i, well in enumerate(wells):
        feedzone = (well.x, well.y, feedzone_depths[i])
        tubes.append(pv.SolidSphere(outer_radius=70, center=feedzone))

    return tubes

def get_layer_polys(mesh: lm.mesh, cmap):

    verts = [[n.pos for n in c.column.node] 
             for c in mesh.layer[-1].cell]

    polys = PolyCollection(verts, cmap=cmap)
    return polys


def plot_data(temp_t, pres_t, enth_t, zs, ts, 
              temp_obs, pres_obs, enth_obs, zs_obs, ts_obs, 
              temp_lims_x, temp_lims_y, pres_lims_x, pres_lims_y,
              enth_lims_x, enth_lims_y, data_end, 
              well_to_plot, fname):
    
    _, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 3))

    axes[0].plot(temp_t, zs, c="k", zorder=3)
    axes[0].scatter(temp_obs[:, well_to_plot], zs_obs, c="k", s=15, zorder=3)

    axes[1].plot(ts, pres_t[:, well_to_plot], c="k", zorder=3)
    axes[1].scatter(ts_obs, pres_obs[:, well_to_plot], c="k", s=15, zorder=3)

    axes[2].plot(ts, enth_t[:, well_to_plot], c="k", zorder=3)
    axes[2].scatter(ts_obs, enth_obs[:, well_to_plot], c="k", s=15, zorder=3)

    axes[1].axvline(data_end, c=COL_DATA_END, ls="--", ymin=1/12, ymax=11/12, zorder=1)
    axes[2].axvline(data_end, c=COL_DATA_END, ls="--", ymin=1/12, ymax=11/12, zorder=1)

    tufte_axis(axes[0], temp_lims_x, temp_lims_y)
    tufte_axis(axes[1], pres_lims_x, pres_lims_y)
    tufte_axis(axes[2], enth_lims_x, enth_lims_y)

    for ax in axes.flat:
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=TICK_SIZE)

    axes[0].set_ylabel(LABEL_ELEV, fontsize=LABEL_SIZE)
    axes[1].set_ylabel(LABEL_PRES, fontsize=LABEL_SIZE)
    axes[2].set_ylabel(LABEL_ENTH, fontsize=LABEL_SIZE)

    axes[0].set_xlabel(LABEL_TEMP, fontsize=LABEL_SIZE)
    axes[1].set_xlabel(LABEL_TIME, fontsize=LABEL_SIZE)
    axes[2].set_xlabel(LABEL_TIME, fontsize=LABEL_SIZE)

    plt.tight_layout()
    plt.savefig(fname)

def plot_predictions(Fs, data_handler, 
                     temp_t, pres_t, enth_t, ts, zs,
                     temp_obs, pres_obs, enth_obs, ts_obs, zs_obs,
                     temp_lims_x, temp_lims_y, pres_lims_x, pres_lims_y,
                     enth_lims_x, enth_lims_y, data_end, 
                     well_num, fname, dim=3):

    fig, axes = plt.subplots(3, 4, figsize=(FULL_WIDTH, 0.9*FULL_WIDTH))

    for j in range(4):

        tufte_axis(axes[0][j], temp_lims_x, temp_lims_y)
        tufte_axis(axes[1][j], pres_lims_x, pres_lims_y)
        tufte_axis(axes[2][j], enth_lims_x, enth_lims_y)

        axes[0][j].plot(temp_t, zs, c="k", zorder=3)
        axes[0][j].scatter(temp_obs[:, well_num], zs_obs, c="k", s=10, zorder=3)

        axes[1][j].plot(ts, pres_t, c="k", zorder=3)
        axes[1][j].scatter(ts_obs, pres_obs[:, well_num], c="k", s=10, zorder=3)

        axes[2][j].plot(ts, enth_t, c="k", zorder=3)
        axes[2][j].scatter(ts_obs, enth_obs[:, well_num], c="k", s=10, zorder=3)

        axes[1][j].axvline(data_end, c=COL_DATA_END, ls="--", ymin=1/12, ymax=11/12, zorder=1)
        axes[2][j].axvline(data_end, c=COL_DATA_END, ls="--", ymin=1/12, ymax=11/12, zorder=1)

        for F_j in Fs[j].T:

            temp_j, pres_j, enth_j = data_handler.get_full_states(F_j)
            
            if dim == 3:
                zs_j, temp_j = data_handler.downhole_temps(temp_j, well_num)
            else: 
                zs_j = data_handler.mesh.zs
                temp_j = data_handler.downhole_temps(temp_j)[:, well_num]
            
            axes[0][j].plot(temp_j, zs_j, c=COL_TEMP, zorder=2, alpha=0.4)
            axes[1][j].plot(ts, pres_j[:, well_num], c=COL_PRES, zorder=2, alpha=0.4)
            axes[2][j].plot(ts, enth_j[:, well_num], c=COL_ENTH, zorder=2, alpha=0.4)

    for i in range(3):
        for j in range(1, 4):
            axes[i][j].spines["left"].set_visible(False)
            axes[i][j].set_yticks([])

    for ax in axes.flat:
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=TICK_SIZE)

    axes[0][0].set_title(ALG_LABELS[0], fontsize=TITLE_SIZE)
    axes[0][1].set_title(ALG_LABELS[1], fontsize=TITLE_SIZE)
    axes[0][2].set_title(ALG_LABELS[2], fontsize=TITLE_SIZE)
    axes[0][3].set_title(ALG_LABELS[3], fontsize=TITLE_SIZE)

    axes[0][0].set_ylabel(LABEL_ELEV, fontsize=LABEL_SIZE)
    axes[1][0].set_ylabel(LABEL_PRES, fontsize=LABEL_SIZE)
    axes[2][0].set_ylabel(LABEL_ENTH, fontsize=LABEL_SIZE)

    axes[0][1].set_xlabel(LABEL_TEMP, fontsize=LABEL_SIZE)
    axes[1][1].set_xlabel(LABEL_TIME, fontsize=LABEL_SIZE)
    axes[2][1].set_xlabel(LABEL_TIME, fontsize=LABEL_SIZE)

    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(fname)

def plot_hyperparams(hps, hps_t, std_lims_x, std_lims_y, lenh_lims_x, 
                     lenh_lims_y, lenv_lims_x, lenv_lims_y, labels, fname):

    fig, axes = plt.subplots(3, 4, figsize=(FULL_WIDTH, 0.9 * FULL_WIDTH))

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

    axes[0][0].set_title(ALG_LABELS[0], fontsize=TITLE_SIZE)
    axes[0][1].set_title(ALG_LABELS[1], fontsize=TITLE_SIZE)
    axes[0][2].set_title(ALG_LABELS[2], fontsize=TITLE_SIZE)
    axes[0][3].set_title(ALG_LABELS[3], fontsize=TITLE_SIZE)

    for i in range(3):
        axes[i][0].set_ylabel("Density", fontsize=LABEL_SIZE)
        axes[i][1].set_xlabel(labels[i], fontsize=LABEL_SIZE)
        
        for j in range(1, 4):
            axes[i][j].spines["left"].set_visible(False)
            axes[i][j].set_yticks([])

    for ax in axes.flat:
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=TICK_SIZE)

    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(fname)

def plot_colourbar(cmap, vmin, vmax, label, fname):
    
    _, ax = plt.subplots(figsize=(3, 3))
    m = ax.pcolormesh(np.zeros((10, 10)), cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(m, ax=ax)
    cbar.set_label(label, fontsize=LABEL_SIZE)
    plt.tight_layout()
    plt.savefig(fname)


def plot_mesh_2d(mesh, prior, wells, fname):

    grid_xticks = [0, 500, 1000, 1500]
    grid_zticks = [-1500, -1000, -500, 0]

    _, ax = plt.subplots(figsize=(0.7 * FULL_WIDTH, 0.5 * FULL_WIDTH))

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
        plt.text(x-110, 40, s=get_well_name(i), color=COL_WELLS, fontsize=TICK_SIZE)

    ax.set_xlabel(LABEL_X1, fontsize=LABEL_SIZE)
    ax.set_ylabel(LABEL_X2, fontsize=LABEL_SIZE)
    ax.set_box_aspect(1)

    ax.set_xticks(grid_xticks)
    ax.set_yticks(grid_zticks)
    ax.tick_params(labelsize=TICK_SIZE, length=0)

    legend_elements = [
        Patch(facecolor=COL_SHAL, edgecolor=COL_GRID, label="$\Omega_{\mathcal{S}}$"),
        Patch(facecolor=COL_CLAY, edgecolor=COL_GRID, label="$\Omega_{\mathcal{C}}$"),
        Patch(facecolor=COL_DEEP, edgecolor=COL_GRID, label="$\Omega_{\mathcal{D}}$"),
        Line2D([0], [0], c="k", ls=(0, (5, 1)), label="$\omega_{\mathcal{S}}$"),
        Line2D([0], [0], c="k", label="$\omega_{\mathcal{D}}$"),
        Line2D([0], [0], c=COL_WELLS, label="Well Tracks"),
        Line2D([0], [0], c=COL_WELLS, marker="o", ms=5, ls="", label="Feedzones"),
    ]

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.45, 0.75), 
              frameon=False, fontsize=TICK_SIZE)

    for s in ax.spines.values():
        s.set_edgecolor(COL_GRID)
        s.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(fname)

def plot_truth_2d(mesh, perm_t, temp_t, fname):

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 0.4*FULL_WIDTH), 
                             layout="constrained")

    perm_mesh = axes[0].pcolormesh(mesh.xs, mesh.zs, perm_t, 
                                   vmin=MIN_PERM_2D, vmax=MAX_PERM_2D, 
                                   cmap=CMAP_PERM)
    temp_mesh = axes[1].pcolormesh(mesh.xs, mesh.zs, temp_t,
                                   vmin=MIN_TEMP_2D, vmax=MAX_TEMP_2D, 
                                   cmap=CMAP_TEMP)

    perm_cbar = fig.colorbar(perm_mesh, ax=axes[0])
    temp_cbar = fig.colorbar(temp_mesh, ax=axes[1])
    
    perm_cbar.set_label(LABEL_PERM, fontsize=LABEL_SIZE)
    temp_cbar.set_label(LABEL_TEMP, fontsize=LABEL_SIZE)

    for ax in axes:
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(fname)

def plot_particles_2d(mesh, vals, fname, vmin=MIN_PERM_2D, 
                      vmax=MAX_PERM_2D, cmap=CMAP_PERM):

    _, axes = plt.subplots(2, 4, figsize=(FULL_WIDTH, 0.52*FULL_WIDTH))

    for i, ax in enumerate(axes.flat):

        val = np.reshape(vals.T[i], (mesh.nx, mesh.nz))
        ax.pcolormesh(mesh.xs, mesh.zs, val, 
                      vmin=vmin, vmax=vmax, cmap=cmap)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(fname)

def plot_upflows_2d(upflow_t, upflows, bnds_x, bnds_y, fname):

    _, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 0.3*FULL_WIDTH))
    bins = np.linspace(bnds_x[0], bnds_x[1], 11)

    for i, ax in enumerate(axes):

        ax.hist(upflows[i], color=COL_UPFL, density=True, bins=bins, zorder=1)
        ax.axvline(upflow_t, c="k", ymin=1/20, ymax=19/20, zorder=2)

        tufte_axis(ax, bnds_x, bnds_y, gap=0.05)
        ax.set_box_aspect(1)
        ax.tick_params(axis="both", which="both", labelsize=TICK_SIZE)

        if i != 0:
            ax.spines["left"].set_visible(False)
            ax.set_yticks([])

    axes[0].set_title(ALG_LABELS[0], fontsize=LABEL_SIZE)
    axes[1].set_title(ALG_LABELS[1], fontsize=LABEL_SIZE)
    axes[2].set_title(ALG_LABELS[2], fontsize=LABEL_SIZE)
    axes[3].set_title(ALG_LABELS[3], fontsize=LABEL_SIZE)

    axes[0].set_ylabel("Density", fontsize=LABEL_SIZE)
    axes[1].set_xlabel(LABEL_UPFL, fontsize=LABEL_SIZE)

    plt.tight_layout()
    plt.savefig(fname)

def plot_grid_2d(vals, meshes, labels, fname, vmin=MIN_PERM_2D, 
                 vmax=MAX_PERM_2D, cmap=CMAP_PERM):

    n_vals = len(vals)

    fig, axes = plt.subplots(1, n_vals, figsize=(n_vals*FULL_WIDTH/4, 0.3*FULL_WIDTH))

    for i, (v, m, l) in enumerate(zip(vals, meshes, labels)):
        
        axes[i].pcolormesh(m.xs, m.zs, v, vmin=vmin, vmax=vmax, cmap=cmap)
        axes[i].set_title(l, fontsize=LABEL_SIZE)

    for ax in axes.flat:
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
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

    _, ax = plt.subplots(figsize=(0.5 * FULL_WIDTH, 0.5 * FULL_WIDTH))

    mesh.m.layer_plot(axes=ax, linewidth=0.75, linecolour=COL_GRID)

    for i, well in enumerate(wells):
        ax.scatter(well.x, well.y, s=20, c=COL_WELLS)
        plt.text(well.x-360, well.y+160, s=get_well_name(i), 
                 color=COL_WELLS, fontsize=TICK_SIZE)
    
    ax.set_xlabel(LABEL_X1, fontsize=LABEL_SIZE)
    ax.set_ylabel(LABEL_X2, fontsize=LABEL_SIZE)
    
    for s in ax.spines.values():
        s.set_edgecolor(COL_GRID)
    
    ax.set_xlim(0, 6000)
    ax.set_ylim(0, 6000)
    ax.set_xticks([0, 3000, 6000])
    ax.set_yticks([0, 3000, 6000])
    ax.tick_params(labelsize=TICK_SIZE, length=0)
    ax.set_facecolor(COL_DEEP)
    
    plt.tight_layout()
    plt.savefig(fname)

def plot_fault_true_3d(mesh, vals, fname):

    total_upflow = sum([c.area * u for u, c in zip(vals, mesh.column)])
    # print(total_upflow)

    fig, ax = plt.subplots(figsize=(0.5 * FULL_WIDTH, 0.4 * FULL_WIDTH))

    polys = get_layer_polys(mesh, CMAP_UPFLOW)
    polys.set_array(vals)
    polys.set_clim(0, 2.75e-4)

    ax.add_collection(polys)
    cbar = plt.colorbar(polys, ax=ax)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.set_label("Upflow [kg s$^{-1}$ m$^{-2}$]", fontsize=LABEL_SIZE)

    ax.set_xlim(0, 6000)
    ax.set_ylim(0, 6000)
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(fname)

def plot_faults_3d(mesh: lm.mesh, upflows, fname):

    # def plot_fault(ax):

    #     polys.set_clim(0, 2e-4)
        
    #     ax.add_collection(polys)
    #     indices = [c.index for c in mesh.layer[-1].cell]
    #     layer_vals = vals[indices]
    #     polys.set_array(layer_vals)
    #     cbar = plt.colorbar(polys, ax=ax)
    #     # cbar.formatter.set_powerlimits((0, 0))
    #     # cbar.set_label("Upflow [kg s$^{-1}$ m$^{-2}$]", fontsize=LABEL_SIZE)

    _, axes = plt.subplots(2, 4, figsize=(FULL_WIDTH, 5.2))
    polys = get_layer_polys(mesh, CMAP_UPFLOW)
    polys.set_clim(0, 2.75e-4)

    for i, ax in enumerate(axes.flat):

        polys_i = deepcopy(polys)
        ax.add_collection(polys_i)
        polys_i.set_array(upflows[i])

        ax.set_xlim(0, 6000)
        ax.set_ylim(0, 6000)
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(fname)

def plot_means_3d(mesh_fine, mesh_crse, fem_mesh_fine, fem_mesh_crse, 
                  means, fname, cmap=CMAP_PERM,
                  vmin=MIN_PERM_3D, vmax=MAX_PERM_3D):
    
    wsize = (5 * SLICE_HEIGHT, SLICE_HEIGHT)
    p = pv.Plotter(shape=(1, 5), window_size=wsize, border=False, off_screen=True)
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
    
    wsize = (4 * SLICE_HEIGHT, SLICE_HEIGHT)
    p = pv.Plotter(shape=(1, 4), window_size=wsize, border=False, off_screen=True)

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
    
    wsize = (4 * SLICE_HEIGHT, SLICE_HEIGHT)
    p = pv.Plotter(shape=(1, 4), window_size=wsize, border=False, off_screen=True)

    outline = fem_mesh.outline()
    p.add_light(pv.Light(position=CAMERA_POSITION, intensity=0.5))

    for i, interval in enumerate(intervals):

        fem_mesh["vals"] = map_to_fem_mesh(mesh, fem_mesh, interval)
        slice = fem_mesh.clip(normal="x")

        p.subplot(0, i)
        p.add_mesh(outline, line_width=6, color=COL_GRID)
        p.add_mesh(slice, cmap="coolwarm", show_scalar_bar=False)
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
    
    p.screenshot(fname)

def plot_slices_3d(mesh, fem_mesh, vals, fname, cmap=CMAP_PERM, vmin=MIN_PERM_3D, vmax=MAX_PERM_3D):

    p = pv.Plotter(shape=(2, 3), window_size=(1200, 650), border=False, off_screen=True)

    outline = fem_mesh.outline()
    p.add_light(pv.Light(position=(4000, 3500, 1000), intensity=0.5))

    for i in range(2):
        for j in range(3):

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
    camera_position_bottom = (1000, 2000, 6000)

    caps = [convert_inds(mesh, fem_mesh, cap_cell_inds[i]) 
            for i in range(3)]

    p = pv.Plotter(shape=(2, 3), window_size=(3600, 2000), border=False, off_screen=True)

    light = pv.Light(position=(0, 0, -2000), intensity=0.1)
    p.add_light(light)

    for j in range(3):

        fem_mesh["cap"] = caps[j]

        cap = fem_mesh.threshold(0.5)
        cap_r = cap.rotate_x(180)

        outline = cap.extract_surface().extract_feature_edges()
        outline_r = cap_r.extract_surface().extract_feature_edges()

        p.subplot(0, j)
        p.add_mesh(cap, color="orange", lighting=True)
        p.add_mesh(outline, line_width=3, color="k")
        p.camera.position = camera_position_top

        p.subplot(1, j)
        p.add_mesh(cap_r, color="orange", lighting=True)
        p.add_mesh(outline_r, line_width=3, color="k")
        p.camera.position = camera_position_bottom

    p.screenshot(fname)
