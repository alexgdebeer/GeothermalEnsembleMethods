from copy import deepcopy

import colorcet
from layermesh import mesh as lm
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pyvista as pv

from src.consts import *

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12

MAX_PERM = -12.5
MIN_PERM = -17.0

MIN_TEMP = 20
MAX_TEMP = 300

MIN_STDS = 0
MAX_STDS = 1.5

CMAP_PERM = "cet_bgy"
CMAP_TEMP = "coolwarm"
CMAP_UPFLOW = "cet_fire"

COL_WELLS = "royalblue"
COL_DOM1 = "whitesmoke"
COL_GRID = "darkgrey"

COL_DATA_END = "darkgrey"

FULL_WIDTH = 10.0
SLICE_HEIGHT = 1200

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

def plot_mesh(fem_mesh: pv.UnstructuredGrid, wells, feedzone_depths, fname):

    tubes = get_well_tubes(wells, feedzone_depths)
    edges = fem_mesh.extract_geometry().extract_all_edges()

    p = pv.Plotter(off_screen=True, window_size=(2400, 2400))
    p.add_mesh(fem_mesh, color=COL_DOM1, opacity=0.2)
    p.add_mesh(tubes, color=COL_WELLS)
    p.add_mesh(edges, line_width=2, color=COL_GRID)

    p.camera.position = CAMERA_POSITION
    p.add_light(pv.Light(position=CAMERA_POSITION, intensity=0.2))
    p.screenshot(fname)

def plot_grid_layer(mesh, wells, fname):

    _, ax = plt.subplots(figsize=(4, 4))

    mesh.m.layer_plot(axes=ax, linewidth=0.75, linecolour=COL_GRID)

    for i, well in enumerate(wells):
        ax.scatter(well.x, well.y, s=20, c=COL_WELLS)
        plt.text(well.x-360, well.y+160, s=get_well_name(i), 
                 color=COL_WELLS, fontsize=TICK_SIZE)
    
    ax.set_xlabel("$x_{1}$ [m]", fontsize=LABEL_SIZE)
    ax.set_ylabel("$x_{2}$ [m]", fontsize=LABEL_SIZE)
    
    for s in ax.spines.values():
        s.set_edgecolor("darkgrey")
    
    ax.set_xlim(0, 6000)
    ax.set_ylim(0, 6000)
    ax.set_xticks([0, 3000, 6000])
    ax.set_yticks([0, 3000, 6000])
    ax.tick_params(labelsize=TICK_SIZE, length=0)
    ax.set_facecolor(COL_DOM1)
    
    plt.tight_layout()
    plt.savefig(fname)

def plot_data(temp_t, pres_t, enth_t, zs, ts, 
              temp_obs, pres_obs, enth_obs, zs_obs, ts_obs, 
              temp_lims_x, temp_lims_y, pres_lims_x, pres_lims_y,
              enth_lims_x, enth_lims_y, data_end, 
              well_to_plot, fname):
    
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 3))

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

    axes[0].set_ylabel("Elevation [m]", fontsize=LABEL_SIZE)
    axes[1].set_ylabel("Pressure [MPa]", fontsize=LABEL_SIZE)
    axes[2].set_ylabel("Enthalpy [kJ/kg]", fontsize=LABEL_SIZE)

    axes[0].set_xlabel("Temperature [$^\circ$C]", fontsize=LABEL_SIZE)
    axes[1].set_xlabel("Time [Years]", fontsize=LABEL_SIZE)
    axes[2].set_xlabel("Time [Years]", fontsize=LABEL_SIZE)

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
                  means, fname, cmap=CMAP_PERM, vmin=MIN_PERM, vmax=MAX_PERM):
    
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
        p.add_mesh(slice, cmap=CMAP_PERM, clim=(MIN_STDS, MAX_STDS), show_scalar_bar=False)
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

def plot_slice(mesh, fem_mesh, vals, fname, cmap=CMAP_PERM, vmin=MIN_PERM, vmax=MAX_PERM):
    
    p = pv.Plotter(window_size=(1200, 1200), off_screen=True)#, border=False)
    p.add_light(pv.Light(position=(4000, 3500, 1000), intensity=0.5))

    outline = fem_mesh.outline()
    fem_mesh["vals"] = map_to_fem_mesh(mesh, fem_mesh, vals)
    slice = fem_mesh.clip(normal="x")

    p.add_mesh(outline, line_width=6, color=COL_GRID)
    p.add_mesh(slice, cmap=cmap, clim=(vmin, vmax), show_scalar_bar=False)
    p.camera.position = CAMERA_POSITION
    
    p.screenshot(fname)

def plot_slices_3d(mesh, fem_mesh, vals, fname, cmap=CMAP_PERM, vmin=MIN_PERM, vmax=MAX_PERM):

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

def plot_intervals(mesh, fem_mesh, cells_in_interval):

    fem_mesh["vals"] = map_to_fem_mesh(mesh, fem_mesh, cells_in_interval)

    p = pv.Plotter()
    p.add_mesh(fem_mesh.clip(normal="x"), cmap="coolwarm")
    p.show()

def plot_colourbar(cmap, vmin, vmax, label, fname):
    
    fig, ax = plt.subplots(figsize=(3, 3))
    m = ax.pcolormesh(np.zeros((10, 10)), cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(m, ax=ax)
    cbar.set_label(label, fontsize=LABEL_SIZE)
    plt.tight_layout()
    plt.savefig(fname)

def plot_predictions(Fs, data_handler, 
                     temp_t, pres_t, enth_t, ts, zs,
                     temp_obs, pres_obs, enth_obs, ts_obs, zs_obs,
                     temp_lims_x, temp_lims_y, pres_lims_x, pres_lims_y,
                     enth_lims_x, enth_lims_y, data_end, 
                     well_num, fname):

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
            zs_j, temp_j = data_handler.downhole_temps(temp_j, well_num)
            
            axes[0][j].plot(temp_j, zs_j, c="limegreen", zorder=2, alpha=0.4)
            axes[1][j].plot(ts, pres_j[:, well_num], c="darkorange", zorder=2, alpha=0.4)
            axes[2][j].plot(ts, enth_j[:, well_num], c="deepskyblue", zorder=2, alpha=0.4)

    for i in range(3):
        for j in range(1, 4):
            axes[i][j].spines["left"].set_visible(False)
            axes[i][j].set_yticks([])

    for ax in axes.flat:
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=TICK_SIZE)

    axes[0][0].set_title("Prior", fontsize=TITLE_SIZE)
    axes[0][1].set_title("EKI", fontsize=TITLE_SIZE)
    axes[0][2].set_title("EKI (Localisation)", fontsize=TITLE_SIZE)
    axes[0][3].set_title("EKI (Inflation)", fontsize=TITLE_SIZE)

    axes[0][0].set_ylabel("Elevation [m]", fontsize=LABEL_SIZE)
    axes[1][0].set_ylabel("Pressure [MPa]", fontsize=LABEL_SIZE)
    axes[2][0].set_ylabel("Enthalpy [kJ/kg]", fontsize=LABEL_SIZE)

    axes[0][1].set_xlabel("Temperature [$^\circ$C]", fontsize=LABEL_SIZE)
    axes[1][1].set_xlabel("Time [Years]", fontsize=LABEL_SIZE)
    axes[2][1].set_xlabel("Time [Years]", fontsize=LABEL_SIZE)

    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(fname)
