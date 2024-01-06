import numpy as np

from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import matplotlib.pyplot as plt 

from setup_slice import *

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

np.random.seed(1)

title_size = 14
label_size = 12
legend_size = 12
tick_size = 8

cmap_grid = ListedColormap(["whitesmoke", "gainsboro", "silver"])

grid_xticks = [0, 500, 1000, 1500]
grid_zticks = [-1500, -1000, -500, 0]

fig, ax = plt.subplots(figsize=(5.0, 3.5))

ws = prior.sample().squeeze()
zs_bound_2 = prior.gp_boundary.transform(ws[:mesh_crse.nx])

zones = np.zeros((mesh_crse.nx, mesh_crse.nz))

for i in range(mesh_crse.nx):
    for j in range(mesh_crse.nz):
        if mesh_crse.zs[j] > -60.0:
            zones[j, i] = 1.0
        elif mesh_crse.zs[j] > zs_bound_2[i]:
            zones[j, i] = 0.5
        else:
            zones[j, i] = 0.0

# Define grid
grid = np.zeros((mesh_crse.nx, mesh_crse.nz))
grid = np.ma.masked_array(grid, grid == 0.0)

# Define coordinates of boundaries
xs_bound_1 = [0, 1500]
zs_bound_1 = [-60, -60]

zs_bound_2 = [zs_bound_2[0], *zs_bound_2, zs_bound_2[-1]]
xs_bound_2 = [0, *mesh_crse.xs, mesh_crse.xmax]

# Define each subdomain
cs_zone_1 = np.array([
    [0, mesh_crse.xmax, mesh_crse.xmax, 0],
    [-60, -60, 0, 0]]).T

cs_zone_2 = np.array([
    [*xs_bound_2, mesh_crse.xmax, 0], 
    [*zs_bound_2, -60, -60]]).T

cs_zone_3 = np.array([
    [*xs_bound_2, mesh_crse.xmax, 0],
    [*zs_bound_2, -mesh_crse.zmax, -mesh_crse.zmax]]).T

zone_1 = Polygon(cs_zone_1, facecolor="silver", zorder=0)
zone_2 = Polygon(cs_zone_2, facecolor="gainsboro", zorder=0)
zone_3 = Polygon(cs_zone_3, facecolor="whitesmoke", zorder=0)

# Plot grid
ax.pcolormesh(mesh_crse.xs, mesh_crse.zs, grid, 
              cmap=cmap_grid, edgecolors="darkgrey")

# Plot subdomains
ax.add_patch(zone_1) 
ax.add_patch(zone_2)
ax.add_patch(zone_3)

# Plot boundaries
ax.plot(xs_bound_1, zs_bound_1, c="k", linewidth=1.5, linestyle=(0, (5, 1)), zorder=1)
ax.plot(xs_bound_2, zs_bound_2, c="k", linewidth=1.5, zorder=1)

for (i, well) in enumerate(wells_fine):
    x, z = well.x, well.feedzone_cell.centre[-1]
    ax.plot([x, x], [0, -1300], linewidth=1.5, color="royalblue", zorder=2)
    ax.scatter([x], [z], color="royalblue", s=20)
    plt.text(x-110, 40, s=r"\texttt{WELL "+ f"{i+1}" + r"}", color="royalblue", fontsize=tick_size)

ax.set_xlabel("$x_{1}$ [m]", fontsize=label_size)
ax.set_ylabel("$x_{2}$ [m]", fontsize=label_size)
ax.set_box_aspect(1)

ax.set_xticks(grid_xticks)
ax.set_yticks(grid_zticks)
ax.tick_params(labelsize=tick_size, length=0)

legend_elements = [
    Patch(facecolor="silver", edgecolor="darkgrey", label="$\Omega_{\mathcal{S}}$"),
    Patch(facecolor="gainsboro", edgecolor="darkgrey", label="$\Omega_{\mathcal{C}}$"),
    Patch(facecolor="whitesmoke", edgecolor="darkgrey", label="$\Omega_{\mathcal{D}}$"),
    Line2D([0], [0], c="k", ls=(0, (5, 1)), label="$\omega_{\mathcal{S}}$"),
    Line2D([0], [0], c="k", label="$\omega_{\mathcal{D}}$"),
    Line2D([0], [0], c="royalblue", label="Well Tracks"),
    Line2D([0], [0], c="royalblue", marker="o", ms=5, ls="", label="Feedzones"),
]

ax.legend(handles=legend_elements, bbox_to_anchor=(1.45, 0.75), 
            frameon=False, fontsize=tick_size)

for s in ax.spines.values():
    s.set_edgecolor("darkgrey")

plt.tight_layout()
plt.savefig("plots/slice/mesh.pdf")
plt.clf()