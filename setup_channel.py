"""Setup script for 3D channel model."""

import numpy as np

from src.consts import *
from src.grfs import *
from src.models import *

# np.random.seed(2)

MESH_NAME = "models/channel/gCH"
MODEL_NAME = "models/channel/CH"

"""
Classes
"""

class ChannelPrior():

    def __init__(self, mesh: IrregularMesh, cap: ClayCap, 
                 fault: Fault, grf_ext: PermField, 
                 grf_flt: PermField, grf_cap: PermField, 
                 grf_upflow: UpflowField, ls_upflows):

        self.mesh = mesh 

        self.cap = cap
        self.fault = fault
        self.grf_ext = grf_ext 
        self.grf_flt = grf_flt
        self.grf_cap = grf_cap
        self.grf_upflow = grf_upflow

        self.compute_upflow_weightings(ls_upflows)

        self.param_counts = [0, cap.n_params, fault.n_params, 
                             grf_ext.n_params, grf_flt.n_params, 
                             grf_cap.n_params, grf_upflow.n_params]
        
        self.n_params = sum(self.param_counts)
        self.param_inds = np.cumsum(self.param_counts)

        self.inds = {
            "cap"        : np.arange(*self.param_inds[0:2]),
            "fault"      : np.arange(*self.param_inds[1:3]),
            "grf_ext"    : np.arange(*self.param_inds[2:4]),
            "grf_flt"    : np.arange(*self.param_inds[3:5]),
            "grf_cap"    : np.arange(*self.param_inds[4:6]),
            "grf_upflow" : np.arange(*self.param_inds[5:7])
        }

    def compute_upflow_weightings(self, lengthscale):

        mesh_centre = self.mesh.m.centre[:2]
        col_centres = np.array([c.centre for c in self.mesh.m.column])
        
        dxy = col_centres - mesh_centre
        ds = np.sum(-((dxy / lengthscale) ** 2), axis=1)

        upflow_weightings = np.exp(ds)
        self.upflow_weightings = upflow_weightings

    def transform(self, params):

        params = np.squeeze(params)

        cap_cell_inds = self.cap.get_cells_in_cap(params[self.inds["cap"]])

        perms_ext = self.grf_ext.get_perms(params[self.inds["grf_ext"]])
        perms_flt = self.grf_flt.get_perms(params[self.inds["grf_flt"]])
        perms_cap = self.grf_cap.get_perms(params[self.inds["grf_cap"]])

        perms_ext = self.grf_ext.level_set(perms_ext)
        perms_flt = self.grf_flt.level_set(perms_flt)
        perms_cap = self.grf_cap.level_set(perms_cap)

        fault_cells, fault_cols = self.fault.get_cells_in_fault(params[self.inds["fault"]])
        fault_cell_inds = [c.index for c in fault_cells]
        fault_col_inds = [c.index for c in fault_cols]

        perms = np.copy(perms_ext)
        perms[fault_cell_inds] = perms_flt[fault_cell_inds]
        perms[cap_cell_inds] = perms_cap[cap_cell_inds]

        upflow_rates = self.grf_upflow.get_upflows(params[self.inds["grf_upflow"]])
        upflow_rates *= self.upflow_weightings
        upflow_rates = upflow_rates[fault_col_inds]
        upflow_cells = [col.cell[-1] for col in fault_cols]

        upflows = [MassUpflow(c, max(r, 0.0)) 
                   for c, r in zip(upflow_cells, upflow_rates)]

        return perms, upflows

    def sample(self, n=1):
        return np.random.normal(size=(n, self.n_params))

"""
Model parameters
"""

mesh = IrregularMesh(MESH_NAME)

well_xs = [450, 750, 1000, 700, 525, 950,  825, 400, 1100]
well_ys = [775, 750,  750, 475, 975, 525, 1025, 425, 1050]
n_wells = len(well_xs)
well_depths = [-750] * n_wells
feedzone_depths = [-600] * n_wells
feedzone_rates = [-0.2] * n_wells

wells = [Well(x, y, depth, mesh, fz_depth, fz_rate)
         for (x, y, depth, fz_depth, fz_rate) 
         in zip(well_xs, well_ys, well_depths, feedzone_depths, feedzone_rates)]

"""
Clay cap
"""

# Bounds for depth of clay cap, width (horizontal and vertical) and dip
bounds_geom_cap = [(-300, -225), (425, 475), (50, 75), (100, 200)]
n_terms = 5
coef_sds = 5

# Bounds for marginal standard deviations and x, y, z lengthscales
bounds_perm_ext = [(0.75, 1.5), (1000, 2000), (1000, 2000), (200, 800)]
bounds_perm_flt = [(0.5, 1.0), (1000, 2000), (1000, 2000), (200, 800)]
bounds_perm_cap = [(0.5, 1.0), (1000, 2000), (1000, 2000), (200, 800)]

grf_2d = MaternField2D(mesh)
grf_3d = MaternField3D(mesh)

def levels_ext(p):
    """Level set mapping for all non-fault and non-cap regions."""
    if   p < -1.5: return -15.0
    elif p < -0.5: return -14.5
    elif p <  0.5: return -14.0
    elif p <  1.5: return -13.5
    else: return -13.0

def levels_flt(p):
    """Level set mapping for fault."""
    if   p < -0.5: return -13.5
    elif p <  0.5: return -13.0
    else: return -12.5

def levels_cap(p):
    """Level set mapping for clay cap."""
    if   p < -0.5: return -17.0
    elif p <  0.5: return -16.5
    else: return -16.0

# Generate the clay cap and permeability fields
clay_cap = ClayCap(mesh, bounds_geom_cap, n_terms, coef_sds)
perm_field_ext = PermField(mesh, grf_3d, bounds_perm_ext, levels_ext)
perm_field_flt = PermField(mesh, grf_3d, bounds_perm_flt, levels_flt)
perm_field_cap = PermField(mesh, grf_3d, bounds_perm_cap, levels_cap)

"""
Fault
"""

bounds_fault = [(400, 1100), (400, 1100), (100, 150)]

mu_upflow = 1.0e-4
bounds_upflow = [(0.1e-6, 0.5e-6), (200, 1000), (200, 1000)]

fault = Fault(mesh, bounds_fault)
upflow_field = UpflowField(mesh, grf_2d, mu_upflow, bounds_upflow)

"""
Prior
"""

ls_upflows = 400

prior = ChannelPrior(mesh, clay_cap, fault, perm_field_ext, 
                     perm_field_flt, perm_field_cap, 
                     upflow_field, ls_upflows)

"""
Timestepping 
"""

dt = SECS_PER_WEEK
tmax = 52 * SECS_PER_WEEK

"""
Plotting functions
"""

def plot_vals_on_mesh(mesh: IrregularMesh, vals):

    cell_centres = mesh.fem_mesh.cell_centers().points
    cell_vals = [vals[mesh.m.find(c, indices=True)] for c in cell_centres]
    
    mesh.fem_mesh["values"] = cell_vals
    slices = mesh.fem_mesh.slice_along_axis(n=5, axis="y")
    slices.plot(scalars="values", cmap="turbo")

def plot_upflows(mesh, upflows):

    values = np.zeros((mesh.m.num_cells, ))
    for upflow in upflows:
        values[upflow.cell.index] = upflow.rate

    mesh.m.layer_plot(value=values, colourmap="coolwarm")

def plot_wells(mesh, perms, wells):

    def get_well_tubes(wells: list[Well]):
        
        lines = pv.MultiBlock()
        for well in wells:
            line = pv.Line(*well.coords)
            lines.append(line)
        bodies = lines.combine().extract_geometry().clean().split_bodies()

        tubes = pv.MultiBlock()
        for body in bodies:
            tubes.append(body.extract_geometry().tube(radius=10))
        return tubes

    cell_centres = mesh.fem_mesh.cell_centers().points
    cell_vals = [perms[mesh.m.find(c, indices=True)] for c in cell_centres]
    mesh.fem_mesh["perms"] = cell_vals
    mesh.fem_mesh.set_active_scalars("perms")

    import pyvista as pv

    tubes = get_well_tubes(wells)

    p = pv.Plotter()
    p.add_mesh(mesh.fem_mesh.threshold([-20.0, -15.5]), cmap="coolwarm")
    p.add_mesh(mesh.fem_mesh.threshold([-15.5, -10.0]),  opacity=0.5, cmap="coolwarm")
    p.add_mesh(tubes, color="k")
    # for well in wells:
    #     p.add_lines(well.coords, color="black", width=5)
    p.show()

"""
Model functions
"""

def run_model(white_noise):

    perms, upflows = prior.transform(white_noise)

    plot_wells(mesh, perms, wells)
    plot_vals_on_mesh(mesh, perms)
    mesh.m.slice_plot(value=perms, colourmap="viridis")
    mesh.m.layer_plot(value=perms, colourmap="viridis")
    plot_upflows(mesh, upflows)

    m = Model3D(MODEL_NAME, mesh, perms, wells, upflows, dt, tmax)
    return m.run()

"""
Truth generation
"""

white_noise = prior.sample()
flag = run_model(white_noise)

print(flag)

import h5py

with h5py.File(f"{MODEL_NAME}_PR.h5", "r") as f:

    cell_inds = f["cell_index"][:, 0]
    src_inds = f["source_index"][:, 0]
    
    temps = f["cell_fields"]["fluid_temperature"][0][cell_inds]

plot_vals_on_mesh(mesh, temps)

# mesh.m.slice_plot(value=ts, colourmap="coolwarm")