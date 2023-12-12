"""Setup script for 3D channel model."""

import numpy as np

from src.consts import SECS_PER_WEEK
from src.data_handlers import DataHandler3D
from src.grfs import *
from src.models import *
from src.priors import ChannelPrior

np.random.seed(6) # 6 good

DATA_FOLDER = "data/channel"
MODEL_FOLDER = "models/channel"
W_TRUE_PATH = f"{DATA_FOLDER}/w_true.npy"
P_TRUE_PATH = f"{DATA_FOLDER}/p_true.npy"
F_TRUE_PATH = f"{DATA_FOLDER}/F_true.npy"
G_TRUE_PATH = f"{DATA_FOLDER}/G_true.npy"
OBS_PATH = f"{DATA_FOLDER}/obs.npy"
COV_PATH = f"{DATA_FOLDER}/C_e.npy"

READ_TRUTH = True 

MESH_NAME = "models/channel/gCH"
MODEL_NAME = "models/channel/CH"

"""
Model parameters
"""

tmax, nt = 104.0 * SECS_PER_WEEK, 24
dt = tmax / nt

mesh = IrregularMesh(MESH_NAME)

well_xs = [450, 750, 1000, 700, 525, 950,  825, 400, 1100]
well_ys = [775, 750,  750, 475, 975, 525, 1025, 425, 1050]
n_wells = len(well_xs)
well_depths = [-800] * n_wells
feedzone_depths = [-500] * n_wells
feedzone_rates = [-1.5] * n_wells # TODO: change...

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
    if   p < -1.5: return -15.5
    elif p < -0.5: return -15.0
    elif p <  0.5: return -14.5
    elif p <  1.5: return -14.0
    else: return -13.5

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
Observations
"""

temp_obs_zs = [-200, -300, -400, -500, -600, -700, -800]
temp_obs_cs = np.array([[x, y, z] 
                        for z in temp_obs_zs 
                        for x, y in zip(well_xs, well_ys)])

prod_obs_ts = np.array([0, 13, 26, 39, 52]) * SECS_PER_WEEK

data_handler = DataHandler3D(mesh, wells, temp_obs_cs, prod_obs_ts, tmax, nt)

"""
Ensemble functions
"""

def generate_particle(p_i, num):
    name = f"{MODEL_NAME}_{num}"
    logks_t, upflows_t = prior.split(p_i)
    model = Model3D(name, mesh, logks_t, wells, upflows_t, dt, tmax)
    return model

def get_result(particle: Model3D):
    F_i = particle.get_pr_data() # May need to change this to just return enthalpies...
    G_i = data_handler.get_obs(F_i)
    return F_i, G_i

"""
Prior
"""

# Parameter associated with Gaussian kernel for upflows
ls_upflows = 400

prior = ChannelPrior(mesh, clay_cap, fault, perm_field_ext, 
                     perm_field_flt, perm_field_cap, 
                     upflow_field, ls_upflows)

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

noise_level = 0.02

def generate_truth():

    w_t = prior.sample()
    p_t = prior.transform(w_t)

    logks_t, upflows_t = prior.split(p_t)

    model_t = Model3D(MODEL_NAME, mesh, logks_t, wells, upflows_t, dt, tmax)

    plot_wells(mesh, logks_t, wells)
    plot_vals_on_mesh(mesh, logks_t)
    plot_upflows(mesh, upflows_t)
    
    if model_t.run() != ExitFlag.SUCCESS:
        raise Exception("Truth failed to run.")

    F_t = model_t.get_pr_data()
    G_t = data_handler.get_obs(F_t)

    np.save(W_TRUE_PATH, w_t)
    np.save(P_TRUE_PATH, p_t)
    np.save(F_TRUE_PATH, F_t)
    np.save(G_TRUE_PATH, G_t)

    temps = data_handler.get_full_temperatures(F_t)
    plot_vals_on_mesh(mesh, temps)
    return w_t, p_t, F_t, G_t

def read_truth():

    w_t = np.load(W_TRUE_PATH)
    p_t = np.load(P_TRUE_PATH)
    F_t = np.load(F_TRUE_PATH)
    G_t = np.load(G_TRUE_PATH)
    
    return w_t, p_t, F_t, G_t

def generate_data(G_t):

    temp_t, pres_t, enth_t = data_handler.split_obs(G_t)

    cov_temp = (noise_level * np.max(temp_t)) ** 2 * np.eye(temp_t.size)
    cov_pres = (noise_level * np.max(pres_t)) ** 2 * np.eye(pres_t.size)
    cov_enth = (noise_level * np.max(enth_t)) ** 2 * np.eye(enth_t.size)

    C_e = sparse.block_diag((cov_temp, cov_pres, cov_enth)).toarray()
    y = np.random.multivariate_normal(G_t, C_e)

    np.save(OBS_PATH, y)
    np.save(COV_PATH, C_e)

    return y, C_e

def read_data():
    y = np.load(OBS_PATH)
    C_e = np.load(COV_PATH)
    return y, C_e

if READ_TRUTH:
    w_t, p_t, F_t, G_t = read_truth()
    y, C_e = read_data()

else:
    w_t, p_t, F_t, G_t = generate_truth()
    y, C_e = generate_data(G_t)

# This won't work with a fine / coarse model setup
Np = len(p_t)
NF = len(F_t)
NG = len(y) # This will though