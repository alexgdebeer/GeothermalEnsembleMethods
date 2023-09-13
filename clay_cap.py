import itertools as it
import numpy as np
from layermesh import mesh as lm
import pyvista as pv
import utils

MESH_NAME = "models/channel/gCH"

# Fourier stuff
N_TERMS = 5
COEF_SDS = 5

@utils.timer
def get_cap_radii(cell_centres, cap_centre, width_h, width_v, dip, cs):
    """Returns the radius of the clay cap in the direction of each cell."""

    ds = cell_centres - cap_centre

    # Give cap a curved appearance
    # TODO: should this be width_h, or mean(width_h)?
    ds[:, 2] += (dip / width_h**2) * (ds[:, 0]**2 + ds[:, 1]**2) 

    rs = np.linalg.norm(ds, axis=1)
    phis = np.arccos(ds[:, 2] / rs)
    thetas = np.arctan2(ds[:, 1], ds[:, 0])

    rs_cap = np.sqrt(((np.sin(phis) * np.cos(thetas) / width_h)**2 + \
                      (np.sin(phis) * np.sin(thetas) / width_h)**2 + \
                      (np.cos(phis) / width_v)**2) ** -1)
    
    for n, m in it.product(range(N_TERMS), range(N_TERMS)):
        
        rs_cap += cs[n, m, 0] * np.cos(n * thetas) * np.cos(m * phis) + \
                  cs[n, m, 1] * np.cos(n * thetas) * np.sin(m * phis) + \
                  cs[n, m, 2] * np.sin(n * thetas) * np.cos(m * phis) + \
                  cs[n, m, 3] * np.sin(n * thetas) * np.sin(m * phis)
        
    cap = rs < rs_cap
    return cap

def plot(mesh, geo, cap):

    cap_mesh = np.zeros((mesh.n_cells, ))

    containing_cells = mesh.find_containing_cell([c.centre for c in geo.cell])

    cap_mesh[containing_cells] = cap[[c.index for c in geo.cell]]

    mesh.cell_data["cap_mesh"] = cap_mesh
    p = pv.Plotter()
    p.add_mesh(mesh.threshold([0.9, 1.1]), cmap="coolwarm")
    p.show()

geo = lm.mesh(f"{MESH_NAME}.h5")
mesh = pv.UnstructuredGrid(f"{MESH_NAME}.vtu")

cell_centres = np.array([c.centre for c in geo.cell])

cap_centre = np.array([np.random.uniform(700, 800), 
                       np.random.uniform(700, 800),
                       np.random.uniform(-300, -225)])
width_h, width_v = 475, 50
dip = np.random.uniform(50, 200) # Average difference in height between the centre of the cap and the edges 

# Fourier coefficients
cs = np.random.normal(loc=0.0, scale=COEF_SDS, size=(N_TERMS, N_TERMS, 4))

cap = get_cap_radii(cell_centres, cap_centre, width_h, width_v, dip, cs)

plot(mesh, geo, cap)
geo.slice_plot("y", value=cap, colourmap="coolwarm")
# geo.layer_plot(elevation=-250, value=cap, colourmap="coolwarm")
