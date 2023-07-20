import random

from python.geo_models import Mesh, Model, MassUpflow, Well, Feedzone


random.seed(16)

SECS_PER_WEEK = 60.0 * 60.0 * 24.0 * 7.0

xmax, nx = 1500.0, 25
ymax, ny = 60.0, 1
zmax, nz = 1500.0, 25
tmax, nt = 104.0 * SECS_PER_WEEK, 24

dt = tmax/nt 
n_blocks = nx * nz

mesh = Mesh(f"gSL{n_blocks}", xmax, ymax, zmax, nx, ny, nz)

upflow_loc = (0.5*xmax, 0.5*ymax, -zmax + 0.5*mesh.dz)

wells = [Well(Feedzone((x, 0.5 * ymax, -500), -2.0), obs_times)
         for x in [200, 475, 750, 1025, 1300]]

def f(thetas):

    upflows = [MassUpflow(upflow_loc, pri.get_rate(thetas[0]))]
    ks = 10 ** pri.get_perms(thetas[1:])
    
    model = Model(f"SL{n_blocks}", mesh, ks, wells, upflows, dt, tmax)