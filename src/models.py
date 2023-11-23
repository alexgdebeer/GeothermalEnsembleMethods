from copy import deepcopy
from enum import Enum
from itertools import product
from scipy import stats

import h5py
import layermesh.mesh as lm
import numpy as np
import os
import pyvista as pv
import pywaiwera
import yaml

from src.consts import *
from src import utils

EPS = 1e-8

def gauss_to_unif(x, lb, ub):
    return lb + stats.norm.cdf(x) * (ub - lb)

class ExitFlag(Enum):
    SUCCESS = 1
    FAILURE = 2

class ModelType(Enum):
    MODEL2D = 1
    MODEL3D = 2

class Mesh():
    pass

class SliceMesh(Mesh):

    def __init__(self, name, xmax, ymax, zmax, nx, nz):

        self.name = name

        self.xmax = xmax 
        self.ymax = ymax 
        self.zmax = zmax

        self.nx = nx 
        self.nz = nz

        self.dx = xmax / nx
        self.dy = ymax
        self.dz = zmax / nz

        dxs = [self.dx] * nx
        dys = [self.dy]
        dzs = [self.dz] * nz

        self.m = lm.mesh(rectangular=(dxs, dys, dzs))

        self.cell_xs = np.array([c.centre[0] for c in self.m.cell])
        self.cell_zs = np.array([c.centre[-1] for c in self.m.cell])

        self.xs = np.array([c.centre[0] for c in self.m.column])
        self.zs = np.array([l.centre for l in self.m.layer])

        points = [c.centre for c in self.m.cell]
        self.fem_mesh = pv.PolyData(points).delaunay_2d()

        if not os.path.exists(f"{self.name}.msh"):
            utils.info("Writing mesh to file...")
            self.write_to_file()

    def write_to_file(self):
        self.m.write(f"{self.name}.h5")
        self.m.export(f"{self.name}.msh", fmt="gmsh22")

class IrregularMesh(Mesh):

    def __init__(self, name):

        self.name = name
        self.m = lm.mesh(f"{self.name}.h5")

        self.cell_centres = [cell.centre for cell in self.m.cell]
        self.col_centres = [col.centre for col in self.m.column]
        self.col_cells = {col.index: [c.index for c in col.cell] 
                          for col in self.m.column}
        
        self.fem_mesh = pv.UnstructuredGrid(f"{self.name}.vtu")
        self.fem_mesh = self.fem_mesh.triangulate()

class MassUpflow():
    def __init__(self, cell: lm.cell, rate: float):
        self.cell = cell
        self.rate = rate

class Well():
    
    def __init__(self, x: float, y: float, depth: float, mesh: Mesh, 
                 feedzone_depth: float, feedzone_rate: float):
        
        self.max_elev = mesh.m.find((x, y, depth)).column.surface
        self.min_elev = depth
        self.coords = np.array([[x, y, self.max_elev], 
                                [x, y, self.min_elev]])

        self.feedzone_cell = mesh.m.find((x, y, feedzone_depth))
        self.feedzone_rate = feedzone_rate

class PermField():

    def __init__(self, mesh, grf, bounds, level_func, 
                 model_type=ModelType.MODEL3D):

        self.mesh = mesh
        self.grf = grf
        self.model_type = model_type

        self.bounds = bounds
        self.level_func = level_func

        self.n_hyperparams = len(self.bounds)
        self.n_params = self.n_hyperparams + self.mesh.fem_mesh.n_points

    def get_perms(self, ps):
        """Returns the set of permeabilities that correspond to a given 
        set of (whitened) parameters."""

        hyperparams, W = ps[:self.n_hyperparams], ps[self.n_hyperparams:]
        hyperparams = [gauss_to_unif(p, *bnds)
                       for p, bnds in zip(hyperparams, self.bounds)]
        
        X = self.grf.generate_field(W, *hyperparams)

        if self.model_type == ModelType.MODEL3D:
            return self.grf.G @ X
        elif self.model_type == ModelType.MODEL2D:
            return X

    def level_set(self, perms):
        return np.array([self.level_func(p) for p in perms])

class UpflowField():

    def __init__(self, mesh, grf, mu, bounds):

        self.mesh = mesh 
        self.grf = grf

        self.mu = mu 
        self.bounds = bounds 

        self.n_hyperps = len(bounds)
        self.n_params = self.n_hyperps + self.mesh.m.num_columns

    def get_upflows(self, params):
        """Returns the set of upflows that correspond to a given set of 
        (whitened) parameters."""

        hyperparams, W = params[:self.n_hyperps], params[self.n_hyperps:]
        hyperparams = [gauss_to_unif(p, *bnds) 
                       for p, bnds in zip(hyperparams, self.bounds)]
        
        X = self.grf.generate_field(W, *hyperparams)
        return self.mu + X

class Channel():
    
    def __init__(self, mesh, bounds):
        
        self.mesh = mesh
        self.bounds = bounds

        self.n_params = 5

    def get_cells_in_channel(self, pars):
        """Returns the indices of the columns that are contained within 
        the channel specified by a given set of parameters."""
        
        def in_channel(x, y, a1, a2, a3, a4, a5):
            ub = a1 * np.sin(2*np.pi*x/a2) + np.tan(a3)*x + a4 
            return ub-a5 <= y <= ub+a5 

        pars = [gauss_to_unif(p, *self.bounds[i]) for i, p in enumerate(pars)]

        cols = [col for col in self.mesh.m.column if in_channel(*col.centre, *pars)]
        cells = [cell for col in cols for cell in col.cell]
        return cells, cols

class ClayCap():

    def __init__(self, mesh, bounds, n_terms, coef_sds):
        
        self.cell_centres = np.array([c.centre for c in mesh.m.cell])
        self.col_centres = np.array([c.centre for c in mesh.m.column])

        self.bounds = bounds 
        self.n_terms = n_terms 
        self.coef_sds = coef_sds
        self.n_params = 6 + 4 * self.n_terms ** 2

    def cartesian_to_spherical(self, ds):

        rs = np.linalg.norm(ds, axis=1)
        phis = np.arccos(ds[:, 2] / rs)
        thetas = np.arctan2(ds[:, 1], ds[:, 0])
        
        return rs, phis, thetas
    
    def compute_cap_radii(self, phis, thetas, width_h, width_v, coefs):
        """Computes the radius of the clay cap in the direction of each 
        cell, by taking the radius of the (deformed) ellipse that forms 
        the base of the cap, then adding the randomised Fourier series 
        to it."""

        rs = np.sqrt(((np.sin(phis) * np.cos(thetas) / width_h)**2 + \
                      (np.sin(phis) * np.sin(thetas) / width_h)**2 + \
                      (np.cos(phis) / width_v)**2) ** -1)
        
        for n, m in product(range(self.n_terms), range(self.n_terms)):
        
            rs += coefs[n, m, 0] * np.cos(n * thetas) * np.cos(m * phis) + \
                  coefs[n, m, 1] * np.cos(n * thetas) * np.sin(m * phis) + \
                  coefs[n, m, 2] * np.sin(n * thetas) * np.cos(m * phis) + \
                  coefs[n, m, 3] * np.sin(n * thetas) * np.sin(m * phis)
        
        return rs
    
    def get_cap_params(self, ps):
        """Given a set of unit normal variables, generates the 
        corresponding set of clay cap parameters."""

        geom = [gauss_to_unif(p, *self.bounds[i]) 
                for i, p in enumerate(ps[:6])]

        coefs = np.reshape(self.coef_sds * ps[6:], 
                           (self.n_terms, self.n_terms, 4))

        return geom, coefs

    def get_cells_in_cap(self, params):
        """Returns an array of booleans that indicate whether each cell 
        is contained within the clay cap."""

        # Unpack parameters
        geom, coefs = self.get_cap_params(params)
        *centre, width_h, width_v, dip = geom

        ds = self.cell_centres - centre
        ds[:, -1] += (dip / width_h**2) * (ds[:, 0]**2 + ds[:, 1]**2) 

        cell_radii, cell_phis, cell_thetas = self.cartesian_to_spherical(ds)

        cap_radii = self.compute_cap_radii(cell_phis, cell_thetas,
                                           width_h, width_v, coefs)
        
        return (cell_radii < cap_radii).nonzero()
    
    def get_upflow_weightings(self, params, mesh):
        """Returns a list of values by which the upflow in each column 
        should be multiplied."""

        cx, cy, *_ = self.get_cap_params(params)[0]

        s = 800 # TODO: make this into an input

        col_centres = [col.centre for col in mesh.m.column]
        col_weightings = np.array([np.exp(-(((c[0]-cx)/s)**2 + ((c[1]-cy)/s)**2)) / 2
                                   for c in col_centres])

        return col_weightings

class Model():
    """Base class for models, with a set of default methods."""

    def __init__(self, path: str, mesh: Mesh, perms: np.ndarray, 
                 wells: list[Well], upflows: list[MassUpflow], 
                 dt: float, tmax: float):

        self.ns_path = f"{path}_NS"
        self.pr_path = f"{path}_PR"
        self.incon_path = f"{path}_incon"

        self.mesh = mesh 
        self.perms = perms 
        self.wells = wells
        self.upflows = upflows

        self.dt = dt
        self.tmax = tmax
        self.ts = np.arange(0, tmax, dt)

        self.feedzone_cell_inds = [w.feedzone_cell.index for w in wells]
        self.n_feedzones = len(self.feedzone_cell_inds)

        self.ns_model = None
        self.pr_model = None
        
        self.generate_ns()
        self.generate_pr()

    def initialise_ns_model(self):

        self.ns_model = {
            "eos": {"name": "we"},
            "gravity": GRAVITY,
            "logfile": {"echo": False},
            "mesh": {"filename": f"{self.mesh.name}.msh"}
        }

    def add_boundaries(self):
        """Adds an atmospheric boundary condition to the top of the 
        model (leaves the sides with no-flow conditions)."""

        self.ns_model["boundaries"] = [{
            "primary": [P_ATM, T_ATM], 
            "region": 1,
            "faces": {
                "cells": [c.index for c in self.mesh.m.surface_cells],
                "normal": [0, 0, 1]
            }
        }]

    def add_upflows(self):
        """Adds the mass upflows to the base of the model. Where there 
        are no mass upflows, a background heat flux of constant 
        magnitude is imposed."""

        upflow_cell_inds = [
            upflow.cell.index 
            for upflow in self.upflows]

        heat_cell_inds = [
            c.cell[-1].index for c in self.mesh.m.column 
            if c.cell[-1].index not in upflow_cell_inds]

        self.ns_model["source"] = [{
            "component": "energy",
            "rate": HEAT_RATE * self.mesh.m.cell[c].column.area,
            "cell": c
        } for c in heat_cell_inds]

        self.ns_model["source"].extend([{
            "component": "water",
            "enthalpy": MASS_ENTHALPY, 
            "rate": u.rate * u.cell.column.area,
            "cell": u.cell.index
        } for u in self.upflows])

        # total_mass = sum([u.rate * u.cell.column.area for u in self.upflows])
        # utils.info(f"Total mass input: {total_mass:.2f} kg/s")

    def add_rocktypes(self):
        """Adds rocks with given permeabilities (and constant porosity, 
        conductivity, density and specific heat) to the model. 
        Permeabilities may be isotropic or anisotropic."""
        
        if len(self.perms) != self.mesh.m.num_cells:
            raise Exception("Incorrect number of permeabilities.")
        
        self.ns_model["rock"] = {"types": [{
            "name": f"{c.index}",
            "porosity": POROSITY, 
            "permeability": 10.0 ** self.perms[c.index],
            "cells": [c.index],
            "wet_conductivity": CONDUCTIVITY,
            "dry_conductivity": CONDUCTIVITY,
            "density": DENSITY,
            "specific_heat": SPECIFIC_HEAT
        } for c in self.mesh.m.cell]}

    def add_wells(self):
        """Adds wells with constant production / injection rates to the 
        model."""

        self.pr_model["source"].extend([{
            "component": "water",
            "rate": w.feedzone_rate,
            "cell": w.feedzone_cell.index
        } for w in self.wells])

    def add_ns_incon(self):
        """Adds path to initial condition file to the model, if the 
        file exists. Otherwise, sets the entire model to a constant 
        temperature and pressure."""

        if os.path.isfile(f"{self.incon_path}.h5"):
            self.ns_model["initial"] = {"filename": f"{self.incon_path}.h5"}
        else:
            # utils.warn("Initial condition not found. Improvising...")
            self.ns_model["initial"] = {"primary": [P0, T0], "region": 1}

    def add_pr_incon(self):
        """Sets the production history initial condition to be the 
        output file from the natural state run."""
        self.pr_model["initial"] = {"filename": f"{self.ns_path}.h5"}
    
    def add_ns_timestepping(self):
        """Sets the natural state timestepping parameters."""

        self.ns_model["time"] = {
            "step": {
                "size": 1.0e+6,
                "adapt": {"on": True}, 
                "maximum": {"number": MAX_NS_TSTEPS},
                "method": "beuler",
                "stop": {"size": {"maximum": NS_STEPSIZE}}
            }
        }

    def add_pr_timestepping(self):
        """Sets the production history timestepping parameters."""

        self.pr_model["time"] = {
            "step": {
                "adapt": {"on": True},
                "size": self.dt,
                "maximum": {"number": MAX_PR_TSTEPS},
            },
            "stop": self.tmax
        }

    def add_ns_output(self):
        """Sets up the natural state simulation such that it only saves 
        the final model state."""

        self.ns_model["output"] = {
            "frequency": 0, 
            "initial": False, 
            "final": True
        }

    def add_pr_output(self):
        """Sets up production history checkpoints."""
        
        self.pr_model["output"] = {
            "checkpoint": {
                "time": [self.dt], 
                "repeat": True
            },
            "frequency": 0,
            "initial": True,
            "final": False
        }

    def generate_ns(self):
        """Generates the natural state model."""
        
        self.initialise_ns_model()
        self.add_boundaries()
        self.add_rocktypes()
        self.add_upflows()
        self.add_ns_incon()
        self.add_ns_timestepping()
        self.add_ns_output()

        utils.save_json(self.ns_model, f"{self.ns_path}.json")

    def generate_pr(self):
        """Generates the production history model."""

        self.pr_model = deepcopy(self.ns_model)

        self.add_wells()
        self.add_pr_timestepping()
        self.add_pr_output()
        self.add_pr_incon()

        utils.save_json(self.pr_model, f"{self.pr_path}.json")

    @utils.timer
    def run(self):
        """Simulates the model and returns a flag that indicates 
        whether the simulation was successful."""

        env = pywaiwera.docker.DockerEnv(check=False, verbose=False)
        env.run_waiwera(f"{self.ns_path}.json", noupdate=True)
        
        flag = self.get_exitflag(self.ns_path)
        if flag == ExitFlag.FAILURE: 
            return flag

        env.run_waiwera(f"{self.pr_path}.json", noupdate=True)
        return self.get_exitflag(self.pr_path)

    def get_exitflag(self, log_path):
        """Determines the outcome of a simulation."""

        with open(f"{log_path}.yaml", "r") as f:
            log = yaml.safe_load(f)

        for msg in log[::-1]:

            if msg[:3] in [MSG_END_TIME, MSG_MAX_STEP]:
                return ExitFlag.SUCCESS

            elif msg[:3] == MSG_MAX_ITS:
                utils.warn("Simulation failed (max iterations).")
                return ExitFlag.FAILURE

            elif msg[:3] == MSG_ABORTED:
                utils.warn("Simulation failed (aborted).")
                return ExitFlag.FAILURE

        raise Exception(f"Unknown exit condition. Check {log_path}.yaml.")

class Model2D(Model):
    """2D Model (note: can currently only be used with a RegularMesh)."""

    def initialise_ns_model(self):
        
        self.ns_model = {
            "eos": {"name": "we"},
            "gravity": GRAVITY,
            "logfile": {"echo": False},
            "mesh": {
                "filename": f"{self.mesh.name}.msh", 
                "thickness": self.mesh.dy
            },
            "title": "2D Model"
        }

    def get_pr_data(self):
        """Returns the temperatures (deg C), pressures (MPa) and
        enthalpies (kJ/kg) from a production history simulation."""

        with h5py.File(f"{self.pr_path}.h5", "r") as f:
        
            cell_inds = f["cell_index"][:, 0]
            src_inds = f["source_index"][:, 0]
            
            temp = f["cell_fields"]["fluid_temperature"]
            pres = f["cell_fields"]["fluid_pressure"]
            enth = f["source_fields"]["source_enthalpy"]

            ns_temp = np.array(temp[0][cell_inds])
            pr_pres = np.array([p[cell_inds][self.feedzone_cell_inds]
                                for p in pres]) / 1e6
            pr_enth = np.array([e[src_inds][-self.n_feedzones:]
                                for e in enth]) / 1e3

        F_i = np.concatenate((ns_temp.flatten(),
                              pr_pres.flatten(),
                              pr_enth.flatten()))

        return F_i

class Model3D(Model):

    def initialise_ns_model(self):
        
        self.ns_model = {
            "eos": {"name": "we"},
            "gravity": GRAVITY,
            "logfile": {"echo": True},
            "mesh": {
                "filename": f"{self.mesh.name}.msh"
            },
            "title": "3D Model"
        }