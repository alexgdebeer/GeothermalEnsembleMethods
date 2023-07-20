from enum import Enum
import copy
import h5py
import json
import layermesh.mesh as lm
import numpy as np
import os
import pywaiwera
import yaml

YELLOW = "\033[93m"
END_COLOUR = "\033[0m"

P0 = 1.0e+5
T0 = 20.0
P_ATM = 1.0e+5
T_ATM = 20.0

GRAVITY = 9.81

POROSITY = 0.10
CONDUCTIVITY = 2.5
DENSITY = 2.5e+3
SPECIFIC_HEAT = 1.0e+3

MASS_ENTHALPY = 1.5e+6
HEAT_RATE = 0.2

MAX_NS_TSTEPS = 500
MAX_PR_TSTEPS = 1000
NS_STEPSIZE = 1.0e+16


class ExitFlag(Enum):
    SUCCESS = 1
    MAX_ITS = 2
    ABORTED = 3


def warn(msg):
    print(f"{YELLOW}Warning: {msg}{END_COLOUR}")


def load_json(fname):
    with open(fname, "r") as f:
        model = json.load(f)
    return model


def save_json(model, fname):
    with open(fname, "w") as f:
        json.dump(model, f, indent=2, sort_keys=True)


class Mesh():

    def __init__(self, fname, xmax, ymax, zmax, nx, ny, nz):

        self.fname = fname

        self.xmax = xmax 
        self.ymax = ymax 
        self.zmax = zmax

        self.nx = nx 
        self.ny = ny 
        self.nz = nz

        self.dx = xmax / nx
        self.dy = ymax / ny
        self.dz = zmax / nz

        dxs = [self.dx] * nx
        dys = [self.dy] * ny
        dzs = [self.dz] * nz

        self.m = lm.mesh(rectangular=(dxs, dys, dzs))

    def write_to_file(self):
        
        self.m.write(f"{self.fname}.h5")
        self.m.export(f"{self.fname}.msh", fmt="gmsh22")


class MassUpflow():
    def __init__(self, loc, rate):
        self.loc = loc
        self.rate = rate


class Feedzone():
    def __init__(self, loc, rate):
        self.loc = loc
        self.rate = rate


class Well():
    def __init__(self, feedzone, obs_times):
        self.feedzone = feedzone 
        self.obs_times = obs_times


class Model():

    def __init__(self, path: str, mesh: Mesh, 
                 perms: list, wells: "list[Well]", upflows: "list[MassUpflow]",
                 dt: float, tmax: float):
        
        self.ns_path = f"{path}_NS"
        self.pr_path = f"{path}_PR"
        self.incon_path = f"{path}_incon"

        self.mesh = mesh
        self.perms = perms
        self.wells = wells 
        self.upflows = upflows

        self.feedzone_cells = [
            self.mesh.find(w.feedzone.loc, indices=True) 
            for w in self.wells]

        self.dt = dt
        self.tmax = tmax

        self.ns_model = None
        self.pr_model = None
        
        self._generate_ns()
        self._generate_pr()

    def _add_boundaries(self):

        self.ns_model["boundaries"] = [{
            "primary": [P_ATM, T_ATM], 
            "region": 1,
            "faces": {
                "cells": [c.index for c in self.mesh.m.surface_cells],
                "normal": [0, 0, 1]
            }
        }]
    
    def _add_upflows(self):

        upflow_cells = [
            self.mesh.find(upflow.loc, indices=True) 
            for upflow in self.upflows]

        heat_cells = [
            c.cell[-1].index for c in self.mesh.column 
            if c.cell[-1].index not in upflow_cells]

        self.ns_model["source"] = [{
            "component": "energy",
            "rate": HEAT_RATE * self.mesh.cell[c].column.area,
            "cell": c
        } for c in heat_cells]

        self.ns_model["source"].extend([{
            "component": "water",
            "enthalpy": MASS_ENTHALPY, 
            "rate": u.rate,
            "cell": self.mesh.find(u.loc, indices=True)
        } for u in self.upflows])

    def _add_rocktypes(self):
        
        if len(self.perms) != self.mesh.ncell:
            raise Exception("Incorrect number of permeabilities.")
        
        self.ns_model["rock"] = {"types": [{
            "name": f"{c.index}",
            "porosity": POROSITY, 
            "permeability": self.perms[c.index],
            "cells": [c.index],
            "wet_conductivity": CONDUCTIVITY,
            "dry_conductivity": CONDUCTIVITY,
            "density": DENSITY,
            "specific_heat": SPECIFIC_HEAT
        } for c in self.mesh.m.cell]}

    def _add_wells(self):

        self.pr_model["source"].extend([{
            "component": "water",
            "rate": w.feedzone.rate,
            "cell": self.mesh.find(w.feedzone.loc, indices=True)
        } for w in self.wells])

    def _add_ns_incon(self):

        if os.path.isfile(f"{self.incon_path}.h5"):
            self.ns_model["initial"] = {"filename": f"{self.incon_path}.h5"}
        else:
            warn("Initial condition not found. Improvising...")
            self.ns_model["initial"] = {"primary": [P0, T0], "region": 1}

    def _add_pr_incon(self):
        self.pr_model["initial"] = {"filename": f"{self.ns_path}.h5"}

    def _add_ns_timestepping(self):

        self.ns_model["time"] = {
            "step": {
                "size": 1.0e+6,
                "adapt": {"on": True}, 
                "maximum": {"number": MAX_NS_TSTEPS},
                "method": "beuler",
                "stop": {"size": {"maximum": NS_STEPSIZE}}
            }
        }
    
    def _add_pr_timestepping(self):

        self.pr_model["time"] = {
            "step": {
                "adapt": {"on": True},
                "size": self.dt,
                "maximum": {"number": MAX_PR_TSTEPS},
            },
            "stop": self.tmax
        }

    def _add_ns_output(self):

        self.ns_model["output"] = {
            "frequency": 0, 
            "initial": False, 
            "final": True
        }

    def _add_pr_output(self):
        
        self.pr_model["output"] = {
            "checkpoint": {
                "time": [self.dt], 
                "repeat": True
            },
            "frequency": 0,
            "initial": True,
            "final": False
        }

    def _generate_ns(self):
        
        self.ns_model = {
            "eos": {"name": "we"},
            "gravity": GRAVITY,
            "logfile": {"echo": False},
            "mesh": {
                "filename": f"{self.mesh.fname}.msh", 
                "thickness": self.mesh.dy
            },
            "title": "Slice model"
        }
        
        self._add_boundaries()
        self._add_rocktypes()
        self._add_upflows()
        self._add_ns_incon()
        self._add_ns_timestepping()
        self._add_ns_output()

        save_json(self.ns_model, f"{self.path_pr}.json")

    def _generate_pr(self):

        self.pr_model = copy.deepcopy(self.ns_model)

        self._add_wells()
        self._add_pr_timestepping()
        self._add_pr_output()
        self._add_pr_incon()

        save_json(self.pr_model, f"{self.ns_path}.json")

    def run(self):

        env = pywaiwera.docker.DockerEnv(check=False, verbose=False)
        
        env.run_waiwera(f"{self.ns_path}.json", noupdate=True)
        flag = self._get_exitflag(self.ns_path)
        if flag != ExitFlag.SUCCESS: return flag 

        env.run_waiwera(f"{self.pr_path}.json", noupdate=True)
        return self._get_exitflag(self.pr_path)

    def _get_exitflag(self, log_path):

        with open(f"{log_path}.yaml", "r") as f:
            log = yaml.safe_load(f)

        for msg in log[:-50:-1]:

            if msg[:3] == ["info", "timestep", "end_time_reached"]:
                return ExitFlag.SUCCESS
            
            elif msg[:3] == ["info", "timestep", "stop_size_maximum_reached"]:
                return ExitFlag.SUCCESS

            elif msg[:3] == ["info", "timestep", "max_timesteps_reached"]:
                return ExitFlag.MAX_ITS

            elif msg[:3] == ["warn", "timestep", "aborted"]:
                return ExitFlag.ABORTED

        raise Exception("Unknown exit condition encountered. Check the log.")

    def get_pr_data(self):

        with h5py.File(f"{self.pr_path}.h5", "r") as f:
        
            cell_inds = f["cell_index"][:, 0]
            src_inds = f["source_index"][:, 0]
            
            ts = f["cell_fields"]["fluid_temperature"][0][cell_inds]
            
            ps = [p[cell_inds][self.feedzone_cells]
                  for p in f["cell_fields"]["fluid_pressure"]]   
            
            es = [e[src_inds][-len(self.wells):] 
                  for e in f["source_fields"]["source_enthalpy"]]

        ts, ps, es
        return ts, ps, es
