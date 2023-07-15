import copy
import h5py
import json
import layermesh.mesh as lm
import matplotlib.pyplot as plt
import os
import pywaiwera
import yaml

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

P0 = 1.0e+5
T0 = 20.0
P_ATM = 1.0e+5
T_ATM = 20.0

GRAVITY = 9.81

POROSITY = 0.20
CONDUCTIVITY = 2.5
DENSITY = 2.5e+3
SPECIFIC_HEAT = 1.0e+3

MASS_ENTHALPY = 1.0e+7
HEAT_RATE = 1.0e+3

MAX_TSTEPS = 500
NS_STEPSIZE = 1.0e+15


def load_json(fname):
    with open(fname, "r") as f:
        model = json.load(f)
    return model


def save_json(model, fname):
    with open(fname, "w") as f:
        json.dump(model, f, indent=2, sort_keys=True)


def build_base_model(xmax, ymax, zmax, nx, ny, nz, 
                     model_path, mesh_path):

    dx = xmax / nx
    dy = ymax / ny
    dz = zmax / nz

    dxs = [dx] * nx
    dys = [dy] * ny
    dzs = [dz] * nz

    m = lm.mesh(rectangular=(dxs, dys, dzs))
    m.write(f"{mesh_path}.h5")
    m.export(f"{mesh_path}.msh", fmt="gmsh22")

    model = {
        "eos": {"name" : "we"},
        "gravity": GRAVITY,
        "logfile": {"echo" : False},
        "mesh": {
            "filename": f"{mesh_path}.msh", 
            "thickness": dy
        },
        "title": "2D model"
    }

    model["rock"] = {"types" : [{
        "name": f"{c.index}",
        "porosity": POROSITY, 
        "cells": [c.index],
        "wet_conductivity": CONDUCTIVITY,
        "dry_conductivity": CONDUCTIVITY,
        "density": DENSITY,
        "specific_heat": SPECIFIC_HEAT
    } for c in m.cell]}

    model["boundaries"] = [{
        "primary": [P_ATM, T_ATM], 
        "region": 1,
        "faces": {
            "cells": [c.index for c in m.surface_cells],
            "normal": [0, 1]
        }
    }]

    model["time"] = {
        "step": {
            "size": 1.0e+6,
            "adapt": {
                "on": True,
                "method": "iteration",
                "minimum": 5, 
                "maximum": 8
            }, 
            "maximum": {"number": MAX_TSTEPS},
            "method": "beuler",
            "stop": {"size": {"maximum": NS_STEPSIZE}}
        }
    }

    save_json(model, f"{model_path}_base.json")


def build_ns_model(model_path, mesh, perms, upflow_locs, upflow_rates):

    model = load_json(f"{model_path}_base.json")

    if os.path.isfile(f"{model_path}_incon.h5"):
        model["initial"] = {"filename": f"{model_path}_incon.h5"}
    else:
        print("Warning: initial condition file not found.")
        model["initial"] = {"primary": [P0, T0], "region": 1}

    upflow_cells = [
        mesh.find(loc, indices=True) 
        for loc in upflow_locs]

    heat_cells = [
        c.cell[-1].index for c in mesh.column 
        if c.cell[-1].index not in upflow_cells]

    model["source"] = [{
        "component": "energy",
        "rate": HEAT_RATE,
        "cells": heat_cells
    }]

    model["source"].extend([{
        "component": "water",
        "enthalpy": MASS_ENTHALPY, 
        "rate": rate,
        "cell": cell
    } for cell, rate in zip(upflow_cells, upflow_rates)])

    for rt in model["rock"]["types"]:
        rt["permeability"] = perms[rt["cells"][0]]

    model["output"] = {
        "filename": f"{model_path}_NS.h5",
        "frequency": 0, 
        "initial": False, 
        "final": True
    }

    return model


def build_pr_model(model, model_path, mesh, feedzone_locs, feedzone_rates):

    model["source"].extend([{
        "component": "water",
        "rate": rate,
        "cell": mesh.find(loc, indices=True)
    } for loc, rate in zip(feedzone_locs, feedzone_rates)])

    model["time"] = {
        "step": {
            "size": 3600*24*7*10,
            "adapt": {"on": True}, 
            "maximum": {"number": 500},
            "method": "beuler",
            "stop": {"size": {"maximum": 1.0e+15}}
        },
        "stop": 3600*24*7*10
    }

    model["output"] = {
        "checkpoint": {
            "time": [3600*24*7*10], 
            "repeat": True
        }, 
        "filename": f"{model_path}_PR.h5"
    }

    model["initial"] = {"filename": f"{model_path}_NS.h5"}

    return model


def build_models(model_path, mesh_path, perms, upflow_locs, upflow_rates, 
                 feedzone_locs, feedzone_rates):

    mesh = lm.mesh(f"{mesh_path}.h5")

    ns_model = build_ns_model(
        model_path, mesh, perms, 
        upflow_locs, upflow_rates)
    
    pr_model = build_pr_model(
        copy.deepcopy(ns_model), model_path, mesh, 
        feedzone_locs, feedzone_rates)

    save_json(ns_model, f"{model_path}_NS.json")
    save_json(pr_model, f"{model_path}_PR.json")


def run_model(model_path):
    env = pywaiwera.docker.DockerEnv(check=False, verbose=False)
    env.run_waiwera(f"{model_path}.json", noupdate=True)


def run_info(model_path):

    with open(f"{model_path}.yaml", "r") as f:
        log = yaml.safe_load(f)

    for msg in log[:-50:-1]:

        if msg[:3] == ["info", "timestep", "end_time_reached"]:
            return "success"
        
        elif msg[:3] == ["info", "timestep", "stop_size_maximum_reached"]:
            return "success"

        elif msg[:3] == ["info", "timestep", "max_timesteps_reached"]:
            return "max_its"

        elif msg[:3] == ["warn", "timestep", "aborted"]:
            return "aborted"

    raise Exception("Unknown exit condition encountered. Check the log.")


def slice_plot(model_folder, mesh_name, quantity, cmap="coolwarm",
               value_label="log(Permeability)", value_unit="m$^2$"):

    mesh = lm.mesh(f"{model_folder}/{mesh_name}.h5")

    mesh.slice_plot(
        value=quantity, 
        value_label=value_label,
        value_unit=value_unit,
        colourmap=cmap,
        xlabel="$x$ (m)",
        ylabel="$z$ (m)"
    )


def get_quantity(model_path, q="fluid_temperature"):

    results = h5py.File(f"{model_path}.h5", "r")
    index = results["cell_index"][:, 0]
    return results["cell_fields"][q][-1][index]