import h5py
import json
import layermesh.mesh as lm
import matplotlib.pyplot as plt
import os
import pywaiwera
import yaml

plt.rc("text", usetex=True)
plt.rc("font", family="serif")


def build_base_model(xmax, ymax, zmax, nx, ny, nz, 
                     mesh_name, model_name, model_folder, mass_cols, 
                     P_atm=1.0e+5, T_atm=20.0, P0=1.0e+5, T0=20.0, 
                     permeability=1.0e-14, porosity=0.25):

    dx = xmax / nx
    dy = ymax / ny
    dz = zmax / nz

    dxs = [dx] * nx
    dys = [dy] * ny
    dzs = [dz] * nz

    mesh = lm.mesh(rectangular=(dxs, dys, dzs))
    mesh.write(f"{model_folder}/{mesh_name}.h5")
    mesh.export(f"{model_folder}/{mesh_name}.msh", fmt="gmsh22")

    incon_name = f"{model_folder}/{model_name}_incon.h5"

    heat_cells = [
        c.cell[-1].index for c in mesh.column 
        if c.index not in mass_cols]

    model = {
        "eos": {"name" : "we"},
        "gravity": 9.81,
        "logfile": {"echo" : False},
        "mesh": {
            "filename": f"{model_folder}/{mesh_name}.msh", 
            "thickness": dy
        },
        "output" : {
            "filename": f"{model_folder}/{model_name}.h5",
            "frequency": 0, 
            "initial": False, 
            "final": True
        },
        "title": "Simple 2D model"
    }

    model["rock"] = {"types" : [
        {
            "name": f"{c.index}", 
            "permeability": permeability,
            "porosity": porosity, 
            "cells": [c.index],
            "wet_conductivity": 2.5,
            "dry_conductivity": 2.5,
            "density": 2.5e+3,
            "specific_heat": 1.0e+3
        }
        for c in mesh.cell
    ]}

    model["source"] = [
        {
            "component": "energy",
            "rate": 1.0e+3,
            "cells": heat_cells
        }
    ]

    if os.path.isfile(incon_name):
        model["initial"] = {"filename": incon_name}
    else:
        print("Warning: initial condition file not found.")
        model["initial"] = {"primary": [P0, T0], "region": 1}

    model["boundaries"] = [{
        "primary": [P_atm, T_atm], 
        "region": 1,
        "faces": {
            "cells": [c.index for c in mesh.surface_cells],
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
            "maximum": {"number": 500},
            "method": "beuler",
            "stop": {"size": {"maximum": 1.0e+15}}
        }
    }

    with open(f"{model_folder}/{model_name}_base.json", "w") as f:
        json.dump(model, f, indent=2, sort_keys=True)


def build_model(model_folder, model_name, mass_rate, mass_cells, permeabilities):

    with open(f"{model_folder}/{model_name}_base.json", "r") as f:
        model = json.load(f)

    model["source"].append({
        "component": "water",
        "enthalpy": 1.0e+7, 
        "rate": mass_rate / len(mass_cells),
        "cells": list([int(c) for c in mass_cells])
    })

    for rt in model["rock"]["types"]:
        rt["permeability"] = permeabilities[rt["cells"][0]]

    model["output"]["filename"] = f"{model_folder}/{model_name}.h5"

    with open(f"{model_folder}/{model_name}.json", "w") as f:
        json.dump(model, f, indent=2, sort_keys=True)


def get_mass_cells(mesh_name, model_folder, mass_cols):
    
    m = lm.mesh(f"{model_folder}/{mesh_name}.h5")
    return [c.cell[-1].index for c in m.column if c.index in mass_cols]


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