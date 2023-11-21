"""Plots the quantities related to the model currently in the slice
model folder.
"""

import h5py
import numpy as np

from layermesh import mesh as lm
from matplotlib import pyplot as plt

MODEL_FOLDER = "models/slice"
MODEL_NAME = "SL625"
GEOM_NAME = "gSL625"

mesh = lm.mesh(f"{MODEL_FOLDER}/{GEOM_NAME}.h5")

with h5py.File(f"{MODEL_FOLDER}/{MODEL_NAME}_PR.h5", "r") as f:

    cell_inds = f["cell_index"][:, 0]
    src_inds = f["source_index"][:, 0]

    ns_temps = f["cell_fields"]["fluid_temperature"][0][cell_inds]
    ns_sats = f["cell_fields"]["fluid_vapour_saturation"][-1][cell_inds]

    enthalpies = [e[src_inds][-5:] 
                  for e in f["source_fields"]["source_enthalpy"]]

mesh.slice_plot(value=ns_temps, colourmap="coolwarm")
mesh.slice_plot(value=ns_sats, colourmap="viridis")

print(enthalpies)
plt.plot(enthalpies)
plt.show()
