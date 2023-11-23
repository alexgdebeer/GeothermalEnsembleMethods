import h5py

from src.methods import run_eki_dmc

from setup_slice import *

Np = n_blocks_crse + 1
NF = n_blocks_crse + 2 * n_wells * (nt + 1)
Ne = 100

ws, ps, Fs, Gs, inds_succ = run_eki_dmc(F, G, prior, y, C_e, Np, NF, Ne)

# TODO: make this into a function
with h5py.File(f"eki_dmc.h5", "w") as f:
    f.create_dataset("ws", data=ws)
    f.create_dataset("ps", data=ps)
    f.create_dataset("Fs", data=Fs)
    f.create_dataset("Gs", data=Gs)
    f.create_dataset("inds_succ", data=inds_succ)