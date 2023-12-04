from src.methods import *
from setup_slice import *

Np = n_blocks_crse + 1
NF = n_blocks_crse + 2 * n_wells * (nt + 1)
Ne = 100

localiser = BootstrapLocaliser()
imputer = GaussianImputer()

ws, ps, Fs, Gs, inds = run_eki_dmc(
    F, G, prior, y, C_e, Np, NF, Ne, 
    localiser, imputer)

fname = "eki_dmc_boot.h5"
results = {
    "ws": ws, 
    "ps": ps, 
    "Fs": Fs, 
    "Gs": Gs, 
    "inds": inds
}
save_results_eki(fname, results)