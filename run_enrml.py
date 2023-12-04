from src.methods import *
from setup_slice import *

Np = n_blocks_crse + 1
NF = n_blocks_crse + 2 * n_wells * (nt + 1)
Ne = 100

localiser = IdentityLocaliser()
imputer = GaussianImputer()

ws, ps, Fs, Gs, Ss, lams, en_ind, inds = run_enrml(
    F, G, prior, y, C_e, Np, NF, Ne, 
    localiser=localiser, imputer=imputer)

fname = "enrml.h5"
results = {
    "ws": ws, "ps": ps, 
    "Fs": Fs, "Gs": Gs, 
    "inds": inds, 
    "Ss": Ss, "lams": lams
}
save_results_enrml(fname, results, en_ind)