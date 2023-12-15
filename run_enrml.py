from src.methods import *
from setup_slice import *

Np = n_blocks_crse + 1
NF = n_blocks_crse + 2 * n_wells * (nt + 1)
NG = len(y)
Ne = 100

ensemble = Ensemble(prior, generate_particle, get_result, Np, NF, NG, Ne)
localiser = IdentityLocaliser()
inflator = IdentityInflator()
imputer = GaussianImputer()

ws, ps, Fs, Gs, Ss, lams, en_ind, inds_succ = run_enrml(
    ensemble, prior, y, C_e,
    localiser=localiser, inflator=inflator, imputer=imputer, nesi=False)

fname = "enrml.h5"
results = {
    "ws": ws, 
    "ps": ps, 
    "Fs": Fs, 
    "Gs": Gs, 
    "inds_succ": inds_succ, 
    "Ss": Ss, 
    "lams": lams
}

save_results_enrml(fname, results, en_ind)