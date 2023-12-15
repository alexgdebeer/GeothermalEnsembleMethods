from src.methods import *
from setup_slice import *

Np = n_blocks_crse + 1
NF = n_blocks_crse + 2 * n_wells * (nt + 1)
NG = len(y)
Ne = 10

ensemble = Ensemble(prior, generate_particle, get_result, Np, NF, NG, Ne)
localiser = IdentityLocaliser()
inflator = IdentityInflator()
imputer = GaussianImputer()

ws, ps, Fs, Gs, inds_succ = run_eki_dmc(
    ensemble, prior, y, C_e,
    localiser=localiser, inflator=inflator, imputer=imputer, nesi=False)

fname = "eki_dmc.h5"
results = {
    "ws": ws, 
    "ps": ps, 
    "Fs": Fs, 
    "Gs": Gs, 
    "inds_succ": inds_succ
}
save_results_eki(fname, results)