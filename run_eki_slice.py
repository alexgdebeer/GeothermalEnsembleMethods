from src.methods import *
from setup_slice import *

Ne = 100

ensemble = Ensemble(prior, generate_particle, get_result, Np, NF, NG, Ne)

localiser = IdentityLocaliser()
inflator = IdentityInflator()
imputer = GaussianImputer()

fname = "data/slice/eki_dmc.h5"

ws, ps, Fs, Gs, inds_succ = run_eki_dmc(
    ensemble, prior, y, C_e,
    localiser=localiser, inflator=inflator, 
    imputer=imputer, nesi=False)

results = {
    "ws": ws, 
    "ps": ps, 
    "Fs": Fs, 
    "Gs": Gs, 
    "inds_succ": inds_succ
}
save_results_eki(fname, results)