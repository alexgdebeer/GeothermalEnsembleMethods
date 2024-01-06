from src.methods import *
from setup_fault import *

Ne = 100

ensemble = Ensemble(prior, generate_particle, get_result, Np, NF, NG, Ne)
localiser = IdentityLocaliser()
inflator = IdentityInflator()
imputer = GaussianImputer()

ws, ps, Fs, Gs, inds = run_eki_dmc(
    ensemble, prior, y, C_e,
    localiser=localiser, inflator=inflator,
    imputer=imputer, nesi=True)

fname = "data/fault/eki_dmc_inf.h5"
results = {
    "ws": ws, 
    "ps": ps, 
    "Fs": Fs, 
    "Gs": Gs, 
    "inds_succ": inds
}
save_results_eki(fname, results)