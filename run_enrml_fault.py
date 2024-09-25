from GeothermalEnsembleMethods.methods import *
from setup_fault import *

Ne = 100

ensemble = Ensemble(prior, generate_particle, get_result, Np, NF, NG, Ne)

localiser = IdentityLocaliser()
inflator = IdentityInflator()
imputer = GaussianImputer()

fname = "data/fault/enrml.h5"

ws, ps, Fs, Gs, Ss, lams, en_ind, inds_succ = run_enrml(
    ensemble, prior, y, C_e,
    localiser=localiser, inflator=inflator, imputer=imputer, nesi=False)

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