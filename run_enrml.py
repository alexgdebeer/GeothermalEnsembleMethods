from src.methods import *
from setup_slice import *

Np = n_blocks_crse + 1
NF = n_blocks_crse + 2 * n_wells * (nt + 1)
NG = len(y)
Ne = 100

ensemble = Ensemble(prior, generate_particle, get_result, Np, NF, NG, Ne)

settings = [
    (IdentityLocaliser(), IdentityInflator(), GaussianImputer()),
    (BootstrapLocaliser(), IdentityInflator(), GaussianImputer())
]

fnames = [
    "data/enrml/enrml.h5",
    "data/enrml/enrml_boot.h5"
]

for (localiser, inflator, imputer), fname in zip(settings, fnames):

    ws, ps, Fs, Gs, Ss, lams, en_ind, inds_succ = run_enrml(
        ensemble, prior, y, C_e,
        localiser=localiser, inflator=inflator, imputer=imputer, nesi=True)

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