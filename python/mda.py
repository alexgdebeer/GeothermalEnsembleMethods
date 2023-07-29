import scipy.sparse as sparse

from setup import *
from ensemble_methods import EnsembleProblem

Ni = 4
Ne = 25

# Define localisation matrix
loc_mat = sparse.block_diag((
    prior.gp_depth_clay.cor,
    prior.rf_perm_shal.cor,
    prior.rf_perm_clay.cor,
    prior.rf_perm_deep.cor,
    [1.0]
)).toarray()

prob = EnsembleProblem(f, g, prior, likelihood, Nf, Ne)

prob.run_es_mda(Ni)
prob.save_results(fname="data/mda_test")