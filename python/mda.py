import ensemble_methods as em
import scipy.sparse as sparse

from setup import *

Ne = 50

# Define localisation matrix
loc_mat = sparse.block_diag((
    prior.gp_depth_clay.cor,
    prior.rf_perm_shal.cor,
    prior.rf_perm_clay.cor,
    prior.rf_perm_deep.cor,
    [1.0]
)).toarray()

prob = em.ESMDAProblem(f, g, prior, likelihood, Nf, Ne)

prob.run()
prob.save_results(fname="data/mda_50_noloc")