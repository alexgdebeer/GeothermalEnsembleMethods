import ensemble_methods as em
import scipy.sparse as sparse

from setup import *

Ne = 25

# # Define localisation matrix
# loc_mat = sparse.block_diag((
#     np.exp(-0.5 * (prior.d_depth_clay.x_dists / (10 * prior.d_depth_clay.l)) ** 2),
#     np.exp(-0.5 * (prior.d_perm_shal.x_dists / (10 * prior.d_perm_shal.lx)) ** 2 + \
#            -0.5 * (prior.d_perm_shal.z_dists / (10 * prior.d_perm_shal.lz)) ** 2),
#     np.exp(-0.5 * (prior.d_perm_clay.x_dists / (10 * prior.d_perm_clay.lx)) ** 2 + \
#            -0.5 * (prior.d_perm_clay.z_dists / (10 * prior.d_perm_clay.lz)) ** 2),
#     np.exp(-0.5 * (prior.d_perm_deep.x_dists / (10 * prior.d_perm_deep.lx)) ** 2 + \
#            -0.5 * (prior.d_perm_deep.z_dists / (10 * prior.d_perm_deep.lz)) ** 2),
#     [1.0]
# )).toarray()

prob = em.ESMDAProblem(f, g, prior, likelihood, Nf, Ne)#, loc_type="linearised", loc_mat=loc_mat)

prob.run()
prob.save_results(fname=f"data/mda_{Ne}")