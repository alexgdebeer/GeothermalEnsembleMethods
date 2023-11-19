import GeothermalEnsembleMethods as gm

from setup_slice import *

Ne = 100

prob = gm.EnRMLProblem(f, g, prior, likelihood, Nf, Ne, i_max=30, loc_type="cycle")
prob.run()
prob.save_results(fname=f"data/enrml_{Ne}_loc_cycle_max_30_tsvd_99")