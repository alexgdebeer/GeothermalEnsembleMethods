import ensemble_methods as em

from setup import *

Ne = 25

prob = em.EnRMLProblem(f, g, prior, likelihood, Nf, Ne)
prob.run()
prob.save_results(fname=f"data/enrml_{Ne}_loc_shuffle")