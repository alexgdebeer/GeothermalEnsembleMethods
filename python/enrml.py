import ensemble_methods as em

from setup import *

Ne = 50
i_max = 10
gamma = 10
l_min = 0.01
max_cuts = 5

prob = em.EnRMLProblem(f, g, prior, likelihood, Nf, Ne)
prob.run(i_max, gamma, l_min, max_cuts)
prob.save_results(fname="data/enrml_test")