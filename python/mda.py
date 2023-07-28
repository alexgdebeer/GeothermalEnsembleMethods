from setup import *
from ensemble_methods import EnsembleProblem

prob = EnsembleProblem(f, g, prior, likelihood, len(fs_t), Ne=5)

prob.run_es_mda(Ni=4)
prob.save_results(fname="mda_test")