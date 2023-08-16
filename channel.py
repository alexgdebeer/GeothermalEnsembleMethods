import numpy as np
import scipy.special as special

from t2grids import *
import layermesh.mesh as lm

geo = mulgrid("models/channel/gCH.dat").layermesh

cs_base = geo.layer[-1].cell
cxs = [c.centre[0] for c in cs_base]
cys = [c.centre[1] for c in cs_base]

rho = 100
nu = 100

