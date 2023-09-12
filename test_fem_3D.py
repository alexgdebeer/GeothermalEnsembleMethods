import numpy as np
from scipy.special import gamma
from scipy import spatial

from t2grids import mulgrid
from matern_fields import *

# np.random.seed(0)

geo = mulgrid("models/channel/gCH.dat").layermesh

# Define number of spatial dimensions of problem and smoothness parameter
d = 2
nu = 2 - d/2.0

# Define lengthscale in each direction
lx = 100
ly = 300
lz = 100

# Define marginal variance and other parameters
sigma = 1.0
alpha = sigma**2 * (2**d * np.pi**(d/2) * gamma(nu + d/2.0)) / gamma(nu)

# TODO: tune Robin parameter
lam = 1.42 * np.sqrt(lx * ly * lz)

# Generate mesh
points = np.array([c.centre for c in geo.cell])
mesh = spatial.Delaunay(points)

# TODO: add check to confirm that points aren't missing from the mesh

elements = mesh.simplices
nodes = mesh.points
boundary_nodes = get_boundary_nodes(mesh)
neighbours = mesh.neighbors
n_nodes = len(nodes)

M, Kx, Ky, Kz, N = generate_fem_matrices_3D(nodes, boundary_nodes, elements, neighbours)
L = np.linalg.cholesky(M.toarray())

W = np.random.normal(loc=0.0, scale=1.0, size=n_nodes)

K = lx**2 * Kx + ly**2 * Ky + lz**2 * Kz
H = M + K + (lx * ly * lz / lam) * N
        
X = sparse.linalg.spsolve(H, np.sqrt(alpha * lx * ly * lz) * L.T @ W)