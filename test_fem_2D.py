import numpy as np
from scipy.special import gamma
from scipy import spatial

from t2grids import mulgrid
from matern_fields import *

# np.random.seed(0)

"""Returns the set of nodes on the boundary of the triangulation."""
def get_boundary_nodes(mesh):

    boundary_nodes = set()

    for simplex, neighbors in zip(mesh.simplices, mesh.neighbors):
        for i, neighbor in enumerate(neighbors):
            if neighbor == -1:
                nodes = [n for j, n in enumerate(simplex) if j != i]
                boundary_nodes.update(nodes)
    
    return boundary_nodes

# Define number of spatial dimensions of problem and smoothness parameter
d = 2
nu = 2 - d/2.0

# Define lengthscale and standard deviation
lx = 100
ly = 300
lam = 1.42 * np.sqrt(lx * ly)
sigma = 1.0
alpha = sigma**2 * (2**d * np.pi**(d/2) * gamma(nu + d/2.0)) / gamma(nu)

# Read in model geometry
geo = mulgrid("models/channel/gCH.dat").layermesh

points = np.array([c.centre for c in geo.column])

mesh = spatial.Delaunay(points)

elements = mesh.simplices
nodes = mesh.points
boundary_nodes = get_boundary_nodes(mesh)
n_nodes = len(nodes)

M, Kx, Ky, N = generate_fem_matrices_2D(nodes, boundary_nodes, elements)
L = np.linalg.cholesky(M.toarray())

W = np.random.normal(loc=0.0, scale=1.0, size=n_nodes)

K = lx**2 * Kx + ly**2 * Ky
H = M + K + (lx * ly / lam) * N
        
X = sparse.linalg.spsolve(H, np.sqrt(alpha * lx * ly) * L.T @ W)

geo.layer_plot(value=[X[c.column.index] for c in geo.cell])

# point_cloud = pv.PolyData(points)
# mesh = point_cloud.delaunay_2d()

# nodes = mesh.points[:, :2]
# n_nodes = len(nodes)
# elements = mesh.regular_faces
# boundary = mesh.extract_feature_edges(boundary_edges=True, 
#                                       non_manifold_edges=False, 
#                                       manifold_edges=False)

# # Surely a better way than this...
# boundary_nodes = [geo.find(c[:2]).index for c in boundary.points]

# from matplotlib import pyplot as plt
# plt.triplot(points[:,0], points[:,1], mesh.simplices)
# plt.scatter(points[list(boundary_nodes), 0], points[list(boundary_nodes), 1], color="pink")
# plt.show()