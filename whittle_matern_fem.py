from collections import deque
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.special import gamma

from skfem import Basis, ElementTriP1, MeshTri
from skfem.visuals.matplotlib import plot

np.random.seed(0)

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# Define number of spatial dimensions of problem and smoothness parameter
d = 2
nu = 2 - d/2.0

# Define lengthscale and standard deviation
l = 0.2
sigma = 1.0
alpha = sigma**2 * (2**d * np.pi**(d/2) * gamma(nu + d/2.0)) / gamma(nu)

mesh = MeshTri.init_circle(5)
basis = Basis(mesh, ElementTriP1())

npoints = mesh.nvertices
elements = mesh.t.T
nodes = mesh.p.T
boundary_nodes = set(mesh.boundary_nodes())

# Define gradients of basis functions in transformed simplex
gradients = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])

M = sparse.lil_matrix((npoints, npoints))
S = sparse.lil_matrix((npoints, npoints))
N = sparse.lil_matrix((npoints, npoints))

for element in elements:
    
    # Extract the global indices of the nodes of the current element
    nodes_e = nodes[element, :]

    inds = deque([0, 1, 2])

    for _ in range(3):

        # Extract the global index of the current point
        pi = element[inds[0]]
        
        # Generate transformation matrix
        T = np.array([nodes_e[inds[1]] - nodes_e[inds[0]],
                      nodes_e[inds[2]] - nodes_e[inds[0]]]).T
    
        # Compute the absolute value of the determinant 
        detT = np.abs(np.linalg.det(T))

        for j in range(3):

            # Find the global index of point k
            pj = element[inds[j]]

            # Add the inner product of the basis functions of nodes i and j
            # to the M matrix
            M[pi, pj] += detT * (1/12 if pi == pj else 1/24) # TODO: tidy

            # Add the inner product of the gradients of the basis functions 
            # of points i and j over the current element to the S matrix
            S[pi, pj] += detT * 0.5 * \
                gradients[0] @ np.linalg.inv(T.T @ T) @ gradients[j].T
            
            # Neumann boundary stuff
            if pi in boundary_nodes and pj in boundary_nodes and pi != pj:
                N[pi, pj] += np.linalg.norm(nodes[pi] - nodes[pj]) / 6

        # Rotate the simplex
        inds.rotate(-1)

PLOT_SAMPLES = False
PLOT_BOUNDARIES = True

if PLOT_SAMPLES:

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))

    W = np.random.normal(loc=0.0, scale=1.0, size=npoints)

    for l, ax in zip([0.05, 0.1, 0.25, 0.5], axes.flat):

        lam = 1.42 * l

        H = M + l**2 * S + (l**2 / lam) * N
        G = (alpha * l ** 2 * M).tocsc()
        L = np.linalg.cholesky(G.toarray())
        X = sparse.linalg.spsolve(H, L.T @ W)

        plot(mesh, X, ax=ax)
        ax.axis("off")
        ax.set_aspect("equal", "box")
        ax.set_title(f"$l$ = {l}")

    plt.savefig("samples.pdf")

if PLOT_BOUNDARIES:

    nruns = 1000

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

    # Neumann
    H = M + l**2 * S
    G = alpha * l ** 2 * M 
    L = np.linalg.cholesky(G.toarray())
    XS = np.zeros((npoints, nruns))

    for i in range(nruns):

        W = np.random.normal(loc=0.0, scale=1.0, size=npoints)
        XS[:,i] = sparse.linalg.spsolve(H, L.T @ W)

        if (i+1) % 10 == 0:
            print(f"{i+1} runs complete.")

    stds = np.std(XS, axis=1)
    plot(mesh, stds, ax=axes[0], colorbar=True, vmin=0, vmax=2)

    # Robin
    lam = 1.42 * l

    H = M + l**2 * S + (l**2 / lam) * N
    G = alpha * l ** 2 * M
    L = np.linalg.cholesky(G.toarray())
    XS = np.zeros((npoints, nruns))

    for i in range(nruns):

        W = np.random.normal(loc=0.0, scale=1.0, size=npoints)
        XS[:,i] = sparse.linalg.spsolve(H, L.T @ W)

        if (i+1) % 10 == 0:
            print(f"{i+1} runs complete.")

    stds = np.std(XS, axis=1)
    plot(mesh, stds, ax=axes[1], colorbar=True, vmin=0, vmax=2)

    axes[0].axis("off")
    axes[1].axis("off")
    axes[0].set_aspect("equal", "box")
    axes[1].set_aspect("equal", "box")
    axes[0].set_title("Neumann")
    axes[1].set_title("Robin")

    plt.savefig("boundary_condition_comparison.pdf")


# spatial.delaunay_plot_2d(mesh)
# plt.show()

pass