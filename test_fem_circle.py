import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma

from skfem import Basis, ElementTriP1, MeshTri
from skfem.visuals.matplotlib import plot

from matern_fields import *

np.random.seed(0)

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# Define number of spatial dimensions of problem and smoothness parameter
d = 2
nu = 2 - d/2.0

# Define variance hyperparameter
sigma = 1.0
alpha = sigma**2 * (2**d * np.pi**(d/2) * gamma(nu + d/2.0)) / gamma(nu)

# Set up mesh
mesh = MeshTri.init_circle(5)
basis = Basis(mesh, ElementTriP1())

points = 10 * mesh.p.T
elements = mesh.t.T
boundary_facets = mesh.facets.T[mesh.boundary_facets()]

M, Kx, Ky, N = generate_fem_matrices_2D(points, elements, boundary_facets)

L = np.linalg.cholesky(M.toarray())

n_nodes = len(points)

PLOT_SAMPLES = True
PLOT_BOUNDARY_CONDITIONS = True

if PLOT_SAMPLES:

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))

    W = np.random.normal(loc=0.0, scale=1.0, size=n_nodes)

    lx = 2.0
    for ly, ax in zip([0.5, 1.0, 2.5, 5.0], axes.flat):

        lam = 1.42 * np.sqrt(lx * ly)

        K = lx**2 * Kx + ly**2 * Ky
        H = M + K + (lx * ly / lam) * N

        X = sparse.linalg.spsolve(H, np.sqrt(alpha * lx * ly) * L.T @ W)

        plot(mesh, X, ax=ax)
        ax.axis("off")
        ax.set_aspect("equal", "box")
        ax.set_title(f"$l_x$ = {lx}, $l_y$ = {ly}")

    plt.savefig("samples.pdf")

if PLOT_BOUNDARY_CONDITIONS:

    n_runs = 10_000

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

    lx = 3
    ly = 3

    # Neumann
    H = M + lx**2 * Kx + ly**2 * Ky
    G = alpha * lx * ly * M.toarray()
    L = np.linalg.cholesky(G)
    XS = np.zeros((n_nodes, n_runs))

    for i in range(n_runs):

        W = np.random.normal(loc=0.0, scale=1.0, size=n_nodes)
        XS[:,i] = sparse.linalg.spsolve(H, L.T @ W)

        if (i+1) % 10 == 0:
            print(f"{i+1} runs complete.")

    stds = np.std(XS, axis=1)
    plot(mesh, stds, ax=axes[0], colorbar=True)#, vmin=0, vmax=2)

    # Robin
    lam = 1.42 * np.sqrt(lx * ly)

    H = M + lx * Kx + ly * Ky
    G = alpha * lx * ly * M.toarray()
    L = np.linalg.cholesky(G)
    XS = np.zeros((n_nodes, n_runs))

    H = M + lx**2 * Kx + ly**2 * Ky + (lx * ly / lam) * N
    G = alpha * lx * ly * M
    L = np.linalg.cholesky(G.toarray())
    XS = np.zeros((n_nodes, n_runs))

    for i in range(n_runs):

        W = np.random.normal(loc=0.0, scale=1.0, size=n_nodes)
        XS[:,i] = sparse.linalg.spsolve(H, L.T @ W)

        if (i+1) % 10 == 0:
            print(f"{i+1} runs complete.")

    stds = np.std(XS, axis=1)
    plot(mesh, stds, ax=axes[1], colorbar=True)#, vmin=0, vmax=2)

    axes[0].axis("off")
    axes[1].axis("off")
    axes[0].set_aspect("equal", "box")
    axes[1].set_aspect("equal", "box")
    axes[0].set_title("Neumann")
    axes[1].set_title("Robin")

    plt.savefig("boundary_conditions.pdf")
        