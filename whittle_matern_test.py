from collections import deque
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.special import gamma

from skfem import Basis, ElementTriP1, MeshTri
from skfem.visuals.matplotlib import plot

np.random.seed(0)

SAMPLE_1D = False
SAMPLE_2D = False
SAMPLE_2D_ANISOTROPIC = False
SAMPLE_FEM = True

# TODO: benchmark on something
# TODO: use non-periodic finite difference boundary conditions

if SAMPLE_1D:

    a = 1
    l = 1

    # Set up coordinates
    xs = np.linspace(0, 10, 1001)
    nx = len(xs)
    h = xs[1] - xs[0]

    H = np.zeros((nx, nx))

    for i in range(nx):
        H[i, i] = 1 + 2 * l**2 / h**2
        H[i, (i-1)%nx] = -l**2 / h**2
        H[i, (i+1)%nx] = -l**2 / h**2

    W = np.sqrt(a * l / h) * np.random.normal(0.0, 1.0, nx)

    X = np.linalg.solve(H, W)

    plt.plot(xs, X)
    plt.show()


if SAMPLE_2D:

    a = 1
    l = 50
    
    xs = np.linspace(1, 100, 100)
    nx = len(xs)
    h = xs[1] - xs[0]

    # Form finite difference operator
    A = sparse.lil_matrix((nx, nx), dtype=np.float64)
    A.setdiag(-2)
    A.setdiag(1, k=1)
    A.setdiag(1, k=-1)
    A[0,-1] = 1
    A[-1,0] = 1

    A /= (h**2)
    A *= l**2

    I = sparse.eye(nx**2)
    J = sparse.eye(nx)

    H = I - sparse.kron(A, J) - sparse.kron(J, A)

    W = np.sqrt(a * l**2 / h**2) * np.random.normal(0.0, 1.0, nx**2)

    X = sparse.linalg.spsolve(H, W)
    
    plt.pcolormesh(np.reshape(X, (nx, nx)))
    plt.show()


if SAMPLE_2D_ANISOTROPIC:

    a = 1
    lx = 50
    ly = 20
    
    xs = np.linspace(1, 100, 100)
    nx = len(xs)
    h = xs[1] - xs[0]

    A = sparse.lil_matrix((nx, nx), dtype=np.float64)
    A.setdiag(-2)
    A.setdiag(1, k=1)
    A.setdiag(1, k=-1)
    A[0, -1] = 1
    A[-1, 0] = 1

    A /= (h**2)

    I = sparse.eye(nx**2)
    J = sparse.eye(nx)

    H = I - lx**2 * sparse.kron(A, J) - ly**2 * sparse.kron(J, A)

    W = np.sqrt(a * lx * ly / h**2) * np.random.normal(0.0, 1.0, nx ** 2)

    X = sparse.linalg.spsolve(H, W)
    
    plt.pcolormesh(np.reshape(X, (nx, nx)))
    plt.show()


# TODO: compare to the ACF
# TODO: figure out how to do this the way in Chada (2018)?

if SAMPLE_FEM:

    # Define number of spatial dimensions of problem and smoothness parameter
    d = 2
    eta = 2 - d/2.0

    # Define lengthscale and standard deviation
    l = 0.3
    sigma = 1.0
    alpha = sigma**2 * (2**d * np.pi**(d/2) * gamma(eta + d/2)) / gamma(eta)

    mesh = MeshTri.init_circle(5)
    basis = Basis(mesh, ElementTriP1())

    npoints = mesh.nvertices
    elements = mesh.t.T
    points = mesh.p.T
    boundary_nodes = set(mesh.boundary_nodes())

    # Define gradients of basis functions in transformed simplex
    gradients = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])

    M = sparse.lil_matrix((npoints, npoints))
    S = sparse.lil_matrix((npoints, npoints))
    N = sparse.lil_matrix((npoints, npoints))

    for element in elements:
        
        # Extract the global indices of the points on the current simplex
        nodes = points[element, :]

        inds = deque([0, 1, 2])

        for _ in range(3):

            # Extract the global index of the current point
            pi = element[inds[0]]
            
            # Generate transformation matrix
            T = np.array([nodes[inds[1]] - nodes[inds[0]],
                          nodes[inds[2]] - nodes[inds[0]]]).T
        
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
                    N[pi, pj] += np.linalg.norm(points[pi] - points[pj]) / 6

            # Rotate the simplex
            inds.rotate(-1)

    lam = 1.42 * l * 1e6

    # TODO: can I avoid the toarray?
    H = (M + l**2 * S + (l**2 / lam) * N)
    G = alpha * l ** 2 * M
    L = np.linalg.cholesky(G.toarray())

    W = np.random.normal(loc=0.0, scale=1.0, size=npoints)
    plot(mesh, sparse.linalg.spsolve(H, L.T @ W), colorbar=True)
    plt.show()

    nruns = 1000

    XS = np.zeros((npoints, nruns))

    for i in range(nruns):

        W = np.random.normal(loc=0.0, scale=1.0, size=npoints)
        XS[:,i] = sparse.linalg.spsolve(H, L.T @ W)

        if (i+1) % 10 == 0:
            print(f"{i+1} runs complete.")

    plot(mesh, np.std(XS, axis=1), colorbar=True, vmin=0, vmax=2)
    plt.show()

    # spatial.delaunay_plot_2d(mesh)
    # plt.show()

    pass