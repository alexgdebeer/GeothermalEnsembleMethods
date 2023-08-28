import itertools as it
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from scipy import sparse
from scipy import spatial

np.random.seed(0)

SAMPLE_1D = True
SAMPLE_2D = True
SAMPLE_2D_ANISOTROPIC = True
SAMPLE_FEM = False

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


# TODO: compute redo manually and compute integrals numerically

if SAMPLE_FEM:

    alpha = 1.0
    l = 0.5

    xs = np.linspace(0, 10, 11)
    ys = np.linspace(0, 10, 11)

    mesh = spatial.Delaunay([(x, y) for x in xs for y in ys])

    # Row and column in the matrix for each basis function
    A = sparse.lil_matrix((mesh.npoints, mesh.npoints))

    # Iterate through the simplices
    for i, simplex in enumerate(mesh.simplices):
        
        points = mesh.points[simplex, :]

        # Iterate through each pair of coordinates in the simplex
        for j in range(3):

            # Extract the index of the point under consideration
            pj = simplex[j]

            # Generate the transformation matrix
            T = np.delete(points, j, axis=0).T # TODO: check by hand that this is actually working
            dT = np.abs(np.linalg.det(T))

            for k in range(3): 
                
                pk = simplex[k]

                integral = 1/12 if j == k else 1/24
                A[pj, pk] += dT * integral


    # Each row of the matrix corresponds to a node 
    # For each node, locate its neighbours -- these will be the non-zero elements of the matrix
    # Extract the coordinates of the simplex each pair of nodes is part of
    # Generate mapping to standard triangle
    # Evaluate integral of basis functions corresponding to the two points over the standardised triangle
    # Transform to get original triangle

    spatial.delaunay_plot_2d(mesh)
    plt.show()

    pass