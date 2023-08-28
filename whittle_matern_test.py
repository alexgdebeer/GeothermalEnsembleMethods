import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from scipy import sparse

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


import skfem as fm
from skfem.helpers import dot, grad
from skfem.visuals.matplotlib import plot

# TODO: compute redo manually and compute integrals numerically

if SAMPLE_FEM:

    alpha = 1.0
    l = 0.5

    @fm.BilinearForm
    def m(phi, varphi, _):
        return phi * varphi
    
    @fm.BilinearForm
    def s(phi, varphi, _):
        return dot(grad(phi), grad(varphi))

    @fm.BilinearForm
    def g(phi, varphi, _):
        return phi * varphi

    # Generate mesh
    mesh = fm.MeshTri().refined(5)

    # Define a basis
    Vh = fm.Basis(mesh, fm.ElementTriP1())

    # Assemble matrices
    M = m.assemble(Vh)
    S = s.assemble(Vh)
    G = g.assemble(Vh)

    H = M + l**2 * S 
    G = alpha * l**2 * G

    H_inv = sparse.linalg.inv(H)

    mu = np.zeros((M.shape[0], ))
    cov = H_inv @ G @ H_inv
    cov = cov.toarray()

    # G_inv = sparse.linalg.inv(G)
    # R = linalg.cholesky(G_inv.toarray())

    print(cov.shape)

    ws = np.random.multivariate_normal(mean=mu, cov=cov)
    nx = int(np.sqrt(len(ws)))

    d = np.diag(cov)

    # Generate mesh
    plt.pcolormesh(cov)
    plt.show()

    pass