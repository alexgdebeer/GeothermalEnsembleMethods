import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from scipy.special import gamma

# Gradients of basis functions
GRAD_2D = np.array([[-1.0, 1.0, 0.0], 
                    [-1.0, 0.0, 1.0]])

GRAD_3D = np.array([[-1.0, 1.0, 0.0, 0.0], 
                    [-1.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 1.0]])


class MaternField2D():

    def __init__(self):
        self.dim = 2
        self.nu = 2 - self.dim / 2

    def build_fem_matrices(self, points, elements, boundary_facets):
        """
        Builds the FEM matrices required for generating Matern fields in two
        dimensions.

        Parameters:
        ----------------
        points: np.ndarray 
            Array of point locations. The ith row contains the coordinates of 
            point i.
        elements: np.ndarray 
            Array of indices of points in each element. The ith row contains 
            the indices of the points of element i.
        boundary_facets: np.ndarray
            Array of indices of points on each boundary facet. The ith row 
            contains the indices of the points of facet i.

        Returns:
        ----------------
        M: scipy.sparse.lil_matrix
            Mass matrix.
        Kx: scipy.sparse.lil_matrix
            Contributions to stiffness matrix in the x direction.
        Ky: scipy.sparse.lil_matrix
            Contributions to stiffness matrix in the y direction.
        N: scipy.sparse.lil_matrix
            Inner products of each pair of basis functions over the mesh 
            boundary (used when implementing Robin boundary conditions).
        """
        
        n_points = len(points)

        M = sparse.lil_matrix((n_points, n_points))
        Kx = sparse.lil_matrix((n_points, n_points))
        Ky = sparse.lil_matrix((n_points, n_points))
        N = sparse.lil_matrix((n_points, n_points))

        for e in elements:        
            
            for i in range(3):
                
                T = np.array([points[e[(i+1)%3]] - points[e[i]],
                              points[e[(i+2)%3]] - points[e[i]]]).T
                detT = np.abs(np.linalg.det(T))
                invT = np.linalg.inv(T)

                for j in range(3):
                    
                    if i == j:
                        M[e[i], e[j]] += detT * 1/12
                    else: 
                        M[e[i], e[j]] += detT * 1/24

                    kl = 1/2 * detT * GRAD_2D[:, 0].T @ invT 
                    kr = invT.T @ GRAD_2D[:, (j-i)%3]

                    Kx[e[i], e[j]] += kl.flat[0] * kr.flat[0]
                    Ky[e[i], e[j]] += kl.flat[1] * kr.flat[1]

        for (fi, fj) in boundary_facets:
            n = np.linalg.norm(points[fi] - points[fj])
            N[fi, fi] += n * 1/3
            N[fj, fj] += n * 1/3
            N[fi, fj] += n * 1/6
            N[fj, fi] += n * 1/6

        self.M = M
        self.L = np.linalg.cholesky(M.toarray())
        self.Kx = Kx 
        self.Ky = Ky
        self.N = N

    def generate_field(self, W, sigma, lx, ly, bcs="robin", lam=None):
        """Given a set of white noise and a set of hyperparameters,
        generates the corresponding Matern field."""

        alpha = sigma**2 * (2**self.dim * np.pi**(self.dim/2) * \
                            gamma(self.nu + self.dim/2)) / gamma(self.nu)
        
        K = lx**2 * self.Kx + ly**2 * self.Ky 

        if bcs == "robin":
            if lam is None:
                lam = 1.42 * np.sqrt(lx * ly)
            self.H = self.M + K + (lx * ly / lam) * self.N 

        elif bcs == "neumann":
            self.H = self.M + K 
        
        # # TEMP: calculate empirical standard deviations
        # inv_H = np.linalg.inv(self.H.toarray())
        # cov = alpha * lx * ly * inv_H @ self.M.toarray() @ inv_H.T
        # return np.sqrt(np.diag(cov))
        # TODO: correct marginal standard deviations?

        X = linalg.spsolve(self.H, np.sqrt(alpha * lx * ly) * self.L.T @ W)
        return X


class MaternField3D():

    def __init__(self):
        self.dim = 3
        self.nu = 2 - self.dim / 2
        
    def build_fem_matrices(self, points, elements, boundary_facets):
        """Builds the FEM matrices required to generate Matern fields in three 
        dimensions."""

        n_points = len(points)

        M = sparse.lil_matrix((n_points, n_points))
        Kx = sparse.lil_matrix((n_points, n_points))
        Ky = sparse.lil_matrix((n_points, n_points))
        Kz = sparse.lil_matrix((n_points, n_points))
        N = sparse.lil_matrix((n_points, n_points))

        for e in elements:

            for i in range(4):

                T = np.array([points[e[(i+1)%4]] - points[e[i]],
                              points[e[(i+2)%4]] - points[e[i]],
                              points[e[(i+3)%4]] - points[e[i]]]).T

                detT = np.abs(np.linalg.det(T))
                invT = np.linalg.inv(T)

                for j in range(4):
                    
                    if i == j: 
                        M[e[i], e[j]] += detT * 1/60
                    else: 
                        M[e[i], e[j]] += detT * 1/120

                    kl = 1/6 * detT * GRAD_3D[:, 0].T @ invT 
                    kr = invT.T @ GRAD_3D[:, (j-i)%4]

                    Kx[e[i], e[j]] += kl.flat[0] * kr.flat[0]
                    Ky[e[i], e[j]] += kl.flat[1] * kr.flat[1]
                    Kz[e[i], e[j]] += kl.flat[2] * kr.flat[2]
        
        for f in boundary_facets:

            for i in range(3):
                
                # Calculate the determinant of the transformation from the 
                # reference facet to the new facet (no division by 1/2
                # because the reference facet has an area of 1/2)
                detTb = np.linalg.norm(np.cross(points[f[(i+1)%3]] - points[f[i]], 
                                                points[f[(i+2)%3]] - points[f[i]]))

                for j in range(3):
                    if i == j:
                        N[f[i], f[j]] += detTb * 1/12
                    else:
                        N[f[i], f[j]] += detTb * 1/24

        self.M = M
        self.L = np.linalg.cholesky(M.toarray())
        self.Kx = Kx
        self.Ky = Ky 
        self.Kz = Kz 
        self.N = N

    def generate_field(self, W, sigma, lx, ly, lz, bcs="robin", lam=None):
        """Generates a Matern field."""

        alpha = sigma**2 * (2**self.dim * np.pi**(self.dim/2) * \
                            gamma(self.nu + self.dim/2)) / gamma(self.nu)

        # Form complete stiffness matrix
        K = lx**2 * self.Kx + ly**2 * self.Ky + lz**2 * self.Kz
        
        if bcs == "robin":
            if lam is None:
                lam = 1.42 * np.sqrt(lx * ly * lz) # TODO: tune Robin parameter
            self.H = self.M + K + (lx * ly * lz / lam) * self.N 

        elif bcs == "neumann":
            self.H = self.M + K
        
        # # TEMP: calculate empirical standard deviations
        # inv_H = np.linalg.inv(self.H.toarray())
        # cov = alpha * lx * ly * lz * inv_H @ self.M.toarray() @ inv_H.T
        # return np.sqrt(np.diag(cov))

        X = linalg.spsolve(self.H, np.sqrt(alpha * lx * ly * lz) * self.L.T @ W)
        return X
    

def generate_fem_matrices_2D(points, elements, boundary_facets):
    """
    Builds the FEM matrices required for generating Matern fields in two
    dimensions.

    Parameters:
    ----------------
    points: np.ndarray 
        Array of point locations. The ith row contains the coordinates of 
        point i.
    elements: np.ndarray 
        Array of indices of points in each element. The ith row contains the 
        indices of the points of element i.
    boundary_facets: np.ndarray
        Array of indices of points on each facet that makes up the boundary.

    Returns:
    ----------------
    M: scipy.sparse.lil_matrix
        Mass matrix.
    Kx: scipy.sparse.lil_matrix
        Contributions to stiffness matrix in the x direction.
    Ky: scipy.sparse.lil_matrix
        Contributions to stiffness matrix in the y direction.
    N: scipy.sparse.lil_matrix
        Inner products of each pair of basis functions over the mesh boundary
        (used when implementing Robin boundary conditions).
    """
    
    n_points = len(points)

    M = sparse.lil_matrix((n_points, n_points))
    Kx = sparse.lil_matrix((n_points, n_points))
    Ky = sparse.lil_matrix((n_points, n_points))
    N = sparse.lil_matrix((n_points, n_points))

    for e in elements:        
        
        for i in range(3):
            
            T = np.array([points[e[(i+1)%3]] - points[e[i]],
                          points[e[(i+2)%3]] - points[e[i]]]).T
            detT = np.abs(np.linalg.det(T))
            invT = np.linalg.inv(T)

            for j in range(3):
                
                if i == j:
                    M[e[i], e[j]] += detT * 1/12
                else: 
                    M[e[i], e[j]] += detT * 1/24

                kl = 1/2 * detT * GRAD_2D[:, 0].T @ invT 
                kr = invT.T @ GRAD_2D[:, (j-i)%3]

                Kx[e[i], e[j]] += kl.flat[0] * kr.flat[0]
                Ky[e[i], e[j]] += kl.flat[1] * kr.flat[1]

    for (pi, pj) in boundary_facets:
        n = np.linalg.norm(points[pi] - points[pj])
        N[pi, pi] += n * 1/3
        N[pj, pj] += n * 1/3
        N[pi, pj] += n * 1/6
        N[pj, pi] += n * 1/6

    return M, Kx, Ky, N


def generate_fem_matrices_3D(points, elements, boundary_facets):
    """Builds the FEM matrices required to generate Matern fields in three 
    dimensions."""

    n_points = len(points)

    M = sparse.lil_matrix((n_points, n_points))
    Kx = sparse.lil_matrix((n_points, n_points))
    Ky = sparse.lil_matrix((n_points, n_points))
    Kz = sparse.lil_matrix((n_points, n_points))
    N = sparse.lil_matrix((n_points, n_points))

    for e in elements:

        for i in range(4):

            T = np.array([points[e[(i+1)%4]] - points[e[i]],
                          points[e[(i+2)%4]] - points[e[i]],
                          points[e[(i+3)%4]] - points[e[i]]]).T

            detT = np.abs(np.linalg.det(T))
            invT = np.linalg.inv(T)

            for j in range(4):
                
                if i == j: 
                    M[e[i], e[j]] += detT * 1/60
                else: 
                    M[e[i], e[j]] += detT * 1/120

                kl = 1/6 * detT * GRAD_3D[:, 0].T @ invT 
                kr = invT.T @ GRAD_3D[:, (j-i)%4]

                Kx[e[i], e[j]] += kl.flat[0] * kr.flat[0]
                Ky[e[i], e[j]] += kl.flat[1] * kr.flat[1]
                Kz[e[i], e[j]] += kl.flat[2] * kr.flat[2]
    
    for f in boundary_facets:

        for i in range(3):
            
            # Calculate the determinant of the transformation from the 
            # reference facet to the new facet (no division by 1/2
            # because the reference facet has an area of 1/2)
            detTb = np.linalg.norm(np.cross(points[f[(i+1)%3]] - points[f[i]], 
                                            points[f[(i+2)%3]] - points[f[i]]))

            for j in range(3):
                if i == j:
                    N[f[i], f[j]] += detTb * 1/12
                else:
                    N[f[i], f[j]] += detTb * 1/24

    return M, Kx, Ky, Kz, N


def generate_mesh_mapping_3D(mesh, geo):
    """Generates an operator that maps the result from the FEM mesh back to 
    the cells in the model geometry."""

    H = sparse.lil_matrix((geo.num_cells, mesh.n_points))
    elements = mesh.find_containing_cell([c.centre for c in geo.cell])

    for c, e in zip(geo.cell, elements):
        
        ps = mesh.cells_dict[10][e]

        # Map cell centre back to the reference triangle
        T = np.array([mesh.points[ps[1]] - mesh.points[ps[0]],
                      mesh.points[ps[2]] - mesh.points[ps[0]],
                      mesh.points[ps[3]] - mesh.points[ps[0]]]).T
        
        x, y, z = np.linalg.inv(T) @ (c.centre - mesh.points[ps[0]])

        H[c.index, ps[0]] = 1 - x - y - z
        H[c.index, ps[1]] = x
        H[c.index, ps[2]] = y
        H[c.index, ps[3]] = z

    return H