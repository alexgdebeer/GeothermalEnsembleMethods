from enum import Enum
import numpy as np
import pyvista as pv
from scipy import sparse
from scipy.sparse import linalg
from scipy.special import gamma

from GeothermalEnsembleMethods import utils

GRAD_2D = np.array([[-1.0, 1.0, 0.0], 
                    [-1.0, 0.0, 1.0]])

GRAD_3D = np.array([[-1.0, 1.0, 0.0, 0.0], 
                    [-1.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 1.0]])

class BC(Enum):
    NEUMANN = 1
    ROBIN = 2

class MaternField2D():

    def __init__(self, mesh):

        self.dim = 2
        self.nu = 2 - self.dim / 2

        self.m = mesh.m
        col_centres = [[*col.centre, 0.0] for col in self.m.column]
        self.fem_mesh = pv.PolyData(col_centres).delaunay_2d()

        self.get_mesh_data()
        self.build_fem_matrices()

    def get_mesh_data(self):
        """Extracts information on the points, elements and boundary facets 
        of the mesh."""

        self.fem_mesh["inds"] = np.arange(self.fem_mesh.n_points, dtype=np.int64)

        self.points = self.fem_mesh.points[:, :2]
        self.elements = self.fem_mesh.regular_faces

        boundary = self.fem_mesh.extract_feature_edges(boundary_edges=True, 
                                                       feature_edges=False, 
                                                       non_manifold_edges=False, 
                                                       manifold_edges=False)

        boundary_points = boundary.cast_to_pointset()["inds"]
        boundary_facets = boundary.lines.reshape(-1, 3)[:, 1:]
        self.boundary_facets = [boundary_points[f] for f in boundary_facets]
        
        self.n_points = self.fem_mesh.n_points
        self.n_elements = self.fem_mesh.n_cells
        self.n_boundary_facets = len(self.boundary_facets)

    def build_fem_matrices(self):
        """Builds the FEM matrices required for generating Matern fields in two
        dimensions."""

        M_i = np.zeros((9 * self.n_elements, ))
        M_j = np.zeros((9 * self.n_elements, ))
        M_v = np.zeros((9 * self.n_elements, ))

        K_i = np.zeros((9 * self.n_elements, ))
        K_j = np.zeros((9 * self.n_elements, ))
        K_v = np.zeros((2, 9 * self.n_elements))

        N_i = np.zeros((4 * self.n_boundary_facets, ))
        N_j = np.zeros((4 * self.n_boundary_facets, ))
        N_v = np.zeros((4 * self.n_boundary_facets, ))

        n = 0
        for e in self.elements:
            
            for i in range(3):
                
                T = np.array([self.points[e[(i+1)%3]] - self.points[e[i]],
                              self.points[e[(i+2)%3]] - self.points[e[i]]]).T
                detT = np.abs(np.linalg.det(T))
                invT = np.linalg.inv(T)

                for j in range(3):

                    M_i[n] = e[i]
                    M_j[n] = e[j]
                    M_v[n] = (detT * 1/12) if i == j else (detT * 1/24)

                    kl = 1/2 * detT * GRAD_2D[:, 0].T @ invT 
                    kr = invT.T @ GRAD_2D[:, (j-i)%3]

                    K_i[n] = e[i]
                    K_j[n] = e[j]
                    K_v[:, n] = kl.flatten() * kr.flatten()

                    n += 1

        n = 0
        for (fi, fj) in self.boundary_facets:
            
            det = np.linalg.norm(self.points[fi] - self.points[fj])

            N_i[n:n+4] = np.array([fi, fj, fi, fj])
            N_j[n:n+4] = np.array([fi, fj, fj, fi])
            N_v[n:n+4] = np.array([det * 1/3, det * 1/3, det * 1/6, det * 1/6])

            n += 4

        shape = (self.n_points, self.n_points)

        self.M = sparse.coo_matrix((M_v, (M_i, M_j)), shape=shape)
        self.Kx = sparse.coo_matrix((K_v[0], (K_i, K_j)), shape=shape)
        self.Ky = sparse.coo_matrix((K_v[1], (K_i, K_j)), shape=shape)
        self.N = sparse.coo_matrix((N_v, (N_i, N_j)), shape=shape)

        self.L = np.linalg.cholesky(self.M.toarray())

    def generate_field(self, W, sigma, lx, ly, bcs=BC.ROBIN, lam=None):
        """Given a set of white noise and a set of hyperparameters,
        generates the corresponding Matern field."""

        alpha = sigma**2 * (2**self.dim * np.pi**(self.dim/2) * \
                            gamma(self.nu + self.dim/2)) / gamma(self.nu)
        
        K = lx**2 * self.Kx + ly**2 * self.Ky 

        if bcs == BC.ROBIN:
            if lam is None:
                lam = 1.42 * np.sqrt(lx * ly)
            A = self.M + K + (lx * ly / lam) * self.N 

        elif bcs == BC.NEUMANN:
            A = self.M + K 
        
        # # TEMP: calculate empirical standard deviations
        # inv_A = np.linalg.inv(A.toarray())
        # cov = alpha * lx * ly * inv_A @ self.M.toarray() @ inv_A.T
        # return np.sqrt(np.diag(cov))

        b = np.sqrt(alpha * lx * ly) * self.L.T @ W
        return linalg.spsolve(A, b)
    
    def plot(self, values, **kwargs):
        """Generates a 3D visualisation of the mesh using PyVista."""
        p = pv.Plotter()
        p.add_mesh(self.fem_mesh, scalars=values, **kwargs)
        p.show()
    
    def layer_plot(self, values, **kwargs):
        """Generates a visualisation of the mesh using Layermesh."""
        col_values = [values[c.column.index] for c in self.m.cell]
        self.m.layer_plot(value=col_values, **kwargs)

class MaternField3D():

    def __init__(self, mesh):

        self.dim = 3
        self.nu = 2 - self.dim / 2

        self.m = mesh.m
        self.fem_mesh = mesh.fem_mesh 

        self.get_mesh_data()
        self.build_fem_matrices()
        self.build_geo_to_mesh_mapping()
        self.build_point_to_cell_mapping()
        
    def get_mesh_data(self):
        """Extracts information on the points, elements and facets of the 
        mesh."""

        self.fem_mesh["inds"] = np.arange(self.fem_mesh.n_points, dtype=np.int64)

        self.points = self.fem_mesh.points 
        self.elements = self.fem_mesh.cells_dict[10]

        boundary = self.fem_mesh.extract_geometry()
        boundary_points = boundary.cast_to_pointset()["inds"]
        boundary_facets = boundary.faces.reshape(-1, 4)[:, 1:]
        self.boundary_facets = [boundary_points[f] for f in boundary_facets]

        self.n_points = self.fem_mesh.n_points 
        self.n_elements = self.fem_mesh.n_cells
        self.n_boundary_facets = len(self.boundary_facets)

    @utils.timer
    def build_fem_matrices(self):
        """Builds the FEM matrices required to generate Matern fields in three 
        dimensions."""

        utils.info(f"Constructing FEM matrices (points: {self.n_points})...")

        M_i = np.zeros((16 * self.n_elements, ))
        M_j = np.zeros((16 * self.n_elements, ))
        M_v = np.zeros((16 * self.n_elements, ))

        K_i = np.zeros((16 * self.n_elements, ))
        K_j = np.zeros((16 * self.n_elements, ))
        K_v = np.zeros((3, 16 * self.n_elements))

        N_i = np.zeros((9 * self.n_boundary_facets, ))
        N_j = np.zeros((9 * self.n_boundary_facets, ))
        N_v = np.zeros((9 * self.n_boundary_facets, ))

        n = 0
        for e in self.elements:

            for i in range(4):

                T = np.array([self.points[e[(i+1)%4]] - self.points[e[i]],
                              self.points[e[(i+2)%4]] - self.points[e[i]],
                              self.points[e[(i+3)%4]] - self.points[e[i]]]).T

                detT = np.abs(np.linalg.det(T))
                invT = np.linalg.inv(T)

                for j in range(4):
                    
                    M_i[n] = e[i]
                    M_j[n] = e[j]
                    M_v[n] = (detT * 1/60) if i == j else (detT * 1/120)

                    kl = 1/6 * detT * GRAD_3D[:, 0].T @ invT 
                    kr = invT.T @ GRAD_3D[:, (j-i)%4]

                    K_i[n] = e[i]
                    K_j[n] = e[j]
                    K_v[:, n] = kl.flatten() * kr.flatten()

                    n += 1
        
        n = 0
        for f in self.boundary_facets:

            for i in range(3):
                
                detTf = np.linalg.norm(np.cross(self.points[f[(i+1)%3]] - self.points[f[i]], 
                                                self.points[f[(i+2)%3]] - self.points[f[i]]))

                for j in range(3):
                    
                    N_i[n] = f[i]
                    N_j[n] = f[j]
                    N_v[n] = (detTf * 1/12) if i == j else (detTf * 1/24)

                    n += 1

        shape = (self.n_points, self.n_points)

        self.M = sparse.coo_matrix((M_v, (M_i, M_j)), shape=shape)
        self.Kx = sparse.coo_matrix((K_v[0], (K_i, K_j)), shape=shape)
        self.Ky = sparse.coo_matrix((K_v[1], (K_i, K_j)), shape=shape)
        self.Kz = sparse.coo_matrix((K_v[2], (K_i, K_j)), shape=shape)
        self.N = sparse.coo_matrix((N_v, (N_i, N_j)), shape=shape)
        self.L = np.linalg.cholesky(self.M.toarray())

        utils.info("FEM matrices constructed.")

    def build_geo_to_mesh_mapping(self):
        """Generates an operator that maps the result from the FEM mesh back to 
        the cells in the model geometry."""

        cell_centres = [c.centre for c in self.m.cell]
        elements = self.fem_mesh.find_containing_cell(cell_centres)

        G_i = np.array([[c.index] * 4 for c in self.m.cell]).flatten()
        G_j = np.array([self.elements[e] for e in elements]).flatten()
        G_v = np.zeros((4 * self.m.num_cells, ))

        n = 0
        for c, e in zip(self.m.cell, elements):
            
            ps = self.elements[e]

            T = np.array([self.points[ps[1]] - self.points[ps[0]],
                          self.points[ps[2]] - self.points[ps[0]],
                          self.points[ps[3]] - self.points[ps[0]]]).T
            
            x, y, z = np.linalg.inv(T) @ (c.centre - self.points[ps[0]])
            G_v[n:n+4] = np.array([1-x-y-z, x, y, z])
            
            n += 4

        shape = (self.m.num_cells, self.n_points)
        self.G = sparse.coo_matrix((G_v, (G_i, G_j)), shape=shape)

    def build_point_to_cell_mapping(self):
        """Generates an operator that maps the value at the points on the mesh 
        to the corresponding values at the cell centres."""

        P_i = np.array([[n] * 4 for n in range(self.n_elements)]).flatten()
        P_j = np.array(self.elements).flatten()
        P_v = np.full((4 * self.n_elements, ), 1/4)

        self.P = sparse.coo_matrix((P_v, (P_i, P_j)), 
                                   shape=(self.n_elements, self.n_points))


    def generate_field(self, W, sigma, lx, ly, lz, bcs=BC.ROBIN, lam=None):
        """Generates a Matern field."""

        alpha = sigma**2 * (2**self.dim * np.pi**(self.dim/2) * \
                            gamma(self.nu + self.dim/2)) / gamma(self.nu)

        K = lx**2 * self.Kx + ly**2 * self.Ky + lz**2 * self.Kz
        
        if bcs == BC.ROBIN:
            if lam is None:
                lam = 1.42 * np.sqrt(lx * ly * lz)
            A = self.M + K + (lx * ly * lz / lam) * self.N 

        elif bcs == BC.NEUMANN:
            A = self.M + K
        
        # # TEMP: calculate empirical standard deviations
        # inv_A = np.linalg.inv(A.toarray())
        # cov = alpha * lx * ly * lz * inv_A @ self.M.toarray() @ inv_A.T
        # return np.sqrt(np.diag(cov))

        b = np.sqrt(alpha * lx * ly * lz) * self.L.T @ W
        return linalg.spsolve(A, b)
    
    def plot_points(self, values, **kwargs):
        p = pv.Plotter()
        p.add_mesh(self.fem_mesh, scalars=values, **kwargs)
        p.show()

    def plot_cells(self, values, **kwargs):
        p = pv.Plotter()
        p.add_mesh(self.fem_mesh, scalars=self.P @ values, **kwargs)
        p.show()

    def plot_slice(self, values, **kwargs):
        self.m.slice_plot(value=self.G @ values, **kwargs)

class Gaussian1D():
    """1D Gaussian distribution with squared-exponential covariance function."""

    def __init__(self, mu, std, l, xs):
        
        self.xs = xs
        self.nx = len(xs)

        self.std = std 
        self.l = l

        self.mu = np.array([mu] * self.nx)
        self.generate_cov()

    def generate_cov(self):

        self.x_dists = self.xs[:, np.newaxis] - self.xs.T 
        self.cor = np.exp(-0.5 * (self.x_dists / self.l) ** 2)
        self.cov = self.std ** 2 * self.cor + 1e-8 * np.eye(self.nx) 
        
class Gaussian2D():
    """2D Gaussian distribution with squared-exponential covariance function."""

    def __init__(self, mu, std, lx, lz, cells):
        
        self.cells = cells
        self.cell_xs = np.array([c.centre[0] for c in cells])
        self.cell_zs = np.array([c.centre[-1] for c in cells])
        self.n_cells = len(cells)

        self.std = std 
        self.lx = lx
        self.lz = lz

        self.mu = np.array([mu] * self.n_cells)
        self.generate_cov()

    def generate_cov(self):

        self.x_dists = self.cell_xs[:, np.newaxis] - self.cell_xs.T
        self.z_dists = self.cell_zs[:, np.newaxis] - self.cell_zs.T

        self.cor = np.exp(-0.5 * (self.x_dists / self.lx) ** 2 + \
                          -0.5 * (self.z_dists / self.lz) ** 2)
        self.cov = self.std ** 2 * self.cor + 1e-8 * np.eye(self.n_cells)