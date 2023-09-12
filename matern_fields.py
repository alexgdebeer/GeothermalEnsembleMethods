import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from scipy.special import gamma
import pyvista as pv
import utils

GRAD_2D = np.array([[-1.0, 1.0, 0.0], 
                    [-1.0, 0.0, 1.0]])

GRAD_3D = np.array([[-1.0, 1.0, 0.0, 0.0], 
                    [-1.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 1.0]])

class MaternField2D():

    def __init__(self, geo, mesh):

        self.dim = 2
        self.nu = 2 - self.dim / 2

        self.geo = geo
        self.mesh = mesh

        self.get_mesh_data()
        self.build_fem_matrices()

    def get_mesh_data(self):
        """Extracts information on the points, elements and boundary facets 
        of the mesh."""

        self.mesh["point_inds"] = np.arange(self.mesh.n_points, dtype=np.int64)

        self.points = self.mesh.points[:, :2]
        self.elements = self.mesh.regular_faces

        boundary = self.mesh.extract_feature_edges(boundary_edges=True, 
                                                   feature_edges=False, 
                                                   non_manifold_edges=False, 
                                                   manifold_edges=False)

        boundary_points = boundary.cast_to_pointset()["point_inds"]
        boundary_facets = boundary.lines.reshape(-1, 3)[:, 1:]
        self.boundary_facets = [boundary_points[f] for f in boundary_facets]
        
        self.n_points = self.mesh.n_points
        self.n_elements = self.mesh.n_cells
        self.n_boundary_facets = len(self.boundary_facets)

    def build_fem_matrices(self):
        """Builds the FEM matrices required for generating Matern fields in two
        dimensions."""

        utils.info("Constructing FEM matrices...")

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

        utils.info("FEM matrices constructed.")

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
    
    def plot(self, **kwargs):
        """Generates a 3D visualisation of the mesh using PyVista."""
        p = pv.Plotter()
        p.add_mesh(self.mesh, **kwargs)
        p.show()
    
    def layer_plot(self, values, **kwargs):
        """Generates a visualisation of the mesh using Layermesh."""
        col_values = [values[c.column.index] for c in self.geo.cell]
        self.geo.layer_plot(value=col_values, **kwargs)


class MaternField3D():

    def __init__(self, geo, mesh):

        self.dim = 3
        self.nu = 2 - self.dim / 2

        self.geo = geo
        self.mesh = mesh 

        self.get_mesh_data()
        self.build_fem_matrices()
        self.build_geo_to_mesh_mapping()
        
    def get_mesh_data(self):
        """Extracts information on the points, elements and facets of the 
        mesh."""

        self.mesh["point_inds"] = np.arange(self.mesh.n_points, dtype=np.int64)

        self.points = self.mesh.points 
        self.elements = self.mesh.cells_dict[10]

        boundary = self.mesh.extract_geometry()
        boundary_points = boundary.cast_to_pointset()["point_inds"]
        boundary_facets = boundary.faces.reshape(-1, 4)[:, 1:]
        self.boundary_facets = [boundary_points[f] for f in boundary_facets]

        self.n_points = self.mesh.n_points 
        self.n_elements = self.mesh.n_cells
        self.n_boundary_facets = len(self.boundary_facets)

    def build_fem_matrices(self):
        """Builds the FEM matrices required to generate Matern fields in three 
        dimensions."""

        utils.info("Constructing FEM matrices...")

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

        utils.info("Constructing geo to mesh mapping...")

        B = sparse.lil_matrix((self.geo.num_cells, self.mesh.n_points))
        elements = self.mesh.find_containing_cell([c.centre for c in self.geo.cell])

        for c, e in zip(self.geo.cell, elements):
            
            ps = self.elements[e]

            # Map cell centre back to the reference triangle
            T = np.array([self.points[ps[1]] - self.points[ps[0]],
                          self.points[ps[2]] - self.points[ps[0]],
                          self.points[ps[3]] - self.points[ps[0]]]).T
            
            x, y, z = np.linalg.inv(T) @ (c.centre - self.points[ps[0]])
            B[c.index, ps[0]] = 1 - x - y - z
            B[c.index, ps[1]] = x
            B[c.index, ps[2]] = y
            B[c.index, ps[3]] = z

        self.B = B

        utils.info("Mapping from geo to mesh constructed.")

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

        b = np.sqrt(alpha * lx * ly * lz) * self.L.T @ W
        X = linalg.spsolve(self.H, b)
        return X
    
    def plot(self, **kwargs):
        p = pv.Plotter()
        p.add_mesh(self.mesh, **kwargs)
        p.show()

    def slice_plot(self, values, **kwargs):
        self.geo.slice_plot(value=self.B@values, **kwargs)