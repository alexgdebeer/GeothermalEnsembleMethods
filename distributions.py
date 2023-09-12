import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats


"""1D Gaussian distribution with exponential-squared 
covariance function."""
class Gaussian1D():
    

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
        

"""2D Gaussian distribution with exponential-squared 
covariance function."""
class Gaussian2D():


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


class SlicePrior():


    def __init__(self, mesh, d_depth_clay, 
                 d_perm_shal, d_perm_clay, d_perm_deep,
                 mass_rate_bounds, level_width):

        self.mesh = mesh

        self.d_depth_clay = d_depth_clay
        self.d_perm_shal = d_perm_shal
        self.d_perm_clay = d_perm_clay
        self.d_perm_deep = d_perm_deep

        self.generate_mu()
        self.generate_cov()

        self.chol_inv = np.linalg.cholesky(self.cov)
        self.chol = np.linalg.inv(self.chol_inv)

        self.mass_rate_bounds = mass_rate_bounds
        self.level_width = level_width

        self.is_depth_clay = np.array(range(self.d_depth_clay.nx))
        self.is_perm_shal = np.array(range(self.is_depth_clay[-1] + 1,
                                           self.is_depth_clay[-1] + 1 + \
                                            self.d_perm_shal.n_cells))
        self.is_perm_clay = np.array(range(self.is_perm_shal[-1] + 1,
                                           self.is_perm_shal[-1] + 1 + \
                                            self.d_perm_clay.n_cells))
        self.is_perm_deep = np.array(range(self.is_perm_clay[-1] + 1,
                                           self.is_perm_clay[-1] + 1 + \
                                            self.d_perm_deep.n_cells))


    def generate_mu(self):
        self.mu = np.hstack((self.d_depth_clay.mu, 
                             self.d_perm_shal.mu,
                             self.d_perm_clay.mu, 
                             self.d_perm_deep.mu, 
                             np.array([0.0])))


    def generate_cov(self):
        self.cov = sparse.block_diag((self.d_depth_clay.cov,
                                      self.d_perm_shal.cov,
                                      self.d_perm_clay.cov,
                                      self.d_perm_deep.cov,
                                      [1.0])).toarray()


    def apply_level_sets(self, perms):

        min_level = np.floor(np.min(perms))
        max_level = np.ceil(np.max(perms)) + 1e-8

        levels = np.arange(min_level, max_level, self.level_width)
        return np.array([levels[np.abs(levels - p).argmin()] for p in perms])


    def transform_mass_rate(self, mass_rate):
        return self.mass_rate_bounds[0] + \
            np.ptp(self.mass_rate_bounds) * stats.norm.cdf(mass_rate)


    def transform_perms(self, perms):

        perms = np.array(perms)

        clay_boundary = perms[self.is_depth_clay]
        perm_shal = perms[self.is_perm_shal]
        perm_clay = perms[self.is_perm_clay]
        perm_deep = perms[self.is_perm_deep]

        perms = np.copy(perm_shal)

        for i in range(self.d_perm_deep.n_cells):
            
            cell = self.d_perm_deep.cells[i]
            cx, cz = cell.centre[0], cell.centre[-1]
            x_ind = np.abs(self.d_depth_clay.xs - cx).argmin()

            if clay_boundary[x_ind] < cz:
                perms = np.append(perms, perm_clay[i])
            else: 
                perms = np.append(perms, perm_deep[i])

        return perms


    def sample(self, n=1):
        ws = np.random.normal(size=(len(self.mu), n))
        return self.mu[:, np.newaxis] + self.chol_inv @ ws


    def transform(self, thetas):
        *perms, mass_rate = np.squeeze(thetas)
        perms = self.transform_perms(perms)
        mass_rate = self.transform_mass_rate(mass_rate)
        perms = self.apply_level_sets(perms)
        perms =  10 ** perms
        return perms, mass_rate


class Likelihood():


    def __init__(self, mu, cov):
        
        self.mu = mu

        self.cov = cov
        self.cov_inv = np.linalg.inv(self.cov)
        self.cov_inv_sq = np.sqrt(self.cov_inv)
        
        self.chol_inv = np.linalg.cholesky(self.cov)


    def sample(self, n):
        ws = np.random.normal(size=(len(self.mu), n))
        return self.mu[:, np.newaxis] + self.chol_inv @ ws