import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats


"""Discretised Gaussian process with exponential-squared 
covariance function."""
class GaussianProcess():
    
    def __init__(self, mu, std, l, xs):
        
        self.xs = xs
        self.nx = len(xs)

        self.std = std 
        self.l = l

        self.mu = np.array([mu] * self.nx)
        self._generate_cov()

    def _generate_cov(self):

        x_dists = self.xs[:, np.newaxis] - self.xs.T 
        cor = np.exp(-0.5 * (x_dists / self.l) ** 2)
        self.cov = self.std ** 2 * cor + 1e-8 * np.eye(self.nx) 
        

"""Discretised Gaussian random field with exponential-squared 
covariance function."""
class GaussianRF():

    def __init__(self, mu, std, lx, lz, cells):
        
        self.cells = cells
        self.cell_xs = np.array([c.centre[0] for c in cells])
        self.cell_zs = np.array([c.centre[-1] for c in cells])
        self.n_cells = len(cells)

        self.std = std 
        self.lx = lx
        self.lz = lz

        self.mu = np.array([mu] * self.n_cells)
        self._generate_cov()

    def _generate_cov(self):

        x_dists = self.cell_xs[:, np.newaxis] - self.cell_xs.T
        z_dists = self.cell_zs[:, np.newaxis] - self.cell_zs.T

        cor = np.exp(-0.5 * (x_dists / self.lx) ** 2 + \
                     -0.5 * (z_dists / self.lz) ** 2)
        self.cov = self.std ** 2 * cor + 1e-8 * np.eye(self.n_cells)


class SlicePrior():

    def __init__(self, mesh, gp_depth_clay, 
                 rf_perm_shal, rf_perm_clay, rf_perm_deep,
                 mass_rate_bounds, level_width):

        self.mesh = mesh

        self.gp_depth_clay = gp_depth_clay
        self.rf_perm_shal = rf_perm_shal
        self.rf_perm_clay = rf_perm_clay
        self.rf_perm_deep = rf_perm_deep

        self._generate_mu()
        self._generate_cov()

        self.chol_inv = np.linalg.cholesky(self.cov)
        self.chol = np.linalg.inv(self.chol_inv)

        self.mass_rate_bounds = mass_rate_bounds
        self.level_width = level_width

        self.is_depth_clay = np.array(range(self.gp_depth_clay.nx))
        self.is_perm_shal = np.array(range(self.is_depth_clay[-1] + 1,
                                           self.is_depth_clay[-1] + 1 + \
                                            self.rf_perm_shal.n_cells))
        self.is_perm_clay = np.array(range(self.is_perm_shal[-1] + 1,
                                           self.is_perm_shal[-1] + 1 + \
                                            self.rf_perm_clay.n_cells))
        self.is_perm_deep = np.array(range(self.is_perm_clay[-1] + 1,
                                           self.is_perm_clay[-1] + 1 + \
                                            self.rf_perm_deep.n_cells))

    def _generate_mu(self):
        self.mu = np.hstack((self.gp_depth_clay.mu, 
                             self.rf_perm_shal.mu,
                             self.rf_perm_clay.mu, 
                             self.rf_perm_deep.mu, 
                             np.array([0.0])))

    def _generate_cov(self):
        self.cov = sparse.block_diag((self.gp_depth_clay.cov,
                                      self.rf_perm_shal.cov,
                                      self.rf_perm_clay.cov,
                                      self.rf_perm_deep.cov,
                                      [1.0])).toarray()

    def _apply_level_sets(self, perms):

        min_level = np.floor(np.min(perms))
        max_level = np.ceil(np.max(perms)) + 1e-8

        levels = np.arange(min_level, max_level, self.level_width)
        return np.array([levels[np.abs(levels - p).argmin()] for p in perms])

    def _transform_mass_rate(self, mass_rate):
        return self.mass_rate_bounds[0] + \
            np.ptp(self.mass_rate_bounds) * stats.norm.cdf(mass_rate)

    def _transform_perms(self, perms):

        perms = np.array(perms)

        clay_boundary = perms[self.is_depth_clay]
        perm_shal = perms[self.is_perm_shal]
        perm_clay = perms[self.is_perm_clay]
        perm_deep = perms[self.is_perm_deep]

        perms = np.copy(perm_shal)

        for i in range(self.rf_perm_deep.n_cells):
            
            cell = self.rf_perm_deep.cells[i]
            cx, cz = cell.centre[0], cell.centre[-1]
            x_ind = np.abs(self.gp_depth_clay.xs - cx).argmin()

            if clay_boundary[x_ind] < cz:
                perms = np.append(perms, perm_clay[i])
            else: 
                perms = np.append(perms, perm_deep[i])

        perms = self._apply_level_sets(perms)
        return 10 ** perms

    def sample(self, n=1):
        ws = np.random.normal(size=(len(self.mu), n))
        return self.mu[:, np.newaxis] + self.chol_inv @ ws

    def transform(self, thetas):
        *perms, mass_rate = np.squeeze(thetas)
        perms = self._transform_perms(perms)
        mass_rate = self._transform_mass_rate(mass_rate)
        return perms, mass_rate


class Likelihood():

    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov
        self.chol_inv = np.linalg.cholesky(self.cov)

    def sample(self, n):
        ws = np.random.normal(size=(len(self.mu), n))
        return self.mu[:, np.newaxis] + self.chol_inv @ ws