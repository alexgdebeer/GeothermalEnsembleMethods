import numpy as np

class Likelihood():
    """Defines a Gaussian likelihood."""

    def __init__(self, mu, cov):
        
        self.mu = mu

        self.cov = cov
        self.cov_inv = np.linalg.inv(self.cov)
        self.cov_inv_sq = np.sqrt(self.cov_inv)
        
        self.chol_inv = np.linalg.cholesky(self.cov)

    def sample(self, n):
        ws = np.random.normal(size=(len(self.mu), n))
        return self.mu[:, np.newaxis] + self.chol_inv @ ws