from copy import copy
import h5py
import itertools as it
import numpy as np

import geo_models as gm
import utils as utils

class EnsembleProblem():

    def __init__(self, f, g, prior, likelihood, Nf, Ne):
        
        self.pri = prior 
        self.lik = likelihood 

        self.f = f
        self.g = g 
        self.y = self.lik.mu

        self.Nt = len(self.pri.mu)
        self.Ng = len(self.lik.mu)
        self.Ny = len(self.lik.mu)
        self.Nf = Nf
        self.Ne = Ne

        self.en_inds = list(range(self.Ne))

        self.ts = []
        self.fs = []
        self.gs = []

    """Samples an initial ensemble from the prior."""
    def _generate_initial_ensemble(self):
        self.ts.append(self.pri.sample(self.Ne))
        
    """Runs the ensemble. Failed simulations are removed from the 
    list of ensemble indices."""
    def _run_ensemble(self):

        fs_i = np.zeros((self.Nf, self.Ne))
        gs_i = np.zeros((self.Ng, self.Ne))

        for (i, t) in enumerate(self.ts[-1].T):
            if i in self.en_inds:
                if type(f_t := self.f(t)) == gm.ExitFlag:
                    self.en_inds.remove(i)
                else:
                    fs_i[:, i] = f_t
                    gs_i[:, i] = self.g(f_t)

        self.fs.append(fs_i)
        self.gs.append(gs_i)
    
    """Returns a matrix of scaled deviations of the individual ensemble 
    memebers from the mean ensemble member."""
    def _compute_dts(self, ts):
        dts_unscaled = ts - np.mean(ts, axis=1)[:, np.newaxis]
        return dts_unscaled / np.sqrt(len(self.en_inds) - 1)
    
    """Returns a matrix of scaled deviations of the individual ensemble
    predictions from the mean prediction of the ensemble."""
    def _compute_dgs(self, gs):
        dgs_unscaled = gs - np.mean(gs, axis=1)[:, np.newaxis]
        return dgs_unscaled / np.sqrt(len(self.en_inds) - 1)

    """Returns the mean and variance of the data misfit term for each 
    ensemble member."""
    def _compute_S(self):

        diffs = self.y[:, None] - self.gs[-1][:, self.en_inds]
        Ss = 0.5 * np.sum((self.lik.cov_inv_sq @ diffs) ** 2, axis=1)

        return np.mean(Ss), np.var(Ss)

    """Returns the ensemble estimate of the model Jacobian."""
    def _compute_jac(self, dts, dgs):
        return dgs @ np.linalg.pinv(dts)

    """Saves results of the inversion to an h5 file."""
    def save_results(self, fname):
        with h5py.File(f"{fname}.h5", "w") as f:
            for k, v in self.results.items():
                f.create_dataset(k, data=v)


class ESMDAProblem(EnsembleProblem):

    def __init__(self, *args, loc_type=None, loc_mat=None):
        
        super().__init__(*args)

        self.loc_type = loc_type
        self.loc_mat = loc_mat
        self.alphas = []
        self.t = 0.0

        self.results = {
            "ts": self.ts, 
            "fs": self.fs, 
            "gs": self.gs, 
            "alphas": self.alphas, 
            "inds": self.en_inds
        }

    """Calculates ensemble (cross-)covariance matrices."""
    def _compute_covs(self, dts, dgs):

        if self.loc_type == "linearised":
            cov_tt = self.loc_mat * (dts @ dts.T)
            jac = self._compute_jac(dts, dgs)
            cov_tg = cov_tt @ jac.T
            cov_gg = jac @ cov_tt @ jac.T
            return cov_tg, cov_gg
        
        cov_tg = dts @ dgs.T
        cov_gg = dgs @ dgs.T
        return cov_tg, cov_gg

    def _compute_gain(self, dts, dgs):

        cov_tg, cov_gg = self._compute_covs(dts, dgs)
        K = cov_tg @ np.linalg.inv(cov_gg + self.alphas[-1] * self.lik.cov) # TODO: TSVD

        return K

    """Carries out the bootstrapping-based localisation procedure described 
    by Zhang and Oliver (2010)."""
    def _localise_bootstrap(self, K, Nb=100, sigalpha_sq=0.6**2):

        Ks = np.zeros((self.Nt, self.Ng, Nb))
        loc_mat = np.zeros((self.Nt, self.Ng))

        for k in range(Nb):

            inds_r = np.random.choice(self.en_inds, size=len(self.en_inds))
            dts = self._compute_dts(self.ts[-1][:,inds_r])
            dgs = self._compute_dgs(self.gs[-1][:,inds_r])

            Ks[:,:,k] = self._compute_gain(dts, dgs)
            
        for (i, j) in it.product(range(self.Nt), range(self.Ng)):
            r_sq = np.mean((Ks[i, j, :] - K[i, j]) ** 2) / K[i, j] ** 2
            loc_mat[i, j] = 1.0 / (1.0 + r_sq * (1 + 1 / sigalpha_sq))

        return K * loc_mat

    def _update_ensemble(self):

        # TODO: clean up
        ys_p = np.random.multivariate_normal(
            self.y, 
            self.alphas[-1] * self.lik.cov, 
            size=(self.Ne, )
        ).T
        
        dts = self._compute_dts(self.ts[-1][:,self.en_inds])
        dgs = self._compute_dgs(self.gs[-1][:,self.en_inds])
        K = self._compute_gain(dts, dgs)

        if self.loc_type == "bootstrap":
            K = self._localise_bootstrap(K)

        self.ts.append(self.ts[-1] + K @ (ys_p - self.gs[-1]))

    def _compute_alpha(self):

        mu_S, var_S = self._compute_S()
        
        q1 = self.Ny / (2 * mu_S)
        q2 = np.sqrt(self.Ny / (2 * var_S))
        alpha_inv = min(max(q1, q2), 1.0 - self.t)

        self.t += alpha_inv
        self.alphas.append(alpha_inv ** -1.0)

        utils.info(f"alpha: {self.alphas[-1]:.4f} | t: {self.t:.4f}")

    def _is_converged(self):
        return np.abs(self.t-1.0) < 1e-8

    def run(self):

        self._generate_initial_ensemble()
        self._run_ensemble()

        while True:

            utils.info(f"Beginning iteration {len(self.ts)}...")
            self._compute_alpha()
            self._update_ensemble()
            self._run_ensemble()

            if self._is_converged():
                return


class EnRMLProblem(EnsembleProblem):

    def __init__(self, *args, 
                 i_max=16, gamma=10, lambda_min=0.01, max_cuts=5, 
                 ds_min = 0.01, dt_min = 0.25):
        
        super().__init__(*args)

        self.ss = []
        self.lambdas = []

        self.i_max = i_max 
        self.gamma = gamma 
        self.lambda_min = lambda_min 
        
        self.n_cuts = 0
        self.max_cuts = max_cuts 

        self.ds_min = ds_min 
        self.dt_min = dt_min

        self.converged = False

        self.results = {
            "ts": self.ts, 
            "fs": self.fs, 
            "gs": self.gs, 
            "ss": self.ss, 
            "inds": self.en_inds
        }
    
    """Computes quantities from the prior ensemble that are used 
    at each iteration of the algorithm."""
    def _compute_prior_quantities(self):

        self.cov_sc_sqi = np.diag((1.0 / np.diag(self.pri.cov)) ** 0.25)

        mu_S = self._compute_S()[0]
        lambda_0 = 10 ** np.floor(np.log10(mu_S / (2 * self.Ny)))
        utils.info(f"Initial value of lambda: {lambda_0}.")

        self.ss.append(mu_S)
        self.lambdas.append(lambda_0)

        dts_pri = self._compute_dts(self.ts[0][:,self.en_inds])
        self.U_t_pri, self.S_t_pri, _ = tsvd(self.cov_sc_sqi @ dts_pri)

    def _compute_gain(self, dts, U_g, S_g, V_g):
        psi = np.diag(1.0 / (self.lambdas[-1] + 1.0 + S_g ** 2))
        K = dts @ V_g @ np.diag(S_g) @ psi @ U_g.T @ self.lik.cov_inv_sq
        return K

    """Computes the maximum change (in prior standard deviations) of any 
    component of any ensemble member."""
    def _compute_dt_max(self):
        diffs = self.ts[-1][:,self.en_inds] - self.ts[-2][:,self.en_inds]
        return np.abs(np.diag(np.diag(self.pri.cov) ** -0.5) @ diffs).max()

    """Removes the data corresponding to a failed step."""
    def _revert_step(self):

        self.ts.pop()
        self.gs.pop()
        self.fs.pop()
        self.ss.pop()

        self.lambdas[-1] *= self.gamma
        
        utils.info(f"Step rejected. Lambda modified to {self.lambdas[-1]}.")

    """Returns whether or not a step has been accepted."""
    def _step_accepted(self):
        return self.ss[-1] <= self.ss[-2]

    """TODO: write."""
    def _check_step(self):

        self.ss.append(self._compute_S()[0])
        
        if self._step_accepted():    

            self.n_cuts = 0
            if self._is_converged():
                utils.info("Terminating (convergence criteria met).")
                self.converged = True
                return
            
            self.lambdas.append(max(self.lambdas[-1]/self.gamma, self.lambda_min))
            utils.info(f"Lambda has been modified to {self.lambdas[-1]}.")
            
        else:

            self._revert_step()
            self.n_cuts += 1
            if self.n_cuts == self.max_cuts:
                utils.info("Terminating (max consecutive cuts).")
                self.converged = True
                return

    def _is_converged(self):

        ds = 1.0 - (self.ss[-1]/self.ss[-2])
        dt_max = self._compute_dt_max()

        utils.info(f"ds: {ds:.2%} | dt_max: {dt_max:.2f}")
        return (ds <= self.ds_min) and (dt_max <= self.dt_min)

    """Carries out the bootstrapping-based localisation procedure described 
    by Zhang and Oliver (2010)."""
    def _localise_bootstrap(self, K, Nb=100, sigalpha_sq=0.6**2):

        Ks = np.zeros((self.Nt, self.Ng, Nb))
        loc_mat = np.zeros((self.Nt, self.Ng))

        for k in range(Nb):

            inds_r = np.random.choice(self.en_inds, size=len(self.en_inds))
            dts = self._compute_dts(self.ts[-1][:,inds_r])
            dgs = self._compute_dgs(self.gs[-1][:,inds_r])

            Ug_r, Sg_r, Vg_r = tsvd(self.lik.cov_inv_sq @ dgs)
            Ks[:,:,k] = self._compute_gain(dts, Ug_r, Sg_r, Vg_r)
            
        for (i, j) in it.product(range(self.Nt), range(self.Ng)):
            r_sq = np.mean((Ks[i, j, :] - K[i, j]) ** 2) / K[i, j] ** 2
            loc_mat[i, j] = 1.0 / (1.0 + r_sq * (1 + 1 / sigalpha_sq))

        return K * loc_mat

    """Carries out the adaptive localisation procedure described by 
    Luo and Bhakta (2020)."""
    def _localise_shuffle(self, K):

        def _compute_cor(dts, dgs):
            
            t_sds_inv = np.diag(np.diag(dts @ dts.T) ** -0.5)
            g_sds_inv = np.diag(np.diag(dgs @ dgs.T) ** -0.5)

            cov = dts @ dgs.T
            cor = t_sds_inv @ cov @ g_sds_inv
            return cor
        
        def _gaspari_cohn(z):
            if 0 <= z <= 1:
                return -(1.4)*z**5 + (1/2)*z**4 + (5/8)*z**3 - (5/3)*z**2 + 1
            elif 1 < z <= 2:
                return (1/12)*z**5 - (1/2)*z**4 + (5/8)*z**3 + \
                       (5/3)*z**2 - 5*z + 4 - (2/3)*z**-1
            return 0.0

        loc_mat = np.zeros((self.Nt, self.Ny))
        cor_mats = np.zeros((self.Nt, self.Ny, len(self.en_inds)-1))

        # Just do things once (on the first iteration)
        # Repeatedly reshuffle / cycle and compute the gain.
        # Compute localisation matrix

        dts = self._compute_dts(self.ts[0][:,self.en_inds])
        dgs = self._compute_dgs(self.gs[0][:,self.en_inds])
        cor = _compute_cor(dts, dgs) 

        for i in range(len(self.en_inds)-1):
            dgs_s = np.roll(dgs, shift=i+1, axis=1)
            cor_mats[:,:,i] = _compute_cor(dts, dgs_s)
        
        print(cor_mats)

        error_sd = np.median(np.abs(cor_mats)) / 0.6745
        thresh = np.sqrt(2.0 * np.log(self.Nt * self.Ny)) * error_sd
        # print(thresh)

        for (i, j) in it.product(range(self.Nt), range(self.Ny)):
            loc_mat[i, j] = _gaspari_cohn((1-np.abs(cor[i, j])) / (1-thresh))

        print(loc_mat)

        return K * loc_mat

    """Carries out a single update step."""
    def _update_ensemble(self):
    
        dts = self._compute_dts(self.ts[-1][:,self.en_inds])
        dgs = self._compute_dgs(self.gs[-1][:,self.en_inds])

        U_g, S_g, V_g = tsvd(self.lik.cov_inv_sq @ dgs)

        # Compute Kalman gain
        K = self._compute_gain(dts, U_g, S_g, V_g)
    
        # TODO: make this into an option
        K = self._localise_shuffle(K)

        # Compute changes to ensemble members based on deviations 
        # from prior mean
        delta_pri = dts @ V_g @ \
            np.diag(1 / (self.lambdas[-1] + 1.0 + S_g ** 2)) @ \
            V_g.T @ dts.T @ self.cov_sc_sqi @ self.U_t_pri @ \
            np.diag(1.0 / self.S_t_pri ** 2) @ self.U_t_pri.T @ \
            self.cov_sc_sqi @ (self.ts[-1] - self.ts[0])

        # Compute changes to ensemble members based on misfit
        delta_lik = K @ (self.gs[-1] - self.ys_p) 

        # Update ensemble and calculate mean misfit term
        self.ts.append(self.ts[-1] - delta_pri - delta_lik)

    def run(self):

        self._generate_initial_ensemble()
        self._run_ensemble()
        self._compute_prior_quantities()
        self.ys_p = self.lik.sample(self.Ne)

        i = 0

        while i < self.i_max:

            i += 1
            utils.info(f"Beginning iteration {i}...")

            self._update_ensemble()
            self._run_ensemble()
            self._check_step()

            if self.converged:
                return

        utils.info("Terminating: maximum iterations reached.")
        return


def tsvd(A, energy=0.99):

    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    S_cum = np.cumsum(S)
    for (i, c) in enumerate(S_cum):
        if c / S_cum[-1] >= energy:
            return U[:, :i], S[:i], Vt.T[:, :i]
        
    raise Exception("Error in TSVD function.")