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

    def _generate_initial_ensemble(self):
        self.ts.append(self.pri.sample(self.Ne))
        
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

    """Returns a matrix of rescaled deviations from the mean (for 
    the ensemble members or the ensemble predictions)."""
    def _compute_deltas(self, q):

        q_filtered = q[:,self.en_inds]
        deltas = q_filtered - np.mean(q_filtered, axis=1)[:, np.newaxis]
        
        return deltas / np.sqrt(len(self.en_inds) - 1)
    
    def _compute_S(self):

        diffs = self.y[:, None] - self.gs[-1][:, self.en_inds]
        Ss = 0.5 * np.sum((self.lik.cov_inv_sq @ diffs) ** 2, axis=1)
        
        return np.mean(Ss), np.var(Ss)

    """Returns the ensemble estimate of the model Jacobian."""
    def _compute_jac(self, delta_t, delta_g):
        return delta_g @ np.linalg.pinv(delta_t)

    def save_results(self, fname):
        with h5py.File(f"{fname}.h5", "w") as f:
            for k, v in self.results.items():
                f.create_dataset(k, data=v)


class ESMDAProblem(EnsembleProblem):

    def __init__(self, *args, loc_mat=None):
        
        super().__init__(*args)

        self.loc_mat = loc_mat
        self._as = []
        self.t = 0.0

        self.results = {
            "ts": self.ts, 
            "fs": self.fs, 
            "gs": self.gs, 
            "alphas": self._as, 
            "inds": self.en_inds
        }

    """Calculates ensemble (cross-)covariance matrices."""
    def _compute_covs(self):

        delta_t = self._compute_deltas(self.ts[-1])
        delta_g = self._compute_deltas(self.gs[-1])

        if not self.loc_mat:
            cov_tg = delta_t @ delta_g.T
            cov_gg = delta_g @ delta_g.T
            return cov_tg, cov_gg

        cov_tt = self.loc_mat * (delta_t @ delta_t.T)
        jac = self._compute_jac(delta_t, delta_g)
        cov_tg = cov_tt @ jac.T
        cov_gg = jac @ cov_tt @ jac.T

        return cov_tg, cov_gg

    def _compute_gain(self):

        cov_tg, cov_gg = self._compute_covs()
        K = cov_tg @ np.linalg.inv(cov_gg + self._as[-1] * self.lik.cov) # TODO: TSVD

        return K
    
    def _update_ensemble(self):

        # TODO: clean up
        ys_p = np.random.multivariate_normal(
            self.y, 
            self._as[-1] * self.lik.cov, 
            size=(self.Ne, )
        ).T
        
        K = self._compute_gain()

        ts_updated = self.ts[-1] + K @ (ys_p - self.gs[-1])
        self.ts.append(ts_updated)
        
        return

    def _compute_alpha(self):

        mu_S, var_S = self._compute_S()
        
        q1 = self.Ny / (2 * mu_S)
        q2 = np.sqrt(self.Ny / (2 * var_S))
        a_inv = min(max(q1, q2), 1.0 - self.t)

        self.t += a_inv
        self._as.append(a_inv ** -1.0)

    def _is_converged(self):
        return np.abs(self.t-1.0) < 1e-8

    def run(self):

        self._generate_initial_ensemble()
        self._run_ensemble()

        while True:

            self._compute_alpha()
            utils.info(f"It: {len(self._as)} | a: {self._as[-1]:.4f} | t: {self.t:.4f}")
            self._update_ensemble()
            self._run_ensemble()

            if self._is_converged():
                return


class EnRMLProblem(EnsembleProblem):

    def __init__(self, *args):
        
        super().__init__(*args)

        self.ss = []
        self.ls = []

        self.results = {
            "ts": self.ts, 
            "fs": self.fs, 
            "gs": self.gs, 
            "ss": self.ss, 
            "inds": self.en_inds
        }
    
    def _compute_dt_max(self):
        dts = self.ts[-1][:,self.en_inds] - self.ts[-2][:,self.en_inds]
        return np.abs(dts).max()

    def _compute_gain(self, dts, Ug, Sg, Vg, l):
        psi = np.diag(1.0 / (l + 1.0 + Sg ** 2))
        return dts @ Vg @ np.diag(Sg) @ psi @ Ug.T @ self.lik.cov_inv_sq

    """Carries out bootstrapping-based localisation procedure described 
    by Zhang and Oliver (2010)."""
    def _localise_bootstrap(self, K, ts, gs, l, 
                            Nb=100, sigalpha_sq=0.6**2):

        Ks = np.zeros(self.Nt, self.Ng, Nb)
        loc_mat = np.zeros(self.Nt, self.Ng)

        inds_r = copy(self.en_inds)

        for k in range(Nb):

            np.random.shuffle(inds_r)
            dts = self._compute_deltas(ts[:,inds_r])
            dgs = self._compute_deltas(gs[:,inds_r])

            Ug_r, Sg_r, Vg_r = tsvd(self.lik.cov_inv_sq * dgs)
            Ks[:,:,k] = self._compute_gain(dts, Ug_r, Sg_r, Vg_r, l)
            
        for (i, j) in it.product(range(self.Nt), range(self.Ng)):
            r_sq = np.mean((Ks[i, j, :] - K[i, j]) ** 2) / K[i, j] ** 2
            loc_mat[i, j] = 1.0 / (1.0 + r_sq * (1 + 1 / sigalpha_sq))

        return K * loc_mat

    def run(self, i_max, gamma, l_min, max_cuts, 
            ds_min = 0.01, dt_min = 0.1):

        # NOTE: This isn't needed for current problem because it is just the identity
        cov_sc_sqi = np.diag((1.0 / np.diag(self.pri.cov)) ** 0.25)

        self._generate_initial_ensemble()
        self._run_ensemble()

        ys_p = self.lik.sample(self.Ne)

        s = self._compute_S()
        l = 10 ** np.floor(np.log10(s / (2 * self.Ny)))

        self.ss.append(s)
        self.ls.append(l)

        dts_prior = self._compute_deltas(self.ts[0])
        Ut_p, St_p, _ = tsvd(cov_sc_sqi @ dts_prior)

        i = 0
        n_cuts = 0

        while i < i_max:

            dts = self._compute_deltas(self.ts[-1])
            dgs = self._compute_deltas(self.gs[-1])

            Ug, Sg, Vg = tsvd(self.lik.cov_inv_sq @ dgs)

            K = self._compute_gain(dts, Ug, Sg, Vg, self.ls[-1])
            
            # TODO: localise

            dt_prior = dts @ Vg @ \
                np.diag((self.ls[-1] + 1.0 + Sg ** 2) ** -1) @ \
                Vg.T @ dts.T @ cov_sc_sqi @ Ut_p @ \
                np.diag(1.0 / St_p ** 2) @ Ut_p.T @ cov_sc_sqi @ \
                (self.ts[-1] - self.ts[0])

            dt_lik = K @ (self.gs[-1] - ys_p) 

            self.ts.append(self.ts[-1] - dt_prior - dt_lik)
            self._run_ensemble()
            self.ss.append(self._compute_S(self.gs[-1]))

            if self.ss[-1] <= self.ss[-2]:

                n_cuts = 0
                ds = 1.0 - (self.ss[-1]/self.ss[-2])
                dt_max = self._compute_dt_max()

                utils.info(f"ds: {ds:.2%} | ds_max: {dt_max:.2f}")

                # Check for convergence
                if (ds <= ds_min) and (dt_max <= dt_min):
                    return
                
                i += 1
                self.ls.append(max(self.ls[-1]/gamma, l_min))
                utils.info(f"Step accepted. λ is now {self.ls[-1]}.")
                
            else:

                n_cuts += 1
                if n_cuts == max_cuts:
                    utils.info("Maximum number of failed steps reached.")
                    return

                # Remove all the stuff corresponding to the failed step
                self.ts.pop()
                self.gs.pop()
                self.fs.pop()
                self.ss.pop()
                self.ls[-1] *= gamma
                utils.info(f"Step rejected. λ is now {self.ls[-1]}.")

        return


def tsvd(A, energy=0.99):

    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    S_cum = np.cumsum(S)
    for (i, c) in enumerate(S_cum):
        if c / S_cum[-1] >= energy:
            return U[:, :i], S[:i], Vt.T[:, :i]
        
    raise Exception("Error in TSVD function.")