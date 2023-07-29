from copy import copy
import h5py
import itertools as it
import numpy as np

import geo_models as gm
import utils


class EnsembleProblem():

    def __init__(self, f, g, prior, likelihood, Nf, Ne):
        
        self.prior = prior 
        self.likelihood = likelihood 

        self.f = f
        self.g = g 

        self.Nt = len(prior.mu)
        self.Nf = Nf
        self.Ng = len(likelihood.mu)
        self.Ne = Ne
        self.ensemble_inds = list(range(Ne))

    def _generate_initial_ensemble(self):
        return self.prior.sample(self.Ne)
        
    def _run_ensemble(self, ts):

        fs = np.zeros((self.Nf, self.Ne))
        gs = np.zeros((self.Ng, self.Ne))

        for i in range(self.Ne):
            if i in self.ensemble_inds:
                
                f_t = self.f(ts[:, i])
                if type(f_t) != gm.ExitFlag:
                    fs[:, i] = f_t
                    gs[:, i] = self.g(f_t)
                else:
                    self.ensemble_inds.remove(i)

        return fs, gs

    """Returns a matrix of rescaled deviations from the mean (for 
    the ensemble members or the ensemble predictions)."""
    def _calculate_deltas(self, q):
        deltas = q - np.mean(q, axis=1)[:, np.newaxis]
        return deltas / np.sqrt(len(self.ensemble_inds) - 1)
    
    """Returns the ensemble estimate of the model Jacobian."""
    def _calculate_J(self, delta_t, delta_g):
        return delta_g @ np.linalg.pinv(delta_t)

    """Calculates ensemble (cross-)covariance matrices."""
    def _calculate_covs(self, delta_t, delta_g):

        if not self.loc_mat:
            cov_tg = delta_t @ delta_g.T
            cov_gg = delta_g @ delta_g.T
            return cov_tg, cov_gg

        cov_tt = self.loc_mat * (delta_t * delta_t.T)
        J = self._calculate_J(delta_t, delta_g)
        cov_tg = cov_tt @ J.T
        cov_gg = J @ cov_tt @ J.T
        return cov_tg, cov_gg

    def save_results(self, fname):
        with h5py.File(f"{fname}.h5", "w") as f:
            for k, v in self.results.items():
                f.create_dataset(k, data=v)


class ESMDAProblem(EnsembleProblem):

    def run(self, Ni, loc_mat=None):

        self.loc_mat = loc_mat

        ts = np.zeros((self.Nt, self.Ne, Ni+1))
        fs = np.zeros((self.Nf, self.Ne, Ni+1))
        gs = np.zeros((self.Ng, self.Ne, Ni+1))

        ts[:,:,0] = self._generate_initial_ensemble()
        fs[:,:,0], gs[:,:,0] = self._run_ensemble(ts[:,:,0])

        alphas = Ni * np.ones(Ni)

        for i in range(Ni):

            utils.info(f"Beginning iteration {i}...")

            # TODO: clean up
            gs_p = np.random.multivariate_normal(self.likelihood.mu, 
                                                 alphas[i] * self.likelihood.cov, 
                                                 size=(self.Ne, )).T

            delta_t = self._calculate_deltas(ts[:,:,0])
            delta_g = self._calculate_deltas(gs[:,:,0])

            cov_tg, cov_gg = self._calculate_covs(delta_t, delta_g)
            K = cov_tg @ np.linalg.inv(cov_gg + alphas[i] * self.likelihood.cov)

            ts[:,:,i+1] = ts[:,:,i] + K @ (gs_p - gs[:,:,i])
            fs[:,:,i+1], gs[:,:,i+1] = self._run_ensemble(ts[:,:,i+1])

        self.results = {
            "ts": ts, 
            "fs": fs, 
            "gs": gs, 
            "alphas": alphas, 
            "inds": self.ensemble_inds 
        }


class EnRMLProblem(EnsembleProblem):

    def _calculate_s(self, gs, ys_p, cov_error_sq_inv):
        log_liks = np.sum((cov_error_sq_inv * \
            (gs[:,self.ensemble_inds]-ys_p[:,self.ensemble_inds])) ** 2, axis=1)
        return np.mean(log_liks)
    
    def _calculate_dt_max(self, ts, ts_prev):
        dts = (ts[:,self.ensemble_inds] - ts_prev[:,self.ensemble_inds]) / \
            ts[:,self.ensemble_inds]
        return np.maximum(np.abs(dts))

    def _compute_gain(self, dts, Ug, Sg, Vg, cov_eta_sqi, l):
        psi = np.diag(1.0 / (l + 1.0 + Sg ** 2))
        return dts @ Vg @ Sg @ psi @ Ug.T @ cov_eta_sqi

    """Carries out bootstrapping-based localisation procedure described 
    by Zhang and Oliver (2010)."""
    def _localise_bootstrap(self, K, ts, gs, cov_eta_sqi, l, 
                            Nb=100, sigalpha_sq=0.6**2):

        Ks = np.zeros(self.Nt, self.Ng, Nb)
        loc_mat = np.zeros(self.Nt, self.Ng)

        inds_r = copy(self.ensemble_inds)

        for k in range(Nb):

            np.random.shuffle(inds_r)
            dts = self._calculate_deltas(ts[:,inds_r])
            dgs = self._calculate_deltas(gs[:,inds_r])

            Ug_r, Sg_r, Vg_r = tsvd(cov_eta_sqi * dgs)
            Ks[:,:,k] = self._compute_gain(dts, Ug_r, Sg_r, Vg_r, 
                                           cov_eta_sqi, l)
            
        for (i, j) in it.product(range(self.Nt), range(self.Ng)):
            r_sq = np.mean((Ks[i, j, :] - K[i, j]) ** 2) / K[i, j] ** 2
            loc_mat[i, j] = 1.0 / (1.0 + r_sq * (1 + 1 / sigalpha_sq))

        return K * loc_mat

    def run(self, i_max, gamma, l_min, max_cuts):

        ds_min = 0.01
        dt_min = 0.01

        ts = np.zeros((self.Nt, self.Ne, i_max+1))
        fs = np.zeros((self.Nf, self.Ne, i_max+1))
        gs = np.zeros((self.Ng, self.Ne, i_max+1))

        ss = np.zeros(i_max+1)
        ls = np.zeros(i_max+1)

        cov_eta_sqi = np.sqrt(np.linalg.inv(self.likelihood.cov))
        cov_sc_i = np.diag((1.0 / np.diag(self.prior.cov)) ** 0.25)

        ts[:,:,0] = self.prior.sample(self.Ne)
        ys_p = self.likelihood.sample(self.Ne)
        fs[:,:,0], gs[:,:,0] = self._run_ensemble(ts[:,:,0])

        ss[0] = self._calculate_s(gs[:,:,1], ys_p, cov_eta_sqi)
        ls[0] = 10 ** np.floor(np.log10(ss[0] / (2 * self.Ny)))

        dts_prior = self._calculate_deltas(ts[:,:,0])
        Ut_p, St_p, _ = tsvd(cov_sc_i * dts_prior)

        i = 0
        n_cuts = 0

        while i < i_max:

            dts = self._calculate_deltas(ts[:,:,i])
            dgs = self._calculate_deltas(gs[:,:,i])

            Ug, Sg, Vg = tsvd(cov_eta_sqi * dgs)

            K = self._compute_gain(dts, Ug, Sg, Vg, cov_eta_sqi, ls[i])
            
            # TODO: localise

            # Calculate corrections based on prior deviations
            dt_prior = dts @ Vg @ \
                np.diag((ls[i] + 1.0 + Sg ** 2) ** -1) @ \
                Vg.T @ dts.T @ cov_sc_i @ Ut_p @ \
                np.diag(1.0 / St_p ** 2) @ Ut_p.T @ cov_sc_i @ \
                (ts[:,:,i] - ts[:,:,0])

            # Calculate corrections based on data misfit
            dt_lik = K @ (fs[:,:,i] - ys_p) 

            ts[:,:,i+1] = ts[:,:,i] - dt_prior - dt_lik
            fs[:,:,i+1], gs[:,:,i+1] = self._run_ensemble(ts[:,:,i+1])
            ss[i+1] = self._calculate_s(gs[:,:,i+1], ys_p, cov_eta_sqi)

            if ss[i+1] <= ss[i]:

                n_cuts = 0
                ds = 1-ss[i+1]/ss[i]
                dt_max = self._calculate_dt_max(ts[:,:,i+1], ts[:,:,i])

                utils.info(f"Δs: {ds} | Δθ_max: {dt_max}")

                # Check for convergence
                if (ds <= ds_min) and (dt_max <= dt_min):
                    self.results = {"ts": ts[:,:,1:i+1], "fs": fs[:,:,1:i+1], 
                                    "gs": gs[:,:,1:i+1], "ss": ss[1:i+1], 
                                    "inds": self.ensemble_inds}
                    return
                
                i += 1
                ls[i] = np.max(ls[i-1]/gamma, l_min)
                utils.info(f"Step accepted. λ is now {ls[i]}.")
                
            else:

                n_cuts += 1
                if n_cuts == max_cuts:
                    utils.info("Maximum number of failed steps reached.")
                    self.results = {"ts": ts[:,:,1:i], "fs": fs[:,:,1:i], 
                                    "gs": gs[:,:,1:i], "ss": ss[1:i], 
                                    "inds": self.ensemble_inds}
                    return

                ls[i] *= gamma
                utils.info(f"Step rejected. λ is now f{ls[i]}.")

        self.results = {"ts": ts[:,:,1:i], "fs": fs[:,:,1:i],
                        "gs": gs[:,:,1:i], "ss": ss[1:i], 
                        "inds": self.ensemble_inds}
        return


def tsvd(A):
    # TODO: make into tsvd...?
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U, S, Vt.T