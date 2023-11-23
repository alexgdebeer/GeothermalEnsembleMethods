import h5py
import itertools as it
import numpy as np

from scipy.linalg import inv, sqrtm

from src import models, utils

TOL = 1e-8

# Could define a localiser object
# It could take in the current ensemble and ensemble predictions (I think this would work with a wide range of procedures).

class EnsembleRunner():
    """Runs an ensemble and returns the results, including a list of 
    indices of any failed simulations."""

    def __init__(self, prior, F, G, Np, NF, NG, Ne):
        
        self.prior = prior 
        self.F = F 
        self.G = G
        
        self.Np = Np 
        self.NF = NF 
        self.NG = NG 
        self.Ne = Ne

    def run(self, ws_i):

        ps_i = np.empty((self.Np, self.Ne))
        Fs_i = np.empty((self.NF, self.Ne))
        Gs_i = np.empty((self.NG, self.Ne))

        inds_succ = []
        inds_fail = []

        for i, w_i in enumerate(ws_i.T):
            utils.info(f"Simulating ensemble member {i+1}...")
            p_i = self.prior.transform(w_i)
            F_i = self.F(p_i)
            if type(F_i) == models.ExitFlag:
                inds_fail.append(i)
            else: 
                inds_succ.append(i)
                ps_i[:, i] = p_i
                Fs_i[:, i] = F_i 
                Gs_i[:, i] = self.G(F_i)

        return ps_i, Fs_i, Gs_i, inds_succ, inds_fail

def impute_gaussian(ws, inds_succ, inds_fail):
    """Replaces any failed ensemble members by sampling from a Gaussian
    with moments constructed using the successful ensemble members."""
    
    n_fail = len(inds_fail)

    mu = np.mean(ws[:, inds_succ], axis=1)
    cov = np.cov(ws[:, inds_succ]) + 1e-4 * np.eye(len(mu))
    
    ws[:, inds_fail] = np.random.multivariate_normal(mu, cov, size=n_fail).T
    return ws

def impute_resample(ws, inds_succ, inds_fail):
    """Replaces any failed ensemble members by sampling (with 
    replacement) from the successful ensemble members."""
    
    inds_rep = np.random.choice(inds_succ, size=len(inds_fail))
    ws[:, inds_fail] = ws[:, inds_rep]
    return ws

IMPUTERS = {
    "gaussian": impute_gaussian,
    "resample": impute_resample
}

def compute_covs(ws_i, Gs_i):

    N = ws_i.shape[1]

    d_ws = ws_i - np.mean(ws_i, axis=1)[:, np.newaxis]
    d_Gs = Gs_i - np.mean(Gs_i, axis=1)[:, np.newaxis]

    C_WG = (1 / (N-1)) * d_ws @ d_Gs.T
    C_GG = (1 / (N-1)) * d_Gs @ d_Gs.T

    return C_WG, C_GG

def eki_update(ws_i, Gs_i, Ne, inds_succ, inds_fail, a_i, y, C_e,
               localiser=None, failure_imputation="gaussian"):
    """Runs a single EKI update."""

    ys_i = np.random.multivariate_normal(y, a_i * C_e, size=Ne).T

    C_WG, C_GG = compute_covs(ws_i[:, inds_succ], Gs_i[:, inds_succ])

    # TODO: localisation

    ws_n = ws_i + C_WG @ inv(C_GG + a_i * C_e) @ (ys_i - Gs_i)

    if inds_fail:
        utils.info(f"Imputing {len(inds_fail)} failed ensemble members...")
        ws_n = IMPUTERS[failure_imputation](ws_n, inds_succ, inds_fail)
    
    return ws_n

def compute_a_dmc(t, Gs, y, NG, C_e_invsqrt):
    """Computes the EKI inflation factor following Iglesias and Yang 
    (2021)."""

    diffs = y[:, np.newaxis] - Gs
    Ss = 0.5 * np.sum((C_e_invsqrt @ diffs) ** 2, axis=0)
    mu_S, var_S = np.mean(Ss), np.var(Ss)

    a_inv = max(NG / (2 * mu_S), np.sqrt(NG / (2 * var_S)))
    a_inv = min(a_inv, 1.0 - t)

    return a_inv ** -1

def run_eki_dmc(F, G, prior, y, C_e, Np, NF, Ne,
                localiser=None, failure_imputation="gaussian"):
    """Runs EKI-DMC, as described in Iglesias and Yang (2021)."""

    C_e_invsqrt = sqrtm(inv(C_e))
    NG = len(y)

    ws_i = prior.sample(n=Ne)

    ensemble = EnsembleRunner(prior, F, G, Np, NF, NG, Ne)
    ps_i, Fs_i, Gs_i, inds_succ, inds_fail = ensemble.run(ws_i)
    
    ws = [ws_i]
    ps = [ps_i]
    Fs = [Fs_i]
    Gs = [Gs_i]

    i = 0
    t = 0
    converged = False

    while not converged:

        a_i = compute_a_dmc(t, Gs_i[:, inds_succ], y, NG, C_e_invsqrt)

        t += (a_i ** -1)
        if np.abs(t - 1.0) < TOL:
            converged = True

        ws_i = eki_update(ws_i, Gs_i, Ne, inds_succ, inds_fail, a_i, y, C_e, 
                          localiser, failure_imputation)
        
        ps_i, Fs_i, Gs_i, inds_succ, inds_fail = ensemble.run(ws_i)
        
        ws.append(ws_i)
        ps.append(ps_i)
        Fs.append(Fs_i)
        Gs.append(Gs_i)

        i += 1
        utils.info(f"Iteration {i} complete. t = {t:.4f}.")

    return ws, ps, Fs, Gs, inds_succ


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
        self.n_failures = 0

        self.ts = []
        self.fs = []
        self.gs = []


    """Samples an initial ensemble from the prior."""
    def generate_initial_ensemble(self):
        self.ts.append(self.pri.sample(self.Ne))
    

    """Returns a matrix of scaled deviations of the individual ensemble 
    memebers from the mean ensemble member."""
    def compute_dts(self, ts):
        dts_unscaled = ts - np.mean(ts, axis=1)[:, np.newaxis]
        return dts_unscaled / np.sqrt(len(self.en_inds) - 1)
    

    """Returns a matrix of scaled deviations of the individual ensemble
    predictions from the mean prediction of the ensemble."""
    def compute_dgs(self, gs):
        dgs_unscaled = gs - np.mean(gs, axis=1)[:, np.newaxis]
        return dgs_unscaled / np.sqrt(len(self.en_inds) - 1)


    """Returns the mean and variance of the data misfit term for each 
    ensemble member."""
    def compute_S(self):

        diffs = self.y[:, np.newaxis] - self.gs[-1][:, self.en_inds]
        Ss = 0.5 * np.sum((self.lik.cov_inv_sq @ diffs) ** 2, axis=1)

        return np.mean(Ss), np.var(Ss)


    """Returns the ensemble estimate of the model Jacobian."""
    def compute_jac(self, dts, dgs):
        return dgs @ np.linalg.pinv(dts)


    """Saves results of the inversion to an h5 file."""
    def save_results(self, fname):
        with h5py.File(f"{fname}.h5", "w") as f:
            for k, v in self.results.items():
                f.create_dataset(k, data=v)


# class ESMDAProblem(EnsembleProblem):


#     def __init__(self, *args, loc_type=None, loc_mat=None):
        
#         super().__init__(*args)

#         self.loc_type = loc_type
#         self.loc_mat = loc_mat
#         self.alphas = []
#         self.t = 0.0

#         self.results = {
#             "ts": self.ts, 
#             "fs": self.fs, 
#             "gs": self.gs, 
#             "alphas": self.alphas, 
#             "inds": self.en_inds
#         }


#     """Runs the ensemble. Failed simulations are removed from the 
#     list of ensemble indices."""
#     def run_ensemble(self):

#         fs_i = np.zeros((self.Nf, self.Ne))
#         gs_i = np.zeros((self.Ng, self.Ne))

#         for (i, t) in enumerate(self.ts[-1].T):
#             if i in self.en_inds:
#                 if type(f_t := self.f(t)) == models.ExitFlag:
#                     self.en_inds.remove(i)
#                 else:
#                     fs_i[:, i] = f_t
#                     gs_i[:, i] = self.g(f_t)

#         self.fs.append(fs_i)
#         self.gs.append(gs_i)


#     """Calculates ensemble (cross-)covariance matrices."""
#     def compute_covs(self, dts, dgs):

#         if self.loc_type == "linearised":
#             cov_tt = self.loc_mat * (dts @ dts.T)
#             jac = self.compute_jac(dts, dgs)
#             cov_tg = cov_tt @ jac.T
#             cov_gg = jac @ cov_tt @ jac.T
#             return cov_tg, cov_gg
        
#         cov_tg = dts @ dgs.T
#         cov_gg = dgs @ dgs.T
#         return cov_tg, cov_gg
    

#     def compute_gain(self, dts, dgs):

#         cov_tg, cov_gg = self.compute_covs(dts, dgs)
#         K = cov_tg @ np.linalg.inv(cov_gg + self.alphas[-1] * self.lik.cov) # TODO: TSVD

#         return K
    

#     """Carries out the bootstrapping-based localisation procedure described 
#     by Zhang and Oliver (2010)."""
#     def localise_bootstrap(self, K, Nb=100, sigalpha_sq=0.6**2):

#         Ks = np.zeros((self.Nt, self.Ng, Nb))
#         loc_mat = np.zeros((self.Nt, self.Ng))

#         for k in range(Nb):

#             inds_r = np.random.choice(self.en_inds, size=len(self.en_inds))
#             dts = self.compute_dts(self.ts[-1][:,inds_r])
#             dgs = self.compute_dgs(self.gs[-1][:,inds_r])

#             Ks[:,:,k] = self._compute_gain(dts, dgs)
            
#         for (i, j) in it.product(range(self.Nt), range(self.Ng)):
#             r_sq = np.mean((Ks[i, j, :] - K[i, j]) ** 2) / K[i, j] ** 2
#             loc_mat[i, j] = 1.0 / (1.0 + r_sq * (1 + 1 / sigalpha_sq))

#         return K * loc_mat
    

#     """Generates the localisation matrix for a version of the adaptive
#     procedure described by Luo and Bhakta (2020) (also used in PEST)."""
#     def localise_cycle(self):

#         def compute_cor(dts, dgs):
            
#             t_sds_inv = np.diag(np.diag(dts @ dts.T) ** -0.5)
#             g_sds_inv = np.diag(np.diag(dgs @ dgs.T) ** -0.5)

#             cov = dts @ dgs.T
#             cor = t_sds_inv @ cov @ g_sds_inv
#             return cor
        
#         def gaspari_cohn(z):
#             if 0 <= z <= 1:
#                 return -(1/4)*z**5 + (1/2)*z**4 + (5/8)*z**3 - (5/3)*z**2 + 1
#             elif 1 < z <= 2:
#                 return (1/12)*z**5 - (1/2)*z**4 + (5/8)*z**3 + \
#                        (5/3)*z**2 - 5*z + 4 - (2/3)*z**-1
#             return 0.0

#         loc_mat = np.zeros((self.Nt, self.Ny))
#         cor_mats = np.zeros((self.Nt, self.Ny, len(self.en_inds)-1))

#         dts = self.compute_dts(self.ts[0][:,self.en_inds])
#         dgs = self.compute_dgs(self.gs[0][:,self.en_inds])
#         cor = compute_cor(dts, dgs) 

#         for i in range(len(self.en_inds)-1):
#             dgs_s = np.roll(dgs, shift=i+1, axis=1)
#             cor_mats[:,:,i] = compute_cor(dts, dgs_s)

#         error_sd = np.median(np.abs(cor_mats), axis=2) / 0.6745

#         for (i, j) in it.product(range(self.Nt), range(self.Ny)):
#             loc_mat[i, j] = gaspari_cohn((1-np.abs(cor[i, j])) / (1-error_sd[i, j]))

#         print(loc_mat)

#         return loc_mat
    

#     def update_ensemble(self):

#         # TODO: clean up
#         ys_p = np.random.multivariate_normal(
#             self.y, 
#             self.alphas[-1] * self.lik.cov, 
#             size=(self.Ne, )
#         ).T
        
#         dts = self.compute_dts(self.ts[-1][:,self.en_inds])
#         dgs = self.compute_dgs(self.gs[-1][:,self.en_inds])
#         K = self.compute_gain(dts, dgs)

#         if self.loc_type == "cycle":
#             if self.loc_mat is None: # First iteration
#                 self.loc_mat = self.localise_cycle()
#             K *= self.loc_mat 

#         elif self.loc_type == "bootstrap":
#             K = self.localise_bootstrap(K)

#         self.ts.append(self.ts[-1] + K @ (ys_p - self.gs[-1]))


#     def compute_alpha(self):

#         mu_S, var_S = self.compute_S()
        
#         q1 = self.Ny / (2 * mu_S)
#         q2 = np.sqrt(self.Ny / (2 * var_S))
#         alpha_inv = min(max(q1, q2), 1.0 - self.t)

#         self.t += alpha_inv
#         self.alphas.append(alpha_inv ** -1.0)

#         utils.info(f"alpha: {self.alphas[-1]:.4f} | t: {self.t:.4f}")


#     def is_converged(self):
#         return np.abs(self.t-1.0) < 1e-8
    

#     def run(self):

#         self.generate_initial_ensemble()
#         self.run_ensemble()

#         while True:

#             utils.info(f"Beginning iteration {len(self.ts)}...")
#             self.compute_alpha()
#             self.update_ensemble()
#             self.run_ensemble()

#             if self.is_converged():
#                 return


class EnRMLProblem(EnsembleProblem):

    def __init__(self, *args, 
                 i_max=16, gamma=10, lambda_min=0.01, max_cuts=5, 
                 ds_min = 0.01, dt_min = 0.25, loc_type=None, 
                 max_fails_per_it=10):
        
        super().__init__(*args)

        self.ss = []
        self.lambdas = []

        self.n_it = 0
        self.max_its = i_max

        self.n_cuts = 0
        self.max_cuts = max_cuts 

        self.gamma = gamma 
        self.lambda_min = lambda_min 

        self.ds_min = ds_min 
        self.dt_min = dt_min

        self.converged = False

        self.loc_type = loc_type
        self.loc_mat = None

        self.failed_inds = []
        self.max_fails_per_it = max_fails_per_it

        self.results = {
            "ts": self.ts, 
            "fs": self.fs, 
            "gs": self.gs, 
            "ss": self.ss, 
            "inds": self.en_inds
        }


    def reset(self):
        self.n_failures = 0
        self.failed_inds = []


    def remove_failed_inds(self):
        for i in self.failed_inds:
            self.en_inds.remove(i)


    """Removes the data corresponding to a failed step."""
    def revert_step(self):
        self.ts.pop()
        self.gs.pop()
        self.fs.pop()
        self.ss.pop()
        self.lambdas[-1] *= self.gamma
        utils.info(f"Step rejected. Lambda modified to {self.lambdas[-1]}.")


    def run_ensemble(self):

        fs_i = np.zeros((self.Nf, self.Ne))
        gs_i = np.zeros((self.Ng, self.Ne))

        for (i, t) in enumerate(self.ts[-1].T):
            if i in self.en_inds:
                
                if type(f_t := self.f(t)) == models.ExitFlag:    
                    self.failed_inds.append(i)
                    if len(self.failed_inds) > self.max_fails_per_it and \
                        self.n_it > 0:
                        utils.info("Iteration ending: maximum failures exceeded.")
                        self.fs.append(fs_i)
                        self.gs.append(gs_i)
                        return
                    
                else:
                    fs_i[:, i] = f_t
                    gs_i[:, i] = self.g(f_t)

        self.fs.append(fs_i)
        self.gs.append(gs_i)

    
    """Computes quantities from the prior ensemble that are used 
    at each iteration of the algorithm."""
    def compute_prior_quantities(self):

        self.cov_sc_sqi = np.diag((1.0 / np.diag(self.pri.cov)) ** 0.25)

        dts_pri = self.compute_dts(self.ts[0][:,self.en_inds])
        self.U_t_pri, self.S_t_pri, _ = tsvd(self.cov_sc_sqi @ dts_pri)

        mu_s = self.compute_S()[0] # TODO: change to lowercase S
        lambda_0 = 10 ** np.floor(np.log10(mu_s / (2 * self.Ny)))

        self.ss.append(mu_s)
        self.lambdas.append(lambda_0)

        utils.info(f"Initial value of lambda: {lambda_0}.")


    def compute_gain(self, dts, U_g, S_g, V_g):

        psi = np.diag(1.0 / (self.lambdas[-1] + 1.0 + S_g ** 2))
        
        K = dts @ V_g @ np.diag(S_g) @ psi @ U_g.T @ self.lik.cov_inv_sq
        return K


    """Computes the maximum change (in standard deivations with respect to 
    the prior) of any component of any ensemble member."""
    def compute_dt_max(self):
        diffs = self.ts[-1][:, self.en_inds] - self.ts[-2][:, self.en_inds]
        return np.abs(np.diag(np.diag(self.pri.cov) ** -0.5) @ diffs).max()


    """Returns whether or not a step has been accepted."""
    def step_accepted(self):
        return self.ss[-1] <= self.ss[-2]


    """Checks whether a step is accepted, and modifies the damping parameter
    accordingly."""
    def check_step(self):

        self.ss.append(self.compute_S()[0])
        
        if self.step_accepted() and \
            len(self.failed_inds) <= self.max_fails_per_it:

            self.remove_failed_inds()
            self.n_cuts = 0
            if self.is_converged():
                utils.info("Terminating: convergence criteria met.")
                self.converged = True
                return
            
            lambda_new = max(self.lambdas[-1]/self.gamma, self.lambda_min)
            self.lambdas.append(lambda_new)
            utils.info(f"Step accepted: lambda = {self.lambdas[-1]}.")
            
        else:

            self.revert_step()
            self.n_cuts += 1
            if self.n_cuts == self.max_cuts:
                utils.info("Terminating: max consecutive cuts.")
                self.converged = True
                return


    def is_converged(self):

        ds = 1.0 - (self.ss[-1]/self.ss[-2])
        dt_max = self.compute_dt_max()

        utils.info(f"ds: {ds:.2%} | dt_max: {dt_max:.2f}")
        return (ds <= self.ds_min) and (dt_max <= self.dt_min)


    """Carries out the bootstrapping-based localisation procedure described 
    by Zhang and Oliver (2010)."""
    def localise_bootstrap(self, K, Nb=100, sigalpha_sq=0.6**2):

        Ks = np.zeros((self.Nt, self.Ng, Nb))
        loc_mat = np.zeros((self.Nt, self.Ng))

        for k in range(Nb):

            inds_r = np.random.choice(self.en_inds, size=len(self.en_inds))
            dts = self.compute_dts(self.ts[-1][:,inds_r])
            dgs = self.compute_dgs(self.gs[-1][:,inds_r])

            Ug_r, Sg_r, Vg_r = tsvd(self.lik.cov_inv_sq @ dgs)
            Ks[:,:,k] = self.compute_gain(dts, Ug_r, Sg_r, Vg_r)
            
        for (i, j) in it.product(range(self.Nt), range(self.Ng)):
            r_sq = np.mean((Ks[i, j, :] - K[i, j]) ** 2) / K[i, j] ** 2
            loc_mat[i, j] = 1.0 / (1.0 + r_sq * (1 + 1 / sigalpha_sq))

        print(loc_mat)

        return K * loc_mat


    """Generates the localisation matrix for a version of the adaptive
    procedure described by Luo and Bhakta (2020) (also used in PEST)."""
    def localise_cycle(self):

        def compute_cor(dts, dgs):
            
            t_sds_inv = np.diag(np.diag(dts @ dts.T) ** -0.5)
            g_sds_inv = np.diag(np.diag(dgs @ dgs.T) ** -0.5)

            cov = dts @ dgs.T
            cor = t_sds_inv @ cov @ g_sds_inv
            return cor
        
        def gaspari_cohn(z):
            if 0 <= z <= 1:
                return -(1/4)*z**5 + (1/2)*z**4 + (5/8)*z**3 - (5/3)*z**2 + 1
            elif 1 < z <= 2:
                return (1/12)*z**5 - (1/2)*z**4 + (5/8)*z**3 + \
                       (5/3)*z**2 - 5*z + 4 - (2/3)*z**-1
            return 0.0

        loc_mat = np.zeros((self.Nt, self.Ny))
        cor_mats = np.zeros((self.Nt, self.Ny, len(self.en_inds)-1))

        dts = self.compute_dts(self.ts[0][:,self.en_inds])
        dgs = self.compute_dgs(self.gs[0][:,self.en_inds])
        cor = compute_cor(dts, dgs) 

        for i in range(len(self.en_inds)-1):
            dgs_s = np.roll(dgs, shift=i+1, axis=1)
            cor_mats[:,:,i] = compute_cor(dts, dgs_s)

        # Separate for each parameter
        error_sd = np.median(np.abs(cor_mats), axis=2) / 0.6745
        print(error_sd)

        for (i, j) in it.product(range(self.Nt), range(self.Ny)):
            loc_mat[i, j] = gaspari_cohn((1-np.abs(cor[i, j])) / (1-error_sd[i, j]))

        print(loc_mat)

        return loc_mat


    """Carries out a single update step."""
    def update_ensemble(self):
    
        dts = self.compute_dts(self.ts[-1][:,self.en_inds])
        dgs = self.compute_dgs(self.gs[-1][:,self.en_inds])

        U_g, S_g, V_g = tsvd(self.lik.cov_inv_sq @ dgs)

        K = self.compute_gain(dts, U_g, S_g, V_g)
    
        if self.loc_type == "cycle":
            if self.loc_mat is None: # First iteration
                self.loc_mat = self.localise_cycle()
            K *= self.loc_mat 

        elif self.loc_type == "bootstrap":
            K = self.localise_bootstrap(K)

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

        self.generate_initial_ensemble()
        self.run_ensemble()
        self.remove_failed_inds()
        self.compute_prior_quantities()
        self.reset()
        self.ys_p = self.lik.sample(self.Ne)

        while self.n_it < self.max_its:

            self.n_it += 1
            utils.info(f"Beginning iteration {self.n_it}...")

            self.update_ensemble()
            self.run_ensemble()
            self.check_step()
            self.reset()

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