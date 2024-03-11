from abc import ABC, abstractmethod
from copy import copy, deepcopy

import h5py
import numpy as np
from numpy.random import default_rng
from scipy.linalg import inv, sqrtm

from src import utils
from src.models import *

EPS_IMPUTERS = 1e-4
TOL = 1e-8

class Imputer(ABC):
    
    @abstractmethod 
    def impute(self):
        pass

class GaussianImputer(Imputer):
    """Replaces any failed ensemble members by sampling from a Gaussian
    with moments constructed using the successful ensemble members."""
    
    def impute(self, ws, inds_succ, inds_fail):

        n_fail = len(inds_fail)

        mu = np.mean(ws[:, inds_succ], axis=1)
        cov = np.cov(ws[:, inds_succ]) + EPS_IMPUTERS * np.eye(len(mu))

        ws[:, inds_fail] = default_rng(0).multivariate_normal(
            mu, cov, method="cholesky", size=n_fail
        ).T

        return ws

class ResamplingImputer(Imputer):
    """Replaces any failed ensemble members by sampling (with 
    replacement) from the successful ensemble members."""

    def impute(self, ws, inds_succ, inds_fail):

        size = (ws.shape[0], len(inds_fail))
        perturbations = EPS_IMPUTERS * np.random.normal(size=size)

        inds_rep = np.random.choice(inds_succ, size=len(inds_fail))
        ws[:, inds_fail] = ws[:, inds_rep] + perturbations
        return ws

class Localiser(ABC):

    def compute_gain_eki(self):
        raise Exception(f"{type(self)} cannot be used with EKI.")

    def compute_gain_enrml(self):
        raise Exception(f"{type(self)} cannot be used with EnRML.")

    def _compute_gain_eki(self, ws, Gs, a_i, C_e):
        C_wG, C_GG = compute_covs(ws, Gs)
        return C_wG @ inv(C_GG + a_i * C_e)
    
    def _compute_gain_enrml(self, dw, UG, SG, VG, C_e_invsqrt, lam):
        psi = np.diag(SG / (lam + 1.0 + SG**2))
        return dw @ VG @ psi @ UG.T @ C_e_invsqrt

class IdentityLocaliser(Localiser):
    """Computes the Kalman gain without using localisation."""

    def compute_gain_eki(self, ws, Gs, a_i, C_e):
        return self._compute_gain_eki(ws, Gs, a_i, C_e)

    def compute_gain_enrml(self, ws, Gs, dw, UG, SG, VG, C_e_invsqrt, lam):
        return self._compute_gain_enrml(dw, UG, SG, VG, C_e_invsqrt, lam)

class BootstrapLocaliser(Localiser):
    """Carries out the localisation procedure described by Zhang and 
    Oliver (2010)."""
    
    def __init__(self, n_boot=100, sigalpha=0.60, regularised=False):

        self.n_boot = n_boot
        self.sigalpha = sigalpha
        self.regularised = regularised

    def compute_P(self, K, Ks_boot):

        var_Kis = np.mean((Ks_boot - K[:, :, np.newaxis]) ** 2, axis=2)
        Rsq = var_Kis / (K ** 2)

        if self.regularised:
            P = 1 / (1 + Rsq * (1 + 1 / self.sigalpha ** 2))
        else: 
            P = (1 - Rsq / (self.n_boot - 1)) / (Rsq + 1)
            P = np.maximum(P, 0.0)

        return P

    def compute_gain_eki(self, ws, Gs, a_i, C_e):

        Nw, Ne = ws.shape
        NG, Ne = Gs.shape
        K = self._compute_gain_eki(ws, Gs, a_i, C_e)

        Ks_boot = np.empty((Nw, NG, self.n_boot))

        for k in range(self.n_boot):
            inds_res = np.random.choice(Ne, Ne)
            ws_res = ws[:, inds_res]
            Gs_res = Gs[:, inds_res]
            K_res = self._compute_gain_eki(ws_res, Gs_res, a_i, C_e)
            Ks_boot[:, :, k] = K_res

        P = self.compute_P(K, Ks_boot)
        return P * K
    
    def compute_gain_enrml(self, ws, Gs, dw, UG, SG, VG, C_e_invsqrt, lam):

        Nw, Ne = ws.shape
        NG, Ne = Gs.shape
        K = self._compute_gain_enrml(dw, UG, SG, VG, C_e_invsqrt, lam)

        Ks_boot = np.empty((Nw, NG, self.n_boot))

        for k in range(self.n_boot):

            inds_res = np.random.choice(Ne, Ne)
            
            dw_res = compute_deltas(ws[:, inds_res])
            dG_res = compute_deltas(Gs[:, inds_res])
            UG_res, SG_res, VG_res = compute_tsvd(dG_res)
            
            Ks_boot[:, :, k] = self._compute_gain_enrml(
                dw_res, UG_res, SG_res, VG_res, 
                C_e_invsqrt, lam 
            )
        
        P = self.compute_P(K, Ks_boot)
        return P * K

class ShuffleLocaliser(Localiser):
    """Carries out a variant of correlation-based localisation 
    (see, e.g., Luo and Bhakta 2017)."""

    def __init__(self, n_shuffle=100):
        self.n_shuffle = n_shuffle
    
    def shuffle_inds(self, Ne):
        
        inds = np.arange(Ne)
        np.random.shuffle(inds)
        
        for i in range(Ne):
            if inds[i] == i:
                inds[(i+1)%Ne], inds[i] = inds[i], inds[(i+1)%Ne]

        return inds
    
    def compute_gain(self, K, ws, Gs):
        
        Nw, Ne = ws.shape
        NG, Ne = Gs.shape 

        R_wG = compute_cors(ws, Gs)[0]

        P = np.zeros((Nw, NG))
        R_wGs = np.zeros((Nw, NG, self.n_shuffle))

        for i in range(self.n_shuffle):
            inds_shuffled = self.shuffle_inds(Ne)
            ws_shuffled = ws[:, inds_shuffled]
            R_wGs[:, :, i] = compute_cors(ws_shuffled, Gs)[0]

        error_sds = np.std(R_wGs, axis=2)

        for i in range(Nw):
            for j in range(NG):
                if np.abs(R_wG[i, j]) > error_sds[i, j]:
                    P[i, j] = 1

        return P * K

    def compute_gain_eki(self, ws, Gs, a_i, C_e):
        K = self._compute_gain_eki(ws, Gs, a_i, C_e)
        return self.compute_gain(K, ws, Gs)
    
    def compute_gain_enrml(self, ws, Gs, dw, UG, SG, VG, C_e_invsqrt, lam):
        K = self._compute_gain_enrml(dw, UG, SG, VG, C_e_invsqrt, lam)
        return self.compute_gain(K, ws, Gs)

class Inflator(ABC):
    
    @abstractmethod
    def update_eki(self):
        pass 

    @abstractmethod
    def update_enrml(self):
        pass

    def _update_eki(self, ws, Gs, Ne, 
                    inds_succ, inds_fail, alpha, y, C_e,
                    localiser: Localiser, imputer: Imputer):
        """Runs a single EKI update."""

        ys_i = np.random.multivariate_normal(y, alpha*C_e, size=Ne).T

        K = localiser.compute_gain_eki(
            ws[:, inds_succ], 
            Gs[:, inds_succ], 
            alpha, C_e
        )

        ws[:, inds_succ] += K @ (ys_i[:, inds_succ] - Gs[:, inds_succ])

        if (n_fail := len(inds_fail)) > 0:
            utils.info(f"Imputing {n_fail} failed ensemble members...")
            ws = imputer.impute(ws, inds_succ, inds_fail)
        
        return ws
    
    def _update_enrml(self, ws, Gs, ys, C_e_invsqrt, ws_pr, Uw_pr, Sw_pr, lam, 
                     inds_succ, inds_fail,
                     localiser: Localiser, imputer: Imputer):
    
        dw = compute_deltas(ws[:, inds_succ])
        dG = compute_deltas(Gs[:, inds_succ])

        UG, SG, VG = compute_tsvd(C_e_invsqrt @ dG)

        psi = np.diag((lam + 1 + SG**2)**-1)

        dw_pr = dw @ VG @ psi @ VG.T @ dw.T @ \
            Uw_pr @ np.diag(Sw_pr**-2) @ Uw_pr.T @ \
                (ws[:, inds_succ] - ws_pr[:, inds_succ])

        K = localiser.compute_gain_enrml(ws[:, inds_succ], Gs[:, inds_succ], 
                                         dw, UG, SG, VG, C_e_invsqrt, lam) 
        
        dw_obs = K @ (Gs[:, inds_succ] - ys[:, inds_succ])
        ws[:, inds_succ] -= (dw_pr + dw_obs) 

        if (n_fail := len(inds_fail)) > 0:
            utils.info(f"Imputing {n_fail} failed ensemble members...")
            ws = imputer.impute(ws, inds_succ, inds_fail)
        
        return ws

class IdentityInflator(Inflator):
    """Updates the current ensemble without applying any inflation."""

    def update_eki(self, *args):
        return self._update_eki(*args)

    def update_enrml(self, *args):
        return self._update_enrml(*args)

class AdaptiveInflator(Inflator):
    """Updates the current ensemble using the adaptive inflation 
    method described by Evensen (2009)."""

    def __init__(self, n_dummy_params=100):
        self.n_dummy_params = n_dummy_params

    def generate_dummy_params(self, n_succ):

        size = (self.n_dummy_params, n_succ)
        dummy_params = np.random.normal(size=size)
        
        for r in dummy_params:
            mean, std = np.mean(r), np.std(r)
            r = (r - mean) / std
        
        return dummy_params
    
    def update_eki(self, ws, Gs, Ne, inds_succ, inds_fail, 
                   alpha, y, C_e, localiser, imputer):
        
        Nw, Ne = ws.shape
        n_succ = len(inds_succ)

        dummy_params = np.zeros((self.n_dummy_params, Ne))
        dummy_params[:, inds_succ] = self.generate_dummy_params(n_succ)

        ws_aug = np.vstack((ws, dummy_params))

        ws_aug = self._update_eki(
            ws_aug, Gs, Ne, inds_succ, inds_fail, 
            alpha, y, C_e, localiser, imputer
        )

        ws_new = ws_aug[:Nw, :]
        dummy_params = ws_aug[Nw:, inds_succ]
        fac = 1 / np.mean(np.std(dummy_params, axis=1))

        utils.info(f"Inflation factor: {fac}")

        mu_w = np.mean(ws_new, axis=1)[:, np.newaxis]
        return fac * (ws_new - mu_w) + mu_w

    def update_enrml(self, ws, Gs, ys, C_e_invsqrt, 
                     ws_pr, Uw_pr, Sw_pr, lam, 
                     inds_succ, inds_fail, 
                     localiser: Localiser, 
                     imputer: Imputer):

        Nw, Ne = ws.shape 
        n_succ = len(inds_succ)
        
        dummy_params = np.zeros((self.n_dummy_params, Ne))
        dummy_params[:, inds_succ] = self.generate_dummy_params(n_succ)

        ws_aug = np.vstack((ws, dummy_params))

        dw = compute_deltas(ws[:, inds_succ])
        dG = compute_deltas(Gs[:, inds_succ])
        dw_aug = compute_deltas(ws_aug[:, inds_succ])

        UG, SG, VG = compute_tsvd(C_e_invsqrt @ dG)

        psi = np.diag((lam + 1 + SG**2)**-1)
        dw_pr = dw @ VG @ psi @ VG.T @ dw.T @ \
            Uw_pr @ np.diag(Sw_pr**-2) @ Uw_pr.T @ \
                (ws[:, inds_succ] - ws_pr[:, inds_succ])

        K = localiser.compute_gain_enrml(ws_aug[:, inds_succ], Gs[:, inds_succ], 
                                         dw_aug, UG, SG, VG, C_e_invsqrt, lam) 
        
        dw_obs = K @ (Gs[:, inds_succ] - ys[:, inds_succ])
        
        ws[:, inds_succ] -= (dw_pr + dw_obs[:Nw, :])

        dummy_params = ws_aug[Nw:, inds_succ] - dw_obs[Nw:, :]
        fac = 1 / np.mean(np.std(dummy_params, axis=1))
        utils.info(f"Inflation factor: {fac}")

        if (n_fail := len(inds_fail)) > 0:
            utils.info(f"Imputing {n_fail} failed ensemble members...")
            ws = imputer.impute(ws, inds_succ, inds_fail)

        mu_w = np.mean(ws, axis=1)[:, np.newaxis]
        return fac * (ws - mu_w) + mu_w

def compute_deltas(xs):
    N = xs.shape[1]
    mu = np.mean(xs, axis=1)[:, np.newaxis]
    deltas = (1 / np.sqrt(N-1)) * (xs - mu)
    return deltas

def compute_covs(ws, Gs):

    dws = compute_deltas(ws)
    dGs = compute_deltas(Gs)
    
    C_wG = dws @ dGs.T
    C_GG = dGs @ dGs.T

    return C_wG, C_GG

def compute_cors(ws, Gs):

    sd_w_inv = inv(np.diag(np.std(ws, axis=1)))
    sd_G_inv = inv(np.diag(np.std(Gs, axis=1)))

    C_wG, C_GG = compute_covs(ws, Gs)

    R_wG = sd_w_inv @ C_wG @ sd_G_inv 
    R_GG = sd_G_inv @ C_GG @ sd_G_inv

    return R_wG, R_GG

def compute_a_dmc(t, Gs, y, NG, C_e_invsqrt):
    """Computes the EKI inflation factor following Iglesias and Yang 
    (2021)."""

    diffs = y[:, np.newaxis] - Gs
    Ss = 0.5 * np.sum((C_e_invsqrt @ diffs) ** 2, axis=0)
    mu_S, var_S = np.mean(Ss), np.var(Ss)

    a_inv = max(NG / (2 * mu_S), np.sqrt(NG / (2 * var_S)))
    a_inv = min(a_inv, 1.0 - t)

    return a_inv ** -1

def run_eki_dmc(ensemble: Ensemble, prior, y, C_e,
                localiser: Localiser=IdentityLocaliser(),
                inflator: Inflator=IdentityInflator(), 
                imputer: Imputer=GaussianImputer(),
                nesi=True):
    """Runs EKI-DMC, as described in Iglesias and Yang (2021)."""

    C_e_invsqrt = sqrtm(inv(C_e))
    NG = len(y)

    ws_i = prior.sample(n=ensemble.Ne)

    ps_i, Fs_i, Gs_i, inds_succ, inds_fail = ensemble.run(ws_i, nesi)
    
    ws = [deepcopy(ws_i)]
    ps = [deepcopy(ps_i)]
    Fs = [deepcopy(Fs_i)]
    Gs = [deepcopy(Gs_i)]
    inds = [copy(inds_succ)]

    i = 0
    t = 0
    converged = False

    while not converged:

        a_i = compute_a_dmc(t, Gs_i[:, inds_succ], y, NG, C_e_invsqrt)

        t += (a_i ** -1)
        if np.abs(t - 1.0) < TOL:
            converged = True

        ws_i = inflator.update_eki(
            ws_i, Gs_i, ensemble.Ne, inds_succ, inds_fail, 
            a_i, y, C_e, localiser, imputer
        )
        
        ps_i, Fs_i, Gs_i, inds_succ, inds_fail = ensemble.run(ws_i, nesi)
        
        ws.append(deepcopy(ws_i))
        ps.append(deepcopy(ps_i))
        Fs.append(deepcopy(Fs_i))
        Gs.append(deepcopy(Gs_i))
        inds.append(copy(inds_succ))

        i += 1
        utils.info(f"Iteration {i} complete. t = {t:.4f}.")

    return ws, ps, Fs, Gs, inds

def compute_tsvd(A, energy=0.99):
    
    U, S, Vt = np.linalg.svd(A)
    V = Vt.T

    if np.min(S) < -TOL:
        raise Exception("Negative eigenvalue encountered in TSVD.")

    n_eigvals = len(S)
    eig_cum = np.cumsum(S)

    for i in range(n_eigvals):
        if eig_cum[i] / eig_cum[-1] >= energy:
            return U[:, 1:i], S[1:i], V[:, 1:i]
    
    raise Exception("Issue with TSVD computation.")

def compute_S(Gs, ys, C_e_invsqrt):
    """Computes mean of EnRML misfit functions for each particle."""
    Ss = np.sum((C_e_invsqrt @ (Gs - ys)) ** 2, axis=0)
    return np.mean(Ss)

def run_enrml(ensemble: Ensemble, prior, y, C_e, 
              gamma=10, lam_min=0.01, 
              max_cuts=5, max_its=30, 
              dS_min=0.01, dw_min=0.5,
              localiser: Localiser=IdentityLocaliser(),
              inflator: Inflator=IdentityInflator(),
              imputer: Imputer=GaussianImputer(),
              nesi=True):
    
    C_e_invsqrt = sqrtm(inv(C_e))
    NG = len(y)

    ys = np.random.multivariate_normal(mean=y, cov=C_e, size=ensemble.Ne).T

    ws_pr = prior.sample(n=ensemble.Ne)
    ps_pr, Fs_pr, Gs_pr, inds_succ, inds_fail = ensemble.run(ws_pr, nesi)
    S_pr = compute_S(Gs_pr[:, inds_succ], ys[:, inds_succ], C_e_invsqrt)
    lam = 10**np.floor(np.log10(S_pr / 2*NG))
    
    ws = [deepcopy(ws_pr)]
    ps = [deepcopy(ps_pr)]
    Fs = [deepcopy(Fs_pr)]
    Gs = [deepcopy(Gs_pr)]
    Ss = [S_pr]
    lams = [lam]
    inds = [copy(inds_succ)]

    dw_pr = compute_deltas(ws_pr)
    Uw_pr, Sw_pr, _ = compute_tsvd(dw_pr)

    i = 1
    en_ind = 0
    n_cuts = 0
    while i <= max_its:

        ws_i = inflator.update_enrml(
            ws[en_ind], Gs[en_ind], ys, C_e_invsqrt, 
            ws_pr, Uw_pr, Sw_pr, lam, inds_succ, inds_fail, 
            localiser, imputer)
        
        ps_i, Fs_i, Gs_i, inds_succ, inds_fail = ensemble.run(ws_i, nesi)
        S_i = compute_S(Gs_i[:, inds_succ], ys[:, inds_succ], C_e_invsqrt)
        
        ws.append(deepcopy(ws_i))
        ps.append(deepcopy(ps_i))
        Fs.append(deepcopy(Fs_i))
        Gs.append(deepcopy(Gs_i))
        Ss.append(S_i)
        lams.append(lam)
        inds.append(copy(inds_succ))
        i += 1

        if S_i <= Ss[en_ind]:

            dS = 1 - (S_i / Ss[en_ind])
            dw_max = np.max(np.abs(ws_i - ws[en_ind]))

            en_ind = i-1
            n_cuts = 0 
            lam = max(lam / gamma, lam_min)

            utils.info(f"Iteration {i}: step accepted.")
            utils.info(f"dS = {dS:.2f}, dw_max = {dw_max:.2f}.")

            if (dS < dS_min) and (dw_max < dw_min):
                utils.info("Convergence criteria met.")
                return ws, ps, Fs, Gs, Ss, lams, en_ind, inds 
            
        else:

            utils.info(f"Iteration {i}: step rejected.")
            n_cuts += 1
            lam *= gamma 

            if n_cuts == max_cuts:
                utils.info(f"Terminating: {n_cuts} consecutive cuts made.")
                return ws, ps, Fs, Gs, Ss, lams, en_ind, inds

    utils.info("Terminating: maximum number of iterations exceeded.")
    return ws, ps, Fs, Gs, Ss, lams, en_ind, inds 

def save_results_eki(fname: str, results: dict):

    with h5py.File(fname, "w") as f:
        
        for key, vals in results.items():
            for i, val in enumerate(vals):
                f.create_dataset(f"{key}_{i}", data=val)

        post_ind = len(results["ws"])-1
        f.create_dataset("algorithm", data=["eki"])
        f.create_dataset("post_ind", data=[post_ind])

def save_results_enrml(fname: str, results: dict, post_ind: int):

    with h5py.File(fname, "w") as f:

        for key, vals in results.items():
            for i, val in enumerate(vals):
                f.create_dataset(f"{key}_{i}", data=val)

        f.create_dataset("algorithm", data=["enrml"])
        f.create_dataset("post_ind", data=[post_ind])