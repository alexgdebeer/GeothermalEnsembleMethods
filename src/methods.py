from abc import ABC, abstractmethod

import h5py
import numpy as np
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
        
        ws[:, inds_fail] = np.random.multivariate_normal(mu, cov, size=n_fail).T
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

class FisherLocaliser(Localiser):
    """Carries out the localisation procedure described by Flowerdew 
    (2015)."""

    def compute_gain_eki(self, ws, Gs, a_i, C_e):
        
        Nw, Ne = ws.shape
        NG, Ne = Gs.shape

        K = self._compute_gain_eki(ws, Gs, a_i, C_e)

        R_wG = compute_cors(ws, Gs)[0]
        P = np.zeros(K.shape)

        for i in range(Nw):
            for j in range(NG):
                cor_ij = R_wG[i, j]
                s = np.log((1+cor_ij) / (1-cor_ij)) / 2
                sig_s = (np.tanh(s + np.sqrt(Ne-3)**-1) - \
                         np.tanh(s - np.sqrt(Ne-3)**-1)) / 2
                P[i, j] = cor_ij**2 / (cor_ij**2 + sig_s**2)

        return P * K

class BootstrapLocaliser(Localiser):
    """Carries out the localisation procedure described by Zhang and 
    Oliver (2010)."""
    
    def __init__(self, n_boot=100, sigalpha=0.60, tol=1e-8):

        self.n_boot = n_boot
        self.sigalpha = sigalpha
        self.tol=tol

    def compute_P(self, K, Ks_boot):

        var_Kis = np.mean((Ks_boot - K[:, :, np.newaxis]) ** 2, axis=2)
        Rsq = var_Kis / np.maximum(K ** 2, self.tol)
        P = 1 / (1 + Rsq * (1 + 1 / self.sigalpha ** 2))

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
    """Carries out the localisation procedure described by Luo and 
    Bhakta (2020)."""

    def __init__(self, n_shuffle=50):
        self.n_shuffle = n_shuffle
        self.P = None

    def gaspari_cohn(self, z):
        if 0 <= z <= 1:
            return -(1/4)*z**5 + (1/2)*z**4 + (5/8)*z**3 - (5/3)*z**2 + 1
        elif 1 < z <= 2:
            return (1/12)*z**5 - (1/2)*z**4 + (5/8)*z**3 + (5/3)*z**2 - 5*z + 4 - (2/3)*z**-1
        return 0.0
    
    def shuffle_inds(self, Ne):
        
        inds = np.arange(Ne)
        np.random.shuffle(inds)
        
        for i in range(Ne):
            if inds[i] == i:
                inds[(i+1)%Ne], inds[i] = inds[i], inds[(i+1)%Ne]

        return inds
    
    def compute_gain(self, K, ws, Gs):

        if self.P is not None:
            return self.P * K
        
        Nw, Ne = ws.shape
        NG, Ne = Gs.shape 

        R_wG = compute_cors(ws, Gs)[0]

        P = np.zeros((Nw, NG))
        R_wGs = np.zeros((Nw, NG, self.n_shuffle))

        for i in range(self.n_shuffle):
            inds_shuffled = self.shuffle_inds(Ne)
            ws_shuffled = ws[:, inds_shuffled]
            R_wGs[:, :, i] = compute_cors(ws_shuffled, Gs)[0]

        error_sds = np.median(np.abs(R_wGs), axis=2) / 0.6745

        for i in range(Nw):
            for j in range(NG):
                z = (1-np.abs(R_wG[i, j])) / (1-error_sds[i, j])
                P[i, j] = self.gaspari_cohn(z)

        self.P = P
        return P * K

    def compute_gain_eki(self, ws, Gs, a_i, C_e):
        K = self._compute_gain_eki(ws, Gs, a_i, C_e)
        return self.compute_gain(K, ws, Gs)
    
    def compute_gain_enrml(self, ws, Gs, dw, UG, SG, VG, C_e_invsqrt, lam):
        K = self._compute_gain_enrml(dw, UG, SG, VG, C_e_invsqrt, lam)
        return self.compute_gain(K, ws, Gs)

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

def eki_update(ws_i, Gs_i, Ne, inds_succ, inds_fail, a_i, y, C_e,
               localiser: Localiser, imputer: Imputer):
    """Runs a single EKI update."""

    ys_i = np.random.multivariate_normal(y, a_i * C_e, size=Ne).T

    K = localiser.compute_gain_eki(
        ws_i[:, inds_succ], 
        Gs_i[:, inds_succ], 
        a_i, C_e
    )

    ws_n = ws_i + K @ (ys_i - Gs_i)

    if (n_fail := len(inds_fail)) > 0:
        utils.info(f"Imputing {n_fail} failed ensemble members...")
        ws_n = imputer.impute(ws_n, inds_succ, inds_fail)
    
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

def run_eki_dmc(ensemble: Ensemble, prior, 
                y, C_e, Ne,
                localiser: Localiser=IdentityLocaliser(), 
                imputer: Imputer=GaussianImputer(),
                nesi=True):
    """Runs EKI-DMC, as described in Iglesias and Yang (2021)."""

    C_e_invsqrt = sqrtm(inv(C_e))
    NG = len(y)

    ws_i = prior.sample(n=Ne)

    ps_i, Fs_i, Gs_i, inds_succ, inds_fail = ensemble.run(ws_i, nesi)
    
    ws = [ws_i]
    ps = [ps_i]
    Fs = [Fs_i]
    Gs = [Gs_i]
    inds = [inds_succ]

    i = 0
    t = 0
    converged = False

    while not converged:

        a_i = compute_a_dmc(t, Gs_i[:, inds_succ], y, NG, C_e_invsqrt)

        t += (a_i ** -1)
        if np.abs(t - 1.0) < TOL:
            converged = True

        ws_i = eki_update(ws_i, Gs_i, Ne, inds_succ, inds_fail, 
                          a_i, y, C_e, localiser, imputer)
        
        ps_i, Fs_i, Gs_i, inds_succ, inds_fail = ensemble.run(ws_i, nesi)
        
        ws.append(ws_i)
        ps.append(ps_i)
        Fs.append(Fs_i)
        Gs.append(Gs_i)
        inds.append(inds_succ)

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

def enrml_update(ws, Gs, ys, C_e_invsqrt, ws_pr, Uw_pr, Sw_pr, lam, 
                 inds_succ, inds_fail,
                 localiser: Localiser, imputer: Imputer):
    
    dw = compute_deltas(ws[:, inds_succ])
    dG = compute_deltas(Gs[:, inds_succ])

    UG, SG, VG = compute_tsvd(C_e_invsqrt @ dG)

    K = localiser.compute_gain_enrml(ws[:, inds_succ], Gs[:, inds_succ], 
                                     dw, UG, SG, VG, C_e_invsqrt, lam) 
    
    psi = np.diag((lam + 1 + SG**2)**-1)

    dw_pr = dw @ VG @ psi @ VG.T @ dw.T @ \
        Uw_pr @ np.diag(Sw_pr**-2) @ Uw_pr.T @ \
            (ws[:, inds_succ] - ws_pr[:, inds_succ])
    
    dw_obs = K @ (Gs[:, inds_succ] - ys[:, inds_succ])
    ws[:, inds_succ] -= (dw_pr + dw_obs) 

    if (n_fail := len(inds_fail)) > 0:
        utils.info(f"Imputing {n_fail} failed ensemble members...")
        ws = imputer.impute(ws, inds_succ, inds_fail)
    
    return ws

def compute_S(Gs, ys, C_e_invsqrt):
    """Computes mean of EnRML misfit functions for each particle."""
    Ss = np.sum((C_e_invsqrt @ (Gs - ys)) ** 2, axis=0)
    return np.mean(Ss)

def run_enrml(F, G, prior, y, C_e, Np, NF, Ne, 
              gamma=10, lam_min=0.01, 
              max_cuts=5, max_its=30, 
              dS_min=0.01, dw_min=0.5,
              localiser: Localiser=IdentityLocaliser(),
              imputer: Imputer=GaussianImputer()):
    
    C_e_invsqrt = sqrtm(inv(C_e))
    NG = len(y)

    raise Exception("Fix me!!")
    ensemble = Ensemble(prior, F, G, Np, NF, NG, Ne)
    ys = np.random.multivariate_normal(mean=y, cov=C_e, size=Ne).T

    ws_pr = prior.sample(n=Ne)
    ps_pr, Fs_pr, Gs_pr, inds_succ, inds_fail = ensemble.run(ws_pr)
    S_pr = compute_S(Gs_pr[:, inds_succ], ys[:, inds_succ], C_e_invsqrt)
    lam = 10**np.floor(np.log10(S_pr / 2*NG))
    
    ws = [ws_pr]
    ps = [ps_pr]
    Fs = [Fs_pr]
    Gs = [Gs_pr]
    Ss = [S_pr]
    lams = [lam]
    inds = [inds_succ]

    dw_pr = compute_deltas(ws_pr)
    Uw_pr, Sw_pr, _ = compute_tsvd(dw_pr)

    i = 1
    en_ind = 0
    n_cuts = 0
    while i <= max_its:

        ws_i = enrml_update(
            ws[en_ind], Gs[en_ind], ys, C_e_invsqrt, 
            ws_pr, Uw_pr, Sw_pr, lam, inds_succ, inds_fail, 
            localiser, imputer)
        
        ps_i, Fs_i, Gs_i, inds_succ, inds_fail = ensemble.run(ws_i)
        S_i = compute_S(Gs_i[:, inds_succ], ys[:, inds_succ], C_e_invsqrt)
        
        ws.append(ws_i)
        ps.append(ps_i)
        Fs.append(Fs_i)
        Gs.append(Gs_i)
        Ss.append(S_i)
        lams.append(lam)
        inds.append(inds_succ)
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

def save_results_enrml(fname: str, results: dict, post_ind: int):

    with h5py.File(fname, "w") as f:

        for key, vals in results.items():
            for i, val in enumerate(vals):
                f.create_dataset(f"{key}_{i}", data=val)

        f.create_dataset("algorithm", data="enrml")
        f.create_dataset("post_ind", data=post_ind)