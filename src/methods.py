import h5py
import itertools as it
import numpy as np

from scipy.linalg import inv, sqrtm

from src import models, utils

TOL = 1e-8

# ----------------
# Imputation functions
# ----------------

class Imputer():
    pass

class GaussianImputer(Imputer):
    """Replaces any failed ensemble members by sampling from a Gaussian
    with moments constructed using the successful ensemble members."""
    
    def impute(ws, inds_succ, inds_fail):

        n_fail = len(inds_fail)

        mu = np.mean(ws[:, inds_succ], axis=1)
        cov = np.cov(ws[:, inds_succ]) + 1e-4 * np.eye(len(mu)) # TODO: read paper on this
        
        ws[:, inds_fail] = np.random.multivariate_normal(mu, cov, size=n_fail).T
        return ws
    
class ResamplingImputer(Imputer):
    """Replaces any failed ensemble members by sampling (with 
    replacement) from the successful ensemble members."""

    def impute(ws, inds_succ, inds_fail):
        inds_rep = np.random.choice(inds_succ, size=len(inds_fail))
        ws[:, inds_fail] = ws[:, inds_rep]
        return ws

# ----------------
# Localisation functions
# ----------------

class Localiser():

    def _compute_gain_eki(ws, Gs, a_i, C_e):
        C_wG, C_GG = compute_covs(ws, Gs)
        return C_wG @ inv(C_GG + a_i * C_e)

class IdentityLocaliser(Localiser):
    """Returns the Kalman gain without applying any localisation."""

    def __init__(self):
        pass

    def compute_gain_eki(self, ws, Gs, a_i, C_e):
        return self._compute_gain_eki(ws, Gs, a_i, C_e)

class BootstrapLocaliser(Localiser):
    """Carries out the bootstrap-based localisation procedure described 
    by Zhang and Oliver (2010)."""
    
    def __init__(self, N_boot=100, sigalpha=0.60, tol=1e-8):

        self.N_boot = N_boot
        self.sigalpha = sigalpha
        self.tol=tol

    def compute_gain_eki(self, ws, Gs, a_i, C_e):

        Nw, Ne = ws.shape
        NG, Ne = Gs.shape
        K = self._compute_gain_eki(ws, Gs, a_i, C_e)

        Ks_boot = np.empty((Nw, NG, self.N_boot))

        for k in range(self.N_boot):
            inds_res = np.random.choice(Ne, Ne)
            ws_res = ws[:, inds_res]
            Gs_res = Gs[:, inds_res]
            K_res = self._compute_gain_eki(ws_res, Gs_res, a_i, C_e)
            Ks_boot[:, :, k] = K_res

        var_Kis = np.mean((Ks_boot - K[:, :, np.newaxis]) ** 2, axis=2)
        Rsq = var_Kis / (K ** 2 + self.tol)
        P = 1 / (1 + Rsq * (1 + 1 / self.sigalpha ** 2))

        # TEMP (use as a check...)
        for i in range(Nw):
            for j in range(NG):
                r_sq = np.mean((Ks_boot[i, j, :] - K[i, j]) ** 2) / (K[i, j] ** 2)
                print(1.0 / (1.0 + r_sq * (1 + 1 / (self.sigalpha ** 2))))
                print(P[i, j])

        return P * K 

class CycleLocaliser(Localiser):
    """Carries out a variant of the localisation procedure described by
    Luo and Bhakta (2020)."""

    def __init__(self):
        pass # Maybe there are some parameters to define?

    def gaspari_cohn(self, z):
        if 0 <= z <= 1:
            return -(1/4)*z**5 + (1/2)*z**4 + (5/8)*z**3 - \
                (5/3)*z**2 + 1
        elif 1 < z <= 2:
            return (1/12)*z**5 - (1/2)*z**4 + (5/8)*z**3 + \
                (5/3)*z**2 - 5*z + 4 - (2/3)*z**-1
        return 0.0

    def compute_gain_eki(self, ws, Gs, a_i, C_e):

        Nw, Ne = ws.shape
        NG, Ne = Gs.shape

        K = self._compute_gain_eki(ws, Gs, a_i, C_e)
        R_wG = compute_cors(ws, Gs)[0]

        P = np.zeros((Nw, NG))
        R_wGs = np.zeros((Nw, NG, Ne-1))

        for i in range(Ne-1):
            ws_cycled = np.roll(ws, shift=i+1, axis=1)
            R_wGs[:, :, i] = compute_cors(ws_cycled, Gs)[0]

        error_sds = np.median(np.abs(R_wGs), axis=2) / 0.6745

        for i in range(Nw):
            for j in range(NG):
                z = (1-np.abs(R_wG[i, j])) / (1-error_sds[i, j])
                P[i, j] = self.gaspari_cohn(z)

        return P * K

# ----------------
# General EKI functions
# ----------------

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

def compute_covs(ws, Gs):

    Nw = ws.shape[0]
    NG = Gs.shape[0]

    C = np.cov(np.vstack((ws, Gs)))
    C_wG = C[:Nw, -NG:]
    C_GG = C[-NG:, -NG:]

    return C_wG, C_GG

def compute_cors(ws, Gs):

    Nw = ws.shape[0]
    NG = Gs.shape[0]

    R = np.corrcoef(np.vstack((ws, Gs)))
    R_wG = R[:Nw, -NG:]
    R_GG = R[-NG:, -NG:]

    return R_wG, R_GG

def eki_update(ws_i, Gs_i, Ne, inds_succ, inds_fail, a_i, y, C_e,
               localiser, imputer):
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

def run_eki_dmc(F, G, prior, y, C_e, Np, NF, Ne,
                localiser: Localiser=IdentityLocaliser, 
                imputer: Imputer=GaussianImputer):
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
        
        ps_i, Fs_i, Gs_i, inds_succ, inds_fail = ensemble.run(ws_i)
        
        ws.append(ws_i)
        ps.append(ps_i)
        Fs.append(Fs_i)
        Gs.append(Gs_i)
        inds.append(inds_succ)

        i += 1
        utils.info(f"Iteration {i} complete. t = {t:.4f}.")

    return ws, ps, Fs, Gs, inds

    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    S_cum = np.cumsum(S)
    for (i, c) in enumerate(S_cum):
        if c / S_cum[-1] >= energy:
            return U[:, :i], S[:i], Vt.T[:, :i]
        
    raise Exception("Error in TSVD function.")