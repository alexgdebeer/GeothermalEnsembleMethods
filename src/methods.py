import h5py
import numpy as np

from abc import ABC, abstractmethod
from scipy.linalg import inv, sqrtm

from src import models, utils

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
        cov = np.cov(ws[:, inds_succ]) + 1e-4 * np.eye(len(mu)) # TODO: read paper on this
        
        ws[:, inds_fail] = np.random.multivariate_normal(mu, cov, size=n_fail).T
        return ws

class ResamplingImputer(Imputer):
    """Replaces any failed ensemble members by sampling (with 
    replacement) from the successful ensemble members."""

    def impute(self, ws, inds_succ, inds_fail):
        inds_rep = np.random.choice(inds_succ, size=len(inds_fail))
        ws[:, inds_fail] = ws[:, inds_rep]
        return ws

class Localiser(ABC):

    @abstractmethod
    def compute_gain_eki(self):
        pass

    def _compute_gain_eki(self, ws, Gs, a_i, C_e):
        C_wG, C_GG = compute_covs(ws, Gs)
        return C_wG @ inv(C_GG + a_i * C_e)

class IdentityLocaliser(Localiser):
    """Computes the Kalman gain without using localisation."""

    def compute_gain_eki(self, ws, Gs, a_i, C_e):
        return self._compute_gain_eki(ws, Gs, a_i, C_e)

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

        var_Kis = np.mean((Ks_boot - K[:, :, np.newaxis]) ** 2, axis=2)
        Rsq = var_Kis / np.maximum(K ** 2, self.tol)
        P = 1 / (1 + Rsq * (1 + 1 / self.sigalpha ** 2))

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

    def compute_gain_eki(self, ws, Gs, a_i, C_e):

        K = self._compute_gain_eki(ws, Gs, a_i, C_e)

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
                localiser: Localiser=IdentityLocaliser(), 
                imputer: Imputer=GaussianImputer()):
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

def compute_tsvd(A, energy=0.99):
    
    U, S, Vt = np.linalg.svd(A)
    V = Vt.T

    if np.minimum(S) < -TOL:
        raise Exception("Negative eigenvalue encountered in TSVD.")

    n_eigvals = len(S)
    eig_cum = np.cumsum(S)

    for i in range(n_eigvals):
        if eig_cum[i] / eig_cum[-1] >= energy:
            return U[:, 1:i], S[1:i], V[:, 1:i]
    
    raise Exception("Issue with TSVD computation.")

def enrml_update(ws, Gs, ys, C_e_invsqrt, ws_pr, Uw_pr, Sw_pr, lam, 
                 localiser, imputer):
    
    pass

def compute_S():
    pass # TODO: write

# Need some additional parameters
def run_enrml(F, G, prior, y, C_e, Np, NF, Ne, 
              gamma=10, lam_min=0.01, 
              max_cuts=5, max_its=30, 
              dS_min=0.01, dw_min=0.5,
              localiser: Localiser=IdentityLocaliser(),
              imputer: Imputer=GaussianImputer()):
    
    C_e_invsqrt = sqrtm(inv(C_e))
    NG = len(y)

    ensemble = EnsembleRunner(prior, F, G, Np, NF, NG, Ne)

    ws_pr = prior.sample(n=Ne)
    ps_pr, Fs_pr, Gs_pr, inds_succ_pr, inds_fail_pr = ensemble.run(ws_i)
    S_pr = compute_S(Gs_pr, ys, C_e_invsqrt) # TODO: define S
    lam = 10**np.floor(np.log10(S_pr / 2*NG))
    
    ws = [ws_pr]
    ps = [ps_pr]
    Fs = [Fs_pr]
    Gs = [Gs_pr]
    Ss = [S_pr]
    lams = [lam]
    inds = [inds_succ_pr]

    ys = np.random.multivariate_normal(mean=y, cov=C_e, size=Ne).T

    dw_pr = compute_deltas(ws_pr)
    Uw_pr, Sw_pr, _ = compute_tsvd(dw_pr)

    i = 0
    en_ind = 0
    n_cuts = 0
    while i <= max_its:

        ws_i = enrml_update(
            ws[en_ind], Gs[en_ind], ys, C_e_invsqrt, 
            ws_pr, Uw_pr, Sw_pr, lam, localiser, imputer) # Probably need inds succ and inds fail in here...
        
        ps_i, Fs_i, Gs_i, inds_succ, inds_fail = ensemble.run(ws_i)
        S_i = compute_S(Gs_i, ys, C_e_invsqrt)
        
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

            en_ind = i
            n_cuts = 0 
            lam = max(lam / gamma, lam_min)

            # TODO: print some stuff

            if (dS < dS_min) and (dw_max < dw_min):
                utils.info("Convergence criteria met.")
                return ws, ps, Fs, Gs, Ss, lams, en_ind, inds_succ 
            
        else:

            # TODO: print some stuff
            n_cuts += 1
            lam *= gamma 

            if n_cuts == max_cuts:
                utils.info(f"Terminating: {n_cuts} consecutive cuts made.")
                return ws, ps, Fs, Gs, Ss, lams, en_ind, inds_succ

    utils.info("Terminating: maximum number of iterations exceeded.")
    return ws, ps, Fs, Gs, Ss, lams, en_ind, inds_succ 

def save_results(fname: str, results: dict):

    with h5py.File(fname, "w") as f:
        for (k, v) in results.items():
            f.create_dataset(k, data=v)