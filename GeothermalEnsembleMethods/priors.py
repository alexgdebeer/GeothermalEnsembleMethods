from abc import ABC, abstractmethod

import numpy as np
from scipy import stats

from .models import *


class Prior(ABC):

    @abstractmethod
    def sample():
        pass

    @abstractmethod
    def transform():
        pass


class SlicePrior(Prior):

    def __init__(self, mesh, depth_shal, gp_boundary, 
                 grf_shal, grf_clay, grf_deep,
                 mass_rate_bounds):

        self.mesh = mesh

        self.depth_shal = depth_shal
        self.gp_boundary = gp_boundary
        
        self.grf_shal = grf_shal
        self.grf_clay = grf_clay
        self.grf_deep = grf_deep

        self.mass_rate_bounds = mass_rate_bounds

        self.param_counts = [0, gp_boundary.n_params, 
                             grf_shal.n_params, grf_clay.n_params, 
                             grf_deep.n_params, 1]

        self.n_params = sum(self.param_counts)
        self.param_inds = np.cumsum(self.param_counts)

        self.inds = {
            "boundary" : np.arange(*self.param_inds[0:2]),
            "grf_shal" : np.arange(*self.param_inds[1:3]),
            "grf_clay" : np.arange(*self.param_inds[2:4]),
            "grf_deep" : np.arange(*self.param_inds[3:5])
        }

    def get_hyperparams(self, ws):
        hps_shal = self.grf_shal.get_hyperparams(ws[self.inds["grf_shal"]])
        hps_clay = self.grf_clay.get_hyperparams(ws[self.inds["grf_clay"]])
        hps_deep = self.grf_deep.get_hyperparams(ws[self.inds["grf_deep"]])
        return hps_shal, hps_clay, hps_deep

    def combine_perms(self, boundary, perms_shal, perms_clay, perms_deep):

        perms = np.zeros((perms_shal.shape))
        for i, cell in enumerate(self.mesh.m.cell):

            cx, _, cz = cell.centre
            x_ind = np.abs(self.gp_boundary.xs - cx).argmin()

            if cz > self.depth_shal:
                perms[i] = perms_shal[i]
            elif cz > boundary[x_ind]:
                perms[i] = perms_clay[i]
            else: 
                perms[i] = perms_deep[i]

        return perms

    def transform_mass_rate(self, mass_rate):
        return self.mass_rate_bounds[0] + \
            np.ptp(self.mass_rate_bounds) * stats.norm.cdf(mass_rate)

    def transform(self, ws):

        ws = np.squeeze(ws)

        perms_shal = self.grf_shal.get_perms(ws[self.inds["grf_shal"]])
        perms_clay = self.grf_clay.get_perms(ws[self.inds["grf_clay"]])
        perms_deep = self.grf_deep.get_perms(ws[self.inds["grf_deep"]])

        perms_shal = self.grf_shal.level_set(perms_shal)
        perms_clay = self.grf_clay.level_set(perms_clay)
        perms_deep = self.grf_deep.level_set(perms_deep)

        boundary = self.gp_boundary.transform(ws[self.inds["boundary"]])
        perms = self.combine_perms(boundary, perms_shal, perms_clay, perms_deep)

        mass_rate = ws[-1]
        mass_rate = self.transform_mass_rate(mass_rate)
        
        ps = np.append(perms, mass_rate)
        return ps

    def sample(self, n=1):
        return np.random.normal(size=(self.n_params, n))


class FaultPrior(Prior):

    def __init__(self, mesh: IrregularMesh, cap: ClayCap, 
                 fault: Fault, grf_ext: PermField, 
                 grf_flt: PermField, grf_cap: PermField, 
                 grf_upflow: UpflowField, ls_upflows):

        self.mesh = mesh 

        self.cap = cap
        self.fault = fault
        self.grf_ext = grf_ext 
        self.grf_flt = grf_flt
        self.grf_cap = grf_cap
        self.grf_upflow = grf_upflow

        self.compute_upflow_weightings(ls_upflows)

        self.param_counts = [0, cap.n_params, fault.n_params, 
                             grf_ext.n_params, grf_flt.n_params, 
                             grf_cap.n_params, grf_upflow.n_params]
        
        self.n_params = sum(self.param_counts)
        self.param_inds = np.cumsum(self.param_counts)

        self.inds = {
            "cap"        : np.arange(*self.param_inds[0:2]),
            "fault"      : np.arange(*self.param_inds[1:3]),
            "grf_ext"    : np.arange(*self.param_inds[2:4]),
            "grf_flt"    : np.arange(*self.param_inds[3:5]),
            "grf_cap"    : np.arange(*self.param_inds[4:6]),
            "grf_upflow" : np.arange(*self.param_inds[5:7])
        }

    def get_hyperparams(self, ws):
        hps_ext = self.grf_ext.get_hyperparams(ws[self.inds["grf_ext"]])
        hps_flt = self.grf_flt.get_hyperparams(ws[self.inds["grf_flt"]])
        hps_cap = self.grf_cap.get_hyperparams(ws[self.inds["grf_cap"]])
        return hps_ext, hps_flt, hps_cap

    def compute_upflow_weightings(self, lengthscale):

        mesh_centre = self.mesh.m.centre[:2]
        col_centres = np.array([c.centre for c in self.mesh.m.column])
        
        dxy = col_centres - mesh_centre
        ds = np.sum(-((dxy / lengthscale) ** 2), axis=1)

        upflow_weightings = np.exp(ds)
        self.upflow_weightings = upflow_weightings

    def transform(self, params):

        params = np.squeeze(params)

        cap_cell_inds = self.cap.get_cells_in_cap(params[self.inds["cap"]])

        perms_ext = self.grf_ext.get_perms(params[self.inds["grf_ext"]])
        perms_flt = self.grf_flt.get_perms(params[self.inds["grf_flt"]])
        perms_cap = self.grf_cap.get_perms(params[self.inds["grf_cap"]])

        perms_ext = self.grf_ext.level_set(perms_ext)
        perms_flt = self.grf_flt.level_set(perms_flt)
        perms_cap = self.grf_cap.level_set(perms_cap)

        fault_cells, fault_cols = self.fault.get_cells_in_fault(params[self.inds["fault"]])
        fault_cell_inds = [c.index for c in fault_cells]
        fault_col_inds = [c.index for c in fault_cols]

        perms = np.copy(perms_ext)
        perms[fault_cell_inds] = perms_flt[fault_cell_inds]
        perms[cap_cell_inds] = perms_cap[cap_cell_inds]

        upflow_rates = self.grf_upflow.get_upflows(params[self.inds["grf_upflow"]])
        upflow_rates *= self.upflow_weightings
        
        upflows = np.zeros(self.mesh.m.num_columns)
        upflows[fault_col_inds] = upflow_rates[fault_col_inds]

        return np.concatenate((perms, upflows))

    def split(self, p_i):

        logks = p_i[:-self.mesh.m.num_columns]
        mass_rates_t = p_i[-self.mesh.m.num_columns:]

        upflows = []
        for rate, col in zip(mass_rates_t, self.mesh.m.column):
            if rate > 0:
                upflow = MassUpflow(col.cell[-1], rate)
                upflows.append(upflow)
        
        return logks, upflows

    def sample(self, n=1):
        return np.random.normal(size=(self.n_params, n))
