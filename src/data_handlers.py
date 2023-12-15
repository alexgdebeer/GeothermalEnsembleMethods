from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

from src.consts import *
from src.models import *

class DataHandler(ABC):

    def __init__(self, mesh: Mesh, wells, temp_obs_cs,
                 prod_obs_ts, tmax, nt):

        self.mesh = mesh
        self.cell_cs = np.array([c.centre for c in mesh.m.cell])

        self.wells = wells
        self.n_wells = len(wells)
        self.feedzone_coords = np.array([w.feedzone_cell.centre for w in wells])

        self.tmax = tmax 
        self.ts = np.linspace(0, tmax, nt+1)
        self.nt = nt

        self.temp_obs_cs = temp_obs_cs
        self.prod_obs_ts = prod_obs_ts

        self.inds_prod_obs = np.searchsorted(self.ts, prod_obs_ts-EPS)
        self.n_prod_obs_ts = len(self.prod_obs_ts)

        self.n_temp_full = self.mesh.m.num_cells
        self.n_pres_full = (self.nt+1) * self.n_wells
        self.n_enth_full = (self.nt+1) * self.n_wells

        self.n_temp_obs = len(self.temp_obs_cs)
        self.n_pres_obs = self.n_prod_obs_ts * self.n_wells
        self.n_enth_obs = self.n_prod_obs_ts * self.n_wells

        self.generate_inds_full()
        self.generate_inds_obs()

    def generate_inds_full(self):
        """Generates indices used to extract temperatures, pressures 
        and enthalpies from a vector of complete data."""
        self.inds_full_temp = np.arange(self.n_temp_full)
        self.inds_full_pres = np.arange(self.n_pres_full) + 1 + self.inds_full_temp[-1]
        self.inds_full_enth = np.arange(self.n_enth_full) + 1 + self.inds_full_pres[-1]

    def generate_inds_obs(self):
        """Generates indices used to extract temperatures, pressures 
        and enthalpy observations from a vector of observations."""
        self.inds_obs_temp = np.arange(self.n_temp_obs)
        self.inds_obs_pres = np.arange(self.n_pres_obs) + 1 + self.inds_obs_temp[-1]
        self.inds_obs_enth = np.arange(self.n_enth_obs) + 1 + self.inds_obs_pres[-1]

    def reshape_to_wells(self, obs):
        """Reshapes a 1D array of observations such that each column 
        contains the observations for a single well."""
        return np.reshape(obs, (-1, self.n_wells))

    @abstractmethod
    def get_full_temperatures(self, F_i):
        raise NotImplementedError()
    
    def get_full_pressures(self, F_i):
        pres = F_i[self.inds_full_pres]
        return self.reshape_to_wells(pres)

    def get_full_enthalpies(self, F_i):
        enth = F_i[self.inds_full_enth]
        return self.reshape_to_wells(enth)
    
    def get_full_states(self, F_i):
        temp = self.get_full_temperatures(F_i)
        pres = self.get_full_pressures(F_i)
        enth = self.get_full_enthalpies(F_i)
        return temp, pres, enth 
    
    @abstractmethod
    def get_obs_temperatures(self):
        raise NotImplementedError()
    
    def get_obs_pressures(self, pres_full):
        return pres_full[self.inds_prod_obs, :]

    def get_obs_enthalpies(self, enth_full):
        return enth_full[self.inds_prod_obs, :]
    
    def get_obs_states(self, F_i):
        """Extracts the observations from a complete vector of model 
        output, and returns each type of observation individually."""
        temp_full, pres_full, enth_full = self.get_full_states(F_i)
        temp_obs = self.get_obs_temperatures(temp_full)
        pres_obs = self.get_obs_pressures(pres_full)
        enth_obs = self.get_obs_enthalpies(enth_full)
        return temp_obs, pres_obs, enth_obs
    
    def get_obs(self, F_i):
        """Extracts the observations from a complete vector of model
        output, and returns them as a vector."""
        temp_obs, pres_obs, enth_obs = self.get_obs_states(F_i)
        obs = np.concatenate((temp_obs.flatten(), 
                              pres_obs.flatten(), 
                              enth_obs.flatten()))
        return obs

    def split_obs(self, G_i):
        """Splits a set of observations into temperatures, pressures 
        and enthalpies."""
        temp_obs = self.reshape_to_wells(G_i[self.inds_obs_temp])
        pres_obs = self.reshape_to_wells(G_i[self.inds_obs_pres])
        enth_obs = self.reshape_to_wells(G_i[self.inds_obs_enth])
        return temp_obs, pres_obs, enth_obs
    
    @abstractmethod
    def downhole_temps(self):
        raise NotImplementedError()

class DataHandler2D(DataHandler):

    def get_full_temperatures(self, F_i):
        temp = F_i[self.inds_full_temp]
        temp = np.reshape(temp, (self.mesh.nz, self.mesh.nx)) # TODO: check
        return temp

    def get_obs_temperatures(self, temp_full):
        """Extracts the temperatures at each observation point from a 
        full set of temperatures."""
        temp_full = np.reshape(temp_full, (self.mesh.nz, self.mesh.nx)) 
        mesh_coords = (self.mesh.xs, self.mesh.zs)
        interpolator = RegularGridInterpolator(mesh_coords, temp_full.T)
        temp_obs = interpolator(self.temp_obs_cs) # TODO: check this
        return self.reshape_to_wells(temp_obs)
    
    def downhole_temps(self, temps):
        """Generates the downhole temperatures for a given well.
        TODO: cut off at well depths?"""
        mesh_coords = (self.mesh.xs, self.mesh.zs)
        interpolator = RegularGridInterpolator(mesh_coords, temps.T)

        downhole_coords = np.array([[w.x, z] 
                                    for z in self.mesh.zs
                                    for w in self.wells])
        
        temp_well = interpolator(downhole_coords)
        return self.reshape_to_wells(temp_well)

class DataHandler3D(DataHandler):

    def get_full_temperatures(self, F_i):
        temp = F_i[self.inds_full_temp]
        return temp

    def get_obs_temperatures(self, temp_full):
        """Extracts the temperatures at each observation point from a 
        full set of temperatures."""
        interp = LinearNDInterpolator(self.mesh.tri, temp_full)
        temp_obs = interp(self.temp_obs_cs)
        return self.reshape_to_wells(temp_obs)

    def downhole_temps(self, temp_full, well_num):
        """Returns interpolated temperatures down a single well."""

        well = self.wells[well_num]
        elevs = [l.centre for l in self.mesh.m.layer
                 if well.min_elev <= l.centre <= well.max_elev]
        if well.min_elev not in elevs:
            elevs.append(well.min_elev)
        
        coords = np.array([[well.x, well.y, e] for e in elevs])
        interp = LinearNDInterpolator(self.mesh.tri, temp_full)
        downhole_temps = interp(coords)
        return elevs, downhole_temps
