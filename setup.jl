using BlockDiagonals
using Distributions
using Interpolations
using LinearAlgebra
using PyCall
using Random
using SimIntensiveInference

include("priors.jl")
@pyinclude "model_functions.py"

# TODO:
# Make a finer grid for the truth

# Random.seed!(10) # 12, 17
Random.seed!(0)

secs_per_week = 60.0 * 60.0 * 24.0 * 7.0

xmax, nx = 1500.0, 25
ymax, ny = 60.0, 1
zmax, nz = 1500.0, 25
tmax, nt = 104.0 * secs_per_week, 24

dx = xmax / nx
dz = zmax / nz
dt = tmax / nt

xs = collect(range(dx, xmax-dx, nx))
zs = collect(range(dz, zmax-dz, nz)) .- zmax
ts = collect(range(0, tmax, nt+1))

n_blocks = nx * nz

model_folder = "models"
mesh_name = "gSQ$n_blocks"
model_name = "SQ$n_blocks"
mesh_path = "$model_folder/$mesh_name"
model_path = "$model_folder/$model_name"

py"build_base_model"(xmax, ymax, zmax, nx, ny, nz, model_path, mesh_path)

upflow_xs = [0.5xmax]
upflow_zs = [-zmax+0.5dz]
upflow_locs = [(x, 0.5ymax, z) for (x, z) ∈ zip(upflow_xs, upflow_zs)]

fz_xs = [200, 475, 750, 1025, 1300]
fz_zs = [-500, -500, -500, -500, -500]
fz_qs = [-2.0, -2.0, -2.0, -2.0, -2.0]
fz_locs = [(x, 0.5ymax, z) for (x, z) ∈ zip(fz_xs, fz_zs)]
fz_cells = py"get_feedzone_cells"(mesh_path, fz_locs)

ts_obs_xlocs = [200, 475, 750, 1025, 1300]
ts_obs_zlocs = [-300, -500, -700, -900, -1100, -1300]
ts_obs_xs = [x for x ∈ ts_obs_xlocs for _ ∈ ts_obs_zlocs]
ts_obs_zs = [z for _ ∈ ts_obs_xlocs for z ∈ ts_obs_zlocs]

t_obs = [1, 4, 7, 10, 13]

nfz = length(fz_cells)
nt_obs = length(t_obs)

# Define indices for extracting raw data and observations
nts_raw = n_blocks 
nps_raw = nfz * (nt+1)
nes_raw = nfz * (nt+1)

n_obs_raw = nts_raw + nps_raw + nes_raw

inds_ts_raw = 1:nts_raw
inds_ps_raw = (1:nps_raw) .+ nts_raw 
inds_es_raw = (1:nes_raw) .+ nts_raw .+ nps_raw

nts_obs = length(ts_obs_xs)
nps_obs = nfz * nt_obs
nes_obs = nfz * nt_obs

inds_ts_obs = 1:nts_obs
inds_ps_obs = (1:nps_obs) .+ nts_obs 
inds_es_obs = (1:nes_obs) .+ nts_obs .+ nps_obs

# Define the distribution of the observation noise
σϵ_ts = 2.0
σϵ_ps = 1.0e+5
σϵ_es = 1.0e+4

Γ_ts = σϵ_ts^2 * Matrix(1.0I, nts_obs, nts_obs)
Γ_ps = σϵ_ps^2 * Matrix(1.0I, nps_obs, nps_obs)
Γ_es = σϵ_es^2 * Matrix(1.0I, nes_obs, nes_obs)

Γ_ϵ = BlockDiagonal([Γ_ts, Γ_ps, Γ_es])
ϵ_dist = MvNormal(Γ_ϵ)

get_raw_temperatures(us) = reshape(us[inds_ts_raw], nx, nz)
get_raw_pressures(us) = reshape(us[inds_ps_raw], nfz, nt+1)
get_raw_enthalpies(us) = reshape(us[inds_es_raw], nfz, nt+1)

"""Runs a combined natural state and production simulation, and returns a 
complete list of steady-state temperatures and transient pressures and 
enthalpies."""
function f(θs::AbstractVector)::Union{AbstractVector, Symbol}

    upflow_q = get_mass_rate(p, θs)
    ks = 10 .^ get_perms(p, θs)
    
    py"build_models"(
        model_path, mesh_path, ks, upflow_locs, [upflow_q], 
        fz_locs, fz_qs, tmax, dt)
    
    py"run_simulation"("$(model_path)_NS")
    flag = py"run_info"("$(model_path)_NS")
    flag != "success" && @warn "NS simulation failed. Flag: $(flag)."
    flag != "success" && return :failure 

    py"run_simulation"("$(model_path)_PR")
    flag = py"run_info"("$(model_path)_PR")
    flag != "success" && @warn "PR simulation failed. Flag: $(flag)."
    flag != "success" && return :failure 

    us = py"get_pr_data"("$(model_path)_PR", nfz, fz_cells)
    return us

end

"""Extracts the observations from the complete output of a simulation."""
function g(us::Union{AbstractVector, Symbol})

    us == :failure && return :failure

    temperatures = get_raw_temperatures(us)
    temperatures = interpolate((xs, zs), temperatures, Gridded(Linear()))
    temperatures = [temperatures(x, z) for (x, z) ∈ zip(ts_obs_xs, ts_obs_zs)]

    pressures = vec(get_raw_pressures(us)[:, t_obs])
    enthalpies = vec(get_raw_enthalpies(us)[:, t_obs])

    return vcat(temperatures, pressures, enthalpies)

end

# Define prior parameters
mass_rate_bnds = [1.0e-1, 2.0e-1]
depth_shal = -100.0

μ_depth_clay = -300.0
k_depth_clay = ExpSquaredKernel(80, 500)

μ_perm_shal = -14.0
μ_perm_clay = -16.0
μ_perm_deep = -14.0
k_perm_shal = ARDExpSquaredKernel(0.25, 1500, 200)
k_perm_clay = ARDExpSquaredKernel(0.25, 1500, 200)
k_perm_deep = ARDExpSquaredKernel(0.50, 1500, 200)

level_width = 0.25

p = GeothermalPrior(
    mass_rate_bnds, 
    depth_shal,
    μ_depth_clay, k_depth_clay,
    μ_perm_shal, μ_perm_clay, μ_perm_deep,
    k_perm_shal, k_perm_clay, k_perm_deep, 
    level_width, 
    xs, reverse(zs)
)

# Generate the true set of parameters and outputs
θs_t = rand(p)
q_t = get_mass_rate(p, θs_t)
logks_t = get_perms(p, θs_t)
ks_t = 10 .^ logks_t

us_t = @time f(vec(θs_t))
us_o = g(us_t) + rand(ϵ_dist)

# Set up the likelihood
L = MvNormal(us_o, Γ_ϵ)

# py"slice_plot"(model_folder, mesh_name, logps_t, cmap="turbo")