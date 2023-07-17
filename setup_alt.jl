using BlockDiagonals
using Distributions 
using Interpolations
using LinearAlgebra
using PyCall
using Random 
using SimIntensiveInference

include("priors.jl")
@pyinclude "model_functions.py"

Random.seed!(4)

secs_per_week = 60.0 * 60.0 * 24.0 * 7.0

xmax, nx = 1500.0, 25
ymax, ny = 50.0, 1
zmax, nz = 1500.0, 25
tmax, nt = 104.0 * secs_per_week, 24

nx_f = 2nx
ny_f = ny 
nz_f = 2nz

dx, dx_f = xmax / nx, xmax / nx_f 
dz, dz_f = zmax / nz, zmax / nz_f 
dt = tmax / nt

xs = collect(range(dx, xmax-dx, nx))
zs = collect(range(dz, zmax-dz, nz)) .- zmax

xs_f = collect(range(dx_f, xmax-dx_f, nx_f))
zs_f = collect(range(dz_f, zmax-dz_f, nz_f)) .- zmax

ts = collect(range(0, tmax, nt+1))

n_blocks = nx * nz

mesh_path = "models/gSQ$n_blocks"
model_path = "models/SQ$n_blocks"

mesh_path_f = "models/gSQ$(4n_blocks)"
model_path_f = "models/SQ$(4n_blocks)"

py"build_base_model"(
    xmax, ymax, zmax, nx, ny, nz,
    model_path, mesh_path)

py"build_base_model"(
    xmax, ymax, zmax, nx_f, ny_f, nz_f, 
    model_path_f, mesh_path_f)

# Upflow locations 
uf_xs = [0.5xmax]
uf_zs = [-zmax+0.5dz]
uf_locs = [(x, 0.5ymax, z) for (x, z) ∈ zip(uf_xs, uf_zs)]

uf_xs_f = [0.5xmax-0.5dx_f, 0.5xmax+0.5dx_f]
uf_zs_f = [-zmax+0.5dz_f, -zmax+0.5dz_f]
uf_locs_f = [(x, 0.5ymax, z) for (x, z) ∈ zip(uf_xs_f, uf_zs_f)]

# Feedzone locations
fz_xs = [200, 475, 750, 1025, 1300]
fz_zs = [-500, -500, -500, -500, -500]
fz_qs = [-2.0, -2.0, -2.0, -2.0, -2.0]
fz_locs = [(x, 0.5ymax, z) for (x, z) ∈ zip(fz_xs, fz_zs)]

fz_cells = py"get_feedzone_cells"(mesh_path, fz_locs)
fz_cells_f = py"get_feedzone_cells"(mesh_path_f, fz_locs)

# Natural state temperature observations
ts_obs_xlocs = [200, 475, 750, 1025, 1300]
ts_obs_zlocs = [-300, -500, -700, -900, -1100, -1300]
ts_obs_xs = [x for x ∈ ts_obs_xlocs for _ ∈ ts_obs_zlocs]
ts_obs_zs = [z for _ ∈ ts_obs_xlocs for z ∈ ts_obs_zlocs]

t_obs = [1, 4, 7, 10, 13]

nfz = length(fz_locs)
nt_obs = length(t_obs)

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

# ----------------
# Prior 
# ----------------

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
    mass_rate_bnds, depth_shal,
    μ_depth_clay, k_depth_clay,
    μ_perm_shal, μ_perm_clay, μ_perm_deep,
    k_perm_shal, k_perm_clay, k_perm_deep, 
    level_width, 
    xs, reverse(zs)
)

θs = rand(p)
q = get_mass_rate(p, θs)
logks = get_perms(p, θs)
ks = reshape(10 .^ logks, nx, nz)

ks_f = zeros(nx_f, nz_f)

for i ∈ 1:nx, j ∈ 1:nz
    ks_f[2i-1, 2j-1] = ks[i, j]
    ks_f[2i, 2j-1] = ks[i, j]
    ks_f[2i-1, 2j] = ks[i, j]
    ks_f[2i, 2j] = ks[i, j]
end

py"build_models"(
    model_path, mesh_path, vec(ks), uf_locs, [q], 
    fz_locs, fz_qs, tmax, dt)

py"build_models"(
    model_path_f, mesh_path_f, vec(ks_f), uf_locs_f, [0.5q, 0.5q], 
    fz_locs, fz_qs, tmax, dt)

py"run_simulation"("$(model_path)_NS")
flag = py"run_info"("$(model_path)_NS")
flag != "success" && @warn "NS simulation failed. Flag: $(flag)."

py"run_simulation"("$(model_path)_PR")
flag = py"run_info"("$(model_path)_PR")
flag != "success" && @warn "PR simulation failed. Flag: $(flag)."

py"run_simulation"("$(model_path_f)_NS")
flag = py"run_info"("$(model_path_f)_NS")
flag != "success" && @warn "NS simulation failed. Flag: $(flag)."

py"run_simulation"("$(model_path_f)_PR")
flag = py"run_info"("$(model_path_f)_PR")
flag != "success" && @warn "PR simulation failed. Flag: $(flag)."

us = py"get_pr_data"("$(model_path)_PR", nfz, fz_cells)
us_f = py"get_pr_data"("$(model_path_f)_PR", nfz, fz_cells)