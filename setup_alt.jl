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

dx = xmax / nx
dz = zmax / nz
dt = tmax / nt

xs = collect(range(dx, xmax-dx, nx))
zs = collect(range(dz, zmax-dz, nz)) .- zmax
ts = collect(range(0, tmax, nt+1))

n_blocks = nx * nz

mesh_path = "models/gSQ$n_blocks"
model_path = "models/SQ$n_blocks"

py"build_base_model"(
    xmax, ymax, zmax, nx, ny, nz,
    model_path, mesh_path)

# Upflow locations 
uf_xs = [0.5xmax]
uf_zs = [-zmax+0.5dz]
uf_locs = [(x, 0.5ymax, z) for (x, z) ∈ zip(uf_xs, uf_zs)]

# Feedzone locations
fz_xs = [200, 475, 750, 1025, 1300]
fz_zs = [-500, -500, -500, -500, -500]
fz_qs = [-2.0, -2.0, -2.0, -2.0, -2.0]
fz_locs = [(x, 0.5ymax, z) for (x, z) ∈ zip(fz_xs, fz_zs)]

fz_cells = py"get_feedzone_cells"(mesh_path, fz_locs)

# Natural state temperature observations
ts_obs_xlocs = [200, 475, 750, 1025, 1300]
ts_obs_zlocs = [-300, -500, -700, -900, -1100, -1300]
ts_obs_xs = [x for x ∈ ts_obs_xlocs for _ ∈ ts_obs_zlocs]
ts_obs_zs = [z for _ ∈ ts_obs_xlocs for z ∈ ts_obs_zlocs]

t_obs = [1, 4, 7, 10, 13]

nfz = length(fz_locs)
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

# ----------------
# Truth and data generation
# ----------------

function generate_data(ks)

    nts_raw_f = 4n_blocks
    inds_ts_raw_f = 1:nts_raw_f
    inds_ps_raw_f = (1:nps_raw) .+ nts_raw_f 
    inds_es_raw_f = (1:nes_raw) .+ nts_raw_f .+ nps_raw

    uf_locs_f = [
        (0.5xmax-0.25dx, 0.5ymax, -zmax+0.25dz), 
        (0.5xmax+0.25dx, 0.5ymax, -zmax+0.25dz)
    ]

    xs_f = collect(range(0.5dx, xmax-0.5dx, 2nx))
    zs_f = collect(range(0.5dz, zmax-0.5dz, 2nz)) .- zmax

    # Upscale permeabilities
    ks = reshape(ks, nx, nz)
    ks_f = zeros(2nx, 2nz)

    for i ∈ 1:nx, j ∈ 1:nz
        ks_f[2i-1, 2j-1] = ks[i, j]
        ks_f[2i, 2j-1] = ks[i, j]
        ks_f[2i-1, 2j] = ks[i, j]
        ks_f[2i, 2j] = ks[i, j]
    end

    mesh_path_f = "models/gSQ$(4n_blocks)"
    model_path_f = "models/SQ$(4n_blocks)"

    py"build_base_model"(
        xmax, ymax, zmax, 2nx, ny, 2nz, 
        model_path_f, mesh_path_f)

    py"build_models"(
        model_path_f, mesh_path_f, vec(ks_f), uf_locs_f, [0.5q_t, 0.5q_t], 
        fz_locs, fz_qs, tmax, dt)

    py"run_simulation"("$(model_path_f)_NS")
    flag = py"run_info"("$(model_path_f)_NS")
    flag != "success" && error("NS simulation failed. Flag: $(flag).")
    
    py"run_simulation"("$(model_path_f)_PR")
    flag = py"run_info"("$(model_path_f)_PR")
    flag != "success" && error("PR simulation failed. Flag: $(flag).")

    fz_cells_f = py"get_feedzone_cells"(mesh_path_f, fz_locs)
    us = py"get_pr_data"("$(model_path_f)_PR", nfz, fz_cells_f)

    temperatures = reshape(us[inds_ts_raw_f], 2nx, 2nz)
    temperatures = interpolate((xs_f, zs_f), temperatures, Gridded(Linear()))
    temperatures = [temperatures(x, z) for (x, z) ∈ zip(ts_obs_xs, ts_obs_zs)]

    pressures = vec(reshape(us[inds_ps_raw_f], nfz, nt+1)[:, t_obs])
    enthalpies = vec(reshape(us[inds_es_raw_f], nfz, nt+1)[:, t_obs])

    return vcat(temperatures, pressures, enthalpies)

end

θs_t = rand(p)
q_t = get_mass_rate(p, θs_t)
logks_t = get_perms(p, θs_t)
ks_t = 10 .^ logks_t

us_o = generate_data(ks_t)

py"run_simulation"("$(model_path)_NS")
flag = py"run_info"("$(model_path)_NS")
flag != "success" && @warn "NS simulation failed. Flag: $(flag)."

py"run_simulation"("$(model_path)_PR")
flag = py"run_info"("$(model_path)_PR")
flag != "success" && @warn "PR simulation failed. Flag: $(flag)."

us = py"get_pr_data"("$(model_path)_PR", nfz, fz_cells)