using Distributions
using Interpolations
using LinearAlgebra
using PyCall
using Random
using SimIntensiveInference

include("priors.jl")

Random.seed!(1)

# TODO:
# Make a finer grid for the truth

# ----------------
# Base model setup
# ----------------

@pyinclude "model_functions.py"

secs_per_week = 60.0 * 60.0 * 24.0 * 7.0

xmax, nx = 1500.0, 25
ymax, ny = 60.0, 1
zmax, nz = 1500.0, 25
tmax, nt = 104.0 * secs_per_week, 52

dx = xmax / nx
dz = zmax / nz
dt = tmax / nt

xs = collect(range(dx, xmax-dx, nx))
zs = collect(range(dz, zmax-dz, nz)) .- zmax

n_blocks = nx * nz

model_folder = "models"
mesh_name = "gSQ$n_blocks"
model_name = "SQ$n_blocks"
mesh_path = "$model_folder/$mesh_name"
model_path = "$model_folder/$model_name"

py"build_base_model"(xmax, ymax, zmax, nx, ny, nz, model_path, mesh_path)

upflow_locs = [(0.5xmax, 0.5ymax, -zmax+0.5dz)]
feedzone_xs = [300, 600, 900, 1200]
feedzone_ys = [-500, -500, -500, -500]
feedzone_qs = [-1.0, -1.0, -1.0, -1.0]
feedzone_locs = [(x, 0.5ymax, z) for (x, z) ∈ zip(feedzone_xs, feedzone_ys)]

# Define the observation locations
temp_obs_xs = [300, 600, 900, 1200]
temp_obs_zs = [-300, -500, -700, -900, -1100, -1300]
n_obs = length(temp_obs_xs) * length(temp_obs_zs)

# ----------------
# Model functions 
# ----------------

function f(θs::AbstractVector)::Union{AbstractVector, Symbol}

    mass_rate = get_mass_rate(p, θs)
    ps = 10 .^ get_perms(p, θs)
    
    py"build_models"(
        model_path, mesh_path, ps, upflow_locs, [mass_rate], 
        feedzone_locs, feedzone_qs, tmax, dt)
    
    py"run_model"("$(model_path)_NS")

    flag = py"run_info"("$(model_path)_NS")
    flag != "success" && @warn "Model failed. Flag: $(flag)."
    flag != "success" && return :failure 

    py"run_model"("$(model_path)_PR")

    return reduce(vcat, py"get_pr_temps"("$(model_path)_PR"))

end

function g(temps::Union{AbstractVector, Symbol})

    temps == :failure && return :failure
    temps = interpolate((xs, zs), reshape(temps, nx, nz), Gridded(Linear()))
    return [temps(x, z) for (x, z) ∈ zip(xs_o, zs_o)]

end

# ----------------
# Prior setup
# ----------------

mass_rate_bnds = [1.0e-1, 2.0e-1]
depth_shal = -100.0

μ_depth_clay = -300.0
k_depth_clay = ExpSquaredKernel(80, 500)

μ_perm_shal = -14.0
μ_perm_clay = -16.0
μ_perm_deep = -14.0
k_perm_shal = ARDExpSquaredKernel(0.25, 1000, 250)
k_perm_clay = ARDExpSquaredKernel(0.25, 1000, 250)
k_perm_deep = ARDExpSquaredKernel(0.50, 1000, 250)

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

# ----------------
# Data generation 
# ----------------

# Generate the true set of parameters and outputs
θs_t = rand(p)
logps_t = get_perms(p, θs_t)
ps_t = 10 .^ logps_t
us_t = @time f(vec(θs_t))
us_t = reshape(us_t, nx, nz, nt+1)

error("Stop.")

# Define the distribution of the observation noise
σ_ϵ_t = 2.0
Γ_ϵ = σ_ϵ_t^2 * Matrix(1.0I, n_obs, n_obs)
ϵ_dist = MvNormal(Γ_ϵ)

# Generate the data and add noise
us_t = interpolate((xs, zs), us_t, Gridded(Linear()))
xs_o = [x for x ∈ x_locs for _ ∈ z_locs]
zs_o = [z for _ ∈ x_locs for z ∈ z_locs]
us_o = [us_t(x, z) for (x, z) ∈ zip(xs_o, zs_o)] + rand(ϵ_dist)

# ----------------
# Likelihood setup 
# ----------------

σ_ϵ = 2.0
Γ_ϵ = σ_ϵ^2 * Matrix(1.0I, n_obs, n_obs)
L = MvNormal(us_o, Γ_ϵ)

# py"slice_plot"(model_folder, mesh_name, logps_t, cmap="turbo")