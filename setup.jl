using Distributions
using Interpolations
using LinearAlgebra
using PyCall
using Random
using SimIntensiveInference

include("priors.jl")

Random.seed!(1)

# TODO: extend things to production history
# Make a finer grid for the truth
# Initial condition stuff?

# ----------------
# Base model setup
# ----------------

@pyinclude "model_functions.py"

xmax, nx = 1500.0, 25
ymax, ny = 60.0, 1
zmax, nz = 1500.0, 25

dx = xmax / nx
dz = zmax / nz

xs = collect(range(dx, xmax-dx, nx))
zs = collect(range(dz, zmax-dz, nz))

n_blocks = nx * nz

model_folder = "models"
mesh_name = "gSQ$n_blocks"
model_name = "SQ$n_blocks"
mesh_path = "$model_folder/$mesh_name"
model_path = "$model_folder/$model_name"

py"build_base_model"(xmax, ymax, zmax, nx, ny, nz, model_path, mesh_path)

# ----------------
# Model functions 
# ----------------

function f(θs::AbstractVector)::Union{AbstractVector, Symbol}

    mass_rate = get_mass_rate(p, θs)
    ps = 10 .^ get_perms(p, θs)
    
    py"build_models"(
        model_path, mesh_path, ps, 
        upflow_locs, [mass_rate], 
        feedzone_locs, feedzone_rates)
    
    py"run_model"("$(model_path)_NS")
    py"run_model"("$(model_path)_PR")

    flag = py"run_info"("$(model_path)_NS")
    flag != "success" && @warn "Model failed. Flag: $(flag)."
    flag != "success" && return :failure 

    temps = py"get_quantity"("$(model_path)_NS", "fluid_temperature")
    return temps

end

function g(temps::Union{AbstractVector, Symbol})

    temps == :failure && return :failure
    temps = interpolate((xs, zs), reshape(temps, nx, nz), Gridded(Linear()))
    return [temps(x, z) for (x, z) ∈ zip(xs_o, zs_o)]

end

# ----------------
# Prior setup
# ----------------

mass_rate_bnds = [0.5e-2, 1.5e-2]
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
    xs, -zs
)

# ----------------
# Data generation 
# ----------------

upflow_locs = [(xmax/2.0, ymax/2.0, -zmax+dz/2.0)]
feedzone_locs = [(x, ymax/2.0, -zmax/2.0) for x in [250, 500, 750, 1000, 1250]]
feedzone_rates = [0.1e-2 for _ in 1:5]

# Generate the true set of parameters and outputs
θs_t = rand(p)
logps_t = get_perms(p, θs_t)
ps_t = 10 .^ logps_t
us_t = @time reshape(f(vec(θs_t)), nx, nz)

# Define the observation locations
x_locs = 300:300:1200
z_locs = 300:200:1300
n_obs = length(x_locs) * length(z_locs)

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

# py"slice_plot"(model_folder, mesh_name, logps_t[1:n_blocks], cmap="turbo")