using BlockDiagonals
using JLD2
include("setup.jl")
include("ensemble_methods.jl")

# Define localisation kernel

lb = 500
lx = 1500
lz = 200

ds_bound = (p.xs .- p.xs').^2 ./ lb^2
ds_shal = (p.cxs_shal .- p.cxs_shal').^2 ./ lx^2 + (p.cys_shal .- p.cys_shal').^2 ./ lz^2
ds_deep = (p.cxs_deep .- p.cxs_deep').^2 ./ lx^2 + (p.cys_deep .- p.cys_deep').^2 ./ lz^2

Φ_bound = exp.(-0.5 * ds_bound)
Φ_shal = exp.(-0.5 * ds_shal)
Φ_clay = exp.(-0.5 * ds_deep)
Φ_deep = exp.(-0.5 * ds_deep)

Φ = BlockDiagonal([[1.0][:,:], Φ_bound, Φ_shal, Φ_clay, Φ_deep])

Ni = 8
Ne = 50
θs, fs, us, αs, inds = SimIntensiveInference.run_es_mda_test(
    f, g, p, L, n_obs_raw, Ni, Ne, Φ)

logks_post = reduce(hcat, get_perms(p, θ) for θ ∈ eachcol(θs[:,inds,end]))
μ_post = reshape(mean(logks_post, dims=2), nx, nz)
σ_post = reshape( std(logks_post, dims=2), nx, nz)