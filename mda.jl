using JLD2
include("setup.jl")
include("ensemble_methods.jl")

Ni = 4
Ne = 25
θs, fs, us, αs, inds = run_es_mda_geothermal(
    f, g, p, L, n_obs_raw, Ni, Ne, localisation=true)#, α_method=:constant)

logks_post = reduce(hcat, get_perms(p, θ) for θ ∈ eachcol(θs[:,inds,end]))
μ_post = reshape(mean(logks_post, dims=2), nx, nz)
σ_post = reshape( std(logks_post, dims=2), nx, nz)

@save "results/mda_$(Ne)_$(Ni).jld2" θs fs us αs inds