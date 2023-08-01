using JLD2
include("setup.jl")

Ni = 8
Ne = 50
θs, fs, us, αs, inds = run_es_mda(
    f, g, p, L, n_obs_raw, Ni, Ne, localisation=true)

logks_post = reduce(hcat, get_perms(p, θ) for θ ∈ eachcol(θs[:,inds,end]))
μ_post = reshape(mean(logks_post, dims=2), nx, nz)
σ_post = reshape( std(logks_post, dims=2), nx, nz)

@save "results/mda_$(Ne)_$(Ni).jld2" θs fs us αs inds