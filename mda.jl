using JLD2
include("setup.jl")

Ni = 4
Ne = 25
θs, fs, us, αs, inds = run_es_mda(f, g, p, L, n_blocks, Ni, Ne, localisation=true)#, α_method=:constant)

@save "mda_$(Ne)_$(Ni).jld2" θs fs us αs inds