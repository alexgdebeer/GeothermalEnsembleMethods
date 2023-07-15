include("setup.jl")

Ne = 25
γ = 10
i_max = 4

θs, fs, us, ss, λs, inds = run_lm_enrml(f, g, p, L, γ, i_max, n_blocks, Ne, localisation=true)

@save "results/enrml_$(Ne).jld2" θs fs us ss λs inds