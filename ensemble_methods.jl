using SimIntensiveInference

function run_ensemble(
    f::Function, 
    g::Function,
    θs::AbstractArray,
    Nf::Int,
    Ny::Int,
    Ne::Int,
    inds::AbstractVector,
    n_it::Int,
    incon_num::Union{Int, Nothing}
)

    fs = zeros(Nf, Ne)
    ys = zeros(Ny, Ne)
    inds_mod = []

    for i ∈ inds

        fθ = f(θs[:,i], n_it, i, incon_num)
        if fθ != :failure 
            fs[:,i] = fθ
            ys[:,i] = g(fθ)
            push!(inds_mod, i)
        end

    end

    return fs, ys, inds_mod

end

"""Runs the geometric ES-MDA algorithm described by Rafiee and Reynolds 
(2018)."""
function run_es_mda_geothermal(
    f::Function,
    g::Function,
    p::Prior,
    L::Distribution,
    Nf::Int,
    Ni::Int,
    Ne::Int;
    α_method::Symbol=:dynamic,
    localisation::Bool=false,
    verbose::Bool=true
)

    Nθ = p.Nθ
    Ny = length(L.μ)

    θs = zeros(Nθ, Ne, Ni+1)
    fs = zeros(Nf, Ne, Ni+1)
    ys = zeros(Ny, Ne, Ni+1)

    inds = 1:Ne

    αs = zeros(Ni)
    β = 0.0

    θs[:,:,1] = rand(p, Ne)
    fs[:,:,1], ys[:,:,1], inds = run_ensemble(f, g, θs[:,:,1], Nf, Ny, Ne, inds, 0, nothing)

    Γ_ϵ_sqi = sqrt(inv(L.Σ))

    for i ∈ 1:Ni

        Δθ = SimIntensiveInference.calculate_deviations(θs[:,inds,i], length(inds))
        Δy = SimIntensiveInference.calculate_deviations(ys[:,inds,i], length(inds))

        if i == 1
            _, Λy, _ = svd(Γ_ϵ_sqi * Δy, full=true)
            αs[1], β = SimIntensiveInference.calculate_inflations(Λy, Ni, α_method)
        else 
            αs[i] = β^(i-1) * αs[1]
        end

        ϵs = rand(MvNormal(zeros(Ny), αs[i]*L.Σ), Ne)

        Uy, Λy, Vy = SimIntensiveInference.tsvd(Γ_ϵ_sqi * Δy)
        K = SimIntensiveInference.compute_mda_gain(Δθ, Uy, Diagonal(Λy), Vy, Γ_ϵ_sqi, αs[i])

        if localisation
            SimIntensiveInference.localise_mda!(K, θs[:,:,i], ys[:,:,i], Γ_ϵ_sqi, αs[i], inds, Nθ, Ny)
        end

        # Find best incon based on fit to data
        incon_num = inds[findmin(sum((Γ_ϵ_sqi * (ys[:,:,i] .- L.μ)).^2, dims=1)[inds])[2]]

        θs[:,:,i+1] = θs[:,:,i] + K * (L.μ .+ ϵs .- ys[:,:,i])
        fs[:,:,i+1], ys[:,:,i+1], inds = run_ensemble(f, g, θs[:,:,i+1], Nf, Ny, Ne, inds, i, incon_num)

        if verbose
            @info "Iteration $i complete."
        end

    end

    return θs, fs, ys, αs, inds

end