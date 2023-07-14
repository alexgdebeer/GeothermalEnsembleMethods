using BlockDiagonals
using Distributions
using LinearAlgebra
using SimIntensiveInference

struct GeothermalPrior <: Prior 

    mass_rate_bnds::AbstractVector
    mass_rate_dist::Distribution

    level_width::Real

    μ::AbstractVector
    Γ::AbstractMatrix
    L::AbstractMatrix

    dist::MvNormal

    xs::AbstractVector
    ys::AbstractVector
    cxs::AbstractVector
    cys::AbstractVector
    cxs_deep::AbstractVector 
    cys_deep::AbstractVector

    Nx::Int
    Nθ::Int
    Nis_shal::Int
    Nis_deep::Int
    
    function GeothermalPrior(
        mass_rate_bnds::AbstractVector,
        depth_shal::Real,
        μ_depth_clay::Real,
        k_depth_clay::KernelFunction,
        μ_perm_shal::Real, 
        μ_perm_clay::Real,
        μ_perm_deep::Real,
        k_perm_shal::KernelFunction,
        k_perm_clay::KernelFunction,
        k_perm_deep::KernelFunction,
        level_width::Real,
        xs::AbstractVector,
        ys::AbstractVector
    )

        function get_is(depth_shal, cys)
            is_shal, is_deep = [], []
            for (i, y) ∈ enumerate(cys)
                y ≥ depth_shal && push!(is_shal, i)
                y < depth_shal && push!(is_deep, i)
            end
            return is_shal, is_deep
        end

        cxs = [x for _ ∈ ys for x ∈ xs]
        cys = [y for y ∈ ys for _ ∈ xs]

        is_shal, is_deep = get_is(depth_shal, cys)

        Nx = length(xs)
        Nis_shal = length(is_shal)
        Nis_deep = length(is_deep)

        μ = vcat(
            [0.0], 
            fill(μ_depth_clay, Nx),
            fill(μ_perm_shal, Nis_shal),
            fill(μ_perm_clay, Nis_deep),
            fill(μ_perm_deep, Nis_deep)
        )

        Γ_depth_clay = generate_cov(xs, xs, k_depth_clay)
        Γ_perm_shal = generate_cov(cxs[is_shal], cys[is_shal], k_perm_shal)
        Γ_perm_clay = generate_cov(cxs[is_deep], cys[is_deep], k_perm_clay)
        Γ_perm_deep = generate_cov(cxs[is_deep], cys[is_deep], k_perm_deep)

        Γ = BlockDiagonal([
            [1.0][:,:], 
            Γ_depth_clay, 
            Γ_perm_shal,
            Γ_perm_clay, 
            Γ_perm_deep
        ])

        L = inv_cholesky(Γ)
        dist = MvNormal(μ, Γ)
        mass_rate_dist = Normal()

        cxs_deep = cxs[is_deep]
        cys_deep = cys[is_deep]

        Nθ = length(μ)

        return new(
            mass_rate_bnds, mass_rate_dist, level_width, 
            μ, Γ, L, dist, xs, ys, cxs, cys, cxs_deep, cys_deep, 
            Nx, Nθ, Nis_shal, Nis_deep
        )

    end

end

function apply_level_sets(ps::AbstractVector, level_width::Real)

    levels = floor(minimum(ps)):level_width:ceil(maximum(ps)+level_width)
    return map(p -> levels[findmin(abs.(levels .- p))[2]], ps)

end

function get_perms(d::GeothermalPrior, θs::AbstractVecOrMat)

    function get_perms_deep(
        d::GeothermalPrior, 
        clay_bounds::AbstractVector,
        perms_clay,
        perms_deep
    )
        perms = []
        for (i, (x, y)) ∈ enumerate(zip(d.cxs_deep, d.cys_deep))
            j = findmin(abs.(d.xs .- x))[2]
            y ≥ clay_bounds[j] && push!(perms, perms_clay[i])
            y < clay_bounds[j] && push!(perms, perms_deep[i])
        end
        return perms
    end

    clay_bounds = θs[2 : 1+d.Nx]
    perms_shal = θs[1+d.Nx+1 : 1+d.Nx+d.Nis_shal]
    perms_clay = θs[1+d.Nx+d.Nis_shal+1 : 1+d.Nx+d.Nis_shal+d.Nis_deep]
    perms_deep = θs[1+d.Nx+d.Nis_shal+d.Nis_deep+1 : end]

    perms_deep = get_perms_deep(d, clay_bounds, perms_clay, perms_deep)

    perms = vcat(
        perms_shal,
        perms_deep
    )

    return apply_level_sets(perms, d.level_width)

end

function get_mass_rate(d::GeothermalPrior, θs::AbstractVecOrMat)::Real

    return d.mass_rate_bnds[1] +
        (d.mass_rate_bnds[2] - d.mass_rate_bnds[1]) * 
        cdf(d.mass_rate_dist, θs[1])

end

function Base.rand(d::GeothermalPrior, n::Int=1)

    return rand(d.dist, n)

end