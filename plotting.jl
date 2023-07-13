using JLD2
using PyPlot
using SimIntensiveInference

fnames = [
    "mda_25_4_no_loc.jld2",
    "mda_25_4_loc.jld2",
    "enrml_25_no_loc.jld2",
    "enrml_25_loc.jld2"
]

fig, ax = PyPlot.subplots(4, 6)
kmin = -16.0
kmax = -13.0

for (i, fname) ∈ enumerate(fnames)

    @load fname θs fs inds

    logps_post = reduce(hcat, get_perms(p, θ) for θ ∈ eachcol(θs[:,inds,end]))
    μ_post = reshape(mean(logps_post, dims=2), nx, nz)
    σ_post = reshape( std(logps_post, dims=2), nx, nz)

    μ_post_preds = reshape(mean(fs[:,inds,end], dims=2), nx, nz)

    ax[i, 1].pcolormesh(rotl90(μ_post), cmap="turbo", vmin=kmin, vmax=kmax)
    ax[i, 2].pcolormesh(rotl90(reshape(logps_post[:, 1], nx, nz)), cmap="turbo", vmin=kmin, vmax=kmax)
    ax[i, 3].pcolormesh(rotl90(reshape(logps_post[:, 2], nx, nz)), cmap="turbo", vmin=kmin, vmax=kmax)
    ax[i, 4].pcolormesh(rotl90(reshape(logps_post[:, 3], nx, nz)), cmap="turbo", vmin=kmin, vmax=kmax)
    ax[i, 5].pcolormesh(rotl90(σ_post), cmap="RdYlGn", vmin=0.0, vmax=0.3)
    ax[i, 6].pcolormesh(rotl90(μ_post_preds), cmap="coolwarm", vmin=0.0, vmax=320)

    for j ∈ 1:6
        ax[i, j].set_aspect("equal")
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].spines["top"].set_visible(false)
        ax[i, j].spines["right"].set_visible(false)
        ax[i, j].spines["bottom"].set_visible(false)
        ax[i, j].spines["left"].set_visible(false)
    end

end

ax[1, 1].set_title("Mean")
ax[1, 2].set_title("Sample 1")
ax[1, 3].set_title("Sample 1")
ax[1, 4].set_title("Sample 1")
ax[1, 5].set_title("Stds")
ax[1, 6].set_title("Mean temps")

ax[1, 1].set_ylabel("MDA")
ax[2, 1].set_ylabel("MDA (loc)")
ax[3, 1].set_ylabel("EnRML")
ax[4, 1].set_ylabel("EnRML (loc)")

PyPlot.tight_layout()
PyPlot.savefig("test.pdf")
PyPlot.clf()