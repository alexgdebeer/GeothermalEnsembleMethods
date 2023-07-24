using JLD2
using LaTeXStrings
using PyPlot
using Random
using Seaborn
using SimIntensiveInference

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")
Seaborn.axes_style("whitegrid")

Random.seed!(1)

SECS_PER_MONTH = tmax / 24
kmin = -16.5
kmax = -13.0

xticks = [0, 500, 1000, 1500]
xticklabels = map(string, xticks)
zticks = [-1500, -1000, -500, 0]
zticklabels = map(string, zticks)

logks_t = reshape(logks_t, nx, nz)
temps_t = reshape(us_t[1:n_blocks], nx, nz)

# ----------------
# Model grid 
# ----------------

fig, ax = PyPlot.subplots(figsize=(6, 6))

ax.pcolormesh(xs, zs, 0.1ones(nx, nz), cmap="Greys", vmin=0, vmax=1, edgecolors="silver")

for (i, (x, z)) ∈ enumerate(zip(fz_xs, fz_zs))
    ax.plot([x, x], [0, -1300], linewidth=1.5, color="k", zorder=1)
    ax.scatter([x], [z], zorder=2, color="k")
    ax.text(x+30, z-15, s="W$i", fontsize=14)
end

ax.set_xlabel(L"x"*" (m)", fontsize=24)
ax.set_ylabel(L"z"*" (m)", fontsize=24)
ax.set_aspect(1)
ax.set_xticks(xticks, labels=xticklabels, fontsize=14)
ax.set_yticks(zticks, labels=zticklabels, fontsize=14)

for spine ∈ ax.spines.values()
    spine.set_edgecolor("silver")
end

PyPlot.tight_layout()
PyPlot.savefig("plots/grid.pdf")
PyPlot.clf()

# ----------------
# Prior samples 
# ----------------

fig, ax = PyPlot.subplots(nrows=2, ncols=2, figsize=(6, 6))

for i ∈ 1:4
    logks_pri = reshape(get_perms(p, rand(p)), nx, nz)
    ax[i].pcolormesh(xs, zs, rotl90(logks_pri), cmap="turbo", vmin=kmin, vmax=kmax)
    ax[i].set_aspect(1)
    ax[i].axis("off")
end

PyPlot.tight_layout()
PyPlot.savefig("plots/prior_samples.pdf")
PyPlot.clf()

# ----------------
# Prior samples 
# ----------------

fig, ax = PyPlot.subplots(nrows=1, ncols=2, figsize=(8, 3))

p1 = ax[1].pcolormesh(xs, zs, rotl90(logks_t), cmap="turbo", vmin=kmin, vmax=kmax)
p2 = ax[2].pcolormesh(xs, zs, rotl90(temps_t), cmap="coolwarm")

c1 = fig.colorbar(p1, ax=ax[1], label="log(Permeability) [log(m"*L"^2"*")]")
c2 = fig.colorbar(p2, ax=ax[2])

c1.set_label("log(Permeability) [log(m"*L"^2"*")]", fontsize=14)
c2.set_label("Temperature "*L"[ ^\circ"*"C]", fontsize=14)

ax[1].set_aspect(1)
ax[2].set_aspect(1)

ax[1].axis("off")
ax[2].axis("off")

PyPlot.tight_layout()
PyPlot.savefig("plots/truth.pdf")
PyPlot.clf()

# Enthalpy plotting 

PyPlot.figure(figsize=(6, 6))

well_num = 2

enthalpies_t = get_raw_enthalpies(us_t, inds_es_raw, nfz, nt)
enthalpies_obs = reshape(us_o[inds_es_obs], nfz, nt_obs)

enthalpies_t = enthalpies_t[well_num, :]
enthalpies_obs = enthalpies_obs[well_num, :]

enthalpies_ensemble = reduce(hcat, [
    get_raw_enthalpies(fs[:,i,end], inds_es_raw, nfz, nt)[well_num, :] 
        for i ∈ inds])

PyPlot.plot(ts ./ SECS_PER_MONTH, enthalpies_ensemble ./ 1000, zorder=1, color="royalblue")
PyPlot.plot(ts ./ SECS_PER_MONTH, enthalpies_t ./ 1000, zorder=2, color="k", linewidth=2)
PyPlot.scatter(ts[t_obs] ./ SECS_PER_MONTH, enthalpies_obs ./ 1000, zorder=3, color="k")

PyPlot.gca().set_box_aspect(1)
PyPlot.xlabel("Time [months]", fontsize=14)
PyPlot.ylabel("Enthalpy [kJ/kg]", fontsize=14)

#PyPlot.tight_layout()
PyPlot.savefig("plots/enthalpies.pdf")

# fnames = [
#     "mda_25_4_no_loc.jld2",
#     "mda_25_4_loc.jld2",
#     "enrml_25_no_loc.jld2",
#     "enrml_25_loc.jld2"
# ]

# fig, ax = PyPlot.subplots(4, 6)
# kmin = -16.0
# kmax = -13.0

# for (i, fname) ∈ enumerate(fnames)

#     @load fname θs fs inds

#     logps_post = reduce(hcat, get_perms(p, θ) for θ ∈ eachcol(θs[:,inds,end]))
#     μ_post = reshape(mean(logps_post, dims=2), nx, nz)
#     σ_post = reshape( std(logps_post, dims=2), nx, nz)

#     μ_post_preds = reshape(mean(fs[:,inds,end], dims=2), nx, nz)

#     ax[i, 1].pcolormesh(rotl90(μ_post), cmap="turbo", vmin=kmin, vmax=kmax)
#     ax[i, 2].pcolormesh(rotl90(reshape(logps_post[:, 1], nx, nz)), cmap="turbo", vmin=kmin, vmax=kmax)
#     ax[i, 3].pcolormesh(rotl90(reshape(logps_post[:, 2], nx, nz)), cmap="turbo", vmin=kmin, vmax=kmax)
#     ax[i, 4].pcolormesh(rotl90(reshape(logps_post[:, 3], nx, nz)), cmap="turbo", vmin=kmin, vmax=kmax)
#     ax[i, 5].pcolormesh(rotl90(σ_post), cmap="RdYlGn", vmin=0.0, vmax=0.3)
#     ax[i, 6].pcolormesh(rotl90(μ_post_preds), cmap="coolwarm", vmin=0.0, vmax=320)

#     for j ∈ 1:6
#         ax[i, j].set_aspect("equal")
#         ax[i, j].set_xticks([])
#         ax[i, j].set_yticks([])
#         ax[i, j].spines["top"].set_visible(false)
#         ax[i, j].spines["right"].set_visible(false)
#         ax[i, j].spines["bottom"].set_visible(false)
#         ax[i, j].spines["left"].set_visible(false)
#     end

# end

# ax[1, 1].set_title("Mean")
# ax[1, 2].set_title("Sample 1")
# ax[1, 3].set_title("Sample 1")
# ax[1, 4].set_title("Sample 1")
# ax[1, 5].set_title("Stds")
# ax[1, 6].set_title("Mean temps")

# ax[1, 1].set_ylabel("MDA")
# ax[2, 1].set_ylabel("MDA (loc)")
# ax[3, 1].set_ylabel("EnRML")
# ax[4, 1].set_ylabel("EnRML (loc)")

# PyPlot.tight_layout()
# PyPlot.savefig("test.pdf")
# PyPlot.clf()