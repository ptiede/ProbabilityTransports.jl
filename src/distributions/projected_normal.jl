# ProjectedNormal — a directional prior that concentrates in a direction (like a
# von Mises) but, unlike von Mises, transports *exactly* to StdNormal/StdUniform/
# TVFlat. Each direction is the 2D Gaussian `N(γ·(cos μ, sin μ), I₂)`; its angular
# marginal `atan(x₂, x₁)` is the projected normal, a smooth von-Mises-like density.
# Because it is an affine shift of `StdNormal(2n)` the transport is exact and smooth
# (no wrap boundary) — the legitimate way to get circular coordinates.
#
# Like `DiagonalVonMises`, it is a (multivariate, independent) distribution: `μ` and
# `γ` may be scalars (one direction) or length-`n` vectors (`n` directions). An
# `n`-direction prior lives in ℝ^{2n}, with direction `i` occupying coordinates
# `(2i-1, 2i)`.

"""
    ProjectedNormal(μ, γ)
    ProjectedNormal(ν::AbstractVector)

A directional prior whose angle(s) `atan(x₂ᵢ, x₂ᵢ₋₁)` concentrate around `μ`
(radians) with concentration `γ ≥ 0` (the length of the mean vector). `μ`, `γ` are
scalars for a single direction or length-`n` vectors for `n` independent
directions; the support is ℝ^{2n}. Each direction is `MvNormal(γᵢ·(cos μᵢ, sin μᵢ), I₂)`,
whose angular marginal is the *projected normal*, a smooth approximation to a
`VonMises`. `γ = 0` is uniform on the circle; larger `γ` concentrates more tightly
around `μ`. The single-argument form takes a length-2 mean vector `ν` directly.

Unlike `VonMises`/`DiagonalVonMises`, it transports **exactly** to `StdNormal()`/
`StdUniform()`/`TVFlat()` (it is an affine shift of `StdNormal(2n)`), so it is the
recommended circular prior when you want smooth Gaussian coordinates. Take
`atan(x[2i], x[2i-1])` for direction `i`'s angle.
"""
struct ProjectedNormal{M, G, V} <: Dists.ContinuousMultivariateDistribution
    μ::M    # direction(s) in radians (scalar or length-n vector)
    γ::G    # concentration(s) ≥ 0   (scalar or length-n vector)
    ν::V    # precomputed mean vector, length 2n: [γᵢcosμᵢ, γᵢsinμᵢ …]
end

# mean vector with contiguous (cos, sin) pairs per direction: [c₁,s₁,c₂,s₂,…]
_projnormal_meanvec(μ::Number, γ::Number) = [γ * cos(μ), γ * sin(μ)]
function _projnormal_meanvec(μ::AbstractVector, γ::AbstractVector)
    @argcheck length(μ) == length(γ)
    # `vec(permutedims(hcat(c, s)))` interleaves without scalar indexing.
    return vec(permutedims(hcat(γ .* cos.(μ), γ .* sin.(μ))))
end

function ProjectedNormal(μ::Number, γ::Number)
    μp, γp = promote(float(μ), float(γ))
    return ProjectedNormal(μp, γp, _projnormal_meanvec(μp, γp))
end
ProjectedNormal(μ::AbstractVector, γ::AbstractVector) =
    ProjectedNormal(μ, γ, _projnormal_meanvec(μ, γ))
function ProjectedNormal(ν::AbstractVector)
    @argcheck length(ν) == 2
    return ProjectedNormal(atan(ν[2], ν[1]), hypot(ν[1], ν[2]))
end

Base.length(d::ProjectedNormal) = length(d.ν)                 # = 2n
Base.eltype(d::ProjectedNormal) = eltype(d.ν)
_ndir(d::ProjectedNormal) = length(d.ν) ÷ 2                    # number of directions n
Dists.mean(d::ProjectedNormal) = d.ν
Dists.insupport(d::ProjectedNormal, x::AbstractVector) = length(x) == length(d)

function Dists.logpdf(d::ProjectedNormal, x::AbstractVector)
    # vectorized (no scalar indexing) so it traces under Reactant
    T = float(eltype(d.ν))
    return -sum(abs2, x .- d.ν) / 2 - _ndir(d) * log(convert(T, 2π))
end

function Dists._rand!(rng::AbstractRNG, d::ProjectedNormal, x::AbstractVector)
    randn!(rng, x)
    x .+= d.ν
    return x
end

# Concatenate directions, mirroring `DiagonalVonMises` (scalar `vcat` → vector).
function Dists.product_distribution(dists::AbstractVector{<:ProjectedNormal})
    μ = mapreduce(Base.Fix2(getproperty, :μ), vcat, dists)
    γ = mapreduce(Base.Fix2(getproperty, :γ), vcat, dists)
    return ProjectedNormal(μ, γ)
end

# Exact transport: an affine *shift* (scale 1) of the `2n` standard normals.
function transport_node(d::ProjectedNormal, space)
    ν = d.ν
    return PushforwardTransport(
        ScaleShift(ν, one(eltype(ν))), transport_node(StdNormal(length(ν)), space)
    )
end
