# ProjectedNormal — a directional prior that concentrates in a direction (like a
# von Mises) but, unlike von Mises, transports *exactly* to StdNormal/StdUniform/
# StdFlat. It is the 2D Gaussian `N(γ·(cos μ, sin μ), I₂)`; its angular marginal
# `atan(x₂, x₁)` is the projected normal, a smooth von-Mises-like distribution.
# Because it is an affine shift of `StdNormal(2)` the transport is exact and smooth
# (no wrap boundary) — the legitimate way to get circular coordinates.

"""
    ProjectedNormal(μ, γ)
    ProjectedNormal(ν::AbstractVector)

A directional prior on ℝ² whose angle `atan(x₂, x₁)` concentrates around `μ`
(radians) with concentration `γ ≥ 0` (the length of the mean vector). Equivalent to
`MvNormal(γ·(cos μ, sin μ), I₂)`; its angular marginal is the *projected normal*, a
smooth approximation to a `VonMises`. `γ = 0` is uniform on the circle; larger `γ`
concentrates more tightly around `μ`.

Unlike `VonMises`, it transports **exactly** to `StdNormal()`/`StdUniform()`/`StdFlat()`
(it is an affine shift of `StdNormal(2)`), so it is the recommended circular prior
when you want smooth Gaussian coordinates. Take `atan(x[2], x[1])` for the angle.
"""
struct ProjectedNormal{T} <: Dists.ContinuousMultivariateDistribution
    μ::T
    γ::T
end
function ProjectedNormal(μ::Real, γ::Real)
    μp, γp = promote(float(μ), float(γ))
    return ProjectedNormal{typeof(μp)}(μp, γp)
end
function ProjectedNormal(ν::AbstractVector)
    @argcheck length(ν) == 2
    return ProjectedNormal(atan(ν[2], ν[1]), hypot(ν[1], ν[2]))
end

_meanvec(d::ProjectedNormal) = (d.γ * cos(d.μ), d.γ * sin(d.μ))

Base.length(::ProjectedNormal) = 2
Base.eltype(::ProjectedNormal{T}) where {T} = T
Dists.mean(d::ProjectedNormal) = [_meanvec(d)...]
Dists.insupport(::ProjectedNormal, x::AbstractVector) = length(x) == 2

function Dists._logpdf(d::ProjectedNormal, x::AbstractVector)
    ν1, ν2 = _meanvec(d)
    return -((x[1] - ν1)^2 + (x[2] - ν2)^2) / 2 - oftype(float(x[1]), log(2π))
end

function Dists._rand!(rng::AbstractRNG, d::ProjectedNormal, x::AbstractVector)
    ν1, ν2 = _meanvec(d)
    x[1] = randn(rng) + ν1
    x[2] = randn(rng) + ν2
    return x
end

# Exact transport: an affine *shift* (scale 1) of the standard-normal pair.
function transport_node(d::ProjectedNormal, space)
    ν = [_meanvec(d)...]
    return PushforwardTransport(ScaleShift(ν, one(eltype(ν))), transport_node(StdNormal(2), space))
end
