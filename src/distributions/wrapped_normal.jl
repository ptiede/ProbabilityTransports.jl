# WrappedNormal — a circular (phase) prior that transports exactly to `StdNormal()` with one
# latent coordinate per angle and no singular point: `θ = wrap(μ + σ u)`, `u ~ N(0, 1)`.
# Every singularity-free map from ℝᵏ onto the circle has this lifted form, so the price is
# that each angle has copies `u + 2πk/σ` in the latent space ("sheets"); the latent density
# itself is exactly N(0, 1).

# Principal value on [−π, π].
function _wrap_angle(x)
    tπ = oftype(float(zero(eltype(x))), 2π)
    return x .- tπ .* round.(x ./ tπ)
end

# `u ↦ wrap(μ + σ u)`; the section picks the sheet nearest `u = 0`, `u = wrap(θ − μ) / σ`.
struct WrapAffine{M, S}
    μ::M
    σ::S
end
(f::WrapAffine)(u) = _wrap_angle(f.μ .+ f.σ .* u)

struct WrapAffineSection{M, S}
    μ::M
    σ::S
end
(g::WrapAffineSection)(θ) = _wrap_angle(θ .- g.μ) ./ g.σ

InverseFunctions.inverse(f::WrapAffine) = WrapAffineSection(f.μ, f.σ)
InverseFunctions.inverse(g::WrapAffineSection) = WrapAffine(g.μ, g.σ)

"""
    WrappedNormal(μ, σ)

The wrapped normal distribution on the circle: the angle `θ = wrap(μ + σ u)` with
`u ~ Normal(0, 1)`, as a proper density on `(−π, π]`. `μ` (radians) and `σ > 0` are scalars
for a single angle or length-`n` vectors for `n` independent angles; the variate is the
angle(s). Its mean resultant length is `exp(−σ²/2)`, so a large `σ` is close to uniform
(`σ = 2.44` matches `VonMises` with `κ = 1/π²`).

It transports exactly to `StdNormal()` / `StdUniform()` with one latent coordinate per
angle, through a map with constant derivative `σ`: unlike [`AngularProjectedNormal`](@ref),
whose latent pair has a singular point at the origin, the latent geometry is uniformly
smooth. Each angle is represented on every sheet `u + 2πk/σ`, all with the same
likelihood; a sampler that stays on one sheet sees the prior factor `φ(u)` in place of the
wrapped sum, which varies by `O(σ⁻²)` across a posterior that is narrow in `θ`.

There is no flat (`TVFlat`) transport: the change of variables would give a density in `u`
that is periodic and so not normalizable.
"""
struct WrappedNormal{M, S} <: Dists.ContinuousMultivariateDistribution
    μ::M    # mean angle(s), radians (scalar or length-n vector)
    σ::S    # scale(s) of the unwrapped normal, > 0 (scalar or length-n vector)
    function WrappedNormal{M, S}(μ, σ) where {M, S}
        _positive_scale(σ) || throw(ArgumentError("WrappedNormal σ must be positive, got $σ"))
        (μ isa Number || σ isa Number || length(μ) == length(σ)) ||
            throw(DimensionMismatch("μ has length $(length(μ)), σ $(length(σ))"))
        return new{M, S}(μ, σ)
    end
end

function WrappedNormal(μ::Number, σ::Number)
    μp, σp = promote(float(μ), float(σ))
    return WrappedNormal{typeof(μp), typeof(σp)}(μp, σp)
end
WrappedNormal(μ::AbstractVector, σ::AbstractVector) = WrappedNormal{typeof(μ), typeof(σ)}(μ, σ)

Base.length(d::WrappedNormal) = d.μ isa Number && d.σ isa Number ? 1 : max(length(d.μ), length(d.σ))
Base.eltype(d::WrappedNormal) = float(promote_type(eltype(d.μ), eltype(d.σ)))
Dists.insupport(d::WrappedNormal, x::AbstractVector) = length(x) == length(d)

# Density on the circle at the wrapped offset δ ∈ [−π, π]. For σ < π the direct sum over
# |k| ≤ 4 sheets drops terms below e⁻⁴⁰; for σ ≥ π the Fourier (Jacobi) series to n = 4
# drops terms below e⁻¹²³. Both are evaluated and selected branchlessly so it traces.
function _wn_density(δ, σ)
    tπ = oftype(float(δ), 2π)
    direct = sum(ntuple(j -> exp(-(δ + tπ * (j - 5))^2 / (2 * σ^2)), Val(9))) / (σ * sqrt(tπ))
    fourier = (1 + 2 * sum(ntuple(n -> exp(-n^2 * σ^2 / 2) * cos(n * δ), Val(4)))) / tπ
    return ifelse(σ < tπ / 2, direct, fourier)
end

function Dists.logpdf(d::WrappedNormal, θ::AbstractVector)
    δ = _wrap_angle(θ .- d.μ)
    return sum(log.(_wn_density.(δ, d.σ .+ zero(δ))))
end

function Dists._rand!(rng::AbstractRNG, d::WrappedNormal, x::AbstractVector)
    x .= WrapAffine(d.μ, d.σ)(randn(rng, eltype(d), length(d)))
    return x
end

function Dists.product_distribution(dists::AbstractVector{<:WrappedNormal})
    μ = mapreduce(Base.Fix2(getproperty, :μ), vcat, dists)
    σ = mapreduce(Base.Fix2(getproperty, :σ), vcat, dists)
    return WrappedNormal(μ, σ)
end

# Exact transport (Std spaces): wrap ∘ (μ + σ u) over n standard normals.
function transport_node(d::WrappedNormal, space::Union{StdNormal, StdUniform})
    return PushforwardTransport(WrapAffine(d.μ, d.σ), transport_node(StdNormal(length(d)), space))
end

function transport_node(::WrappedNormal, ::TVFlat)
    throw(
        ArgumentError(
            "WrappedNormal has no flat transport (its density in the flat coordinate is " *
                "periodic and not normalizable); use transport_to(d, StdNormal())"
        )
    )
end
