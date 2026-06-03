# Angular / circular distributions (ported from VLBIImagePriors). Primal only —
# no ChainRules rrules; Enzyme/Reactant differentiate through these directly.

"""
    DiagonalVonMises(μ, κ)

A (multivariate, independent) von Mises distribution with mean `μ` and
concentration `κ`. Custom implementation (vs `Distributions.VonMises`) with full
support on the circle and a `product_distribution` that preserves the type.
"""
struct DiagonalVonMises{M, K, C} <: Dists.ContinuousMultivariateDistribution
    μ::M
    κ::K
    lnorm::C
end

Base.length(d::DiagonalVonMises) = length(d.μ)
Base.eltype(d::DiagonalVonMises) = promote_type(eltype(d.μ), eltype(d.κ))
Dists.insupport(::DiagonalVonMises, x) = true

DiagonalVonMises(μ::AbstractVector, κ::AbstractVector) = DiagonalVonMises(μ, κ, _vonmisesnorm(μ, κ))
DiagonalVonMises(μ::Number, κ::Number) = DiagonalVonMises(μ, κ, _vonmisesnorm(μ, κ))

# log-normalisation. `besseli0x` is the exponentially-scaled I₀; the matching `-1`
# in the kernel keeps the whole thing bounded (a numerical-stability trick).
function _vonmisesnorm(μ::Number, κ::Number)
    Ts = promote_type(typeof(μ), typeof(κ))
    return -convert(Ts, log(2π)) - log(besseli0x(κ))
end
function _vonmisesnorm(μ, κ)
    @argcheck length(μ) == length(κ)
    n = length(μ)
    Ts = promote_type(eltype(μ), eltype(κ))
    return -n * convert(Ts, log(2π)) - sum(x -> log(besseli0x(x)), κ)
end

_vonlogpdf(μ::Number, κ::Number, x::Number) = (cos(x - μ) - 1) * κ
function _vonlogpdf(μ, κ, x)
    # `sum(zip(...))` form (not an indexed loop) — Enzyme traces through it.
    return sum(zip(μ, κ, x); init = zero(eltype(μ))) do (μs, κs, xs)
        return (cos(xs - μs) - 1) * κs
    end
end

Dists.logpdf(d::DiagonalVonMises{<:Real, <:Real, <:Real}, x::Number) = _vonlogpdf(d.μ, d.κ, x) + d.lnorm
function Dists.logpdf(d::DiagonalVonMises, x::Union{Number, AbstractVector})
    return _vonlogpdf(d.μ, d.κ, x) + d.lnorm
end

function Dists._rand!(rng::AbstractRNG, d::DiagonalVonMises, x::AbstractVector)
    dv = Dists.product_distribution(Dists.VonMises.(d.μ, d.κ))
    return rand!(rng, dv, x)
end
Dists.rand(rng::AbstractRNG, d::DiagonalVonMises{<:Real, <:Real}) = rand(rng, Dists.VonMises.(d.μ, d.κ))

function Dists.product_distribution(dists::AbstractVector{<:DiagonalVonMises})
    μ = mapreduce(x -> x.μ, vcat, dists)
    κ = mapreduce(x -> x.κ, vcat, dists)
    lnorm = mapreduce(x -> x.lnorm, +, dists)
    return DiagonalVonMises(μ, κ, lnorm)
end


"""
    WrappedUniform(period)
    WrappedUniform(period, n)

A (multivariate, independent) uniform distribution wrapped over `period`, i.e.
`logpdf(d, x) ≈ logpdf(d, x + period)`.
"""
struct WrappedUniform{T, L} <: Dists.ContinuousMultivariateDistribution
    periods::T
    lnorm::L
end

Base.length(d::WrappedUniform) = length(d.periods)
Base.eltype(d::WrappedUniform) = eltype(d.periods)
Dists.insupport(::WrappedUniform, x) = true

function WrappedUniform(p::AbstractVector)
    all(>(0), p) || throw(ArgumentError("Periods must be positive"))
    return WrappedUniform(p, sum(log, p))
end
WrappedUniform(p::Number, n::Int) = WrappedUniform(fill(p, n), n * log(p))
WrappedUniform(p::Number) = WrappedUniform(p, log(p))

Dists.logpdf(d::WrappedUniform{<:Real}, ::Number) = -d.lnorm
Dists._logpdf(d::WrappedUniform, ::AbstractVector) = -d.lnorm
Dists.rand(rng::AbstractRNG, d::WrappedUniform{<:Real}) = rand(rng) * d.periods
function Dists._rand!(rng::AbstractRNG, d::WrappedUniform, x::AbstractVector{T}) where {T <: Real}
    rand!(rng, x)
    x .= x .* d.periods
    return x
end

function Dists.product_distribution(dists::AbstractVector{<:WrappedUniform})
    periods = mapreduce(x -> x.periods, vcat, dists)
    return WrappedUniform(periods)
end

# ----- no exact transport to StdNormal / StdUniform ------------------------
# `transport_to(prior, StdNormal()/StdUniform())` must transport the prior's
# random variables to be *exactly* standard normal / uniform. A circular variable
# (von Mises, wrapped uniform) cannot: it has no usable quantile, and the smooth
# 2D projected-normal embedding is a *different* distribution, not a transport of
# the circular one. So error, and point at the supported options.
function _no_circular_transport(d, space)
    throw(
        ArgumentError(
            "Cannot transport the circular distribution `$(nameof(typeof(d)))` to " *
            "`$(nameof(typeof(space)))`: there is no measure-preserving map from a " *
            "circle to the line, so the transported variables would not be " *
            "$(nameof(typeof(space)))-distributed. Use `StdFlat()` (a smooth ℝⁿ " *
            "embedding via `AngleTransform`), or replace the prior with a projected-" *
            "normal prior, whose transport to `StdNormal()` is exact.",
        ),
    )
end

transport_node(d::DiagonalVonMises, space::Union{StdNormal, StdUniform}) = _no_circular_transport(d, space)
transport_node(d::WrappedUniform, space::Union{StdNormal, StdUniform}) = _no_circular_transport(d, space)
transport_node(d::Dists.VonMises, space::Union{StdNormal, StdUniform}) = _no_circular_transport(d, space)
