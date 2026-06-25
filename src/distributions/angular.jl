# Angular / circular distributions (ported from VLBIImagePriors). Primal only —
# no ChainRules rrules; Enzyme/Reactant differentiate through these directly.

"""
    DiagonalVonMises(μ, κ)

A (multivariate, independent) von Mises distribution with mean `μ` and
concentration `κ`. Custom implementation (vs `Distributions.VonMises`) with full
support on the circle, a `product_distribution` that preserves the type, and a
type-generic sampler that honours `Float32` (unlike `Distributions.VonMisesSampler`,
which is hardcoded to `Float64`).
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

# Type-generic, Reactant-traceable von Mises sampler. `Distributions.VonMisesSampler`
# hardcodes `Float64` (its fields and the `rand`/`randn` draws inside are all
# Float64), so sampling a Float32 von Mises silently promotes to Float64. This
# implementation carries the element type through, so a Float32 (μ, κ) yields a
# Float32 draw.
#
# Algorithm: DJ Best & NI Fisher (1979), "Efficient Simulation of the von Mises
# Distribution", J. R. Stat. Soc. C, 28(2), 152-157, in its single-uniform form
# (cf. NumPy `random.vonmises` / R `rvm`): `z = cos(π u₁)` replaces the quarter-
# disk rejection of the textbook version, leaving a single rejection loop.
#
# That loop is written condition-driven (a loop-carried `accepted`, no `break`)
# and wrapped in `@trace while`: under Reactant compilation with traced arguments
# it lowers to `stablehlo.while`; otherwise `@trace` expands to the plain loop, so
# CPU sampling is unchanged. `@trace` forbids `break`/`continue`/`return` in its
# body, hence the boolean-flag form.
function _rand_vonmises(rng::AbstractRNG, μ::Number, κ::Number)
    T = float(promote_type(typeof(μ), typeof(κ)))
    m = T(μ)
    k = T(κ)
    # Large-κ limit: von Mises → Normal(μ, 1/κ). The rejection step below both
    # loses efficiency and becomes numerically delicate here, so fall back.
    if k > T(700)
        return m + randn(rng, T) / sqrt(k)
    end
    τ = one(T) + sqrt(one(T) + 4 * abs2(k))
    ρ = (τ - sqrt(2 * τ)) / (2 * k)
    r = (one(T) + abs2(ρ)) / (2 * ρ)
    f = zero(T)
    accepted = false
    # Hoist `π` out of the loop. `@trace while` threads every symbol referenced in
    # its body as loop-carried state with `track_numbers=true`; if `π` (an
    # `Irrational`) appears inside, Reactant tries to build an `RNumber{Irrational}`
    # and errors. A concrete `T(π)` computed beforehand threads fine.
    cπ = T(π)
    @trace while !accepted
        u1 = rand(rng, T)
        u2 = rand(rng, T)
        z = cos(cπ * u1)
        f = (one(T) + r * z) / (r + z)
        c = k * (r - f)
        # `c > 0` always here (r > 1 ≥ f), so `log(c / u2)` is well-defined; use
        # the non-short-circuit `|` since `@trace` can't trace `||`'s branch.
        accepted = (c * (2 - c) > u2) | (log(c / u2) + one(T) >= c)
    end
    # branch-free sign (`±acos(f)`); `sign(u₃ - ½) ∈ {-1, +1}` w.p. 1.
    u3 = rand(rng, T)
    return m + sign(u3 - T(0.5)) * acos(f)
end

function Dists._rand!(rng::AbstractRNG, d::DiagonalVonMises, x::AbstractVector)
    @argcheck length(x) == length(d.μ) == length(d.κ)
    # `@trace for` mirrors the Std* samplers: a `stablehlo` loop under Reactant,
    # a plain loop otherwise.
    @trace for i in eachindex(x)
        _rsetindex!(x, _rand_vonmises(rng, _rgetindex(d.μ, i), _rgetindex(d.κ, i)), i)
    end
    return x
end
Dists.rand(rng::AbstractRNG, d::DiagonalVonMises{<:Real, <:Real}) = _rand_vonmises(rng, d.μ, d.κ)

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
Dists.logpdf(d::WrappedUniform, ::AbstractVector) = -d.lnorm
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
                "$(nameof(typeof(space)))-distributed. Use `TVFlat()` (a smooth ℝⁿ " *
                "embedding via `AngleTransform`), or replace the prior with a projected-" *
                "normal prior, whose transport to `StdNormal()` is exact.",
        ),
    )
end

transport_node(d::DiagonalVonMises, space::Union{StdNormal, StdUniform}) = _no_circular_transport(d, space)
transport_node(d::WrappedUniform, space::Union{StdNormal, StdUniform}) = _no_circular_transport(d, space)
transport_node(d::Dists.VonMises, space::Union{StdNormal, StdUniform}) = _no_circular_transport(d, space)
