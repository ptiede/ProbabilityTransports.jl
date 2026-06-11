# Reactant-friendly truncated wrapper (ported from VLBIImagePriors).
#
# Mirrors `Distributions.Truncated` but with the branching ternaries replaced by
# `ifelse` masks and no exception thrown on invalid bounds — caller is responsible
# for `lower <= upper`. The base distribution must provide Reactant-traceable
# `logpdf`, `cdf`, and `quantile`. Because it exposes `cdf`/`quantile`, it plugs
# straight into the generic scalar transport path (`transport_node(::UnivariateDistribution, ...)`).
#
# Not exported (clashes with `Distributions.Truncated`); use `ProbabilityTransports.Truncated`.
# VLBIImagePriors re-exports it as `VLBITruncated`.

"""
    Truncated(d, lower, upper)
    Truncated(d; lower=nothing, upper=nothing)

Distribution of `X | lower <= X <= upper` where `X ~ d`. Either bound may be
`nothing` for one-sided truncation. The normalisation `log P(lower <= X <= upper)`
is cached at construction. Reactant-friendly: branchless, inverse-CDF sampling.
"""
struct Truncated{D <: Dists.UnivariateDistribution, Tl, Tu, T} <:
    Dists.ContinuousUnivariateDistribution
    untruncated::D
    lower::Tl
    upper::Tu
    logtp::T   # log P(lower <= X <= upper)
    lcdf::T    # cdf(untruncated, lower) — zero when `lower === nothing`
end

function Truncated(d::Dists.UnivariateDistribution, lower::Number, upper::Number)
    lcdf = Dists.cdf(d, lower)
    # use logcdf (not log(cdf)) so deep-tail bounds don't underflow to log(0) = -Inf → NaN
    loglcdf = Dists.logcdf(d, lower)
    logucdf = Dists.logcdf(d, upper)
    logtp = logucdf + log1p(-exp(loglcdf - logucdf))   # stable log(ucdf - lcdf)
    T = promote_type(typeof(logtp), typeof(lcdf))
    return Truncated{typeof(d), typeof(lower), typeof(upper), T}(d, lower, upper, T(logtp), T(lcdf))
end
function Truncated(d::Dists.UnivariateDistribution, ::Nothing, upper::Number)
    ucdf = Dists.cdf(d, upper)
    logtp = Dists.logcdf(d, upper)   # logcdf avoids underflow when upper is deep in the left tail
    T = promote_type(typeof(logtp), typeof(ucdf))
    return Truncated{typeof(d), Nothing, typeof(upper), T}(d, nothing, upper, T(logtp), zero(T))
end
function Truncated(d::Dists.UnivariateDistribution, lower::Number, ::Nothing)
    lcdf = Dists.cdf(d, lower)
    logtp = Dists.logccdf(d, lower)   # log(1 - cdf(lower)), avoids precision loss when lcdf ≈ 1
    T = promote_type(typeof(logtp), typeof(lcdf))
    return Truncated{typeof(d), typeof(lower), Nothing, T}(d, lower, nothing, T(logtp), T(lcdf))
end
Truncated(d::Dists.UnivariateDistribution; lower = nothing, upper = nothing) =
    Truncated(d, lower, upper)

# Support endpoints intersect the truncation bounds with the base support (matching
# `Distributions.truncated`): a one-sided (or out-of-support) bound must not widen the
# base support — `Truncated(Exponential(); upper = 1)` has support (0, 1], not (-∞, 1].
# The flat transform is built from these, so getting them wrong exposes a reachable
# logpdf = -Inf region in latent space (in practice: negative background flux and a
# frozen NUTS chain).
Base.minimum(d::Truncated{<:Any, <:Number}) = max(d.lower, Dists.minimum(d.untruncated))
Base.minimum(d::Truncated{<:Any, Nothing}) = Dists.minimum(d.untruncated)
Base.maximum(d::Truncated{<:Any, <:Any, <:Number}) = min(d.upper, Dists.maximum(d.untruncated))
Base.maximum(d::Truncated{<:Any, <:Any, Nothing}) = Dists.maximum(d.untruncated)
Dists.params(d::Truncated) = (Dists.params(d.untruncated)..., d.lower, d.upper)

# bound checks (dispatched on Nothing vs Real, branchless)
@inline _ge_lower(::Nothing, _) = true
@inline _ge_lower(lower, x) = x >= lower
@inline _le_upper(::Nothing, _) = true
@inline _le_upper(upper, x) = x <= upper

function unnormed_logpdf(d::Truncated, x::Number)
    in_supp = _ge_lower(d.lower, x) & _le_upper(d.upper, x)
    base_lpdf = Dists.logpdf(d.untruncated, x)
    return ifelse(in_supp, base_lpdf, oftype(base_lpdf, -Inf))
end
@inline lognorm(d::Truncated) = -d.logtp

Dists.logpdf(d::Truncated, x::Number) = unnormed_logpdf(d, x) + lognorm(d)
Dists.logpdf(d::Truncated, x::Real) = unnormed_logpdf(d, x) + lognorm(d)

function Dists.cdf(d::Truncated, x::Number)
    raw = (Dists.cdf(d.untruncated, x) - d.lcdf) * exp(-d.logtp)
    return clamp(raw, zero(raw), one(raw))
end
function Dists.quantile(d::Truncated, p::Number)
    return Dists.quantile(d.untruncated, d.lcdf + p * exp(d.logtp))
end

# inverse-CDF sampling (no rejection loop)
Random.rand(rng::AbstractRNG, d::Truncated) = Dists.quantile(d, rand(rng))

Dists.insupport(d::Truncated, x::Number) = _ge_lower(d.lower, x) & _le_upper(d.upper, x)
Dists.insupport(d::Truncated, x::Real) = _ge_lower(d.lower, x) & _le_upper(d.upper, x)
