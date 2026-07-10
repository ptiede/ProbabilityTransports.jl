"""
    StdInverseGamma(α, [dims])

Inverse-gamma with shape `α` and unit scale: `pdf(z; α) = z^(-α-1) exp(-1/z) / Γ(α)` for
`z > 0`. `α` may be a scalar (broadcast over `dims`) or an array matching the distribution's
shape. A transportable base distribution, **not** a valid target space for
[`transport_to`](@ref) (no `space_*` trait). The normalization is cached at construction;
pass `lognorm = false` to skip the cache (`lognorm(d)` then recomputes on demand).
"""
struct StdInverseGamma{T, Tα, N, Tl} <: AbstractStdDist{T, N}
    α::Tα
    lognorm::Tl
    dims::Dims{N}
end

# Compute the normalization
@inline _lognorm_igamma(d::Number, N) = -N * loggamma(d)
@inline _lognorm_igamma(d::AbstractArray, N) = -sum(loggamma, d)

# Constructors + cached/uncached `lognorm` + per-element `_elem_dist` (shared with StdTDist).
@cached_scalar_std StdInverseGamma α _lognorm_igamma

# ----- log-pdf split ------------------------------------------------------
# `loggamma(α)` is the expensive piece for an array `α`; folding it into
# `lognorm` lets a caller cache it across many `logpdf` evaluations.

@inline function _ig_unnormed_elem(α, z)
    pos = z > zero(z)
    zsafe = ifelse(pos, z, oftype(z, 1))
    val = -(α + one(α)) * log(zsafe) - inv(zsafe)
    return ifelse(pos, val, oftype(z, -Inf))
end
@inline _unnormed_kernel(d::StdInverseGamma, z) = _ig_unnormed_elem(d.α, z)
@inline _unnormed_kernel_sum(d::StdInverseGamma, z) = sum(_ig_unnormed_elem.(d.α, z))

function unnormed_logpdf(d::StdInverseGamma{T, <:Number, 0}, x::Number) where {T}
    return _unnormed_kernel(d, x)
end
function unnormed_logpdf(
        d::StdInverseGamma{T, Tα, N}, x::AbstractArray{<:Number, N}
    ) where {T, Tα, N}
    return _unnormed_kernel_sum(d, x)
end

# ----- sampling -----------------------------------------------------------

# `InverseGamma(α, 1)` sample = `1 / Gamma(α, 1)`.
function Random.rand(rng::AbstractRNG, d::StdInverseGamma{T, <:Number, 0}) where {T}
    return inv(_rand_gamma(rng, d.α))
end

function _std_rand!(rng::AbstractRNG, d::StdInverseGamma{T}, x::AbstractArray) where {T}
    α = d.α
    @trace for i in eachindex(x)
        _rsetindex!(x, inv(_rand_gamma(rng, _getith(α, i))), i)
    end
    return x
end


# ----- support / moments --------------------------------------------------

# `@with_real` also emits the `::Real` overload that breaks the ambiguity with
# Distributions' generic `insupport(::ContinuousUnivariateDistribution, ::Real)`.
@with_real Dists.insupport(::StdInverseGamma, x::Number) = x > 0
function Dists.insupport(d::StdInverseGamma, x::AbstractArray)
    size(d) == size(x) || return false
    return all(>(0), x)
end
Base.minimum(::StdInverseGamma{T, <:Any, 0}) where {T} = zero(T)
Base.maximum(::StdInverseGamma{T, <:Any, 0}) where {T} = T(Inf)

function Dists.mean(d::StdInverseGamma{T, <:Real, 0}) where {T}
    return d.α > 1 ? T(1 / (d.α - 1)) : T(Inf)
end
function Dists.var(d::StdInverseGamma{T, <:Real, 0}) where {T}
    return d.α > 2 ? T(1 / ((d.α - 1)^2 * (d.α - 2))) : T(Inf)
end
@inline _ig_elemmean(α::Number, T) = α > 1 ? T(1 / (α - 1)) : T(Inf)
@inline _ig_elemvar(α::Number, T) = α > 2 ? T(1 / ((α - 1)^2 * (α - 2))) : T(Inf)
function Dists.mean(d::StdInverseGamma{T, <:Real, N}) where {T, N}
    return fill(_ig_elemmean(d.α, T), size(d))
end
function Dists.var(d::StdInverseGamma{T, <:Real, N}) where {T, N}
    return fill(_ig_elemvar(d.α, T), size(d))
end
function Dists.mean(d::StdInverseGamma{T, <:AbstractArray, N}) where {T, N}
    return _ig_elemmean.(d.α, T)
end
function Dists.var(d::StdInverseGamma{T, <:AbstractArray, N}) where {T, N}
    return _ig_elemvar.(d.α, T)
end


# ----- cdf / quantile -----------------------------------------------------
# StdInverseGamma(α): cdf(x) = Q(α, 1/x), where Q is the regularised upper
# incomplete gamma. SpecialFunctions provides both directions.
# Element-wise kernels take `α` directly so they broadcast against either a
# scalar or a per-element parameter array (used by the array-shaped transport path).

@inline _ig_elem_cdf(α, x) = last(SpecialFunctions.gamma_inc(α, inv(x), 0))
@inline _ig_elem_quantile(α, p) = inv(SpecialFunctions.gamma_inc_inv(α, one(p) - p, p))

function Dists.cdf(d::StdInverseGamma{T, <:Number, 0}, x::Number) where {T}
    return _ig_elem_cdf(d.α, x)
end
function Dists.quantile(d::StdInverseGamma{T, <:Number, 0}, p::Number) where {T}
    return _ig_elem_quantile(d.α, p)
end
