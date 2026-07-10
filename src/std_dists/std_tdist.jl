"""
    StdTDist(ν, [dims])

Standard (zero-location, unit-scale) Student-t with `ν` degrees of freedom. `ν` may be a scalar
(broadcast over `dims`) or an array matching the distribution's shape. A transportable base
distribution, **not** a valid target space for [`transport_to`](@ref) (no `space_*` trait).
The normalization is cached at construction since it costs a `loggamma` per element; pass
`lognorm = false` to skip the cache (`lognorm(d)` then recomputes on demand).
"""
struct StdTDist{T, Tν, N, Tl} <: AbstractStdDist{T, N}
    ν::Tν
    lognorm::Tl
    dims::Dims{N}
end


# ----- log-pdf split ------------------------------------------------------
# `lognorm` for an array `ν` involves `loggamma((ν+1)/2)` per element — that's
# the expensive piece a caller will want to cache.

function _lognorm_tdist_per_elem(ν::Number)
    return loggamma((ν + 1) / 2) - loggamma(ν / 2) - oftype(float(ν), log(π)) / 2 - log(ν) / 2
end

@inline function _lognorm_tdist(ν::Number, N)
    return N * _lognorm_tdist_per_elem(ν)
end

@inline function _lognorm_tdist(ν::AbstractArray, N)
    return sum(_lognorm_tdist_per_elem, ν)
end

# Constructors + cached/uncached `lognorm` + per-element `_elem_dist` (shared with
# StdInverseGamma). `T` is derived from the parameter *eltype* promoted to a float, so an
# integer `ν` doesn't make `T = Int` (which would break the `T(±Inf)`/`T(NaN)` moments)
# and traced parameter arrays are not copied; the per-element kernels coerce via `float(ν)`.
@cached_scalar_std StdTDist ν _lognorm_tdist

@inline function _unnormed_kernel(d::StdTDist, z)
    ν = d.ν
    return -((ν + one(ν)) / 2) * log1p(z * z / ν)
end
@inline function _unnormed_kernel_sum(d::StdTDist, z)
    ν = d.ν
    s = zero(eltype(z))
    @trace for i in eachindex(z)
        νi = _getith(ν, i)
        log_term = log1p(abs2(_rgetindex(z, i)) / νi)
        s += ((νi + one(νi)) / 2) * log_term
    end
    return -s
end

unnormed_logpdf(d::StdTDist{T, <:Number, 0}, x::Number) where {T} = _unnormed_kernel(d, x)
function unnormed_logpdf(
        d::StdTDist{T, Tν, N}, x::AbstractArray{<:Number, N}
    ) where {T, Tν, N}
    return _unnormed_kernel_sum(d, x)
end

# ----- sampling -----------------------------------------------------------

# `T = Z / sqrt(W/ν)` with `Z ~ N(0, 1)` and `W ~ χ²(ν) = 2·Gamma(ν/2, 1)`.
@inline function _rand_tdist(rng::AbstractRNG, ν::Number)
    z = randn(rng)
    g = _rand_gamma(rng, ν / 2)
    return z / sqrt(2 * g / ν)
end

function Random.rand(rng::AbstractRNG, d::StdTDist{T, <:Number, 0}) where {T}
    return _rand_tdist(rng, d.ν)
end

function _std_rand!(rng::AbstractRNG, d::StdTDist{T}, x::AbstractArray) where {T}
    ν = d.ν
    @trace for i in eachindex(x)
        _rsetindex!(x, _rand_tdist(rng, _getith(ν, i)), i)
    end
    return x
end


# ----- support / moments --------------------------------------------------

# `@with_real` also emits the `::Real` overload that breaks the ambiguity with
# Distributions' generic `insupport(::ContinuousUnivariateDistribution, ::Real)`.
@with_real Dists.insupport(::StdTDist, ::Number) = true
Dists.insupport(d::StdTDist, x::AbstractArray) = size(d) == size(x)
Base.minimum(::StdTDist{T, <:Any, 0}) where {T} = T(-Inf)
Base.maximum(::StdTDist{T, <:Any, 0}) where {T} = T(Inf)

function Dists.mean(d::StdTDist{T, <:Real, 0}) where {T}
    return d.ν > 1 ? zero(T) : T(NaN)
end
function Dists.var(d::StdTDist{T, <:Real, 0}) where {T}
    return d.ν > 2 ? T(d.ν / (d.ν - 2)) : T(Inf)
end
@inline _t_elemmean(ν::Number, T) = ν > 1 ? zero(T) : T(NaN)
@inline _t_elemvar(ν::Number, T) = ν > 2 ? T(ν / (ν - 2)) : T(Inf)
function Dists.mean(d::StdTDist{T, <:Real, N}) where {T, N}
    return fill(_t_elemmean(d.ν, T), size(d))
end
function Dists.var(d::StdTDist{T, <:Real, N}) where {T, N}
    return fill(_t_elemvar(d.ν, T), size(d))
end
function Dists.mean(d::StdTDist{T, <:AbstractArray, N}) where {T, N}
    return _t_elemmean.(d.ν, T)
end
function Dists.var(d::StdTDist{T, <:AbstractArray, N}) where {T, N}
    return _t_elemvar.(d.ν, T)
end


# ----- cdf / quantile

@inline function _t_elem_cdf(ν, x)
    a = ν / 2
    b = oftype(float(ν), 0.5)
    arg = ν / (ν + x * x)
    P_arg = first(SpecialFunctions.beta_inc(a, b, arg))
    return ifelse(x >= zero(x), one(x) - P_arg / 2, P_arg / 2)
end
@inline function _t_elem_quantile(ν, p)
    a = ν / 2
    b = oftype(float(ν), 0.5)
    p_in = ifelse(p < oftype(p, 0.5), 2 * p, 2 * (one(p) - p))
    q_in = one(p) - p_in
    arg = first(SpecialFunctions.beta_inc_inv(a, b, p_in, q_in))
    z_abs = sqrt(ν * (one(ν) / arg - one(ν)))
    return ifelse(p < oftype(p, 0.5), -z_abs, z_abs)
end

function Dists.cdf(d::StdTDist{T, <:Number, 0}, x::Number) where {T}
    return _t_elem_cdf(d.ν, x)
end
function Dists.quantile(d::StdTDist{T, <:Number, 0}, p::Number) where {T}
    return _t_elem_quantile(d.ν, p)
end
