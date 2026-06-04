struct StdTDist{T, Tν, N} <: AbstractStdDist{T, N}
    ν::Tν
    dims::Dims{N}
end
# Store `ν` as-is; derive the output eltype `T` as the parameter type promoted to
# a float. This keeps an integer `ν` from making `T = Int` (which would break the
# `T(±Inf)`/`T(NaN)` moments) without copying float/traced parameter arrays.
# `float(::Type)` resolves for traced eltypes inside a trace — the only place they
# exist — and the per-element kernels coerce via the value-level `float(ν)`.
StdTDist(ν::Number) = StdTDist{float(typeof(ν)), typeof(ν), 0}(ν, ())
StdTDist(ν::Number, dims::Dims{N}) where {N} = StdTDist{float(typeof(ν)), typeof(ν), N}(ν, dims)
StdTDist(ν::Number, dims::Int...) = StdTDist(ν, dims)
StdTDist(ν::AbstractArray{<:Number, N}) where {N} =
    StdTDist{float(eltype(ν)), typeof(ν), N}(ν, size(ν))


# ----- log-pdf split ------------------------------------------------------
# `lognorm` for an array `ν` involves `loggamma((ν+1)/2)` per element — that's
# the expensive piece a caller will want to cache.

@inline function _t_lognorm_per_elem(ν)
    return loggamma((ν + 1) / 2) - loggamma(ν / 2) - oftype(float(ν), log(π)) / 2 - log(ν) / 2
end

@inline function _unnormed_kernel(d::StdTDist, z)
    ν = d.ν
    return -((ν + one(ν)) / 2) * log1p(z * z / ν)
end
@inline function _unnormed_kernel_sum(d::StdTDist, z)
    ν = d.ν
    sq = abs2.(z)
    log_terms = log1p.(sq ./ ν)
    return -sum(((ν .+ 1) ./ 2) .* log_terms)
end

unnormed_logpdf(d::StdTDist{T, <:Number, 0}, x::Number) where {T} = _unnormed_kernel(d, x)
function unnormed_logpdf(
        d::StdTDist{T, Tν, N}, x::AbstractArray{<:Number, N}
    ) where {T, Tν, N}
    return _unnormed_kernel_sum(d, x)
end

@inline lognorm(d::StdTDist{T, <:Number}) where {T} = length(d) * _t_lognorm_per_elem(d.ν)
@inline lognorm(d::StdTDist{T, <:AbstractArray}) where {T} = sum(_t_lognorm_per_elem, d.ν)


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
# NOTE: the array sampler still uses a per-element loop. On CPU it is correct; under
# Reactant `@compile` the loop does not thread the RNG (see `_rand_gamma` in
# `misc.jl`), so array `StdTDist` sampling is not yet traceable — scalar sampling is.
function _std_rand!(rng::AbstractRNG, d::StdTDist{T, <:Number}, x::AbstractArray) where {T}
    ν = d.ν
    @trace for i in eachindex(x)
        x[i] = _rand_tdist(rng, ν)
    end
    return x
end
function _std_rand!(rng::AbstractRNG, d::StdTDist{T, <:AbstractArray}, x::AbstractArray) where {T}
    @trace for i in eachindex(x)
        x[i] = _rand_tdist(rng, d.ν[i])
    end
    return x
end


# ----- support / moments --------------------------------------------------

Dists.insupport(::StdTDist, ::Number) = true
# `<:Real` overload breaks ambiguity with Distributions' generic
# `insupport(::ContinuousUnivariateDistribution, ::Number)`.
Dists.insupport(::StdTDist, ::Real) = true
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


# ----- cdf / quantile -----------------------------------------------------
# Built on the regularised incomplete beta function. With `a = ν/2`, `b = 1/2`,
# and `arg = ν / (ν + z²)`:
#   cdf(z) = 1 - I(arg; a, b) / 2   for z >= 0
#   cdf(z) =     I(arg; a, b) / 2   for z <  0
# Quantile inverts the same identity.
# Element-wise kernels take `ν` directly so they broadcast against either a
# scalar or a per-element parameter array — used by the ArrayHC ascube path
# below for the per-element-ν specialisation.

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

@inline _std_cdf(d::StdTDist, x) = _t_elem_cdf(d.ν, x)
@inline _std_quantile(d::StdTDist, p) = _t_elem_quantile(d.ν, p)

function Dists.cdf(d::StdTDist{T, <:Number, 0}, x::Number) where {T}
    return _std_cdf(d, x)
end
function Dists.quantile(d::StdTDist{T, <:Number, 0}, p::Number) where {T}
    return _std_quantile(d, p)
end
