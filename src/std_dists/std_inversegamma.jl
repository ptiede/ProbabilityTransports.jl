# StdInverseGamma — inverse-gamma with shape `α` and scale 1.
# pdf(z; α) = z^(-α-1) exp(-1/z) / Γ(α) for z > 0.
# `α` may be a scalar (broadcast across the support) or an array of the same
# shape as the distribution.

struct StdInverseGamma{T, Tα, N, Tl} <: AbstractStdDist{T, N}
    α::Tα
    lognorm::Tl
    dims::Dims{N}
    # Precompute the normalization because it is expensive. Store `α` as-is
    # (`Tα = typeof(α)` — scalar or array); derive the float output eltype `T` from its
    # element type so an integer `α` doesn't make `T = Int`.
    function StdInverseGamma(α::Union{Number, AbstractArray}, dims::Dims{N}) where {N}
        Tα = typeof(α)
        T = float(eltype(α))
        lognorm = _lognorm_igamma(α, prod(dims))
        return new{T, Tα, N, typeof(lognorm)}(α, lognorm, dims)
    end
end

# Compute the normalization 
@inline _lognorm_igamma(d::Number, N) = -N * loggamma(d)
@inline _lognorm_igamma(d::AbstractArray, N) = -sum(loggamma, d)

StdInverseGamma(α::Number) = StdInverseGamma(α, ())
StdInverseGamma(α::Number, dims::Int...) = StdInverseGamma(α, dims)
StdInverseGamma(α::AbstractArray) = StdInverseGamma(α, size(α))

# ----- log-pdf split ------------------------------------------------------
# `loggamma(α)` is the expensive piece for an array `α`; folding it into
# `lognorm` lets a caller cache it across many `logpdf` evaluations.

@inline function _unnormed_kernel(d::StdInverseGamma, z)
    α = d.α
    zsafe = ifelse(z > zero(z), z, oftype(z, 1))
    val = -(α + one(α)) * log(zsafe) - inv(zsafe)
    return ifelse(z > zero(z), val, oftype(z, -Inf))
end
@inline function _unnormed_kernel_sum(d::StdInverseGamma, z)
    α = d.α
    log_z = log.(z)
    inv_z = inv.(z)
    return -sum((α .+ 1) .* log_z) - sum(inv_z)
end

function unnormed_logpdf(d::StdInverseGamma{T, <:Number, 0}, x::Number) where {T}
    return _unnormed_kernel(d, x)
end
function unnormed_logpdf(
        d::StdInverseGamma{T, Tα, N}, x::AbstractArray{<:Number, N}
    ) where {T, Tα, N}
    return _unnormed_kernel_sum(d, x)
end

@inline lognorm(d::StdInverseGamma) = d.lognorm

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

Dists.insupport(::StdInverseGamma, x::Number) = x > 0
# `<:Real` overload breaks ambiguity with Distributions' generic
# `insupport(::ContinuousUnivariateDistribution, ::Real)`.
Dists.insupport(::StdInverseGamma, x::Real) = x > 0
function Dists.insupport(d::StdInverseGamma, x::AbstractArray)
    return size(d) == size(x) && all(>(0), x)
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
# scalar or a per-element parameter array — used by the ArrayHC ascube path
# below for the per-element-α specialisation.

@inline _ig_elem_cdf(α, x) = last(SpecialFunctions.gamma_inc(α, inv(x), 0))
@inline _ig_elem_quantile(α, p) = inv(SpecialFunctions.gamma_inc_inv(α, one(p) - p, p))

@inline _std_cdf(d::StdInverseGamma, x) = _ig_elem_cdf(d.α, x)
@inline _std_quantile(d::StdInverseGamma, p) = _ig_elem_quantile(d.α, p)

function Dists.cdf(d::StdInverseGamma{T, <:Number, 0}, x::Number) where {T}
    return _std_cdf(d, x)
end
function Dists.quantile(d::StdInverseGamma{T, <:Number, 0}, p::Number) where {T}
    return _std_quantile(d, p)
end