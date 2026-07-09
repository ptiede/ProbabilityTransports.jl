"""
    StdExponential([T=Float64], dims...)

Standard exponential (unit rate) of shape `dims`, supported on `z ≥ 0`. A transportable base
distribution, **not** a valid target space for [`transport_to`](@ref) (it lacks the `space_*`
trait, so targeting it errors at build time).
"""
struct StdExponential{T, N} <: AbstractStdDist{T, N}
    dims::Dims{N}
end
StdExponential(dims::Dims{N}) where {N} = StdExponential{Float64, N}(dims)
StdExponential(dims::Int...) = StdExponential(dims)
StdExponential() = StdExponential{Float64, 0}(())

StdExponential{T}(dims::Dims{N}) where {T, N} = StdExponential{T, N}(dims)
StdExponential{T}(dims::Int...) where {T} = StdExponential{T}(dims)

# ----- log-pdf split ------------------------------------------------------

@inline function _unnormed_kernel(d::StdExponential, z)
    return ifelse(Dists.insupport(d, z), -z, oftype(z, -Inf))
end
@inline _unnormed_kernel_sum(d::StdExponential, z) = ifelse(Dists.insupport(d, z), -sum(z), oftype(zero(eltype(z)), -Inf))

function unnormed_logpdf(d::StdExponential{T, 0}, x::Number) where {T}
    return _unnormed_kernel(d, x)
end
function unnormed_logpdf(
        d::StdExponential{T, N}, x::AbstractArray{<:Number, N}
    ) where {T, N}
    return _unnormed_kernel_sum(d, x)
end

@inline lognorm(d::StdExponential) = zero(eltype(d))


# ----- sampling

Random.rand(rng::AbstractRNG, ::StdExponential{T, 0}) where {T} = randexp(rng, T)
_std_rand!(rng::AbstractRNG, ::StdExponential, x::AbstractArray) = randexp!(rng, x)


# ----- support / moments
# `@with_real` also emits the `::Real` overload that breaks the ambiguity with
# Distributions' generic `insupport(::ContinuousUnivariateDistribution, ::Real)`.
@with_real Dists.insupport(::StdExponential, x::Number) = x >= 0
function Dists.insupport(d::StdExponential, x::AbstractArray)
    size(d) == size(x) || return false
    return all(>=(0), x)
end
Base.minimum(::StdExponential{T, 0}) where {T} = zero(T)
Base.maximum(::StdExponential{T, 0}) where {T} = T(Inf)

Dists.mean(::StdExponential{T, 0}) where {T} = one(T)
Dists.var(::StdExponential{T, 0}) where {T} = one(T)
Dists.mean(d::StdExponential) = fill(one(eltype(d)), size(d))
Dists.var(d::StdExponential) = fill(one(eltype(d)), size(d))


# ----- cdf / quantile -----------------------------------------------------

Dists.cdf(::StdExponential{T, 0}, x::Number) where {T} = -expm1(-x)
Dists.quantile(::StdExponential{T, 0}, p::Number) where {T} = -log1p(-p)

# the array-transport element (see `_elem_dist` in interface.jl): parameter-free, ignores `i`
_elem_dist(::StdExponential{T}, i; lognorm::Bool = false) where {T} = StdExponential{T}()
