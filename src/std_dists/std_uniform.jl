"""
    StdUniform([T=Float64], dims...)

Standard uniform on the unit hypercube `[0,1]^dims` (a scalar `Uniform(0,1)` when `dims` is
empty). Both a transportable base distribution **and** a valid target space for
[`transport_to`](@ref); targeting it gives a flat latent density (e.g. for nested sampling).
"""
struct StdUniform{T, N} <: AbstractStdDist{T, N}
    dims::Dims{N}
end
StdUniform(dims::Dims{N}) where {N} = StdUniform{Float64, N}(dims)
StdUniform(dims::Int...) = StdUniform(dims)
StdUniform() = StdUniform{Float64, 0}(())
# element-type-preserving constructors (used by `basemeasure`)
StdUniform{T}(dims::Dims{N}) where {T, N} = StdUniform{T, N}(dims)
StdUniform{T}(dims::Int...) where {T} = StdUniform{T}(dims)


# ----- log-pdf split ------------------------------------------------------

@inline function _unnormed_kernel(d::StdUniform, z)
    return ifelse(Dists.insupport(d, z), zero(z), oftype(z, -Inf))
end
# sum the branchless per-element kernel: 0 if every element ∈ [0,1], else -Inf
# (uses `ifelse`, so it still traces under Reactant).
@inline _unnormed_kernel_sum(d::StdUniform, z) = sum(Base.Fix1(_unnormed_kernel, d), z)

unnormed_logpdf(d::StdUniform{T, 0}, x::Number) where {T} = _unnormed_kernel(d, x)
function unnormed_logpdf(d::StdUniform{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return _unnormed_kernel_sum(d, x)
end
@inline lognorm(d::StdUniform) = zero(eltype(d))

# ----- sampling 

Random.rand(rng::AbstractRNG, ::StdUniform{T, 0}) where {T} = rand(rng, T)
_std_rand!(rng::AbstractRNG, ::StdUniform, x::AbstractArray) = rand!(rng, x)


# ----- support / moments 

# `@with_real` also emits the `::Real` overload that breaks the ambiguity with
# Distributions' generic `insupport(::ContinuousUnivariateDistribution, ::Real)`.
@with_real Dists.insupport(::StdUniform, x::Number) = (0 <= x <= 1)
function Dists.insupport(d::StdUniform, x::AbstractArray)
    return size(d) == size(x) && all(xi -> 0 <= xi <= 1, x)
end
Base.minimum(::StdUniform{T, 0}) where {T} = zero(T)
Base.maximum(::StdUniform{T, 0}) where {T} = one(T)

Dists.mean(::StdUniform{T, 0}) where {T} = T(0.5)
Dists.var(::StdUniform{T, 0}) where {T} = T(1) / T(12)
Dists.mean(d::StdUniform) = fill(eltype(d)(0.5), size(d))
Dists.var(d::StdUniform) = fill(eltype(d)(1) / eltype(d)(12), size(d))


# ----- cdf / quantile 

@inline _std_cdf(::StdUniform, x) = clamp(x, zero(x), one(x))
@inline _std_quantile(::StdUniform, p) = p

Dists.cdf(d::StdUniform{T, 0}, x::Number) where {T} = _std_cdf(d, x)
Dists.quantile(d::StdUniform{T, 0}, p::Number) where {T} = _std_quantile(d, p)

# ----- transport interface
space_cdf(::StdUniform, y) = clamp(y, zero(y), one(y))
space_quantile(::StdUniform, u) = u
space_logpdf(::StdUniform, y) = zero(y)
space_dimension(::Type{<:StdUniform}) = 1
