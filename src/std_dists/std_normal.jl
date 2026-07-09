"""
    StdNormal([T=Float64], dims...)

Standard zero-mean unit-variance normal of shape `dims` (a scalar when `dims` is empty).
Both a transportable base distribution **and** a valid target space for [`transport_to`](@ref):
it is the matching base for every Gaussian image prior, where its transport is the vectorized
identity (no `cdf`/`quantile`), keeping those priors fast and Reactant-traceable.
"""
struct StdNormal{T, N} <: AbstractStdDist{T, N}
    dims::Dims{N}
end
StdNormal(d::Dims{N}) where {N} = StdNormal{Float64, N}(d)
StdNormal(d::Int...) = StdNormal(d)
# element-type-preserving constructors (used by `basemeasure`)
StdNormal{T}(d::Dims{N}) where {T, N} = StdNormal{T, N}(d)
StdNormal{T}(d::Int...) where {T} = StdNormal{T}(d)

@with_real Dists.insupport(::StdNormal, ::Number) = true
# Support is all of ℝ, but still validate the shape (matching the other `Std*`).
Dists.insupport(d::StdNormal, x::AbstractArray) = size(d) == size(x)
Base.minimum(::StdNormal{T, 0}) where {T} = T(-Inf)
Base.maximum(::StdNormal{T, 0}) where {T} = T(Inf)


# ----- log-pdf split
@inline _unnormed_kernel(::StdNormal, z) = -z * z / 2
# `init` keeps a zero-length reference (e.g. the latent measure of an empty prior) at
# logpdf 0 instead of erroring on an empty reduction.
@inline _unnormed_kernel_sum(::StdNormal, z) = -sum(abs2, z; init = zero(eltype(z))) / 2

unnormed_logpdf(d::StdNormal{T, 0}, x::Number) where {T} = _unnormed_kernel(d, x)
function unnormed_logpdf(d::StdNormal{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return _unnormed_kernel_sum(d, x)
end

@inline lognorm(d::StdNormal) = -length(d) * oftype(zero(eltype(d)), log(2π) / 2)


# ----- sampling

Random.rand(rng::AbstractRNG, ::StdNormal{T, 0}) where {T} = randn(rng, T)
_std_rand!(rng::AbstractRNG, ::StdNormal, x::AbstractArray) = randn!(rng, x)


# ----- moments
Dists.mean(::StdNormal{T, 0}) where {T} = zero(T)
Dists.var(::StdNormal{T, 0}) where {T} = one(T)
Dists.std(::StdNormal{T, 0}) where {T} = one(T)
Dists.mean(d::StdNormal) = zeros(eltype(d), size(d))
Dists.var(d::StdNormal) = ones(eltype(d), size(d))
Dists.cov(d::StdNormal) = I(length(d))


# cdf / quantile. Only the 0-dim form is meaningful (a multivariate Std dist has no scalar
# cdf); the `space_*` trait and pushforward reach it through the 0-dim element / base — see
# interface.jl and pushforward_distribution.jl.
Dists.cdf(::StdNormal{T, 0}, x::Number) where {T} = (one(x) + erf(x / sqrt(oftype(x, 2)))) / 2        # Φ
Dists.quantile(::StdNormal{T, 0}, p::Number) where {T} = sqrt(oftype(p, 2)) * erfinv(2 * p - one(p)) # Φ⁻¹

# the array-transport element (see `_elem_dist` in interface.jl): parameter-free, ignores `i`
_elem_dist(::StdNormal{T}, i; lognorm::Bool = false) where {T} = StdNormal{T}()
