struct StdExponential{T, N} <: AbstractStdDist{T, N}
    dims::Dims{N}
end
StdExponential(dims::Dims{N}) where {N} = StdExponential{Float64, N}(dims)
StdExponential(dims::Int...) = StdExponential(dims)
StdExponential() = StdExponential{Float64, 0}(())

# ----- log-pdf split ------------------------------------------------------

@inline function _unnormed_kernel(::StdExponential, z, _)
    return ifelse(z >= zero(z), -z, oftype(z, -Inf))
end
@inline _unnormed_kernel_sum(::StdExponential, z) = -sum(z)

function unnormed_logpdf(d::StdExponential{T, 0}, x::Number) where {T}
    return _unnormed_kernel(d, x, 1)
end
function unnormed_logpdf(
        d::StdExponential{T, N}, x::AbstractArray{<:Number, N}
    ) where {T, N}
    return _unnormed_kernel_sum(d, x)
end

@inline lognorm(d::StdExponential) = zero(eltype(d))


# ----- sampling -----------------------------------------------------------

# `randexp(rng, T)` (not `T(randexp(rng))`): see `std_normal.jl` — the typed draw
# avoids the `T(::TracedRNumber)` coercion that has no method under Reactant.
Random.rand(rng::AbstractRNG, ::StdExponential{T, 0}) where {T} = randexp(rng, T)
_std_rand!(rng::AbstractRNG, ::StdExponential, x::AbstractArray) = randexp!(rng, x)


# ----- support / moments --------------------------------------------------

Dists.insupport(::StdExponential, x::Number) = x >= 0
# `<:Real` overload breaks ambiguity with Distributions' generic
# `insupport(::ContinuousUnivariateDistribution, ::Real)`.
Dists.insupport(::StdExponential, x::Real) = x >= 0
function Dists.insupport(d::StdExponential, x::AbstractArray)
    return size(d) == size(x) && all(>=(0), x)
end
Base.minimum(::StdExponential{T, 0}) where {T} = zero(T)
Base.maximum(::StdExponential{T, 0}) where {T} = T(Inf)

Dists.mean(::StdExponential{T, 0}) where {T} = one(T)
Dists.var(::StdExponential{T, 0}) where {T} = one(T)
Dists.mean(d::StdExponential) = fill(one(eltype(d)), size(d))
Dists.var(d::StdExponential) = fill(one(eltype(d)), size(d))


# ----- cdf / quantile -----------------------------------------------------

@inline _std_cdf(::StdExponential, x) = -expm1(-x)
@inline _std_quantile(::StdExponential, p) = -log1p(-p)

Dists.cdf(d::StdExponential{T, 0}, x::Number) where {T} = _std_cdf(d, x)
Dists.quantile(d::StdExponential{T, 0}, p::Number) where {T} = _std_quantile(d, p)