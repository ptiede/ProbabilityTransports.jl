"""
    AbstractStdDist{T, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}

Constructs a standardized distribution with element type `T` and `N` dimensions.
The `Std*` distributions are all subtypes of `AbstractStdDist` and share a common
interface, which is defined here.

To implement a new `Std*` distribution, define a struct subtype of `AbstractStdDist` and
implement the following methods:
- `unnormed_logpdf(d::YourStdDist, x)`
- `lognorm(d::YourStdDist)`
- `_std_rand!(rng, d::YourStdDist, x)`

You can also optionally implement
- `mean(d::YourStdDist)`
- `var(d::YourStdDist)`
- `std(d::YourStdDist)`
- `cov(d::YourStdDist)`
- `cdf(d::YourStdDist, x)`
- `quantile(d::YourStdDist, p)`
- `Base.size(d::YourStdDist)`
- `Base.length(d::YourStdDist)`
- `Base.eltype(d::YourStdDist)`
"""
abstract type AbstractStdDist{T, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}} end
dims(d::AbstractStdDist) = getfield(d, :dims)
Base.size(d::AbstractStdDist) = dims(d)
Base.length(d::AbstractStdDist) = prod(size(d))
Base.eltype(::Type{<:AbstractStdDist{T}}) where {T} = T
Base.eltype(::AbstractStdDist{T}) where {T} = T

function Dists.logpdf(d::AbstractStdDist{T, 0}, x::Number) where {T}
    return unnormed_logpdf(d, x) + lognorm(d)
end

@with_real function Dists.logpdf(d::AbstractStdDist{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end

# Array sampling, delegating to the per-distribution `_std_rand!` so the actual sampler
# lives in exactly one place. `@with_real` emits the `<:Number` overload (admits traced
# Reactant arrays) and the `<:Real` companion that breaks the ambiguity with
# `Distributions._rand!(::Sampleable{<:ArrayLikeVariate}, ::AbstractArray{<:Real})`.
@with_real function Dists._rand!(rng::AbstractRNG, d::AbstractStdDist{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return _std_rand!(rng, d, x)
end
