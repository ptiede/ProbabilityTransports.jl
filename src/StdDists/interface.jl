abstract type AbstractStdDist{T, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}} end
dims(d::AbstractStdDist) = getfield(d, :dims)
Base.size(d::AbstractStdDist) = dims(d)
Base.length(d::AbstractStdDist) = prod(size(d))
Base.eltype(::AbstractStdDist{T}) where {T} = T

function Dists.logpdf(d::AbstractStdDist{T, 0}, x::Number) where {T}
    return unnormed_logpdf(d, x) + lognorm(d)
end

function Dists.logpdf(d::AbstractStdDist{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end

function Dists.logpdf(d::AbstractStdDist{T, N}, x::AbstractArray{<:Real, N}) where {T, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end

