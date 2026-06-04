# StdNormal — standard zero-mean unit-variance normal of arbitrary shape.
# Pre-existed in `src/srf.jl` (used by `StationaryRandomField`); we moved it
# here for consistency with the other Std bases. The struct itself is
# unchanged so existing references in `srf.jl` and `markovrf/gmrf.jl` remain
# valid.

struct StdNormal{T, N} <: AbstractStdDist{T, N}
    dims::Dims{N}
end
StdNormal(d::Dims{N}) where {N} = StdNormal{Float64, N}(d)
StdNormal(d::Int...) = StdNormal(d)
# element-type-preserving constructors (used by `basemeasure`)
StdNormal{T}(d::Dims{N}) where {T, N} = StdNormal{T, N}(d)
StdNormal{T}(d::Int...) where {T} = StdNormal{T}(d)

Dists.insupport(::StdNormal, ::Number) = true
Dists.insupport(::StdNormal, ::Real) = true
Dists.insupport(::StdNormal, x::AbstractArray) = true
Base.minimum(::StdNormal{T, 0}) where {T} = T(-Inf)
Base.maximum(::StdNormal{T, 0}) where {T} = T(Inf)


# ----- log-pdf split ------------------------------------------------------
# `unnormed_logpdf(d, x)` returns only the data-dependent part; `lognorm(d)`
# returns the constant. `logpdf = unnormed_logpdf + lognorm`.

@inline _unnormed_kernel(::StdNormal, z, _) = -z * z / 2

# `sum(abs2, z)` is non-allocating on CPU and Reactant supports the
# mapreduce form (see existing test at `test/reactant.jl` / `srf.jl:412`).
@inline _unnormed_kernel_sum(::StdNormal, z) = -sum(abs2, z) / 2

unnormed_logpdf(d::StdNormal{T, 0}, x::Number) where {T} = _unnormed_kernel(d, x, 1)
function unnormed_logpdf(d::StdNormal{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return _unnormed_kernel_sum(d, x)
end

@inline lognorm(d::StdNormal) = -length(d) * oftype(zero(eltype(d)), log(2π) / 2)


# ----- sampling -----------------------------------------------------------

Random.rand(rng::AbstractRNG, ::StdNormal{T, 0}) where {T} = T(randn(rng))
function Dists._rand!(
        rng::AbstractRNG, ::StdNormal{T, N}, x::AbstractArray{<:Real, N}
    ) where {T, N}
    return randn!(rng, x)
end


# ----- moments ------------------------------------------------------------

Dists.mean(::StdNormal{T, 0}) where {T} = zero(T)
Dists.var(::StdNormal{T, 0}) where {T} = one(T)
Dists.std(::StdNormal{T, 0}) where {T} = one(T)
Dists.mean(d::StdNormal) = zeros(eltype(d), size(d))
Dists.var(d::StdNormal) = ones(eltype(d), size(d))
Dists.cov(d::StdNormal) = I(length(d))


# ----- cdf / quantile -----------------------------------------------------

@inline _std_cdf(::StdNormal, x) = (one(x) + erf(x / sqrt(oftype(x, 2)))) / 2
@inline _std_quantile(::StdNormal, p) = sqrt(oftype(p, 2)) * erfinv(2 * p - one(p))

Dists.cdf(d::StdNormal, x::Number) = _std_cdf(d, x)
Dists.quantile(d::StdNormal, p::Number) = _std_quantile(d, p)