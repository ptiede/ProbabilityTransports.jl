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


# cdf / quantile

# Branchless elementary `erf` (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7), built from only
# +,-,*,/,exp,abs,copysign — so it lowers to `stablehlo`/`arith` constants rather than
# `chlo.erf`. This sidesteps an Enzyme-JAX batching bug (EnzymeAD/Enzyme-JAX#2559): the reverse
# rule of `chlo.erf` emits its coefficient `2/√π` as a `chlo.constant`; when the surrounding
# function is then batched (e.g. an N-sample Monte-Carlo estimator), `EnzymeBatchPass` resizes
# the constant's result type but not its `value` attribute, producing a malformed `chlo.constant`
# that fails MLIR verification. An elementary erf keeps every constant in dialects whose batched
# constants are repaired downstream, so it compiles cleanly. Remove once #2559 lands.
@inline function _erf_poly(x)
    o = one(x)
    p = oftype(x, 0.3275911)
    a1 = oftype(x, 0.254829592)
    a2 = oftype(x, -0.284496736)
    a3 = oftype(x, 1.421413741)
    a4 = oftype(x, -1.453152027)
    a5 = oftype(x, 1.061405429)
    ax = abs(x)
    t = o / (o + p * ax)
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    return copysign(o - poly * exp(-ax * ax), x)
end

# TODO: simplify when EnzymeAD/Enzyme-JAX#2559 is merged and a new version is tagged and released
@inline _erf_std(x) = within_compile() ? _erf_poly(x) : erf(x)
@inline _std_cdf(::StdNormal, x) = (one(x) + _erf_std(x / sqrt(oftype(x, 2)))) / 2
@inline _std_quantile(::StdNormal, p) = sqrt(oftype(p, 2)) * erfinv(2 * p - one(p))

Dists.cdf(d::StdNormal, x::Number) = _std_cdf(d, x)
Dists.quantile(d::StdNormal, p::Number) = _std_quantile(d, p)

# transport spaces
space_cdf(d::StdNormal, y) = _std_cdf(d, y)        # Φ  (defined in StdDists/std_normal.jl)
space_quantile(d::StdNormal, u) = _std_quantile(d, u)  # Φ⁻¹
space_logpdf(::StdNormal, y) = -y * y / 2 - oftype(y, log(2π) / 2)
space_dimension(::Type{<:StdNormal}) = 1
