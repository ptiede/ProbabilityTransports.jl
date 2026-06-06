# ----- PushforwardDistribution --------------------------------------------
#
# The distribution of `f(Z)` for `Z ~ base` and an invertible map `f` that exposes
# `InverseFunctions.inverse` and `ChangesOfVariables.with_logabsdet_jacobian`
# (e.g. `ScaleShift`, `AffineTransform`). This is the distribution-side dual of
# `PushforwardTransport`: it reuses the exact same map machinery, so the density
# is the change of variables and the transport is just `f` wrapped around the
# base's transport. Concretely:
#
#   PushforwardDistribution(AffineTransform(μ, A), StdNormal(K))   # ≡ MvNormal(μ, A*A')
#   PushforwardDistribution(ScaleShift(loc, scale), base)          # element-wise loc-scale
#
# Transporting to a *matching* space (e.g. `StdNormal -> StdNormal`) collapses the
# inner to the identity, so the whole transport is just `f` — no cdf/quantile.
#
# `lognorm` (the constant component of the logpdf) is computed at construction and stored.
# It splits as the base's own `lognorm(base)` plus the map's inverse-map log-det
# `map_lognorm(f, base)` — the latter is the extension point: to support a custom `f`
# whose log-det is data-independent, add a `map_lognorm` method for it.

struct PushforwardDistribution{F, D <: Dists.Distribution, N, L} <:
       Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}
    f::F
    base::D
    lognorm::L
end
function PushforwardDistribution(
        f, base::Dists.Distribution{Dists.ArrayLikeVariate{N}}
    ) where {N}
    ℓ = lognorm(base) + map_lognorm(f, base)
    return PushforwardDistribution{typeof(f), typeof(base), N, typeof(ℓ)}(f, base, ℓ)
end

# The map's inverse-map log-det `logabsdet(∂f⁻¹/∂x)` — the map's contribution to the
# constant `lognorm`. Defined ONLY for maps whose log-det is data-independent (the shipped
# affine maps); there is intentionally no generic fallback, so a map without a
# `map_lognorm` method hits a `MethodError` *here* (a clear "define this for your `f`"
# signal) instead of silently storing a wrong "constant" for an `f` whose log-det varies
# with the data.
map_lognorm(f::ScaleShift{<:Any, <:Number}, base) = -length(base) * log(abs(f.s))
map_lognorm(f::ScaleShift{<:Any, <:AbstractArray}, _) = -sum(log ∘ abs, f.s)
map_lognorm(f::AffineTransform, _) = -_logabsdet(f.L)

Base.size(d::PushforwardDistribution) = size(d.base)
Base.length(d::PushforwardDistribution) = length(d.base)
Base.eltype(d::PushforwardDistribution) = _pf_eltype(d.f, d.base)
_pf_eltype(f::ScaleShift, base) = promote_type(eltype(f.μ), eltype(f.s), eltype(base))
_pf_eltype(f::AffineTransform, base) = promote_type(eltype(f.μ), eltype(f.L), eltype(base))
_pf_eltype(::Any, base) = eltype(base)

# loc / scale accessors (both affine maps carry `μ`; the scale is `s` or `L`)
_pf_scale(f::ScaleShift) = f.s
_pf_scale(f::AffineTransform) = f.L

# support: x is in support iff f⁻¹(x) is in the base's support
Dists.insupport(d::PushforwardDistribution{<:Any, <:Any, 0}, x::Number) =
    Dists.insupport(d.base, inverse(d.f)(x))
Dists.insupport(d::PushforwardDistribution{<:Any, <:Any, 0}, x::Real) =
    Dists.insupport(d.base, inverse(d.f)(x))
function Dists.insupport(d::PushforwardDistribution, x::AbstractArray)
    size(x) == size(d) || return false
    return all(zi -> Dists.insupport(d.base, zi), inverse(d.f)(x))
end

# Change of variables: `x = f(z)`, `z ~ base`  ⇒
#   logpdf(x) = logpdf(base, f⁻¹(x)) + logabsdet(∂f⁻¹/∂x)
# and `with_logabsdet_jacobian(inverse(f), x)` returns exactly `(f⁻¹(x), that log-det)`.
@inline function _pushforward_logpdf(d::PushforwardDistribution, x)
    z, ℓ = with_logabsdet_jacobian(inverse(d.f), x)
    return Dists.logpdf(d.base, z) + ℓ
end
Dists.logpdf(d::PushforwardDistribution{<:Any, <:Any, 0}, x::Number) = _pushforward_logpdf(d, x)
# `<:Number` accepts traced arrays (`TracedRNumber <: Number`); the `<:Real` method
# is more specific on both args and breaks the ambiguity with Distributions' generic
# `logpdf(::ContinuousDistribution{ArrayLikeVariate{N}}, ::AbstractArray{<:Real,N})`.
Dists.logpdf(d::PushforwardDistribution{<:Any, <:Any, N}, x::AbstractArray{<:Number, N}) where {N} =
    _pushforward_logpdf(d, x)
Dists.logpdf(d::PushforwardDistribution{<:Any, <:Any, N}, x::AbstractArray{<:Real, N}) where {N} =
    _pushforward_logpdf(d, x)

# Cache split (the AbstractStdDist convention `logpdf = unnormed_logpdf + lognorm`):
# the inverse-map log-det is data-INDEPENDENT, so it lives in `lognorm` — precomputed at
# construction as `lognorm(base) + map_lognorm(f, base)` and stored in `d.lognorm`; only
# `unnormed_logpdf(base, f⁻¹(x))` depends on the data. The split exists exactly when
# `map_lognorm` is defined for `f` (the shipped affine maps); a `f` whose log-det varies
# with the data has no `map_lognorm` method, so no such distribution is built in the first place.
unnormed_logpdf(d::PushforwardDistribution, x) = unnormed_logpdf(d.base, inverse(d.f)(x))
lognorm(d::PushforwardDistribution) = d.lognorm

# affine maps push the mean through exactly: E[f(Z)] = f(E[Z])
Dists.mean(d::PushforwardDistribution) = d.f(Dists.mean(d.base))

Dists.rand(rng::AbstractRNG, d::PushforwardDistribution) = d.f(rand(rng, d.base))

# Two thin entries delegating to `_pf_rand!`. The `<:Real` method breaks the ambiguity
# with `Distributions._rand!(::Sampleable{<:ArrayLikeVariate}, ::AbstractArray{<:Real})`
# (strictly more specific in the distribution argument); the `<:Number` method admits
# traced (Reactant) arrays, whose eltype is not `<:Real`.
function _pf_rand!(rng::AbstractRNG, d::PushforwardDistribution, x::AbstractArray)
    x .= d.f(rand(rng, d.base))
    return x
end
Dists._rand!(rng::AbstractRNG, d::PushforwardDistribution, x::AbstractArray{<:Number}) =
    _pf_rand!(rng, d, x)
Dists._rand!(rng::AbstractRNG, d::PushforwardDistribution, x::AbstractArray{<:Real}) =
    _pf_rand!(rng, d, x)

# moments: element-wise scale ⇒ `var = scale² var(base)`; matrix scale ⇒
# `cov = A cov(base) Aᵀ` (a genuine covariance), `var = diag(cov)`.
Dists.var(d::PushforwardDistribution{<:ScaleShift}) = _pf_scale(d.f) .^ 2 .* Dists.var(d.base)
Dists.std(d::PushforwardDistribution) = sqrt.(Dists.var(d))
function Dists.cov(d::PushforwardDistribution{<:AffineTransform, <:Any, 1})
    A = d.f.L
    return A * Dists.cov(d.base) * transpose(A)
end
Dists.var(d::PushforwardDistribution{<:AffineTransform, <:Any, 1}) = diag(Dists.cov(d))

# cdf/quantile for a scalar (0-dim) base: `cdf_x(x) = cdf_z(f⁻¹(x))`,
# `quantile_x(p) = f(quantile_z(p))` (branchless ⇒ traces under Reactant).
Dists.cdf(d::PushforwardDistribution{<:ScaleShift, <:Any, 0}, x::Number) =
    _std_cdf(d.base, inverse(d.f)(x))
Dists.quantile(d::PushforwardDistribution{<:ScaleShift, <:Any, 0}, p::Number) =
    d.f(_std_quantile(d.base, p))

# log-cdf / log-ccdf (used by `Truncated`'s constructor). `log(cdf)`/`log1p(-cdf)`
# — reuses `cdf`, accepts traced numbers (not just `<:Real`), and traces. The
# `<:Real` methods break the ambiguity with Distributions' generic logcdf/logccdf.
Dists.logcdf(d::PushforwardDistribution{<:ScaleShift, <:Any, 0}, x::Number) = log(Dists.cdf(d, x))
Dists.logcdf(d::PushforwardDistribution{<:ScaleShift, <:Any, 0}, x::Real) = log(Dists.cdf(d, x))
Dists.logccdf(d::PushforwardDistribution{<:ScaleShift, <:Any, 0}, x::Number) = log1p(-Dists.cdf(d, x))
Dists.logccdf(d::PushforwardDistribution{<:ScaleShift, <:Any, 0}, x::Real) = log1p(-Dists.cdf(d, x))

function Base.show(io::IO, d::PushforwardDistribution)
    print(io, "PushforwardDistribution(f=", nameof(typeof(d.f)), ", base=", nameof(typeof(d.base)))
    sz = size(d)
    isempty(sz) || print(io, ", size=", sz)
    return print(io, ")")
end

# Transport (Std spaces): wrap the base's transport in `f`. For `base === space` the
# inner is the matching-base identity, so this is `f` over an identity (no cdf/quantile).
# (`TVFlat` is handled in the TV extension, where `f` wraps the base's TV transform.)
transport_node(d::PushforwardDistribution, space) =
    PushforwardTransport(d.f, transport_node(d.base, space))
