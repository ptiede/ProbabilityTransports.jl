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

struct PushforwardDistribution{F, D <: Dists.Distribution, N} <:
       Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}
    f::F
    base::D
end
function PushforwardDistribution(
        f, base::Dists.Distribution{Dists.ArrayLikeVariate{N}}
    ) where {N}
    return PushforwardDistribution{typeof(f), typeof(base), N}(f, base)
end

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
# the inverse-map log-det of an affine map is data-INDEPENDENT, so it lives in
# `lognorm`; only `unnormed_logpdf(base, f⁻¹(x))` depends on the data. (For a
# general non-affine `f` use `logpdf` directly — there is no constant split.)
unnormed_logpdf(d::PushforwardDistribution, x) = unnormed_logpdf(d.base, inverse(d.f)(x))
lognorm(d::PushforwardDistribution) = lognorm(d.base) + _inv_map_logabsdet(d.f, d.base)
_inv_map_logabsdet(f::ScaleShift{<:Any, <:Number}, base) = -length(base) * log(abs(f.s))
_inv_map_logabsdet(f::ScaleShift{<:Any, <:AbstractArray}, _) = -sum(log ∘ abs, f.s)
_inv_map_logabsdet(f::AffineTransform, _) = -_logabsdet(f.L)

# affine maps push the mean through exactly: E[f(Z)] = f(E[Z])
Dists.mean(d::PushforwardDistribution) = d.f(Dists.mean(d.base))

Dists.rand(rng::AbstractRNG, d::PushforwardDistribution) = d.f(rand(rng, d.base))

function Dists._rand!(rng::AbstractRNG, d::PushforwardDistribution, x::AbstractArray)
    x .= d.f(rand(rng, d.base))
    return x
end

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

# Transport: wrap the base's transport in `f`. For `base === space` the inner is
# the matching-base identity, so this is `f` over an identity (no cdf/quantile).
transport_node(d::PushforwardDistribution, space) =
    PushforwardTransport(d.f, transport_node(d.base, space))
# A 0-dim pushforward is a `ContinuousUnivariateDistribution`, so disambiguate the
# flat space from the generic univariate `transport_node(_, ::StdFlat)` (TV ext).
transport_node(d::PushforwardDistribution, space::StdFlat) =
    PushforwardTransport(d.f, transport_node(d.base, space))
