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
# A general `f` (e.g. exp) is supported: `logpdf` splits its inverse log-det into a
# data-independent CONSTANT, cached in `lognorm` at construction via `map_lognorm`,
# plus a data-dependent part evaluated per call via `with_logabsdet_jacobian`. For the
# affine family the constant IS the whole log-det (`const_logdet(f) == true`), so the
# per-call Jacobian is skipped entirely; for a general map the constant is zero and the
# genuine change of variables runs each call. To opt a custom constant-log-det map into
# the fast path, define `map_lognorm` AND `const_logdet` for it (they must agree:
# `const_logdet(f) == true` asserts `with_logabsdet_jacobian(inverse(f), x)`'s log-det
# always equals `map_lognorm(f, base)`).

"""
    PushforwardDistribution(f, base)

The distribution of `f(Z)` for `Z ~ base` and an invertible map `f` exposing
`InverseFunctions.inverse` and `ChangesOfVariables.with_logabsdet_jacobian` (e.g. `ScaleShift`,
`AffineTransform`). The density is the change of variables of `base` through `f`; the
data-independent part of the inverse log-det is cached in `lognorm` at construction. For example
`PushforwardDistribution(AffineTransform(μ, A), StdNormal(K))` is `MvNormal(μ, A*A')`.
"""
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

# The data-independent part of the map's inverse log-det `logabsdet(∂f⁻¹/∂x)` — the
# map's contribution to the cached `lognorm`. A general map contributes no constant
# (zero); its full log-det is computed per call in `unnormed_logpdf`.
map_lognorm(f, base) = zero(float(eltype(base)))
map_lognorm(f::ScaleShift{<:Any, <:Number}, base) = -length(base) * log(abs(f.s))
map_lognorm(f::ScaleShift{<:Any, <:AbstractArray}, _) = -sum(log ∘ abs, f.s)
map_lognorm(f::AffineTransform, _) = -_logabsdet(f.L)

# trait: is the inverse log-det a data-independent constant (fully captured by
# `map_lognorm`)? Compile-time constant per map type, so the `logpdf` branch folds.
const_logdet(::Any) = false
const_logdet(::ScaleShift) = true
const_logdet(::AffineTransform) = true

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
@with_real Dists.insupport(d::PushforwardDistribution{<:Any, <:Any, 0}, x::Number) =
    Dists.insupport(d.base, inverse(d.f)(x))
function Dists.insupport(d::PushforwardDistribution, x::AbstractArray)
    size(x) == size(d) || return false
    return all(zi -> Dists.insupport(d.base, zi), inverse(d.f)(x))
end

# Change of variables: `x = f(z)`, `z ~ base`  ⇒
#   logpdf(x) = logpdf(base, f⁻¹(x)) + logabsdet(∂f⁻¹/∂x)
# split as `unnormed_logpdf(d, x) + lognorm(d)` (the AbstractStdDist convention):
# `lognorm` holds every data-independent constant (the base's plus the map's via
# `map_lognorm`), `unnormed_logpdf` holds everything data-dependent. For a
# constant-log-det map (`const_logdet(f)`) the per-call Jacobian is skipped — the
# branch below folds at compile time since the trait is a constant per map type.
@inline function _pushforward_logpdf(d::PushforwardDistribution, x)
    return unnormed_logpdf(d, x) + lognorm(d)
end
Dists.logpdf(d::PushforwardDistribution{<:Any, <:Any, 0}, x::Number) = _pushforward_logpdf(d, x)
# `@with_real` emits the `<:Number` array overload (accepts traced arrays, since
# `TracedRNumber <: Number`) and the `<:Real` companion that breaks the ambiguity with
# Distributions' generic `logpdf(::ContinuousDistribution{ArrayLikeVariate{N}}, ::AbstractArray{<:Real,N})`.
@with_real Dists.logpdf(d::PushforwardDistribution{<:Any, <:Any, N}, x::AbstractArray{<:Number, N}) where {N} =
    _pushforward_logpdf(d, x)

# The data-dependent part of the density. Constant-log-det maps need only the base's
# unnormed density at `f⁻¹(x)` (their whole log-det sits in the cached `lognorm`);
# general maps add their per-call inverse log-det here.
@inline function unnormed_logpdf(d::PushforwardDistribution, x)
    if const_logdet(d.f)
        return unnormed_logpdf(d.base, inverse(d.f)(x))
    else
        z, ℓ = with_logabsdet_jacobian(inverse(d.f), x)
        return unnormed_logpdf(d.base, z) + ℓ
    end
end
lognorm(d::PushforwardDistribution) = d.lognorm

# affine maps push the mean through exactly: E[f(Z)] = f(E[Z]). A general nonlinear
# map does not, so no generic method.
Dists.mean(d::PushforwardDistribution{<:Union{ScaleShift, AffineTransform}}) =
    d.f(Dists.mean(d.base))

# Support endpoints of a scalar element-wise affine pushforward. The map is monotone
# (increasing for s > 0, decreasing for s < 0), so the endpoints are the mapped base
# endpoints in either order. Needed so `Truncated`/flat transforms see the true support.
function Base.minimum(d::PushforwardDistribution{<:ScaleShift{<:Number, <:Number}, <:Any, 0})
    return min(d.f(Dists.minimum(d.base)), d.f(Dists.maximum(d.base)))
end
function Base.maximum(d::PushforwardDistribution{<:ScaleShift{<:Number, <:Number}, <:Any, 0})
    return max(d.f(Dists.minimum(d.base)), d.f(Dists.maximum(d.base)))
end

Dists.rand(rng::AbstractRNG, d::PushforwardDistribution) = d.f(rand(rng, d.base))

# Delegates to `_pf_rand!`. `@with_real` emits the `<:Real` overload that breaks the
# ambiguity with `Distributions._rand!(::Sampleable{<:ArrayLikeVariate}, ::AbstractArray{<:Real})`
# (strictly more specific in the distribution argument) and the `<:Number` overload that admits
# traced (Reactant) arrays, whose eltype is not `<:Real`. Both are restricted to the variate
# dimension `N` (one draw): an unrestricted signature would also capture the stacked array
# Distributions passes for `rand(rng, d, n)`, filling all `n` variates with a single broadcast draw.
function _pf_rand!(rng::AbstractRNG, d::PushforwardDistribution, x::AbstractArray)
    x .= d.f(rand(rng, d.base))
    return x
end
@with_real Dists._rand!(
    rng::AbstractRNG, d::PushforwardDistribution{<:Any, <:Any, N}, x::AbstractArray{<:Number, N}
) where {N} = _pf_rand!(rng, d, x)

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
    Dists.cdf(d.base, inverse(d.f)(x))
Dists.quantile(d::PushforwardDistribution{<:ScaleShift, <:Any, 0}, p::Number) =
    d.f(Dists.quantile(d.base, p))

# log-cdf / log-ccdf (used by `Truncated`'s constructor). `log(cdf)`/`log1p(-cdf)` — reuses
# `cdf`, accepts traced numbers, and traces. `@with_real` also emits the `<:Real` overloads
# that break the ambiguity with Distributions' generic `logcdf`/`logccdf`.
@with_real Dists.logcdf(d::PushforwardDistribution{<:ScaleShift, <:Any, 0}, x::Number) = log(Dists.cdf(d, x))
@with_real Dists.logccdf(d::PushforwardDistribution{<:ScaleShift, <:Any, 0}, x::Number) = log1p(-Dists.cdf(d, x))

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
