# ----- pushforward via an invertible map ----------------------------------
#
# Many distributions are the law of `f(Z)` for an invertible `f` and a simpler
# base `Z`. We transport them by wrapping the inner transport (base -> latent)
# with `f`, reusing the `ChangesOfVariables`/`InverseFunctions` interface for the
# Jacobian and inverse. We ship our own `AffineTransform` (Reactant-friendly:
# no data-dependent control flow, allocation-light) as the concrete map.

# ----- ScaleShift: element-wise location-scale `x = μ .+ s .* z` ------------
# `s` is a scalar (broadcast over `z`) or an array the same shape as `z`. The log
# absolute Jacobian is `n·log|s|` for a scalar (n = length(z)) or `Σ log|sᵢ|` for
# an array — this is what makes a per-pixel Gaussian image prior transport exactly.

struct ScaleShift{Tμ, Ts}
    μ::Tμ
    s::Ts
end
(f::ScaleShift)(z) = f.μ .+ f.s .* z

struct InvScaleShift{Tμ, Ts}
    μ::Tμ
    s::Ts
end
(g::InvScaleShift)(x) = (x .- g.μ) ./ g.s

InverseFunctions.inverse(f::ScaleShift) = InvScaleShift(f.μ, f.s)
InverseFunctions.inverse(g::InvScaleShift) = ScaleShift(g.μ, g.s)

_scaleshift_logabsdet(s::Number, z) = length(z) * log(abs(s))
_scaleshift_logabsdet(s::AbstractArray, _) = sum(log ∘ abs, s)
function ChangesOfVariables.with_logabsdet_jacobian(f::ScaleShift, z)
    return f(z), _scaleshift_logabsdet(f.s, z)
end

# ----- AffineTransform: linear-operator `x = μ .+ L * z` --------------------
# `L` is a matrix (a Cholesky factor for MvNormal, or a general linear operator).
# For element-wise scaling use `ScaleShift` instead.

struct AffineTransform{Tμ, TL}
    μ::Tμ
    L::TL
end

(f::AffineTransform)(z) = f.μ .+ f.L * z

struct InvAffineTransform{Tμ, TL}
    μ::Tμ
    L::TL
end
(g::InvAffineTransform)(x) = g.L \ (x .- g.μ)

InverseFunctions.inverse(f::AffineTransform) = InvAffineTransform(f.μ, f.L)
InverseFunctions.inverse(g::InvAffineTransform) = AffineTransform(g.μ, g.L)

# cheap path for triangular / diagonal Cholesky factors (the MvNormal path)
_logabsdet(L::Union{LinearAlgebra.AbstractTriangular, Diagonal}) = sum(log ∘ abs, diag(L))
# general (possibly full) matrix scale
_logabsdet(L::AbstractMatrix) = first(logabsdet(L))

function ChangesOfVariables.with_logabsdet_jacobian(f::AffineTransform, z)
    return f(z), _logabsdet(f.L)
end

# ----- the pushforward transport node -------------------------------------

struct PushforwardTransport{F, Inner} <: AbstractTransport
    f::F
    inner::Inner
end

dimension(c::PushforwardTransport) = dimension(c.inner)   # invertible f ⇒ equal-dim
space(c::PushforwardTransport) = space(c.inner)

function transport_step(c::PushforwardTransport, y, index)
    z, ℓi, index′ = transport_step(c.inner, y, index)
    x, ℓf = with_logabsdet_jacobian(c.f, z)
    return x, ℓi + ℓf, index′
end

function pullback_step!(y, index, c::PushforwardTransport, x)
    z = inverse(c.f)(x)
    return pullback_step!(y, index, c.inner, z)
end

pullback_eltype(c::PushforwardTransport, ::Type{T}) where {T} = pullback_eltype(c.inner, T)

# ----- identity inner nodes -----------------------------------------------
# Used by the cheap analytic specializations: when the base of an affine
# distribution is exactly the latent reference of the target space, the inner
# map is the identity (no cdf/quantile, zero log-Jacobian).

struct ScalarIdentity{S} <: AbstractTransport
    space::S
end
dimension(::ScalarIdentity) = 1
function transport_step(c::ScalarIdentity, y, index)
    return y[index], zero(_ensure_float(eltype(y))), index + 1
end
pullback_step!(y, index, ::ScalarIdentity, z::Real) = (y[index] = z; index + 1)
pullback_eltype(::ScalarIdentity, ::Type{T}) where {T} = _ensure_float(eltype(T))

struct ArrayIdentity{S} <: AbstractTransport
    n::Int
    space::S
end
dimension(c::ArrayIdentity) = c.n
function transport_step(c::ArrayIdentity, y, index)
    n = c.n
    # a view is enough: the result is immediately consumed by the wrapping affine
    # map (`μ .+ L*z`), so there is no need to copy.
    z = @view y[index:(index + n - 1)]
    return z, zero(_ensure_float(eltype(y))), index + n
end
function pullback_step!(y, index, c::ArrayIdentity, z)
    n = c.n
    @inbounds for i in 1:n
        y[index + i - 1] = z[i]
    end
    return index + n
end
pullback_eltype(::ArrayIdentity, ::Type{V}) where {V <: AbstractArray} = _ensure_float(eltype(V))
