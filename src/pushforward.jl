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

# inverse of `x = μ .+ s .* z` is `z = (x .- μ) ./ s = (-μ./s) .+ (1 ./s) .* x`
InverseFunctions.inverse(f::ScaleShift) = ScaleShift(-f.μ ./ f.s, inv.(f.s))

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
# inverse map `z = L \ (x - μ)` has log-det `-logabsdet(L)` (needed by the
# distribution-side change of variables; `ScaleShift`'s inverse is a `ScaleShift`
# so it already has one).
function ChangesOfVariables.with_logabsdet_jacobian(g::InvAffineTransform, x)
    return g(x), -_logabsdet(g.L)
end

# ----- the pushforward transport node -------------------------------------

struct PushforwardTransport{F, Inner} <: AbstractTransport
    f::F
    inner::Inner
end

dimension(c::PushforwardTransport) = dimension(c.inner)   # invertible f ⇒ equal-dim
space(c::PushforwardTransport) = space(c.inner)

function pfwd_step(c::PushforwardTransport, y, index)
    z, index′ = pfwd_step(c.inner, y, index)
    return c.f(z), index′
end

# (Under `TVFlat`, a pushforward is a TV `PushforwardTransform` — see the TV extension
# — so this core node only serves the Jacobian-free Std spaces.)

function pback_step!(y, index, c::PushforwardTransport, x)
    z = inverse(c.f)(x)
    return pback_step!(y, index, c.inner, z)
end

# (`pback_eltype` falls to the generic `AbstractTransport` method off `space(c)`, which
# for a `PushforwardTransport` delegates to `space(c.inner)`.)

# ----- identity inner nodes -----------------------------------------------
# Used by the cheap analytic specializations: when the base of an affine
# distribution is exactly the latent reference of the target space, the inner
# map is the identity (no cdf/quantile, zero log-Jacobian).

struct ScalarIdentity{S} <: AbstractTransport
    space::S
end
dimension(::ScalarIdentity) = 1
function pfwd_step(c::ScalarIdentity, y, index)
    return _rgetindex(y, index), index + 1
end
pback_step!(y, index, ::ScalarIdentity, z::Number) = (_rsetindex!(y, z, index); index + 1)
