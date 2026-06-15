# DeltaDist — a Dirac/fixed-value "distribution" for clamped parameters (ported
# from HypercubeTransform's DeltaDist concept). It transports to a 0-dimensional
# node: it consumes no latent coordinates and always returns the fixed value, so
# it can sit inside a `NamedDist` to clamp a component. `logpdf` is zero by
# convention (it is a clamped value, not a literal density).

"""
    DeltaDist(x0)

A point mass at `x0`. Used to clamp a parameter: under any space it transports to
a 0-dimensional node that always yields `x0` and consumes no latent coordinates.
`logpdf(DeltaDist(x0), x)` is `0`.
"""
struct DeltaDist{T} <: Dists.ContinuousMultivariateDistribution
    x0::T
end

Base.length(d::DeltaDist) = length(d.x0)
Dists.insupport(::DeltaDist, x) = true
Dists.mean(d::DeltaDist) = d.x0
# clamped value ⇒ density is 0 by convention. Specific signatures avoid ambiguity
# with the Distributions fallbacks.
_delta_zero(d::DeltaDist) = zero(_ensure_float(eltype(d.x0)))
Dists.logpdf(d::DeltaDist, ::AbstractArray) = _delta_zero(d)
Dists.logpdf(d::DeltaDist, ::Number) = _delta_zero(d)
Dists.logpdf(d::DeltaDist, ::AbstractVector) = _delta_zero(d)
Dists.rand(::AbstractRNG, d::DeltaDist) = d.x0

# ----- transport: a 0-dimensional constant node ---------------------------

struct ConstantTransport{T} <: AbstractTransport
    value::T
end

dimension(::ConstantTransport) = 0
transport_node(d::DeltaDist, space) = ConstantTransport(d.x0)

pfwd_step(c::ConstantTransport, y, index) = (c.value, index)
# (Under `TVFlat`, a clamped value is `TV.Constant` — see the TV extension — so this
# core node only serves the Std spaces.)
pback_step!(y, index, ::ConstantTransport, x) = index
pback_eltype(::ConstantTransport) = Bool

# A standalone clamped value transports to a 0-dimensional latent reference, whose latent
# log-density is identically 0 (no coordinates to score). Resolve it here, on the constant
# node, rather than letting an empty reference reach a per-space reducer — that keeps the
# `StdUniform`/`StdExponential`/… kernels free of a defensive empty-`sum` `init`. (The `stop`
# bound excludes the `TVFlat`/`Nothing` reference, avoiding ambiguity with the flat method.)
Dists.logpdf(d::TransportedDistribution{<:ConstantTransport, <:Any, <:AbstractStdDist}, y::AbstractVector) =
    zero(eltype(y))
