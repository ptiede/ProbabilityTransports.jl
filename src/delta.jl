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
Dists.logpdf(d::DeltaDist, x) = zero(_ensure_float(eltype(d.x0)))
Dists._logpdf(d::DeltaDist, ::AbstractVector) = zero(_ensure_float(eltype(d.x0)))
Dists.rand(::AbstractRNG, d::DeltaDist) = d.x0

# ----- transport: a 0-dimensional constant node ---------------------------

struct ConstantTransport{T} <: AbstractTransport
    value::T
end

dimension(::ConstantTransport) = 0
transport_node(d::DeltaDist, space) = ConstantTransport(d.x0)

function transport_step(c::ConstantTransport, y, index)
    return c.value, zero(_ensure_float(eltype(y))), index
end
pullback_step!(y, index, ::ConstantTransport, x) = index
pullback_eltype(::ConstantTransport, ::Type) = Bool
