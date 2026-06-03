# ----- TransportedDistribution --------------------------------------------
#
# Behaves like the latent reference `stop` (so it can be sampled / scored in the
# nice space), while carrying the transport node needed to push a realization to
# the original distribution `start` and to evaluate the pulled-back target
# density `logpdf_fwd`.

struct TransportedDistribution{T, S, E} <: Dists.ContinuousMultivariateDistribution
    transport::T   # AbstractTransport node tree
    start::S       # original (target) distribution
    stop::E        # latent reference (StdNormal/StdUniform of dim n), or `nothing` (flat)
end

"""
    transport_to(distribution, space)

Build a [`TransportedDistribution`](@ref) that maps the latent `space` (an
instance of `StdNormal()`, `StdUniform()`, or `StdFlat()`) to `distribution`.
"""
function transport_to(dist, space)
    node = transport_node(dist, space)
    n = dimension(node)
    stop = _reference(space, n)
    return TransportedDistribution(node, dist, stop)
end

# latent reference of the correct (flattened) dimension
_reference(::StdNormal, n::Int) = StdNormal(n)
_reference(::StdUniform, n::Int) = StdUniform(n)

transport(d::TransportedDistribution) = getfield(d, :transport)
Base.length(d::TransportedDistribution) = dimension(getfield(d, :transport))
Base.eltype(::Type{<:TransportedDistribution}) = Float64
dimension(d::TransportedDistribution) = dimension(getfield(d, :transport))

"""
    transport(dto::TransportedDistribution, y)

Push the latent point `y` to the original distribution's space.
"""
transport(d::TransportedDistribution, y) = transport(d.transport, y)
transport_and_logjac(d::TransportedDistribution, y) = transport_and_logjac(d.transport, y)

"""
    pullback(dto::TransportedDistribution, x)

A latent point mapping to the original-space value `x` (a section; see [`pullback`](@ref)).
"""
pullback(d::TransportedDistribution, x) = pullback(d.transport, x)

"""
    logpdf_fwd(dto, y)

The pulled-back *target* density at latent `y`:
`logpdf(start, transport(dto, y)) + logjac + logprior`. This is the density to put
a sampler on.
"""
function logpdf_fwd(d::TransportedDistribution, y)
    x, ℓ = transport_and_logjac(getfield(d, :transport), y)
    return Dists.logpdf(getfield(d, :start), x) + ℓ
end

# ----- Distributions interface: behave like the latent reference ----------

# Std spaces: `logpdf(dto, y) == logpdf(stop, y)`.
Dists.logpdf(d::TransportedDistribution, y::AbstractVector) = Dists.logpdf(getfield(d, :stop), y)
# Flat space (stop === nothing): there is no separate reference, so logpdf == logpdf_fwd.
Dists.logpdf(d::TransportedDistribution{<:Any, <:Any, Nothing}, y::AbstractVector) = logpdf_fwd(d, y)

function Dists._rand!(rng::AbstractRNG, d::TransportedDistribution, x::AbstractVector)
    rand!(rng, getfield(d, :stop), x)
    return x
end
