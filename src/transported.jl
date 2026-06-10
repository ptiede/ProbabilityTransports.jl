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
instance of `StdNormal()`, `StdUniform()`, or `TVFlat()`) to `distribution`.
"""
function transport_to(dist, space::AbstractStdDist{T}) where {T}
    isconcretetype(T) || error("transport_to only supports concrete eltypes; got $T")
    node = transport_node(dist, space)
    n = dimension(node)
    stop = basemeasure(space, n)
    return TransportedDistribution(node, dist, stop)
end

# latent reference of the correct (flattened) dimension
basemeasure(::StdNormal{T}, n::Int) where {T} = StdNormal{T}(n)
basemeasure(::StdUniform{T}, n::Int) where {T} = StdUniform{T}(n)
# The other Std* distributions are transportable *bases*, not target spaces: they
# lack the `space_*` trait, so error clearly at build time instead of deep in a
# transport. A new target space must define `space_*` AND a `basemeasure` method.
function basemeasure(space::AbstractStdDist, n::Int)
    throw(
        ArgumentError(
            "`$(nameof(typeof(space)))` is a transportable base distribution, not a " *
            "target space: `transport_to` supports `StdNormal()`, `StdUniform()`, and " *
            "`TVFlat()` targets. To add a new target space define the `space_cdf`/" *
            "`space_quantile`/`space_logpdf` trait and a `basemeasure` method for it.",
        ),
    )
end

transport(d::TransportedDistribution) = getfield(d, :transport)
Base.length(d::TransportedDistribution) = dimension(getfield(d, :transport))
# eltype is the latent reference's (`stop`): it is the type of the points this
# distribution samples/scores. Flat (`stop === nothing`) has no reference; keep Float64.
Base.eltype(::Type{<:TransportedDistribution{T, S, E}}) where {T, S, E <: AbstractStdDist} = eltype(E)
Base.eltype(::Type{<:TransportedDistribution{T, S, Nothing}}) where {T, S} = Float64
dimension(d::TransportedDistribution) = dimension(getfield(d, :transport))

"""
    transport(dto::TransportedDistribution, y)

Push the latent point `y` to the original distribution's space.
"""
transport(d::TransportedDistribution, y) = transport(d.transport, y)

"""
    pullback(dto::TransportedDistribution, x)

A latent point mapping to the original-space value `x` (a section; see [`pullback`](@ref)).
"""
pullback(d::TransportedDistribution, x) = pullback(getfield(d, :transport), x)

"""
    pullback!(y, dto::TransportedDistribution, x)

In-place [`pullback`](@ref): write the latent coordinates of `x` into the caller-owned
buffer `y` (its array type sets the backend — `Vector`/`TracedRArray`/`GPUArray`).
"""
pullback!(y, d::TransportedDistribution, x) = pullback!(y, getfield(d, :transport), x)

# `transport_and_logdensity` dispatches on the *space* (the `stop` field).
#
# Std spaces: the transport is exact, so the pulled-back density is the closed-form
# reference `logpdf(stop, y)` — we return the transported point (for the likelihood)
# without ever forming a Jacobian. The `TVFlat` method (`stop === nothing`), which does
# the genuine change of variables on a TV transform, lives in the TV extension.
function transport_and_logdensity(d::TransportedDistribution, y)
    return transport(getfield(d, :transport), y), Dists.logpdf(d, y)
end

"""
    logpdf_fwd(dto, y)

The pulled-back *target* density at latent `y` — the prior contribution a sampler
targets. Equals `last(transport_and_logdensity(dto, y))`: `logpdf(reference, y)` for the
Std spaces (exact transport) and `logpdf(start, transport(dto, y)) + logjac` for `TVFlat`.
"""
logpdf_fwd(d::TransportedDistribution, y) = last(transport_and_logdensity(d, y))

# ----- Distributions interface: behave like the latent reference ----------

# Std spaces: `logpdf(dto, y) == logpdf(stop, y)`.
Dists.logpdf(d::TransportedDistribution, y::AbstractVector) = Dists.logpdf(getfield(d, :stop), y)
# Flat space (stop === nothing): there is no separate reference, so logpdf == logpdf_fwd.
Dists.logpdf(d::TransportedDistribution{<:Any, <:Any, Nothing}, y::AbstractVector) = logpdf_fwd(d, y)

function Dists._rand!(rng::AbstractRNG, d::TransportedDistribution, x::AbstractVector)
    rand!(rng, getfield(d, :stop), x)
    return x
end
