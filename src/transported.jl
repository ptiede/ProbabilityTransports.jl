# ----- TransportedDistribution --------------------------------------------
#
# Behaves like the latent reference `stop` (so it can be sampled / scored in the
# nice space), while carrying the transport node needed to push a realization to
# the original distribution `start` and to evaluate the pulled-back target
# density `logpdf_pfwd`.

"""
    TransportedDistribution

The result of [`transport_to`](@ref). It *behaves like the latent reference space* for the
`Distributions` interface (so it can be sampled and scored there) while carrying the transport
node tree that pushes a latent draw to the original distribution via [`latent_pfwd`](@ref) and
evaluates the pulled-back target density via [`logpdf_pfwd`](@ref). Construct it with
`transport_to(distribution, space)` rather than directly.
"""
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

# Accessor for the stored `AbstractTransport` node tree. (The 2-arg `transport_node(dist,
# space)` is the build-time extension point; this 1-arg method just fetches the node from an
# already-built distribution.)
transport_node(d::TransportedDistribution) = getfield(d, :transport)
Base.length(d::TransportedDistribution) = dimension(getfield(d, :transport))
# eltype is the latent reference's (`stop`): it is the type of the points this
# distribution samples/scores. Flat (`stop === nothing`) has no reference; keep Float64.
Base.eltype(::Type{<:TransportedDistribution{T, S, E}}) where {T, S, E <: AbstractStdDist} = eltype(E)
Base.eltype(::Type{<:TransportedDistribution{T, S, Nothing}}) where {T, S} = Float64
dimension(d::TransportedDistribution) = dimension(getfield(d, :transport))

# Scalar latents, mirroring TransformVariables: a transported distribution whose node is
# *scalar-kind* (`is_scalar_transport` — TV's `ScalarTransform` analogue) speaks scalars
# at every latent entry point (`latent_pfwd`, `latent_pfwd_and_logdensity`,
# `logpdf_pfwd`, `logpdf`), and `latent_pback` hands a bare `Number` back. Scalars are
# normalized here, once, at the user-facing driver — boxed to a 1-vector and sent down
# the ordinary vector path — so the contract is uniform across entry points and across
# the Std and flat spaces. The node protocol itself stays vector-only, and (as in TV) a
# 1-dimensional *vector-kind* node keeps vector semantics.
function _box_latent(d::TransportedDistribution, y::Number)
    is_scalar_transport(getfield(d, :transport)) || throw(
        ArgumentError(
            "scalar latent passed to a vector-kind transport; pass an " *
                "`AbstractVector` of length $(dimension(d))",
        ),
    )
    return [y]
end

"""
    latent_pfwd(dto::TransportedDistribution, y)

Push the latent point `y` to the original distribution's space. For a scalar-kind
transport (univariate target; see `is_scalar_transport`) `y` may be a plain `Number`.
"""
latent_pfwd(d::TransportedDistribution, y) = latent_pfwd(d.transport, y)
latent_pfwd(d::TransportedDistribution, y::Number) = latent_pfwd(d, _box_latent(d, y))

"""
    latent_pback(dto::TransportedDistribution, x)

A latent point mapping to the original-space value `x` (a section; see [`latent_pback`](@ref)).
For a scalar-kind transport this is a bare `Number` (mirroring `TransformVariables.inverse`
on a `ScalarTransform`); otherwise an `AbstractVector`.
"""
function latent_pback(d::TransportedDistribution, x)
    c = getfield(d, :transport)
    y = latent_pback(c, x)
    # `is_scalar_transport` is a type constant, so this branch folds away
    return is_scalar_transport(c) ? y[begin] : y
end

"""
    latent_pback!(y, dto::TransportedDistribution, x)

In-place [`latent_pback`](@ref): write the latent coordinates of `x` into the caller-owned
buffer `y` (its array type sets the backend — `Vector`/`TracedRArray`/`GPUArray`).
"""
latent_pback!(y, d::TransportedDistribution, x) = latent_pback!(y, getfield(d, :transport), x)

# `latent_pfwd_and_logdensity` dispatches on the *space* (the `stop` field).
#
# Std spaces: the transport is exact, so the pulled-back density is the closed-form
# reference `logpdf(stop, y)` — we return the transported point (for the likelihood)
# without ever forming a Jacobian. The `TVFlat` method (`stop === nothing`), which does
# the genuine change of variables on a TV transform, lives in the TV extension.
function latent_pfwd_and_logdensity(d::TransportedDistribution, y)
    return latent_pfwd(getfield(d, :transport), y), Dists.logpdf(d, y)
end
latent_pfwd_and_logdensity(d::TransportedDistribution, y::Number) =
    latent_pfwd_and_logdensity(d, _box_latent(d, y))

"""
    logpdf_pfwd(dto, y)

The pulled-back *target* density at latent `y` — the prior contribution a sampler
targets. Equals `last(latent_pfwd_and_logdensity(dto, y))`: `logpdf(reference, y)` for the
Std spaces (exact transport) and `logpdf(start, latent_pfwd(dto, y)) + logjac` for `TVFlat`.
For a scalar-kind transport `y` may be a plain `Number`.
"""
logpdf_pfwd(d::TransportedDistribution, y) = last(latent_pfwd_and_logdensity(d, y))

# ----- Distributions interface: behave like the latent reference ----------

# Std spaces: `logpdf(dto, y) == logpdf(stop, y)`.
Dists.logpdf(d::TransportedDistribution, y::AbstractVector) = Dists.logpdf(getfield(d, :stop), y)
# Flat space (stop === nothing): there is no separate reference, so logpdf == logpdf_pfwd.
Dists.logpdf(d::TransportedDistribution{<:Any, <:Any, Nothing}, y::AbstractVector) = logpdf_pfwd(d, y)
# Scalar latent: box and re-dispatch on the vector methods above.
Dists.logpdf(d::TransportedDistribution, y::Number) = Dists.logpdf(d, _box_latent(d, y))

function Dists._rand!(rng::AbstractRNG, d::TransportedDistribution, x::AbstractVector)
    rand!(rng, getfield(d, :stop), x)
    return x
end
