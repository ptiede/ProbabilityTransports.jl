_ensure_float(::Type{T}) where {T <: Real} = float(T)
_ensure_float(::Type) = Float64

# ----- core generics ------------------------------------------------------

"""
    transport(c, y)

The pushforward map: take a latent point `y` (in the space `c` was built for) to
the original distribution's space. Always well defined.
"""
function transport end

"""
    transport_and_logjac(c, y)

Return `(x, logjac)` where `x = transport(c, y)` and `logjac` is the log absolute
Jacobian determinant of the (bijective part of the) map at `y`.
"""
function transport_and_logjac end

"""
    pullback(c, x)

A *section* of `transport`: return a canonical latent point mapping to `x`, i.e.
`transport(c, pullback(c, x)) == x`. It is the exact inverse only for bijective
(dimension-preserving) nodes; for dimension-expanding nodes it picks a documented
representative. Deliberately *not* called `inverse`.
"""
function pullback end

"""
    dimension(c)

Number of latent coordinates the transport `c` consumes (may differ from the
original distribution's dimension).
"""
function dimension end

"""
    transport_step(c, y, index) -> (value, logjac, index′)

The composable, index-threaded core of a transport node. Consume the latent
coordinates of `y` starting at `index`, returning the transported `value`, the log
absolute Jacobian determinant `logjac`, and the next index `index′ = index +
dimension(c)`. It always returns `logjac`; `transport` simply discards it.

This is the performance-critical extension point for defining a **new node type**
(most extensions instead add [`transport_node`](@ref) methods or a `space_*` trait,
or wrap a bijection in a `PushforwardTransport`). To add a new `AbstractTransport`
subtype implement, all index-threaded for zero-allocation composition:

  - `dimension(c)` — latent coordinates consumed
  - `transport_step(c, y, index) -> (value, logjac, index′)`
  - `pullback_step!(y, index, c, x) -> index′` — write the latent coords for `x`
  - `pullback_eltype(c, ::Type)` — element type for the pullback buffer
"""
function transport_step end

"""
    pullback_step!(y, index, c, x) -> index′

Backward (section) counterpart of [`transport_step`](@ref): write the latent
coordinates mapping to `x` into `y` starting at `index`, returning the next index.
See [`transport_step`](@ref) for the full node-authoring protocol.
"""
function pullback_step! end

abstract type AbstractTransport end

space(c::AbstractTransport) = getfield(c, :space)

"""
    transport_node(dist, space) -> AbstractTransport

Build the transport node that maps the latent `space` to `dist`. This is the
extension point: to support a new distribution (or a distribution in a particular
space) add a method here returning an `AbstractTransport`. `transport_to` wraps the
result in a [`TransportedDistribution`](@ref).

Dispatch is on *both* the distribution and the space, so the same distribution can
map to different nodes per space (e.g. `MvNormal` becomes an affine-Cholesky
pushforward under `StdNormal` but a TransformVariables transform under `StdFlat`).
Composites (`Tuple`/`NamedTuple`/`NamedDist`) recurse into a `TupleTransport`, and an
already-built `AbstractTransport` is passed through unchanged.
"""
function transport_node end

# Pass an already-built transport through unchanged so composites may mix raw
# distributions with pre-built nodes.
transport_node(c::AbstractTransport, space) = c

function transport(c::AbstractTransport, y)
    @argcheck dimension(c) == length(y)
    return first(transport_step(c, y, firstindex(y)))
end

function transport_and_logjac(c::AbstractTransport, y)
    @argcheck dimension(c) == length(y)
    x, ℓ, _ = transport_step(c, y, firstindex(y))
    return x, ℓ
end

function pullback(c::AbstractTransport, x)
    y = Vector{pullback_eltype(c, x)}(undef, dimension(c))
    pullback_step!(y, firstindex(y), c, x)
    return y
end

pullback_eltype(c::AbstractTransport, x) = pullback_eltype(c, typeof(x))

# ----- scalar node: the universal unit-interval factoring ------------------
#
#   y --F_S--> u ∈ [0,1] --Q_D--> x
#   logjac = logpdf_S(y) - logpdf_D(x)

struct ScalarTransport{D, S} <: AbstractTransport
    dist::D
    space::S
end

dimension(::ScalarTransport{D, S}) where {D, S} = space_dimension(S)

# The generic scalar path needs `quantile` (forward) and `cdf` (pullback) to build
# the exact transport. Check at build time via a compile-time trait so a missing
# method is a clear error rather than a deep stack trace.
function transport_node(d::Dists.UnivariateDistribution, space)
    static_hasmethod(quantile, Tuple{typeof(d), Float64}) || throw(
        ArgumentError(
            "Cannot transport `$(nameof(typeof(d)))` to `$(nameof(typeof(space)))`: it has " *
            "no `quantile` method, so there is no exact transport of its variables to " *
            "`$(nameof(typeof(space)))`. Provide `quantile`/`cdf`, add a `transport_node` " *
            "specialization, or use `StdFlat()`.",
        ),
    )
    return ScalarTransport(d, space)
end

function transport_step(c::ScalarTransport, y, index)
    yi = _rgetindex(y, index)
    u = space_cdf(c.space, yi)
    x = quantile(c.dist, u)
    ℓ = space_logpdf(c.space, yi) - Dists.logpdf(c.dist, x)
    return x, ℓ, index + 1
end

function pullback_step!(y, index, c::ScalarTransport, x::Real)
    u = cdf(c.dist, x)
    _rsetindex!(y, space_quantile(c.space, u), index)
    return index + 1
end

pullback_eltype(::ScalarTransport, ::Type{T}) where {T} = _ensure_float(eltype(T))

# ----- array node ---------------------------------------------------------
# Phase 1 supports `Product` (independent, per-coordinate). Correlated /
# dimension-changing array distributions are specialized in later phases.

struct ArrayTransport{D, M, S} <: AbstractTransport
    dist::D
    dims::NTuple{M, Int}
    space::S
end

ArrayTransport(d, space) = ArrayTransport(d, size(d), space)

transport_node(d::Union{Dists.MultivariateDistribution, Dists.MatrixDistribution}, space) =
    ArrayTransport(d, size(d), space)

dimension(c::ArrayTransport) = prod(c.dims) * space_dimension(typeof(c.space))

pullback_eltype(::ArrayTransport, ::Type{V}) where {V <: AbstractArray} = _ensure_float(eltype(V))

function transport_step(c::ArrayTransport{<:Dists.Product}, y, index)
    comps = c.dist.v
    T = _ensure_float(eltype(y))
    out = Vector{T}(undef, length(comps))
    ℓ = zero(T)
    @inbounds for i in eachindex(comps)
        xi, ℓi, index = transport_step(ScalarTransport(comps[i], c.space), y, index)
        out[i] = xi
        ℓ += ℓi
    end
    return out, ℓ, index
end

function pullback_step!(y, index, c::ArrayTransport{<:Dists.Product}, x)
    comps = c.dist.v
    @inbounds for i in eachindex(comps)
        index = pullback_step!(y, index, ScalarTransport(comps[i], c.space), x[i])
    end
    return index
end

# ----- empty composites ---------------------------------------------------

struct EmptyTupleTransport{S} <: AbstractTransport
    space::S
end
struct EmptyNamedTupleTransport{S} <: AbstractTransport
    space::S
end

dimension(::EmptyTupleTransport) = 0
dimension(::EmptyNamedTupleTransport) = 0

transport_node(::Tuple{}, space) = EmptyTupleTransport(space)
transport_node(::NamedTuple{()}, space) = EmptyNamedTupleTransport(space)

transport_step(::EmptyTupleTransport, y, index) =
    (), zero(_ensure_float(eltype(y))), index
transport_step(::EmptyNamedTupleTransport, y, index) =
    (;), zero(_ensure_float(eltype(y))), index
pullback_step!(y, index, ::EmptyTupleTransport, ::Tuple{}) = index
pullback_step!(y, index, ::EmptyNamedTupleTransport, ::NamedTuple{()}) = index

pullback_eltype(::EmptyTupleTransport, ::Type) = Bool
pullback_eltype(::EmptyNamedTupleTransport, ::Type) = Bool
