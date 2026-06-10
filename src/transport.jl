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
    transport_and_logdensity(dto, y)

Return `(x, ℓ)` where `x = transport(dto, y)` is the target-space point and `ℓ` is the
*pulled-back log density at `y`* — the prior's contribution to put a sampler on. This
is the model-facing primitive: feed `x` to the rest of the generative model (the
likelihood) and add `ℓ`. Dispatched on the latent space:

  - Std spaces (`StdNormal`/`StdUniform`): `ℓ == logpdf(reference, y)`, the closed-form
    reference density — **no Jacobian is computed**, because the transport is exact.
  - `TVFlat`: `ℓ == logpdf(start, x) + logjac`, the genuine change of variables.

Std-space methods live in `transported.jl`; the `TVFlat` method lives in the
TransformVariables extension (where the whole flat tree is a single TV transform).
"""
function transport_and_logdensity end

"""
    pullback(c, x)

A *section* of `transport`: return a canonical latent point mapping to `x`, i.e.
`transport(c, pullback(c, x)) == x`. It is the exact inverse only for bijective
(dimension-preserving) nodes; for dimension-expanding nodes it picks a documented
representative.
"""
function pullback end

"""
    dimension(c)

Number of latent coordinates the transport `c` consumes (may differ from the
original distribution's dimension).
"""
function dimension end

"""
    transport_step(c, y, index) -> (value, index′)

The composable, index-threaded core of a transport node. Consume the latent
coordinates of `y` starting at `index`, returning the transported `value` and the next
index `index′ = index + dimension(c)`. It carries **no Jacobian**: an exact transport to
a Std space has pulled-back density equal to the closed-form reference, so the Jacobian
is never needed there. The flat space does the genuine change of variables entirely in
TransformVariables (every `TVFlat` node is a TV transform), so no core node forms one.

This is the performance-critical extension point for defining a **new node type**
(most extensions instead add [`transport_node`](@ref) methods or a `space_*` trait,
or wrap a bijection in a `PushforwardTransport`). To add a new `AbstractTransport`
subtype implement, all index-threaded for zero-allocation composition:

  - `dimension(c)` — latent coordinates consumed
  - `transport_step(c, y, index) -> (value, index′)`
  - `pullback_step!(y, index, c, x) -> index′` — write the latent coords for `x`
  - `pullback_eltype(c)` — latent-buffer element type (defaults to the reference space's
    eltype via `space(c)`; only override for a node with no `space`, e.g. a 0-dim constant)
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
pushforward under `StdNormal` but a TransformVariables transform under `TVFlat`).
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

"""
    pullback!(y, c::AbstractTransport, x)

Write the latent coordinates of `x` into the buffer `y` (an `AbstractVector` of length
`dimension(c)`) and return it. The caller owns `y`, so its array type sets the backend —
a `Vector` on CPU, a `TracedRArray`/`GPUArray` under Reactant/GPU. This is the
allocation-free primitive; [`pullback`](@ref) is the convenience that allocates a `Vector`.
"""
function pullback!(y::AbstractVector, c::AbstractTransport, x)
    @argcheck length(y) == dimension(c)
    pullback_step!(y, firstindex(y), c, x)
    return y
end

# Convenience: allocate a plain CPU `Vector` (eltype fixed by the reference space) and fill
# it. For Reactant/GPU, call `pullback!` with a backend-appropriate buffer instead.
function pullback(c::AbstractTransport, x)
    return pullback!(Vector{pullback_eltype(c)}(undef, dimension(c)), c, x)
end

pullback_eltype(c::AbstractTransport) = _ensure_float(eltype(space(c)))

# ----- scalar node: the universal unit-interval factoring ------------------
#
#   y --F_S--> u ∈ [0,1] --Q_D--> x

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
            "specialization, or use `TVFlat()`.",
        ),
    )
    return ScalarTransport(d, space)
end

function transport_step(c::ScalarTransport, y, index)
    yi = _rgetindex(y, index)
    u = space_cdf(c.space, yi)
    x = quantile(c.dist, u)
    return x, index + 1
end

function pullback_step!(y, index, c::ScalarTransport, x::Number)
    u = cdf(c.dist, x)
    _rsetindex!(y, space_quantile(c.space, u), index)
    return index + 1
end

# ----- array node ---------------------------------------------------------
# Phase 1 supports `Product` (independent, per-coordinate). Correlated /
# dimension-changing array distributions are specialized in later phases.

struct ArrayTransport{D, M, S} <: AbstractTransport
    dist::D
    dims::NTuple{M, Int}
    space::S
end

ArrayTransport(d, space) = ArrayTransport(d, size(d), space)

# Mirror the univariate build-time check: only distributions with a `transport_step`
# specialization for their `ArrayTransport` (Product, the Std* bases, Dirichlet, …)
# have an exact array transport. Anything else must error here, at build, instead of
# with a deep `MethodError` on the first `transport` call.
function transport_node(d::Union{Dists.MultivariateDistribution, Dists.MatrixDistribution}, space)
    c = ArrayTransport(d, size(d), space)
    static_hasmethod(transport_step, Tuple{typeof(c), Vector{Float64}, Int}) || throw(
        ArgumentError(
            "Cannot transport `$(nameof(typeof(d)))` to `$(nameof(typeof(space)))`: there " *
            "is no exact array transport for it. Add a `transport_node` (or " *
            "`transport_step`) specialization, reparameterize via " *
            "`PushforwardDistribution`, or use `TVFlat()`.",
        ),
    )
    return c
end

dimension(c::ArrayTransport) = prod(c.dims) * space_dimension(typeof(c.space))

function transport_step(c::ArrayTransport{<:Dists.Product}, y, index)
    comps = c.dist.v
    T = _ensure_float(eltype(y))
    out = similar(y, T, length(comps))
    indexr = promote_index(index)
    @trace track_numbers = false for i in eachindex(comps)
        xi, indexr = transport_step(ScalarTransport(comps[i], c.space), y, indexr)
        _rsetindex!(out, xi, i)
    end
    # `indexr` may be traced inside the loop; the advance is static, so return it
    # statically — composites after this node must see the consumed coordinates.
    return out, index + dimension(c)
end

function pullback_step!(y, index, c::ArrayTransport{<:Dists.Product}, x)
    comps = c.dist.v
    indexr = promote_index(index)
    @trace track_numbers = false for i in eachindex(comps)
        indexr = pullback_step!(y, indexr, ScalarTransport(comps[i], c.space), x[i])
    end
    return index + dimension(c)
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

transport_step(::EmptyTupleTransport, y, index) = (), index
transport_step(::EmptyNamedTupleTransport, y, index) = (;), index
pullback_step!(y, index, ::EmptyTupleTransport, ::Tuple{}) = index
pullback_step!(y, index, ::EmptyNamedTupleTransport, ::NamedTuple{()}) = index

pullback_eltype(::EmptyTupleTransport) = Bool
pullback_eltype(::EmptyNamedTupleTransport) = Bool
