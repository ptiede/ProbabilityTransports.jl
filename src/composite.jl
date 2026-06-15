# ----- Tuple / NamedTuple composites --------------------------------------
#
# A composite over a Tuple or NamedTuple of inner transports. Stepping mirrors
# TransformVariables' `_transform_tuple`: recurse with `Base.tail`, threading the
# flag, accumulating the log-Jacobian, and advancing the index by whatever each
# inner node consumed (so dimension-changing inner nodes compose for free).

struct TupleTransport{T, S} <: AbstractTransport
    transports::T
    dimension::Int
    space::S
end

# Use separate `reduce`+`map` (not `mapreduce`) for heterogeneous-tuple type
# stability (see TransformVariables PR #80).
_sum_dimensions(ts::Tuple) = reduce(+, map(dimension, ts); init = 0)
_sum_dimensions(ts::NamedTuple) = _sum_dimensions(values(ts))

function TupleTransport(transports::Union{Tuple, NamedTuple}, space)
    return TupleTransport(transports, _sum_dimensions(transports), space)
end

dimension(c::TupleTransport) = getfield(c, :dimension)

transport_node(t::Tuple, space) = TupleTransport(map(x -> transport_node(x, space), t), space)
function transport_node(nt::NamedTuple, space)
    return TupleTransport(NamedTuple{keys(nt)}(map(x -> transport_node(x, space), values(nt))), space)
end

# --- forward stepping (value only) ---

_transport_tuple(y, index, ::Tuple{}) = (), index

function _transport_tuple(y, index, ts)
    y1, index1 = pfwd_step(first(ts), y, index)
    yr, index2 = _transport_tuple(y, index1, Base.tail(ts))
    return (y1, yr...), index2
end

function pfwd_step(c::TupleTransport{<:Tuple}, y, index)
    return _transport_tuple(y, index, c.transports)
end

function pfwd_step(c::TupleTransport{<:NamedTuple}, y, index)
    yt, index′ = _transport_tuple(y, index, values(c.transports))
    return NamedTuple{keys(c.transports)}(yt), index′
end

# (Under `TVFlat`, composites are a native `TV.as(...)` transform — see the TV
# extension — so `TupleTransport` only ever serves the Jacobian-free Std spaces.)

# --- backward stepping (per-component section) ---
# `Base.tail` recursion (mirroring `_transport_tuple`) — a `for`/`zip` loop over a
# heterogeneous tuple infers the element as a union and dynamically dispatches
# `pback_step!` (type-unstable `index`), which also blocks Reactant tracing.

_pback_tuple!(y, index, ::Tuple{}, ::Tuple{}) = index
function _pback_tuple!(y, index, ts::Tuple, xs::Tuple)
    index1 = pback_step!(y, index, first(ts), first(xs))
    return _pback_tuple!(y, index1, Base.tail(ts), Base.tail(xs))
end

function pback_step!(y, index, c::TupleTransport{<:Tuple}, x::Tuple)
    return _pback_tuple!(y, index, c.transports, x)
end

function pback_step!(y, index, c::TupleTransport{<:NamedTuple}, x::NamedTuple)
    xv = NamedTuple{keys(c.transports)}(x)   # reorder / select to match the transports
    return _pback_tuple!(y, index, values(c.transports), values(xv))
end

# (`pback_eltype` falls to the generic `AbstractTransport` method off `space(c)`: the
# latent buffer's eltype is the reference space's, so no per-leaf promotion is needed.)
