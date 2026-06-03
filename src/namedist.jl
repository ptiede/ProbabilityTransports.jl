# ----- NamedDist / TupleDist ----------------------------------------------
#
# Ported from HypercubeTransform: lightweight (Named)Tuple-backed multivariate
# distributions, plus `_distize` to normalize mixed nested inputs. `transport_node`
# delegates them to the corresponding `TupleTransport` shape.

"""
    TupleDist(dists::Tuple)

A multivariate distribution backed by a tuple of (possibly heterogeneous)
distributions.
"""
struct TupleDist{N, D <: NTuple{N, Dists.Distribution}} <: Dists.ContinuousMultivariateDistribution
    dists::D
end
TupleDist() = TupleDist{0, Tuple{}}(())

Base.length(::TupleDist{N}) where {N} = N

function Dists.logpdf(d::TupleDist{N}, x::Tuple) where {N}
    return sum(map((di, xi) -> Dists.logpdf(di, xi), d.dists, x))
end
Dists.logpdf(::TupleDist{0}, ::Tuple{}) = 0.0

function Dists.rand(rng::AbstractRNG, d::TupleDist{N}) where {N}
    return ntuple(i -> rand(rng, d.dists[i]), N)
end

"""
    NamedDist(d::NamedTuple)
    NamedDist(; dists...)

A multivariate distribution with named components. Values may themselves be
distributions, tuples/arrays of distributions, or nested NamedTuples; `_distize`
normalizes them.
"""
struct NamedDist{Names, D} <: Dists.ContinuousMultivariateDistribution
    dists::D
end

Base.propertynames(::NamedDist{N}) where {N} = N
function Base.getproperty(d::NamedDist{N}, s::Symbol) where {N}
    return getproperty(NamedTuple{N}(getfield(d, :dists)), s)
end
Base.length(d::NamedDist) = reduce(+, map(length, getfield(d, :dists)); init = 0)

function NamedDist(d::NamedTuple{N}) where {N}
    dd = map(_distize, values(d))
    return NamedDist{N, typeof(dd)}(dd)
end
NamedDist(; kwargs...) = NamedDist((; kwargs...))

# normalize mixed inputs into NamedDist / TupleDist / product_distribution
_distize(d::Dists.Distribution) = d
_distize(d::NTuple{N, <:Dists.Distribution}) where {N} = TupleDist(d)
_distize(d::Tuple) = TupleDist(map(_distize, d))
_distize(d::AbstractArray{<:Dists.Distribution}) = Dists.product_distribution(d)
_distize(d::NamedTuple{N}) where {N} = NamedDist(NamedTuple{N}(map(_distize, d)))

function Dists.logpdf(d::NamedDist{N}, x::NamedTuple) where {N}
    xs = NamedTuple{N}(x)
    return sum(map((di, xi) -> Dists.logpdf(di, xi), getfield(d, :dists), values(xs)))
end
Dists.logpdf(::NamedDist{()}, ::NamedTuple{()}) = 0.0

function Dists.rand(rng::AbstractRNG, d::NamedDist{N}) where {N}
    rngF = Base.Fix1(rand, rng)
    return NamedTuple{N}(map(rngF, getfield(d, :dists)))
end

# ----- build transport nodes from the named/tuple distributions -----------

transport_node(d::TupleDist, space) = transport_node(getfield(d, :dists), space)
function transport_node(d::NamedDist{N}, space) where {N}
    # `dists` is a plain tuple (names live in the `N` type param); map the
    # component nodes directly and re-attach the names — equivalent to, but
    # without the round-trip through `transport_node(::NamedTuple, space)`.
    nodes = map(x -> transport_node(x, space), getfield(d, :dists))
    return TupleTransport(NamedTuple{N}(nodes), space)
end
