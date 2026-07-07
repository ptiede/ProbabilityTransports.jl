module ProbabilityTransportsTransformVariablesExt

using ProbabilityTransports
const PT = ProbabilityTransports
import Distributions
const Dists = Distributions

using TransformVariables
const TV = TransformVariables

using ReactantCore: @trace
using LinearAlgebra: norm

# ----- TransformVariables transforms ARE transport nodes --------------------
#
# Under `TVFlat`, *every* node is a `TV.AbstractTransform` (the `transport_node`
# methods below cover composites, clamped values and pushforwards too), so the whole
# flat path runs on TV's own machinery — no core Jacobian plumbing is involved. A
# `TV.AbstractTransform` cannot subtype our `AbstractTransport` (no retroactive
# subtyping across packages), so we forward the value/latent_pback drivers here.

PT.dimension(t::TV.AbstractTransform) = TV.dimension(t)
# The kind trait for TV transforms (they can't subtype `AbstractTransport`, so they need
# their own fallback): TV's scalar kind is our scalar kind, with one addition — our
# `PushforwardTransform` wrapper inherits its inner's kind (below), matching the core
# `PushforwardTransport`, so a distribution keeps the same kind across Std and flat spaces.
PT.is_scalar_transport(::TV.AbstractTransform) = false
PT.is_scalar_transport(::TV.ScalarTransform) = true

# Value-only step: ask TV for `NoLogJac` (exactly what `TV.transform` does) so we don't
# materialize the per-element log-Jacobian buffer that `LogJac` allocates.
function PT.pfwd_step(t::TV.AbstractTransform, y, index)
    val, _, index′ = TV.transform_with(TV.NoLogJac(), t, y, index)
    return val, index′
end

PT.pback_step!(y, index, t::TV.AbstractTransform, x) = TV.inverse_at!(y, index, t, x)
PT.pback_eltype(t::TV.AbstractTransform, ::Type{T}) where {T} = TV.inverse_eltype(t, T)

# Standalone-leaf drivers: TV transforms are not `<: AbstractTransport`, so they do not
# inherit the core `latent_pfwd`/`latent_pback` drivers. These delegate through the step
# protocol (which handles scalar-in-a-vector correctly).
PT.latent_pfwd(t::TV.AbstractTransform, y) = first(PT.pfwd_step(t, y, firstindex(y)))
function PT.latent_pback!(y, t::TV.AbstractTransform, x)
    PT.pback_step!(y, firstindex(y), t, x)
    return y
end
# Flat path keeps the x-dependent `pback_eltype(t, ::Type)` (TV's `inverse_eltype` — a
# genuine change of variables can change the type), unlike the Std-space core.
PT.latent_pback(t::TV.AbstractTransform, x) =
    PT.latent_pback!(Vector{PT.pback_eltype(t, typeof(x))}(undef, TV.dimension(t)), t, x)

# `transport_to` is specialized on `AbstractStdDist` in core (with a concrete-eltype
# check); `TVFlat` has no Std reference eltype to constrain, so it gets its own method
# here. `basemeasure(::TVFlat, n)` is `nothing`, so the flat density path takes over.
function PT.transport_to(dist, space::PT.TVFlat)
    node = PT.transport_node(dist, space)
    n = PT.dimension(node)
    stop = PT.basemeasure(space, n)
    return PT.TransportedDistribution(node, dist, stop)
end

# Flat-space density. `stop === nothing`, and the transport node is always a TV
# transform, so the genuine change of variables comes straight from TV's `LogJac`
# (`transform_with` via the index protocol handles a scalar transform fed a 1-vector).
# This is the *only* `latent_pfwd_and_logdensity` method that forms a Jacobian; the Std
# spaces (in `transported.jl`) return the closed-form reference instead.
# `y::AbstractVector` keeps this disjoint from the core `y::Number` boxing method
# (which boxes scalars and re-dispatches here) — new spaces should type their vector
# methods the same way.
function PT.latent_pfwd_and_logdensity(d::PT.TransportedDistribution{<:Any, <:Any, Nothing}, y::AbstractVector)
    x, ℓ, _ = TV.transform_with(TV.LogJac(), getfield(d, :transport), y, firstindex(y))
    return x, Dists.logpdf(getfield(d, :start), x) + ℓ
end

# ----- per-distribution flat building blocks (the asflat dispatch table) -----
#
# Each method is more specific (on the TVFlat space) than the core `transport_node`
# methods, so there is no ambiguity. Composites, clamped values and pushforwards are
# handled below too, so under `TVFlat` the whole tree is a single TV transform.
# `stop === nothing` makes `logpdf == logpdf_pfwd` for the flat space.

function _interval(d::Dists.UnivariateDistribution)
    s = Dists.support(d)
    lb = isfinite(s.lb) ? s.lb : -TV.∞
    ub = isfinite(s.ub) ? s.ub : TV.∞
    return as(Real, lb, ub)
end

PT.transport_node(d::Dists.ContinuousUnivariateDistribution, ::PT.TVFlat) = _interval(d)
PT.transport_node(d::Dists.AffineDistribution, ::PT.TVFlat) = _interval(d)
PT.transport_node(d::Dists.Dirichlet, ::PT.TVFlat) = TV.UnitSimplex(length(d.alpha))
PT.transport_node(d::Dists.MvNormal, ::PT.TVFlat) = as(Vector, length(d))
PT.transport_node(d::Dists.MvLogNormal, ::PT.TVFlat) = as(Vector, as(Real, 0, TV.∞), length(d))
# `as(Vector, t, n)` applies ONE transform to every coordinate, so it is only correct
# when every component shares a support. Check at build time; a heterogeneous Product
# would otherwise silently push components through the wrong constraint.
function PT.transport_node(d::Dists.Product, ::PT.TVFlat)
    s1 = Dists.support(first(d.v))
    all(c -> Dists.support(c) == s1, d.v) || throw(
        ArgumentError(
            "Cannot transport this `Product` to `TVFlat()`: its components have " *
                "different supports, but the flat transform applies one constraint to " *
                "every coordinate. Use a `Tuple`/`TupleDist` of the component " *
                "distributions instead, which transforms each component separately.",
        ),
    )
    return as(Vector, _interval(first(d.v)), length(d.v))
end

# array-shaped Std* bases: 0-dim → scalar interval; N ≥ 1 → unconstrained array.
_TVFlat(d, inner) = isempty(size(d)) ? inner : as(Array, inner, size(d)...)
PT.transport_node(d::PT.StdNormal, ::PT.TVFlat) = _TVFlat(d, as(Real, -TV.∞, TV.∞))
PT.transport_node(d::PT.StdExponential, ::PT.TVFlat) = _TVFlat(d, as(Real, 0, TV.∞))
PT.transport_node(d::PT.StdInverseGamma, ::PT.TVFlat) = _TVFlat(d, as(Real, 0, TV.∞))
PT.transport_node(d::PT.StdTDist, ::PT.TVFlat) = _TVFlat(d, as(Real, -TV.∞, TV.∞))
PT.transport_node(d::PT.StdUniform, ::PT.TVFlat) = _TVFlat(d, as(Real, 0, 1))

# Truncated: flat transform is the SUPPORT interval — the truncation bounds intersected
# with the base support (which `minimum`/`maximum` compute) — never the explicit bounds
# alone. A one-sided truncation of a bounded base (e.g. `Truncated(Exponential(); upper=1)`)
# would otherwise map ℝ → (-∞, 1), exposing a reachable logpdf = -Inf region in flat space
# that an optimizer/sampler can walk into.
PT.transport_node(d::PT.Truncated, ::PT.TVFlat) = _interval(d)

# ----- composites: a native TV transform tuple ------------------------------
# Tuples / NamedTuples / NamedDist / TupleDist become a `TV.as(...)` so the whole flat
# tree is one TV transform (vs. the core `TupleTransport`, which carries no Jacobian).
# The `Tuple{}` / `NamedTuple{()}` methods resolve the ambiguity with the core empties.
PT.transport_node(t::Tuple, s::PT.TVFlat) = TV.as(map(x -> PT.transport_node(x, s), t))
PT.transport_node(t::Tuple{}, ::PT.TVFlat) = TV.as(())
PT.transport_node(nt::NamedTuple, s::PT.TVFlat) = TV.as(map(x -> PT.transport_node(x, s), nt))
PT.transport_node(nt::NamedTuple{()}, ::PT.TVFlat) = TV.as((;))
PT.transport_node(d::PT.TupleDist, s::PT.TVFlat) = PT.transport_node(getfield(d, :dists), s)
function PT.transport_node(d::PT.NamedDist{N}, s::PT.TVFlat) where {N}
    return TV.as(NamedTuple{N}(map(x -> PT.transport_node(x, s), getfield(d, :dists))))
end

# ----- clamped value: TV ships a 0-dimensional constant transform ------------
PT.transport_node(d::PT.DeltaDist, ::PT.TVFlat) = TV.Constant(d.x0)

# ----- pushforward of a flat base by an invertible map ----------------------
# Distributions that are the law of `f(Z)` for an invertible `f` (a `ChangesOfVariables`
# map: our `ScaleShift`/`AffineTransform`) over a base `Z` whose flat transform is
# `inner`. This is the flat-space form of the core `PushforwardTransport`, recast as a TV
# transform so TV threads its Jacobian. `f`'s log|det| comes from `with_logabsdet_jacobian`.
struct PushforwardTransform{F, I <: TV.AbstractTransform} <: TV.VectorTransform
    f::F
    inner::I
end
TV.dimension(t::PushforwardTransform) = TV.dimension(t.inner)
# Inherit the inner's kind, matching the core `PushforwardTransport`: a pushforward of a
# scalar base stays scalar-kind whether transported to a Std space or to `TVFlat`.
PT.is_scalar_transport(t::PushforwardTransform) = PT.is_scalar_transport(t.inner)

function TV.transform_with(flag::TV.LogJacFlag, t::PushforwardTransform, y::AbstractVector, index)
    z, ℓi, index′ = TV.transform_with(flag, t.inner, y, index)
    flag isa TV.NoLogJac && return t.f(z), ℓi, index′
    x, ℓf = PT.with_logabsdet_jacobian(t.f, z)
    return x, ℓi + ℓf, index′
end

TV.inverse_eltype(t::PushforwardTransform, ::Type{T}) where {T} = TV.inverse_eltype(t.inner, T)
function TV.inverse_at!(x, index, t::PushforwardTransform, y)
    return TV.inverse_at!(x, index, t.inner, PT.inverse(t.f)(y))
end

# ProjectedNormal: an affine *shift* (scale 1) of the 2n flat standard normals.
function PT.transport_node(d::PT.ProjectedNormal, s::PT.TVFlat)
    ν = d.ν
    return PushforwardTransform(PT.ScaleShift(ν, one(eltype(ν))), PT.transport_node(PT.StdNormal(length(ν)), s))
end

# PushforwardDistribution: `f(base)` — wrap the base's flat transform with `f`.
PT.transport_node(d::PT.PushforwardDistribution, s::PT.TVFlat) =
    PushforwardTransform(d.f, PT.transport_node(d.base, s))

# ----- angular TV transforms (ported from VLBIImagePriors; primal only) -----

"""
    AngleTransform()

Maps two reals `(x, y)` to an angle `θ = atan(x, y)`. If `(x, y)` are standard
normal then `θ` is uniform on the circle; the log-Jacobian uses a log-normal
weight on the radius (μ=0, σ=1/4).
"""
struct AngleTransform <: TV.VectorTransform end
TV.dimension(::AngleTransform) = 2

function TV.transform_with(flag::TV.LogJacFlag, ::AngleTransform, y::AbstractVector, index)
    T = eltype(y)
    ℓi = TV.logjac_zero(flag, T)
    x1 = PT._rgetindex(y, index)
    x2 = PT._rgetindex(y, index + 1)
    r = sqrt(x1^2 + x2^2)
    σ = oftype(r, 1 / 4)
    if !(flag isa TV.NoLogJac)
        lr = log(r)
        ℓi = -lr^2 * inv(2 * σ^2) - lr
    end
    return atan(x1, x2), ℓi, index + 2
end

function TV.transform_with(
        flag::TV.LogJacFlag, t::TV.ArrayTransformation{<:AngleTransform}, y::AbstractVector, index
    )
    (; inner_transformation, dims) = t
    T = eltype(y)
    ℓ = TV.logjac_zero(flag, T)
    out = similar(y, dims)
    index0 = index
    @trace for i in eachindex(out)
        θ, ℓi, index2 = TV.transform_with(flag, inner_transformation, y, index0)
        index0 = index2
        ℓ += ℓi
        PT._rsetindex!(out, θ, i)
    end
    return out, ℓ, index + TV.dimension(inner_transformation) * length(out)
end

function TV.inverse_at!(x, index, ::AngleTransform, y::Number)
    x[index:(index + 1)] .= sincos(y)
    return index + 2
end
TV.inverse_eltype(::AngleTransform, ::Type{T}) where {T} = T

"""
    SphericalUnitVector{N}()

Maps `N+1` reals (assumed iid standard normal) to a unit vector on the `N`-sphere
by normalisation; the log-Jacobian is `-‖y‖²/2`.
"""
struct SphericalUnitVector{N} <: TV.VectorTransform
    function SphericalUnitVector{N}() where {N}
        N ≥ 1 || throw(ArgumentError("Dimension should be positive."))
        return new{N}()
    end
end
TV.dimension(::SphericalUnitVector{N}) where {N} = N + 1
TV.inverse_eltype(::SphericalUnitVector{N}, ::Type{T}) where {N, T} = eltype(T)
TV.inverse_eltype(::TV.ArrayTransformation{<:SphericalUnitVector}, ::Type{NTuple{N, T}}) where {N, T} = eltype(T)

function TV.transform_with(flag::TV.LogJacFlag, ::SphericalUnitVector{N}, y::AbstractVector, index) where {N}
    T = eltype(y)
    index2 = index + N + 1
    vy = ntuple(i -> PT._rgetindex(y, index + i - 1), Val(N + 1))
    sly = sum(abs2, vy)
    x = ifelse(
        sly > 0,
        ntuple(n -> vy[n] / sqrt(sly), Val(N + 1)),
        ntuple(i -> ifelse(i == 1, one(T), zero(T)), Val(N + 1)),
    )
    ℓi = TV.logjac_zero(flag, T)
    if !(flag isa TV.NoLogJac)
        ℓi -= sly / 2
    end
    return x, ℓi, index2
end

function TV.transform_with(
        flag::TV.LogJacFlag, t::TV.ArrayTransformation{<:SphericalUnitVector{N}}, y::AbstractVector, index
    ) where {N}
    (; inner_transformation, dims) = t
    T = eltype(y)
    ℓ = TV.logjac_zero(flag, T)
    out = ntuple(_ -> similar(y, dims), Val(N + 1))
    index0 = index
    @trace for i in eachindex(out...)
        θ, ℓi, index2 = TV.transform_with(flag, inner_transformation, y, index0)
        ℓ += ℓi
        index0 = index2
        _set_output!(out, θ, i)
    end
    return out, ℓ, index + TV.dimension(inner_transformation) * length(out[1])
end
function _set_output!(out::NTuple{M}, x, i) where {M}
    return ntuple(Val(M)) do n
        PT._rsetindex!(out[n], PT._rgetindex(x, n), i)
    end
end

function TV.inverse_at!(
        x::AbstractArray, index, t::TV.ArrayTransformation{<:SphericalUnitVector{N}}, y::NTuple
    ) where {N}
    @assert length(y) == N + 1
    ix = 1
    itr = index:(N + 1):(index + TV.dimension(t) - 1)
    M = N + 1
    @trace track_numbers = false for i in itr
        ntuple(Val(M)) do j
            PT._rsetindex!(x, PT._rgetindex(y[j], ix), i + j - 1)
        end
        ix += 1
    end
    return index + TV.dimension(t)
end

# Scalar (single) `SphericalUnitVector` inverse: `y` is the unit-vector `NTuple`; write its
# components straight back as the representative latent point (the section of the
# normalisation map). The `ArrayTransformation` method above handles the vector-of-directions
# case; without this one, `latent_pback`/`inverse` of a standalone transform `MethodError`s.
function TV.inverse_at!(x, index, ::SphericalUnitVector{N}, y::NTuple) where {N}
    @assert length(y) == N + 1
    ntuple(Val(N + 1)) do j
        PT._rsetindex!(x, PT._rgetindex(y, j), index + j - 1)
    end
    return index + N + 1
end

# Constructor functions declared in core (so callers can build these transforms
# without ProbabilityTransports depending on TransformVariables).
PT.angle_transform() = AngleTransform()
PT.spherical_unit_vector(N::Integer) = SphericalUnitVector{N}()

# Flat transport for the angular distributions: each angle is an `AngleTransform`.
PT.transport_node(d::PT.DiagonalVonMises, ::PT.TVFlat) = as(Vector, AngleTransform(), length(d))
PT.transport_node(::PT.DiagonalVonMises{<:Real, <:Real, <:Real}, ::PT.TVFlat) = AngleTransform()
PT.transport_node(d::PT.WrappedUniform, ::PT.TVFlat) = as(Vector, AngleTransform(), length(d))
PT.transport_node(::PT.WrappedUniform{<:Real}, ::PT.TVFlat) = AngleTransform()

PT.basemeasure(::PT.TVFlat, ::Int) = nothing

end # module
