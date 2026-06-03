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
# A `TV.AbstractTransform` cannot subtype our `AbstractTransport` (no retroactive
# subtyping across packages), but it already implements exactly our node protocol
# under different names, so we just forward to TV's index-threaded primitives. The
# core composite machinery then threads TV transforms through Tuples/NamedTuples/
# NamedDist as leaves with no extra glue.

PT.dimension(t::TV.AbstractTransform) = TV.dimension(t)

# `transform_with` still needs a TV flag; we always ask for the log-Jacobian and let
# the caller discard it (the value-only path is cold for the flat space).
function PT.transport_step(t::TV.AbstractTransform, y, index)
    return TV.transform_with(TV.LogJac(), t, y, index)
end

PT.pullback_step!(y, index, t::TV.AbstractTransform, x) = TV.inverse_at!(y, index, t, x)
PT.pullback_eltype(t::TV.AbstractTransform, ::Type{T}) where {T} = TV.inverse_eltype(t, T)

# Standalone-leaf drivers: TV transforms are not `<: AbstractTransport`, so they do
# not inherit the core `transport`/`transport_and_logjac`/`pullback` drivers. These
# delegate through the step protocol (which handles scalar-in-a-vector correctly).
PT.transport(t::TV.AbstractTransform, y) = first(PT.transport_step(t, y, firstindex(y)))
function PT.transport_and_logjac(t::TV.AbstractTransform, y)
    val, ℓ, _ = PT.transport_step(t, y, firstindex(y))
    return val, ℓ
end
function PT.pullback(t::TV.AbstractTransform, x)
    y = Vector{PT.pullback_eltype(t, typeof(x))}(undef, TV.dimension(t))
    PT.pullback_step!(y, firstindex(y), t, x)
    return y
end

# ----- per-distribution flat building blocks (the asflat dispatch table) -----
#
# Each method is more specific (on the StdFlat space) than the core `transport_node`
# methods, so there is no ambiguity. Tuples / NamedTuples / NamedDist fall through to
# the core composite `transport_node`. `stop === nothing` makes `logpdf == logpdf_fwd`
# for the flat space.

function _interval(d::Dists.UnivariateDistribution)
    s = Dists.support(d)
    lb = isfinite(s.lb) ? s.lb : -TV.∞
    ub = isfinite(s.ub) ? s.ub : TV.∞
    return as(Real, lb, ub)
end

PT.transport_node(d::Dists.ContinuousUnivariateDistribution, ::PT.StdFlat) = _interval(d)
PT.transport_node(d::Dists.AffineDistribution, ::PT.StdFlat) = _interval(d)
PT.transport_node(d::Dists.Dirichlet, ::PT.StdFlat) = TV.UnitSimplex(length(d.alpha))
PT.transport_node(d::Dists.MvNormal, ::PT.StdFlat) = as(Vector, length(d))
PT.transport_node(d::Dists.MvLogNormal, ::PT.StdFlat) = as(Vector, as(Real, 0, TV.∞), length(d))
PT.transport_node(d::Dists.Product, ::PT.StdFlat) = as(Vector, _interval(first(d.v)), length(d.v))

# array-shaped Std* bases: 0-dim → scalar interval; N ≥ 1 → unconstrained array.
_stdflat(d, inner) = isempty(size(d)) ? inner : as(Array, inner, size(d)...)
PT.transport_node(d::PT.StdNormal, ::PT.StdFlat) = _stdflat(d, as(Real, -TV.∞, TV.∞))
PT.transport_node(d::PT.StdExponential, ::PT.StdFlat) = _stdflat(d, as(Real, 0, TV.∞))
PT.transport_node(d::PT.StdInverseGamma, ::PT.StdFlat) = _stdflat(d, as(Real, 0, TV.∞))
PT.transport_node(d::PT.StdTDist, ::PT.StdFlat) = _stdflat(d, as(Real, -TV.∞, TV.∞))
PT.transport_node(d::PT.StdUniform, ::PT.StdFlat) = _stdflat(d, as(Real, 0, 1))

# Truncated: flat transform is the constrained interval implied by the bounds.
PT.transport_node(d::PT.Truncated{<:Any, <:Real, <:Real}, ::PT.StdFlat) = as(Real, d.lower, d.upper)
PT.transport_node(d::PT.Truncated{<:Any, <:Real, Nothing}, ::PT.StdFlat) = as(Real, d.lower, TV.∞)
PT.transport_node(d::PT.Truncated{<:Any, Nothing, <:Real}, ::PT.StdFlat) = as(Real, -TV.∞, d.upper)
PT.transport_node(d::PT.Truncated{<:Any, Nothing, Nothing}, ::PT.StdFlat) = as(Real, -TV.∞, TV.∞)

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

# Constructor functions declared in core (so callers can build these transforms
# without ProbabilityTransports depending on TransformVariables).
PT.angle_transform() = AngleTransform()
PT.spherical_unit_vector(N::Integer) = SphericalUnitVector{N}()

# Flat transport for the angular distributions: each angle is an `AngleTransform`.
PT.transport_node(d::PT.DiagonalVonMises, ::PT.StdFlat) = as(Vector, AngleTransform(), length(d))
PT.transport_node(::PT.DiagonalVonMises{<:Real, <:Real, <:Real}, ::PT.StdFlat) = AngleTransform()
PT.transport_node(d::PT.WrappedUniform, ::PT.StdFlat) = as(Vector, AngleTransform(), length(d))
PT.transport_node(::PT.WrappedUniform{<:Real}, ::PT.StdFlat) = AngleTransform()

PT._reference(::PT.StdFlat, ::Int) = nothing

end # module
