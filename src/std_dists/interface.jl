"""
    AbstractStdDist{T, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}}

Constructs a standardized distribution with element type `T` and `N` dimensions.
The `Std*` distributions are all subtypes of `AbstractStdDist` and share a common
interface, which is defined here.

To implement a new `Std*` distribution, define a struct subtype of `AbstractStdDist` and
implement the following methods:
- `unnormed_logpdf(d::YourStdDist, x)`
- `lognorm(d::YourStdDist)`
- `_std_rand!(rng, d::YourStdDist, x)`

You can also optionally implement
- `mean(d::YourStdDist)`
- `var(d::YourStdDist)`
- `std(d::YourStdDist)`
- `cov(d::YourStdDist)`
- `cdf(d::YourStdDist, x)`
- `quantile(d::YourStdDist, p)`
- `Base.size(d::YourStdDist)`
- `Base.length(d::YourStdDist)`
- `Base.eltype(d::YourStdDist)`

To make the distribution transportable to a *non-matching* space, implement `cdf`/
`quantile` on the 0-dim form; to transport its array-shaped (`N ≥ 1`) instances, also
implement
- `_elem_dist(d::YourStdDist, i; lognorm = false)`
"""
abstract type AbstractStdDist{T, N} <: Dists.ContinuousDistribution{Dists.ArrayLikeVariate{N}} end

"""
    _elem_dist(d::AbstractStdDist, i; lognorm = false)

The 0-dim (univariate) element distribution for linear slot `i` of an array-shaped `Std*`
distribution — the per-element factor the array transport path (`std_transport.jl`) maps
through the public 0-dim `cdf`/`quantile`. Linear indexing `[i]` is vec order, matching
the flattened latent block, NOT the distribution's own array axes. Families with scalar
or no parameters ignore `i`.

`lognorm` defaults to `false`: the element skips the normalization cache — `cdf`/
`quantile` never read it and `lognorm(d)` recomputes on demand — and a literal default
const-props through the fused step broadcasts, where a passed kwarg would become a
runtime field of the broadcast closure and destroy inference. Pass `lognorm = true` for
a fully-cached element.

Implemented per family in `std_dists/`; there is deliberately no generic fallback, so a
family without a method fails here rather than deep inside a broadcast kernel.
"""
function _elem_dist end

"""
    @cached_scalar_std StdFoo param _lognorm_foo

Generate the shared "cached-scalar Std" plumbing for a twin family (`StdInverseGamma`,
`StdTDist`) whose single parameter field is `param` and whose normalization is
`_lognorm_foo(param, N)`. Emits, once per family:

- the keyword constructor (with the `lognorm`-caching type-stability branch),
- the `Number` / `Number, Int...` / `AbstractArray` convenience constructors,
- the cached `lognorm(d)` accessor and the uncached (`lognorm = false`) recompute, and
- the per-element `_elem_dist` (scalar- and array-parameter forms).

Only the pdf kernels, sampling, moments, and cdf/quantile differ between the families and
stay in each family file. The family `struct` must use the default inner constructor with
fields `(param, lognorm, dims)`; this macro supplies the outer constructors that derive
the type parameters.
"""
macro cached_scalar_std(Std, param, lognormfn)
    return esc(
        quote
            # Precompute the (expensive) normalization at construction. Store the parameter
            # as-is (`Tp = typeof(p)` — scalar or array); derive the float output eltype `T`
            # from its element type so an integer parameter doesn't make `T = Int` (which
            # would break the `T(±Inf)`/`T(NaN)` moments). Each branch constructs a concrete
            # type (never `new{…, typeof(l)}` of a Union), so the return infers as a 2-type
            # union that call sites split. `float(::Type)` resolves for traced eltypes inside
            # a trace — the only place they exist.
            function $Std(p::Union{Number, AbstractArray}, dims::Dims{N}; lognorm::Bool = true) where {N}
                T = float(eltype(p))
                if lognorm
                    l = $lognormfn(p, prod(dims))
                    return $Std{T, typeof(p), N, typeof(l)}(p, l, dims)
                else
                    return $Std{T, typeof(p), N, Nothing}(p, nothing, dims)
                end
            end
            $Std(p::Number; lognorm::Bool = true) = $Std(p, (); lognorm)
            $Std(p::Number, dims::Int...; lognorm::Bool = true) = $Std(p, dims; lognorm)
            $Std(p::AbstractArray; lognorm::Bool = true) = $Std(p, size(p); lognorm)

            @inline lognorm(d::$Std) = d.lognorm
            # Uncached (constructed with `lognorm = false`): recompute on demand, so the
            # skipped cache is an optimization detail, never observable behavior.
            @inline lognorm(d::$Std{<:Any, <:Any, <:Any, Nothing}) = $lognormfn(d.$param, prod(d.dims))

            # the array-transport element (see `_elem_dist` above): 0-dim with the slot's
            # parameter — linear `[i]` is vec order. Scalar-parameter instances ignore `i`.
            _elem_dist(d::$Std{<:Any, <:Number}, i; lognorm::Bool = false) = $Std(d.$param; lognorm)
            _elem_dist(d::$Std{<:Any, <:AbstractArray}, i; lognorm::Bool = false) = $Std(d.$param[i]; lognorm)
        end,
    )
end

dims(d::AbstractStdDist) = getfield(d, :dims)
Base.size(d::AbstractStdDist) = dims(d)
Base.length(d::AbstractStdDist) = prod(size(d))
Base.eltype(::Type{<:AbstractStdDist{T}}) where {T} = T
Base.eltype(::AbstractStdDist{T}) where {T} = T

function Dists.logpdf(d::AbstractStdDist{T, 0}, x::Number) where {T}
    return unnormed_logpdf(d, x) + lognorm(d)
end

@with_real function Dists.logpdf(d::AbstractStdDist{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return unnormed_logpdf(d, x) + lognorm(d)
end

# Array sampling, delegating to the per-distribution `_std_rand!` so the actual sampler
# lives in exactly one place. `@with_real` emits the `<:Number` overload (admits traced
# Reactant arrays) and the `<:Real` companion that breaks the ambiguity with
# `Distributions._rand!(::Sampleable{<:ArrayLikeVariate}, ::AbstractArray{<:Real})`.
@with_real function Dists._rand!(rng::AbstractRNG, d::AbstractStdDist{T, N}, x::AbstractArray{<:Number, N}) where {T, N}
    return _std_rand!(rng, d, x)
end

# ----- transport target space (the `space_*` trait from spaces.jl) -----------
# A `Std*` distribution doubles as a target space for `transport_to`. The trait is
# per-coordinate, so it derives from the 0-dim element distribution (`_elem_dist`) via the
# ordinary `cdf`/`quantile`/`logpdf`: a family that implements those (0-dim) plus `_elem_dist`
# is usable as a space with no separate `space_*` methods. `space_dimension` keeps the
# type-keyed default of 1 (spaces.jl); a family only overrides it if a scalar dof consumes
# ≠ 1 latent coordinate.
space_cdf(d::AbstractStdDist, y) = Dists.cdf(_elem_dist(d, 1), y)
space_quantile(d::AbstractStdDist, u) = Dists.quantile(_elem_dist(d, 1), u)
space_logpdf(d::AbstractStdDist, y) = Dists.logpdf(_elem_dist(d, 1), y)
