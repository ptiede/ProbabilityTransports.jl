# Transport for array-shaped Std* bases (N ≥ 1): per-element scalar transport,
# each element through the family's scalar cdf/quantile kernel. This is what lets an affine
# distribution over an array Std base (e.g. a per-pixel Gaussian) transport via
# `PushforwardTransport(AffineTransform(loc, scale), transport_node(base, space))`.
#
# (0-dim Std* are univariate and already go through the generic scalar path.)

const _ArrayStd = Union{StdNormal, StdUniform, StdExponential, StdInverseGamma, StdTDist}

# The element distribution for latent slot `i`, constructed inside the fused step
# broadcasts below (isbits — no per-element array is ever materialized). Only the
# array-parameter families depend on `i`: linear `[i]` is vec order, lining up with the
# flat latent block `y[index:index+m-1]`, NOT the distribution's own array axes. They use
# the lognorm-free constructor — the cdf/quantile kernels never read `lognorm`, so the
# per-element `loggamma` would be pure waste. Everything else is `i`-independent and
# returns itself (the `_std_cdf`/`_std_quantile` kernels ignore the shape).
_elem_dist(d::AbstractStdDist, i) = d
_elem_dist(d::StdInverseGamma{<:Any, <:AbstractArray}, i) = StdInverseGamma(d.α[i], nothing, ())
_elem_dist(d::StdTDist{<:Any, <:AbstractArray}, i) = StdTDist(d.ν[i], nothing, ())

# Matching base → space (StdNormal→StdNormal, StdUniform→StdUniform) is the
# identity: no cdf/quantile. Vectorized as a range-view + reshape, so it traces
# under Reactant and avoids a wasteful `erf∘erfinv` round-trip. This is the inner
# of every affine image prior over its matching space.
# 0-dim matching base → space is the scalar identity (no quantile → traces).
transport_node(::StdNormal{T, 0}, space::StdNormal) where {T} = ScalarIdentity(space)
transport_node(::StdUniform{T, 0}, space::StdUniform) where {T} = ScalarIdentity(space)

for (B, S) in ((:StdNormal, :StdNormal), (:StdUniform, :StdUniform))
    @eval function pfwd_step(c::ArrayTransport{<:$B, M, <:$S}, y, index) where {M}
        m = prod(c.dims)
        z = reshape(@view(y[index:(index + m - 1)]), c.dims)
        return z, index + m
    end
    @eval function pback_step!(y, index, c::ArrayTransport{<:$B, M, <:$S}, x) where {M}
        m = prod(c.dims)
        @views y[index:(index + m - 1)] .= vec(x)
        return index + m
    end
end

# Generic *non-matching* base→space transport (e.g. an array StdExponential base targeting
# StdNormal, or any array base whose space differs from it). Same scalar factoring as
# `ScalarTransport` — `x = Q_D(F_S(y))` forward, `u = F_D(x); y = Q_S(u)` back — but applied to a
# contiguous `m`-block of `y` with *broadcasts* instead of a scalar `out[i] = …` loop. That keeps
# it allocation-shaped to the backend AND lowers fully under Reactant: the scalar `setindex!` the
# old `@trace` loop used is disallowed on traced arrays, whereas the broadcasts below become
# elementwise traced ops (the per-element `cdf`/`quantile` — erf/erfinv, log, … — lower as scalar
# kernels). The dist enters the broadcast as a `Ref` plus the element index — `_elem_dist`
# builds the isbits element dist per slot inside the kernel, and the `_std_cdf`/
# `_std_quantile` kernels are shape-agnostic (the public `Dists.cdf`/`quantile` overloads
# stay deliberately 0-dim-only). Array-parameter families are eager only, as before.
#
# `_clamp_unit` (defined in transport.jl) keeps the cdf strictly inside (0,1) so
# `quantile` of an unbounded target stays finite — see the note there.
function pfwd_step(c::ArrayTransport{<:_ArrayStd}, y, index)
    m = prod(c.dims)
    yv = @view y[index:(index + m - 1)]
    # x = Q_D(F_S(y)), fused so only the output allocates
    x = _std_quantile.(_elem_dist.(Ref(c.dist), 1:m), _clamp_unit.(space_cdf.(Ref(c.space), yv)))
    return reshape(x, c.dims), index + m
end

function pback_step!(y, index, c::ArrayTransport{<:_ArrayStd}, x)
    m = prod(c.dims)
    # y = Q_S(F_D(x)), fused straight into the latent block
    @views y[index:(index + m - 1)] .=
        space_quantile.(Ref(c.space), _clamp_unit.(_std_cdf.(_elem_dist.(Ref(c.dist), 1:m), vec(x))))
    return index + m
end
