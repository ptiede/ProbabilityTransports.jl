# Transport for array-shaped Std* bases (N ≥ 1): per-element scalar transport,
# each element through the family's scalar cdf/quantile kernel. This is what lets an affine
# distribution over an array Std base (e.g. a per-pixel Gaussian) transport via
# `PushforwardTransport(AffineTransform(loc, scale), transport_node(base, space))`.
#
# (0-dim Std* are univariate and already go through the generic scalar path.)

const _ArrayStd = Union{StdNormal, StdUniform, StdExponential, StdInverseGamma, StdTDist}

# `_elem_dist` (part of the `AbstractStdDist` interface — declared in
# `std_dists/interface.jl`, implemented per family) supplies the 0-dim element for each
# latent slot; the steps below map it through the public univariate `cdf`/`quantile`.

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
# builds the isbits 0-dim element dist per slot inside the kernel, so the steps go
# through the ordinary univariate `cdf`/`quantile` of the `AbstractStdDist` interface.
# Array-parameter families are eager only, as before.
#
# `_clamp_unit` (defined in transport.jl) keeps the cdf strictly inside (0,1) so
# `quantile` of an unbounded target stays finite — see the note there.
#
# The element dist is built inside the broadcast function, not broadcast as an argument:
# Reactant cannot trace a broadcast whose arguments are distribution objects.
function pfwd_step(c::ArrayTransport{<:_ArrayStd}, y, index)
    m = prod(c.dims)
    yv = @view y[index:(index + m - 1)]
    d, s = c.dist, c.space
    # x = Q_D(F_S(y)), fused so only the output allocates
    x = ((yi, i) -> quantile(_elem_dist(d, i), _clamp_unit(space_cdf(s, yi)))).(yv, 1:m)
    return reshape(x, c.dims), index + m
end

function pback_step!(y, index, c::ArrayTransport{<:_ArrayStd}, x)
    m = prod(c.dims)
    d, s = c.dist, c.space
    # y = Q_S(F_D(x)), fused straight into the latent block
    @views y[index:(index + m - 1)] .=
        ((xi, i) -> space_quantile(s, _clamp_unit(cdf(_elem_dist(d, i), xi)))).(vec(x), 1:m)
    return index + m
end
