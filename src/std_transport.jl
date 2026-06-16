# Transport for array-shaped Std* bases (N ≥ 1): per-element scalar transport,
# each element using the 0-dim scalar base. This is what lets an affine
# distribution over an array Std base (e.g. a per-pixel Gaussian) transport via
# `PushforwardTransport(AffineTransform(loc, scale), transport_node(base, space))`.
#
# (0-dim Std* are univariate and already go through the generic scalar path.)

const _ArrayStd = Union{StdNormal, StdUniform, StdExponential, StdInverseGamma, StdTDist}

# the 0-dim scalar base for element `i` (per-element params for InverseGamma/TDist)
_elem_base(::StdNormal{T}, i) where {T} = StdNormal{T, 0}(())
_elem_base(::StdUniform{T}, i) where {T} = StdUniform{T, 0}(())
_elem_base(::StdExponential{T}, i) where {T} = StdExponential{T, 0}(())
_elem_base(d::StdInverseGamma{T, <:Number}, i) where {T} = StdInverseGamma(d.α, ())
_elem_base(d::StdInverseGamma{T, <:AbstractArray}, i) where {T} = StdInverseGamma(d.α[i], ())
_elem_base(d::StdTDist{T, <:Number}, i) where {T} = StdTDist(d.ν, ())
_elem_base(d::StdTDist{T, <:AbstractArray}, i) where {T} = StdTDist(d.ν[i], ())

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
# kernels). Element bases are homogeneous for the scalar-parameter Std families
# (StdNormal/StdUniform/StdExponential and Number-parameter StdInverseGamma/StdTDist); the
# array-parameter families broadcast their per-element params (eager only, as before).
_elem_bases(d::Union{StdNormal, StdUniform, StdExponential}) = Ref(_elem_base(d, 1))
_elem_bases(d::StdInverseGamma{T, <:Number}) where {T} = Ref(_elem_base(d, 1))
_elem_bases(d::StdTDist{T, <:Number}) where {T} = Ref(_elem_base(d, 1))
_elem_bases(d::StdInverseGamma{T, <:AbstractArray}) where {T} = StdInverseGamma.(d.α, Ref(()))
_elem_bases(d::StdTDist{T, <:AbstractArray}) where {T} = StdTDist.(d.ν, Ref(()))

function pfwd_step(c::ArrayTransport{<:_ArrayStd}, y, index)
    m = prod(c.dims)
    yv = @view y[index:(index + m - 1)]
    u = space_cdf.(Ref(c.space), yv)              # F_S : latent → [0,1]
    x = quantile.(_elem_bases(c.dist), u)         # Q_D : [0,1] → target value
    return reshape(x, c.dims), index + m
end

function pback_step!(y, index, c::ArrayTransport{<:_ArrayStd}, x)
    m = prod(c.dims)
    u = cdf.(_elem_bases(c.dist), vec(x))                                  # F_D : target → [0,1]
    @views y[index:(index + m - 1)] .= space_quantile.(Ref(c.space), u)    # Q_S : [0,1] → latent
    return index + m
end
