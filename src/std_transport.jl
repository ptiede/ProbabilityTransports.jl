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

# Generic per-element fallback for a *non-matching* base→space (e.g. an array StdExponential
# base targeting StdNormal, or any array base whose space differs from it). It walks the scalar
# cdf/quantile path element-by-element. The buffer follows the backend (`similar(y, …)`) and the
# loop is `@trace`d, so the structure itself traces — but the per-element scalar `cdf`/`quantile`
# (erf/erfinv, gamma_inc, …) are host calls, so a *mismatched*-base array transport does not
# fully lower under Reactant. That's fine: the Reactant-fast path is the matching base→space
# identity vectorized above (`:27-38`), which is the one Gaussian image priors actually take.
function pfwd_step(c::ArrayTransport{<:_ArrayStd}, y, index)
    d = c.dist
    m = prod(c.dims)
    T = _ensure_float(eltype(y))
    out = similar(y, T, m)
    index0 = promote_index(index)
    @trace track_numbers=false for i in eachindex(out)
        xi, index0 = pfwd_step(ScalarTransport(_elem_base(d, i), c.space), y, index0)
        out[i] = xi
        nothing
    end
    return reshape(out, c.dims), index + m
end

function pback_step!(y, index, c::ArrayTransport{<:_ArrayStd}, x)
    d = c.dist
    xv = vec(x)
    index0 = promote_index(index)
    @trace track_numbers=false for i in eachindex(xv)
        index0 = pback_step!(y, index0, ScalarTransport(_elem_base(d, i), c.space), xv[i])
    end
    return index0
end
