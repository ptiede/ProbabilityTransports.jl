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
_elem_base(d::StdInverseGamma{T, <:Number}, i) where {T} = StdInverseGamma{T, T, 0}(d.α, ())
_elem_base(d::StdInverseGamma{T, <:AbstractArray}, i) where {T} = StdInverseGamma{T, T, 0}(d.α[i], ())
_elem_base(d::StdTDist{T, <:Number}, i) where {T} = StdTDist{T, T, 0}(d.ν, ())
_elem_base(d::StdTDist{T, <:AbstractArray}, i) where {T} = StdTDist{T, T, 0}(d.ν[i], ())

function transport_step(c::ArrayTransport{<:_ArrayStd}, y, index)
    d = c.dist
    m = prod(c.dims)
    T = _ensure_float(eltype(y))
    out = Vector{T}(undef, m)
    ℓ = zero(T)
    @inbounds for i in 1:m
        xi, ℓi, index = transport_step(ScalarTransport(_elem_base(d, i), c.space), y, index)
        out[i] = xi
        ℓ += ℓi
    end
    return reshape(out, c.dims), ℓ, index
end

function pullback_step!(y, index, c::ArrayTransport{<:_ArrayStd}, x)
    d = c.dist
    xv = vec(x)
    @inbounds for i in eachindex(xv)
        index = pullback_step!(y, index, ScalarTransport(_elem_base(d, i), c.space), xv[i])
    end
    return index
end
