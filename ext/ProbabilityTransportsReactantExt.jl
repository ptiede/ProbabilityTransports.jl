module ProbabilityTransportsReactantExt

using ProbabilityTransports
const PT = ProbabilityTransports
using Reactant

# Scalar indexing into a traced array is disallowed by default, but the scalar
# transport nodes read/write one latent coordinate at a time. Route those through
# `@allowscalar` (the same approach as ComradeBase's `rgetindex`). The vectorized
# array nodes (matching-base identity, affine `ScaleShift`) use whole-array / range
# views and never hit this path, so the hot image-prior transports stay fast.

# Indices are left untyped: under `@jit` the index can itself be a traced
# integer (`TracedRNumber{Int}`) — e.g. a loop counter inside an array
# transform's `transform_with` — which is not `<:Integer`, so a narrower
# signature would fall through to a disallowed scalar `getindex`.
Base.@propagate_inbounds function PT._rgetindex(I::Reactant.AnyTracedRArray, i...)
    return Reactant.@allowscalar I[i...]
end
Base.@propagate_inbounds function PT._rsetindex!(I::Reactant.AnyTracedRArray, v, i...)
    Reactant.@allowscalar I[i...] = v
    return v
end

end # module
