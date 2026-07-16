module ProbabilityTransportsReactantExt

using ProbabilityTransports
const PT = ProbabilityTransports
using Reactant
import Distributions

# `TransportedDistribution{T,S,E,V}` carries its variate form `V` (an abstract phantom tag,
# e.g. `Multivariate = ArrayLikeVariate{1}`) purely to dispatch `rand`/`_rand!`; no field
# stores a value of that type. Reactant only walks a struct's *type parameters* when tracing
# one of its *fields* changed the type (Tracing.jl: `if !changed; return T`) — so on most
# priors the phantom `V` is never visited. But as soon as a field is device-traced (an array
# somewhere in the transport/prior tree), Reactant re-instantiates the type, walks every
# parameter, hits the abstract `V`, and throws `Unhandled abstract type ...Multivariate`.
# Variate-form tags carry nothing to trace, so return them unchanged. This keeps the phantom
# `V` (so `rand`/`_rand!` dispatch is preserved) while making the type Reactant-traceable
# regardless of what the prior holds.
function Reactant.traced_type_inner(
        @nospecialize(T::Type{<:Distributions.VariateForm}),
        seen::Dict{Type, Type},
        mode::Reactant.TraceMode,
        track_numbers::Type,
        ndevices,
        runtime,
    )
    return T
end

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
