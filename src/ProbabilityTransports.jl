module ProbabilityTransports

import Distributions
const Dists = Distributions
using Distributions: quantile, cdf

using Random
using Random: AbstractRNG, randn!, rand!, randexp, randexp!

using LinearAlgebra
using LinearAlgebra: I, diag, cholesky, Diagonal, logabsdet

using SpecialFunctions
using SpecialFunctions: erf, erfinv, loggamma

import PDMats

import ChangesOfVariables
import InverseFunctions
using ChangesOfVariables: with_logabsdet_jacobian
using InverseFunctions: inverse

using ArgCheck: @argcheck
using Tricks: static_hasmethod

using ReactantCore: @trace

using Bessels: besseli0x

# Reactant-safe scalar indexing helpers. Plain indexing on CPU; a Reactant
# extension can specialize these to traced get/setindex when needed.
@inline _rgetindex(x, i...) = @inbounds x[i...]
@inline _rsetindex!(x, v, i...) = (@inbounds setindex!(x, v, i...); v)

# ----- standard reference distributions ("spaces") -----------------------
include("std_dists.jl")

# ----- transport engine ---------------------------------------------------
include("spaces.jl")
include("transport.jl")
include("pushforward.jl")
include("composite.jl")
include("namedist.jl")
include("specialized.jl")
include("std_transport.jl")
include("angular.jl")
include("truncated.jl")
include("delta.jl")
include("projected_normal.jl")
include("transported.jl")

# ----- constructors for transforms that live in the TransformVariables ext --
# These return `TV.VectorTransform`s (`AngleTransform`, `SphericalUnitVector{N}`)
# defined in the extension. Declaring them here lets callers build those transforms
# without ProbabilityTransports taking a hard TransformVariables dependency — the
# methods are added when TransformVariables is loaded.

"""
    angle_transform()

A TransformVariables transform mapping two reals `(x, y)` to an angle `atan(x, y)`
(requires `TransformVariables` to be loaded). See the manual on circular variables.
"""
function angle_transform end

"""
    spherical_unit_vector(N)

A TransformVariables transform mapping `N+1` reals to a unit vector on the
`N`-sphere (requires `TransformVariables` to be loaded).
"""
function spherical_unit_vector end

export StdNormal, StdUniform, StdExponential, StdTDist, StdInverseGamma, StdFlat
export NamedDist, TupleDist, DiagonalVonMises, WrappedUniform, DeltaDist, ProjectedNormal
export transport_to, transport_node, transport, transport_and_logjac, pullback, logpdf_fwd, dimension
export TransportedDistribution
export angle_transform, spherical_unit_vector

end # module
