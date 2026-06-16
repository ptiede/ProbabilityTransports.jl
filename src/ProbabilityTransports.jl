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

using ReactantCore: @trace, within_compile, promote_to_traced

using Bessels: besseli0x

export StdNormal, StdUniform, StdExponential, StdTDist, StdInverseGamma, TVFlat
export NamedDist, TupleDist, DiagonalVonMises, WrappedUniform, DeltaDist, ProjectedNormal,
    AngularProjectedNormal
# Don't export dimension due to potential method ambiguities with TransformVariables
export transport_to, transport_node, latent_pfwd, latent_pfwd_and_logdensity, latent_pback, latent_pback!, logpdf_pfwd
export TransportedDistribution, PushforwardDistribution
export angle_transform, spherical_unit_vector


# Reactant-safe scalar indexing helpers. Plain indexing on CPU; a Reactant
# extension can specialize these to traced get/setindex when needed.
Base.@propagate_inbounds _rgetindex(x, i...) = x[i...]
Base.@propagate_inbounds _rsetindex!(x, v, i...) = (setindex!(x, v, i...); v)
function promote_index(i)
    if within_compile()
        return promote_to_traced(i)
    else
        return i
    end
end

# Interface
include("transport.jl")
include("spaces.jl")
include("std_dists/std_dists.jl")


# ----- standard reference distributions ("spaces") -----------------------

# ----- transport engine ---------------------------------------------------
include("pushforward.jl")
include("composite.jl")
include("specialized.jl")
include("std_transport.jl")
include("transported.jl")
include("distributions/distributions.jl")


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


end # module
