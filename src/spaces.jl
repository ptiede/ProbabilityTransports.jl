# ----- the "space" trait --------------------------------------------------
#
# A *space* is the latent reference we transport into. It is specified by an
# instance of a `Std*` distribution (or the `TVFlat()` marker, handled in the
# TransformVariables extension). For the quantile-based spaces, a space only has
# to supply a per-coordinate map to/from the unit interval plus a per-coordinate
# reference log-density:
#
#   space_cdf(space, y)       F_S : latent coordinate -> u ∈ [0,1]
#   space_quantile(space, u)  Q_S : u -> latent coordinate
#   space_logpdf(space, y)    per-coordinate latent log density
#   space_dimension(S)        latent coordinates consumed per original scalar dof
#
# `space_dimension` is keyed on the *type* so node dimensions stay compile-time
# constants (critical for type-stable index threading).

function space_cdf end
function space_quantile end
function space_logpdf end

space_dimension(::Type) = 1

"""
    TVFlat

Uses the TransformVariables transformation as the transport. Unlike StdNormal, StdUniform, we preserve the
original distribution but move the support to ℝⁿ, so the transport is a TV transform rather than a pushforward. 
"""
struct TVFlat end
