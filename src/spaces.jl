# ----- the "space" trait --------------------------------------------------
#
# A *space* is the latent reference we transport into. It is specified by an
# instance of a `Std*` distribution (or the `StdFlat()` marker, handled in the
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

# ----- StdUniform: the unit hypercube (identity latent<->unit map) ---------

space_cdf(::StdUniform, y) = clamp(y, zero(y), one(y))
space_quantile(::StdUniform, u) = u
space_logpdf(::StdUniform, y) = zero(y)
space_dimension(::Type{<:StdUniform}) = 1

# ----- StdNormal: standard normal latent space ----------------------------

space_cdf(d::StdNormal, y) = _std_cdf(d, y)        # Φ  (defined in StdDists/std_normal.jl)
space_quantile(d::StdNormal, u) = _std_quantile(d, u)  # Φ⁻¹
space_logpdf(::StdNormal, y) = -y * y / 2 - oftype(y, log(2π) / 2)
space_dimension(::Type{<:StdNormal}) = 1

# ----- StdFlat: the TransformVariables (unconstrained ℝⁿ) space ------------
# Marker only; all of its behavior lives in the TransformVariables extension so
# that the marker can be named without loading TransformVariables.

struct StdFlat end
