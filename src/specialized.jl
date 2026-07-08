# ----- cheap analytic / pushforward specializations -----------------------
#
# These `transport_node` methods take precedence over the generic cdf/quantile
# `ScalarTransport`/`ArrayTransport` paths when a distribution is an affine
# reparameterization of the matching Std base — avoiding `erf`/`erfinv` etc. and
# staying Reactant-friendly.

# Affine / LocationScale: μ + σ·base. Wrap the base's transport with the affine.
function transport_node(d::Dists.AffineDistribution, space)
    return PushforwardTransport(ScaleShift(d.μ, d.σ), transport_node(d.ρ, space))
end

# Normal -> StdNormal: x = μ + σ·y directly (no erf/erfinv).
function transport_node(d::Dists.Normal, space::StdNormal)
    return PushforwardTransport(ScaleShift(d.μ, d.σ), ScalarIdentity(space))
end

# Uniform -> StdUniform: x = a + (b-a)·u directly.
function transport_node(d::Dists.Uniform, space::StdUniform)
    return PushforwardTransport(ScaleShift(d.a, d.b - d.a), ScalarIdentity(space))
end

# ----- MvNormal: affine pushforward of independent standard normals --------

_chol_scale(Σ) = cholesky(Σ).L
_chol_scale(Σ::PDMats.PDiagMat) = Diagonal(sqrt.(Σ.diag))
_chol_scale(Σ::PDMats.ScalMat) = Diagonal(fill(sqrt(Σ.value), Σ.dim))

# `x = μ + L·z` with `L = chol(Σ)` and `z` a vector of `n` iid standard normals.
# The inner that produces those `n` iid normals is just the array-`StdNormal`
# base's own transport: the matching-base identity under `StdNormal`, and the
# per-element `Φ⁻¹` quantile loop under any other space. (This is the same shape
# VLBI's matrix-scale `AffineDistribution` uses.)
function transport_node(d::Dists.MvNormal, space)
    n = length(d)
    return PushforwardTransport(
        AffineTransform(Vector(d.μ), _chol_scale(d.Σ)), transport_node(StdNormal(n), space)
    )
end

# ----- Dirichlet: stick-breaking, dimension-reducing (K -> K-1) ------------
#
# The stick-breaking map (per-coordinate Beta quantiles) is an *exact* measure
# transport from the per-coordinate reference to the Dirichlet (dimension-reducing,
# K -> K-1). Because it is exact, its pulled-back density under a Std space is the
# closed-form reference and no Jacobian is needed (only `TVFlat` Dirichlet, handled by
# TV's `UnitSimplex`, carries a Jacobian). The latent_pback is the re-derived exact inverse.

dimension(c::ArrayTransport{<:Dists.Dirichlet}) =
    (prod(c.dims) - 1) * space_dimension(typeof(c.space))

# Regularised incomplete-beta cdf / quantile straight from SpecialFunctions — the same
# primitives `StdTDist` uses — rather than building a `Dists.Beta` and calling its generic
# `quantile`/`cdf` (which allocates a distribution object and is not Reactant-traceable).
# `beta_inc`/`beta_inc_inv` return `(lower, upper)` tails; the lower tail is the cdf.
@inline _beta_cdf(a, b, x) = first(SpecialFunctions.beta_inc(a, b, x))
@inline _beta_quantile(a, b, p) = first(SpecialFunctions.beta_inc_inv(a, b, p))
function pfwd_step(c::ArrayTransport{<:Dists.Dirichlet}, y, index)
    d = c.dist
    α = d.alpha
    K = length(α)
    m = K - 1
    T = _ensure_float(eltype(y))
    x = similar(y, T, K)
    remaining = one(T)
    β = T(sum(α))   # running suffix sum: after subtracting α[i] it equals Σ α[(i+1):K]
    indexr = promote_index(index)
    @trace for i in 1:m
        αi = T(_rgetindex(α, i))
        u = space_cdf(c.space, _rgetindex(y, indexr + (i - 1)))
        β -= αi
        φ = _beta_quantile(αi, β, u)
        xi = remaining * φ
        _rsetindex!(x, xi, i)
        remaining -= xi
    end
    _rsetindex!(x, remaining, K)
    return x, index + m
end

function pback_step!(y, index, c::ArrayTransport{<:Dists.Dirichlet}, x)
    d = c.dist
    α = d.alpha
    K = length(α)
    m = K - 1
    T = _ensure_float(eltype(x))
    remaining = one(T)
    β = T(sum(α))   # running suffix sum, as in `pfwd_step`
    indexr = promote_index(index)
    @trace for i in 1:m
        αi = T(_rgetindex(α, i))
        β -= αi
        xi = _rgetindex(x, i)
        φ = xi / remaining
        u = _beta_cdf(αi, β, φ)
        _rsetindex!(y, space_quantile(c.space, u), indexr + (i - 1))
        remaining -= xi
    end
    return index + m
end
