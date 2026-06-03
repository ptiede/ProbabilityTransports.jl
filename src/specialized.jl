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

# Cheap path: under StdNormal the iid-normal inner is the identity.
function transport_node(d::Dists.MvNormal, space::StdNormal)
    n = length(d)
    return PushforwardTransport(AffineTransform(Vector(d.μ), _chol_scale(d.Σ)), ArrayIdentity(n, space))
end

# Inner node producing `n` iid standard normals from the latent space (via the
# scalar quantile path). Used by the generic MvNormal builder; a dedicated node
# avoids `product_distribution` collapsing iid Normals back into an MvNormal.
struct IIDNormalInner{S} <: AbstractTransport
    n::Int
    space::S
end
dimension(c::IIDNormalInner) = c.n * space_dimension(typeof(c.space))

function transport_step(c::IIDNormalInner, y, index)
    T = _ensure_float(eltype(y))
    out = Vector{T}(undef, c.n)
    ℓ = zero(T)
    nrm = Dists.Normal()
    @inbounds for i in 1:(c.n)
        zi, ℓi, index = transport_step(ScalarTransport(nrm, c.space), y, index)
        out[i] = zi
        ℓ += ℓi
    end
    return out, ℓ, index
end
function pullback_step!(y, index, c::IIDNormalInner, z)
    nrm = Dists.Normal()
    @inbounds for i in 1:(c.n)
        index = pullback_step!(y, index, ScalarTransport(nrm, c.space), z[i])
    end
    return index
end
pullback_eltype(::IIDNormalInner, ::Type{V}) where {V <: AbstractArray} = _ensure_float(eltype(V))

# Generic space: the inner draws iid standard normals via the quantile path.
function transport_node(d::Dists.MvNormal, space)
    n = length(d)
    return PushforwardTransport(AffineTransform(Vector(d.μ), _chol_scale(d.Σ)), IIDNormalInner(n, space))
end

# ----- Dirichlet: stick-breaking, dimension-reducing (K -> K-1) ------------
#
# The stick-breaking map (per-coordinate Beta quantiles) is an *exact* measure
# transport from the per-coordinate reference to the Dirichlet, so the log
# Jacobian is given by the change-of-variables identity
#   logjac = Σ_i logpdf_S(y_i) - logpdf(Dirichlet, x)
# which avoids hand-deriving the Beta Jacobian and makes logpdf_fwd reduce to the
# reference density (≈ 0 for StdUniform). The pullback is the re-derived exact
# inverse.

dimension(c::ArrayTransport{<:Dists.Dirichlet}) =
    (prod(c.dims) - 1) * space_dimension(typeof(c.space))

function transport_step(c::ArrayTransport{<:Dists.Dirichlet}, y, index)
    d = c.dist
    α = d.alpha
    K = length(α)
    m = K - 1
    T = _ensure_float(eltype(y))
    x = zeros(T, K)
    ℓs = zero(T)
    remaining = one(T)
    @inbounds for i in 1:m
        yi = y[index + i - 1]
        u = space_cdf(c.space, yi)
        β = sum(@view α[(i + 1):K])
        φ = quantile(Dists.Beta(T(α[i]), T(β)), u)
        x[i] = remaining * φ
        remaining -= x[i]
        ℓs += space_logpdf(c.space, yi)
    end
    x[K] = remaining
    ℓ = ℓs - Dists.logpdf(d, x)
    return x, ℓ, index + m
end

function pullback_step!(y, index, c::ArrayTransport{<:Dists.Dirichlet}, x)
    d = c.dist
    α = d.alpha
    K = length(α)
    m = K - 1
    remaining = one(eltype(x))
    @inbounds for i in 1:m
        β = sum(@view α[(i + 1):K])
        φ = x[i] / remaining
        u = cdf(Dists.Beta(α[i], β), φ)
        y[index + i - 1] = space_quantile(c.space, u)
        remaining -= x[i]
    end
    return index + m
end
