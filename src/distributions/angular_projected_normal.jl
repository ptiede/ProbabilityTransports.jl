# AngularProjectedNormal — the *angle-valued* projected normal. Where `ProjectedNormal`'s
# variate is the 2n-vector `x`, this distribution's variate is the n angle(s)
# `θᵢ = atan(x₂ᵢ, x₂ᵢ₋₁)` — the projected normal's angular marginal. It therefore has a
# genuine circular density (closed form below) yet still transports *exactly* to
# `StdNormal(2n)` / `StdUniform(2n)`: the transport is an `atan²` pushforward of the 2n
# shifted standard normals. Unlike `DiagonalVonMises` (which has no measure-preserving map
# to the line/cube) it standardizes exactly, so it is the circular prior to reach for when
# you need standard-normal coordinates — e.g. GeoVI — but want a *scalar phase* that drops
# into scalar-phase machinery (a single value per site/time) unchanged.
#
# Like `ProjectedNormal`/`DiagonalVonMises`, `μ` and `γ` may be scalars (one angle) or
# length-`n` vectors (`n` independent angles). The latent space is ℝ^{2n}; the variate is
# ℝ^n (the angles).

# ----- pairwise atan² map (the dimension-reducing pushforward) ----------------
# Forward: ℝ^{2n} → ℝ^n, `(z₂ᵢ₋₁, z₂ᵢ) ↦ atan(z₂ᵢ, z₂ᵢ₋₁)`. It is dimension-reducing and
# not invertible; the `inverse` is the canonical unit-radius section
# `θ ↦ (cos θ, sin θ)` (radius is not recoverable, and is only needed to seed a latent
# point from a parameter value via `latent_pback`). Used only on the transport side — the
# density side is the closed-form `logpdf` below, never a change of variables through `f`.
struct PairwiseAtan2 end
struct PairwiseUnitVector end

function (::PairwiseAtan2)(z::AbstractVector)
    n2 = length(z)
    return atan.(@view(z[2:2:n2]), @view(z[1:2:n2]))
end
# interleave (cos, sin) pairs: [cosθ₁, sinθ₁, cosθ₂, sinθ₂, …] (matches `ν`'s layout)
(::PairwiseUnitVector)(θ::AbstractVector) = vec(permutedims(hcat(cos.(θ), sin.(θ))))

InverseFunctions.inverse(::PairwiseAtan2) = PairwiseUnitVector()
InverseFunctions.inverse(::PairwiseUnitVector) = PairwiseAtan2()

# ----- the distribution -------------------------------------------------------

"""
    AngularProjectedNormal(μ, γ)
    AngularProjectedNormal(ν::AbstractVector)

A circular (phase) prior whose variate is the angle(s) `θ` that concentrate around `μ`
(radians) with concentration `γ ≥ 0` (the length of the mean vector). `μ`, `γ` are scalars
for a single angle or length-`n` vectors for `n` independent angles. It is the *angular
marginal* of `ProjectedNormal`: drawing `X ~ MvNormal(γ·(cos μ, sin μ), I₂)` and returning
`atan(X₂, X₁)`. `γ = 0` is uniform on the circle; larger `γ` concentrates more tightly
around `μ`. The single-argument form takes a length-2 mean vector `ν` directly.

Unlike `VonMises`/`DiagonalVonMises`, it transports **exactly** to `StdNormal()` /
`StdUniform()` (an `atan²` pushforward of an affine shift of `StdNormal(2n)`), so it is the
recommended circular prior when you need smooth standard-normal coordinates (e.g. GeoVI)
but want a scalar phase. The latent space is ℝ^{2n}; the variate (`length`) is `n` angles.

See also [`ProjectedNormal`](@ref), whose variate is the 2n-vector instead of the angle.
"""
struct AngularProjectedNormal{M, G, V} <: Dists.ContinuousMultivariateDistribution
    μ::M    # mean angle(s) in radians (scalar or length-n vector)
    γ::G    # concentration(s) ≥ 0     (scalar or length-n vector)
    ν::V    # cached mean vector, length 2n: [γᵢcosμᵢ, γᵢsinμᵢ …]
end

function AngularProjectedNormal(μ::Number, γ::Number)
    μp, γp = promote(float(μ), float(γ))
    return AngularProjectedNormal(μp, γp, _projnormal_meanvec(μp, γp))
end
AngularProjectedNormal(μ::AbstractVector, γ::AbstractVector) =
    AngularProjectedNormal(μ, γ, _projnormal_meanvec(μ, γ))
function AngularProjectedNormal(ν::AbstractVector)
    @argcheck length(ν) == 2
    return AngularProjectedNormal(atan(ν[2], ν[1]), hypot(ν[1], ν[2]))
end

Base.length(d::AngularProjectedNormal) = length(d.ν) ÷ 2          # n angles
Base.eltype(d::AngularProjectedNormal) = float(eltype(d.ν))
Dists.insupport(d::AngularProjectedNormal, x::AbstractVector) = length(x) == length(d)

# Closed-form projected-normal angular density, summed over independent directions. For one
# direction with `η = γ cos(θ − μ)`:
#   f(θ) = e^{−γ²/2}/2π  +  (η/√{2π}) · e^{−γ² sin²(θ−μ)/2} · Φ(η),   Φ(x) = (1+erf(x/√2))/2.
# Vectorized and branchless so it traces under Reactant. `_erf_poly` (an elementary,
# `chlo.erf`-free erf defined in `std_dists/std_normal.jl`) is used instead of `erf` to
# sidestep the Enzyme-JAX constant-batching bug EnzymeAD/Enzyme-JAX#2559 (see the comment
# there). Revert to `erf` once #2559 lands.
function Dists.logpdf(d::AngularProjectedNormal, θ::AbstractVector)
    T = float(eltype(d.ν))
    dθ = θ .- d.μ
    η = d.γ .* cos.(dθ)
    Φ = (1 .+ _erf_poly.(η ./ sqrt(T(2)))) ./ 2
    f = exp.(-(d.γ .^ 2) ./ 2) ./ (2 * T(π)) .+
        (η ./ sqrt(2 * T(π))) .* exp.(-(d.γ .^ 2) .* sin.(dθ) .^ 2 ./ 2) .* Φ
    return sum(log, f)
end

function Dists._rand!(rng::AbstractRNG, d::AngularProjectedNormal, x::AbstractVector)
    n2 = length(d.ν)
    z = randn(rng, float(eltype(d.ν)), n2) .+ d.ν
    x .= atan.(@view(z[2:2:n2]), @view(z[1:2:n2]))
    return x
end

# Concatenate directions, mirroring `ProjectedNormal` (scalar `vcat` → vector).
function Dists.product_distribution(dists::AbstractVector{<:AngularProjectedNormal})
    μ = mapreduce(Base.Fix2(getproperty, :μ), vcat, dists)
    γ = mapreduce(Base.Fix2(getproperty, :γ), vcat, dists)
    return AngularProjectedNormal(μ, γ)
end

# Exact transport (Std spaces): atan² ∘ (affine shift by ν) over the 2n standard normals.
# `dimension` is 2n (latents consumed); `pfwd` emits n angles. `TVFlat` is intentionally
# not provided here — `asflat` on a posterior carrying this prior errors loudly rather than
# silently using the wrong (dimension-preserving) core node; the StdNormal/StdUniform paths
# are what GeoVI and nested sampling need.
function transport_node(d::AngularProjectedNormal, space::Union{StdNormal, StdUniform})
    ν = d.ν
    inner = PushforwardTransport(
        ScaleShift(ν, one(eltype(ν))), transport_node(StdNormal(length(ν)), space)
    )
    return PushforwardTransport(PairwiseAtan2(), inner)
end
