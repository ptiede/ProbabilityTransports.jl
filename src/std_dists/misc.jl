# Internal sampling helpers shared by the Std* distributions.

#
#    _rand_gamma(rng, α)
#
# Draw a `Gamma(α, 1)` (shape `α`, unit scale) variate using the Marsaglia–Tsang squeeze-
# rejection method. For `α < 1` the boost identity `Gamma(α) = Gamma(α+1) · U^(1/α)` is used.
# Internal; backs `StdTDist` (via a χ² draw) and `StdInverseGamma`.
#
# Two finite-precision notes (both match `Distributions`' own gamma/InverseGamma sampler):
#   * The rejection `while` is capped at 32 iterations so it traces under Reactant (a plain
#     `while` cannot). The acceptance probability per iteration is high, so exhausting the cap
#     (returning the `dv == 0` sentinel) has probability ~1e-54 — negligible.
#   * For *very small* shape (α ≲ 0.02) the boost `U^(1/α)` underflows to 0, so the draw is 0.
#     This is NOT a defect: such a `Gamma(α,1)` genuinely places that much mass below `floatmin`
#     (≈49% of the mass at α=0.001), so the matching `InverseGamma`/`StudentT` draw is genuinely
#     above `floatmax` and returns `Inf` — exactly as `Distributions` does (verified). Draws are
#     finite for any practical shape (α ≳ 0.05).
function _rand_gamma(rng::AbstractRNG, α::Number)
    af = float(α)
    a = within_compile() ? promote_to_traced(af) : af
    T = typeof(a)
    boost = rand(rng, T)
    @trace if a < one(T)
        boost = boost^inv(a)
        a = a + one(T)
    else
        boost = one(T)
    end
    d = a - one(T) / 3
    c = inv(sqrt(9 * d))

    # ensures dv is Traced
    dv = zero(T)
    i = 0
    @trace while (dv == zero(dv)) & (i < 32)
        x = randn(rng, T)
        v = (one(T) + c * x)^3
        u = rand(rng, T)
        vpos = v > zero(T)
        vsafe = ifelse(vpos, v, one(T))                # keep `log(vsafe)` finite when v <= 0
        # Check if the sample is accepted.
        cand = vpos & (log(u) < x * x / 2 + d - d * vsafe + d * log(vsafe))
        dv = ifelse(cand, d * vsafe, dv)               # keep the first accepted draw
        i = i + 1
    end
    return boost * dv
end

_getith(x::Number, i) = x
_getith(x::AbstractArray, i) = _rgetindex(x, i)

#
#    _rand_gamma!(rng, out, α)
#
# Fill `out` with independent `Gamma(α, 1)` draws, where `α` is either a scalar (broadcast
# over `out`) or an array matching `size(out)`. This is the array counterpart of the scalar
# `_rand_gamma` and backs the `_std_rand!` samplers for `StdInverseGamma`/`StdTDist` (and
# `VLBIBeta` in VLBIImagePriors).
#
# It differs from a per-element `@trace for i … _rand_gamma(rng, …)` loop in one crucial
# way: all randomness is drawn as *whole arrays* (`rand`/`randn`) and the fixed 32-iteration
# rejection is a plain (compile-time-unrolled) Julia loop, so there is NO nested `@trace`
# region. That matters under Reactant — a nested `@trace while` (the scalar sampler's
# rejection loop) inside an outer per-element `@trace for` does not thread `rng.seed` across
# the outer iterations, collapsing every element to the *same* draw. Array-level `rand`
# advances the seed once per whole-array call and threads correctly.
#
# The boost uniform `U^(1/a)` (the `a < 1` identity) is drawn unconditionally and masked to
# 1 for `a ≥ 1`, since the shape may be small for some elements and not others; the extra
# draw for large-`a` elements is harmless. The unrolled loop always runs all 32 iterations
# (no early exit) — the same bound the scalar `@trace while` traces to under Reactant.
function _rand_gamma!(rng::AbstractRNG, out::AbstractArray, α)
    T = eltype(out)
    a_in = float.(α .+ zero(out))                       # length/shape of `out`; scalar or array α
    small = a_in .< one(T)
    a = ifelse.(small, a_in .+ one(T), a_in)
    boost = ifelse.(small, rand(rng, T, size(out)...) .^ inv.(a_in), one(T))
    d = a .- one(T) / 3
    c = inv.(sqrt.(9 .* d))
    dv = zero(out)
    for _ in 1:32
        x = randn(rng, T, size(out)...)
        u = rand(rng, T, size(out)...)
        v = (one(T) .+ c .* x) .^ 3
        vpos = v .> zero(T)
        vsafe = ifelse.(vpos, v, one(T))               # keep `log(vsafe)` finite when v <= 0
        cand = vpos .& (log.(u) .< x .^ 2 ./ 2 .+ d .- d .* vsafe .+ d .* log.(vsafe))
        newacc = cand .& (dv .== zero(T))              # keep the first accepted draw per element
        dv = ifelse.(newacc, d .* vsafe, dv)
    end
    out .= boost .* dv
    return out
end
