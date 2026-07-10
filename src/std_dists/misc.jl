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
