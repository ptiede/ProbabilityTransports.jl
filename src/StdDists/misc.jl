# Internal sampling helpers shared by the Std* distributions.

#
#    _rand_gamma(rng, α)

#Draw a `Gamma(α, 1)` (shape `α`, unit scale) variate using the Marsaglia–Tsang
#method. For `α < 1` the boost identity `Gamma(α) = Gamma(α+1) · U^(1/α)` is used.
#Internal; backs `StdTDist` (via a χ² draw) and `StdInverseGamma`.
#
# Reactant-traceable Marsaglia–Tsang. The subtlety is that a scalar accumulated
# through a `@trace` loop is only carried if it is a *traced* value; a concrete
# `zero(T)` returns its initial value under `@compile`. So under compilation we make
# the working element type `T` traced (via `promote_to_traced`), which makes every
# `zero(T)`/`one(T)`/draw traced and lets the loop carry `dv`. On CPU `T` is the plain
# float type and the same code runs as an ordinary loop. `α` may itself be traced
# (it arrives promoted from inside an array `@trace for`), so the `a < 1` boost uses
# `@trace if`, not a plain branch.
function _rand_gamma(rng::AbstractRNG, α::Number)
    af = float(α)
    a = within_compile() ? promote_to_traced(af) : af
    T = typeof(a)
    # shape < 1 via the boost identity `Gamma(a) = Gamma(a+1) · U^(1/a)`
    @trace if a < one(T)
        boost = rand(rng, T)^inv(a)
        a = a + one(T)
    else
        boost = one(T)
    end
    d = a - one(T) / 3
    c = inv(sqrt(9 * d))
    # Bounded rejection loop with early exit: `dv == 0` means "not yet accepted", so
    # the loop stops on the first acceptance (≈1 iteration on CPU) but is bounded for
    # Reactant. `dv` is traced under @compile (T is), so `@trace while` carries it.
    dv = zero(T)
    i = 0
    @trace while (dv == zero(dv)) & (i < 32)
        x = randn(rng, T)
        v = (one(T) + c * x)^3
        u = rand(rng, T)
        vpos = v > zero(T)
        vsafe = ifelse(vpos, v, one(T))                # keep `log(vsafe)` finite when v <= 0
        cand = vpos & (log(u) < x * x / 2 + d - d * vsafe + d * log(vsafe))
        dv = ifelse(cand, d * vsafe, dv)               # keep the first accepted draw
        i = i + 1
    end
    return boost * dv
end
