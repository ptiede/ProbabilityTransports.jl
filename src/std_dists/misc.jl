# Internal sampling helpers shared by the Std* distributions.

#
#    _rand_gamma(rng, α)

#Draw a `Gamma(α, 1)` (shape `α`, unit scale) variate using the Marsaglia–Tsang
#method. For `α < 1` the boost identity `Gamma(α) = Gamma(α+1) · U^(1/α)` is used.
#Internal; backs `StdTDist` (via a χ² draw) and `StdInverseGamma`.
#
# Essentially does very smart rejection sampling. 
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