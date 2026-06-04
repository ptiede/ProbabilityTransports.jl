# Internal sampling helpers shared by the Std* distributions.

#
#    _rand_gamma(rng, α)

#Draw a `Gamma(α, 1)` (shape `α`, unit scale) variate using the Marsaglia–Tsang
#method. For `α < 1` the boost identity `Gamma(α) = Gamma(α+1) · U^(1/α)` is used.
#Internal; backs `StdTDist` (via a χ² draw) and `StdInverseGamma`.
#
function _rand_gamma(rng::AbstractRNG, α::Real)
    T = float(typeof(α))
    a = T(α)
    if a < one(T)
        u = rand(rng, T)
        return _rand_gamma(rng, a + one(T)) * u^inv(a)
    end
    d = a - one(T) / 3
    c = inv(sqrt(9 * d))
    while true
        x = randn(rng, T)
        v = (one(T) + c * x)^3
        v <= zero(T) && continue
        u = rand(rng, T)
        if log(u) < x * x / 2 + d - d * v + d * log(v)
            return d * v
        end
    end
end
