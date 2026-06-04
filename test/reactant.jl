# Reactant tracing tests. Included from `runtests.jl` only when Reactant loads
# successfully (it is skipped on, e.g., Julia nightly where Reactant often won't
# precompile — see the guard in `runtests.jl`).
#
# These lock in the Reactant-friendly angular code: the von Mises / wrapped-uniform
# `logpdf` and the `@trace while` von Mises sampler must lower under `@compile`.

using ProbabilityTransports
using Distributions
using Random
using Reactant
using Test

const PT = ProbabilityTransports

Reactant.set_default_backend("cpu")

# `cf(args)` returns a Reactant concrete scalar; pull a plain `Float64` out of it.
_val(x) = Float64(Reactant.to_number(x))

@testset "Reactant tracing" begin

    @testset "DiagonalVonMises logpdf traces" begin
        d = DiagonalVonMises(Float32[0.0, 1.0], Float32[2.0, 5.0])
        x0 = Float32[0.1, 0.9]
        cpu = logpdf(d, x0)
        cf = @compile (x -> logpdf(d, x))(Reactant.to_rarray(x0))
        @test _val(cf(Reactant.to_rarray(x0))) ≈ cpu rtol = 1.0f-5
    end

    @testset "WrappedUniform logpdf traces" begin
        wu = WrappedUniform(Float32(2π), 3)
        x0 = Float32[0.1, 0.2, 0.3]
        cpu = logpdf(wu, x0)
        cf = @compile (x -> logpdf(wu, x))(Reactant.to_rarray(x0))
        @test _val(cf(Reactant.to_rarray(x0))) ≈ cpu rtol = 1.0f-5
    end

    @testset "von Mises @trace-while sampler traces" begin
        # Concrete (μ, κ) with a traced RNG: this is the realistic use (sample on
        # device inside a larger traced program) and exercises the `@trace while`
        # rejection loop. `sum(z) * 0` just gives `@compile` a traced array input.
        μ = 0.5f0
        dummy = Reactant.to_rarray(Float32[0.0])
        draw(z) = PT._rand_vonmises(Random.default_rng(), μ, 3.0f0) + sum(z) * 0.0f0
        cf = @compile draw(dummy)
        θ = _val(cf(dummy))
        # support is μ ± π; allow a little slack rather than asserting exact bounds.
        @test (μ - π - 1) ≤ θ ≤ (μ + π + 1)
    end

    @testset "non-rejection scalar samplers trace under @compile" begin
        # `StdNormal`/`StdExponential` use `randn(rng, T)`/`randexp(rng, T)` (not
        # `T(randn(rng))`, which has no `T(::TracedRNumber)` method); `StdUniform` uses
        # `rand(rng, T)`. None has a rejection loop, so they lower. The RNG is obtained
        # *inside* the traced function so Reactant overlays a traced RNG; a baked seed
        # makes each call deterministic, so we assert a finite, in-support draw.
        dummy = Reactant.to_rarray(Float64[0.0])
        draw_n(z) = rand(Random.default_rng(), StdNormal()) + sum(z) * 0.0
        draw_e(z) = rand(Random.default_rng(), StdExponential()) + sum(z) * 0.0
        draw_u(z) = rand(Random.default_rng(), StdUniform()) + sum(z) * 0.0
        @test isfinite(_val((@compile draw_n(dummy))(dummy)))
        let e = _val((@compile draw_e(dummy))(dummy)); @test isfinite(e) && e ≥ 0 end
        let u = _val((@compile draw_u(dummy))(dummy)); @test 0 ≤ u ≤ 1 end
    end

    @testset "non-rejection array samplers trace under @compile" begin
        # The ambiguity-free `_rand!` (delegating to `_std_rand!`) lets traced arrays
        # — whose eltype is `TracedRNumber`, not `<:Real` — dispatch. These three use
        # whole-array `randn!`/`randexp!`/`rand!`, so they sample correctly on device.
        a = Reactant.to_rarray(zeros(4))
        arr_n(x) = rand!(Random.default_rng(), StdNormal(4), x)
        arr_e(x) = rand!(Random.default_rng(), StdExponential(4), x)
        arr_u(x) = rand!(Random.default_rng(), StdUniform(4), x)
        @test all(isfinite, Array((@compile arr_n(a))(a)))
        @test all(≥(0), Array((@compile arr_e(a))(a)))
        let u = Array((@compile arr_u(a))(a)); @test all(0 .≤ u .≤ 1) end
    end

    @testset "gamma rejection samplers (StdTDist/StdInverseGamma) trace" begin
        # `_rand_gamma` (Marsaglia–Tsang, backing `StdTDist`/`StdInverseGamma`) lowers
        # under `@compile`: the working element type is made traced so the bounded
        # `@trace while` carries the accumulator, and the `α<1` boost uses `@trace if`.
        dummy = Reactant.to_rarray(Float64[0.0])
        draw_g(z) = PT._rand_gamma(Random.default_rng(), 3.0) + sum(z) * 0.0
        draw_t(z) = rand(Random.default_rng(), StdTDist(5.0)) + sum(z) * 0.0
        draw_i(z) = rand(Random.default_rng(), StdInverseGamma(3.0)) + sum(z) * 0.0
        @test _val((@compile draw_g(dummy))(dummy)) > 0          # Gamma support (0, ∞)
        @test isfinite(_val((@compile draw_t(dummy))(dummy)))
        @test _val((@compile draw_i(dummy))(dummy)) > 0          # InverseGamma support
    end

    @testset "_rand_gamma is statistically correct under @compile" begin
        # Strong guard: with the seed as a runtime input the sampler must reproduce
        # the Gamma(α, 1) mean (= α), not merely return a finite value — catches a
        # regression to a non-threading loop (which returns a constant / 0).
        gdraw(seed) = PT._rand_gamma(Reactant.ReactantRNG(seed), 3.7)
        cf = @compile gdraw(Reactant.to_rarray(UInt64[1, 2]))
        N = 50_000
        s = 0.0
        for _ in 1:N
            s += _val(cf(Reactant.to_rarray(UInt64[rand(UInt64), rand(UInt64)])))
        end
        @test isapprox(s / N, 3.7; atol = 0.1)   # mean of Gamma(3.7, 1) is 3.7
    end
end
