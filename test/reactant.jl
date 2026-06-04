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
end
