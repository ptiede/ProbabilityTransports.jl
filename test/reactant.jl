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
        let e = _val((@compile draw_e(dummy))(dummy))
            @test isfinite(e) && e ≥ 0
        end
        let u = _val((@compile draw_u(dummy))(dummy))
            @test 0 ≤ u ≤ 1
        end
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
        let u = Array((@compile arr_u(a))(a))
            @test all(0 .≤ u .≤ 1)
        end
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

    @testset "transport path traces under @compile" begin
        # The calls a sampler makes in its hot loop: transport, logpdf_pfwd,
        # latent_pfwd_and_logdensity, and latent_pback! — over the Reactant-fast nodes
        # (matching-base identity + affine; no cross-space quantiles).

        # affine image prior: ScaleShift over the matching array StdNormal base
        μ = Float32.(reshape(1:4, 2, 2)) ./ 4
        σ = fill(0.5f0, 2, 2)
        d = PushforwardDistribution(PT.ScaleShift(μ, σ), StdNormal{Float32}(2, 2))
        dto = transport_to(d, StdNormal{Float32}())
        y0 = randn(MersenneTwister(42), Float32, 4)
        cx = latent_pfwd(dto, y0)
        cl = logpdf_pfwd(dto, y0)
        yr = Reactant.to_rarray(y0)
        fx = @compile (y -> latent_pfwd(dto, y))(yr)
        @test Array(fx(yr)) ≈ cx rtol = 1.0f-5
        fl = @compile (y -> logpdf_pfwd(dto, y))(yr)
        @test _val(fl(yr)) ≈ cl rtol = 1.0f-5
        ft = @compile (y -> last(latent_pfwd_and_logdensity(dto, y)))(yr)
        @test _val(ft(yr)) ≈ cl rtol = 1.0f-5

        # latent_pback! into a caller-owned traced buffer (the buffer sets the backend)
        xr = Reactant.to_rarray(collect(cx))
        yb = Reactant.to_rarray(zeros(Float32, 4))
        fb = @compile ((b, x) -> latent_pback!(b, dto, x))(yb, xr)
        @test Array(fb(yb, xr)) ≈ y0 rtol = 1.0f-5

        # MvNormal: affine-Cholesky pushforward
        Σ = [2.0 0.5; 0.5 1.0]
        μ2 = [1.0, -1.0]
        dmv = transport_to(Distributions.MvNormal(μ2, Σ), StdNormal())
        z0 = randn(MersenneTwister(43), 2)
        zr = Reactant.to_rarray(z0)
        fmv = @compile (y -> latent_pfwd(dmv, y))(zr)
        @test Array(fmv(zr)) ≈ latent_pfwd(dmv, z0) rtol = 1.0e-6

        # NamedTuple composite of traceable leaves under the matching space
        nc = transport_to((a = Distributions.Normal(1.0, 2.0), b = Distributions.MvNormal(μ2, Σ)), StdNormal())
        w0 = randn(MersenneTwister(44), 3)
        wr = Reactant.to_rarray(w0)
        fc = @compile (y -> latent_pfwd(nc, y).b)(wr)
        @test Array(fc(wr)) ≈ latent_pfwd(nc, w0).b rtol = 1.0e-6
        flc = @compile (y -> logpdf_pfwd(nc, y))(wr)
        @test _val(flc(wr)) ≈ logpdf_pfwd(nc, w0) rtol = 1.0e-6
    end

    @testset "index threading: pfwd_step/pback_step! advance correctly under @compile" begin
        # The danger under `@trace`/tracing is a DROPPED index update: a node that consumes
        # several latent coordinates must advance the shared index so a *later* node reads
        # fresh coordinates. We put a multi-coordinate node BEFORE a distinct scalar leaf, so
        # a dropped/duplicated index makes the trailing leaf read the wrong coordinate and the
        # compiled result diverges from CPU. The pback!∘pfwd round-trip then exercises the same
        # threading through `pback_step!`. (Only the Reactant-traceable nodes are used here:
        # affine-Cholesky `MvNormal`, the matching-base array identity, and the composite
        # recursion. The generic `Product`/cross-space array nodes are CPU-only — covered in
        # runtests.jl — and intentionally not compiled.)
        Σ3 = [2.0 0.3 0.1; 0.3 1.5 0.2; 0.1 0.2 1.0]
        μ3 = [1.0, 2.0, 3.0]

        # MvNormal (3 coords) THEN a scalar Normal that must consume coordinate 4.
        A = transport_to((mv = Distributions.MvNormal(μ3, Σ3), n = Normal(50.0, 0.1)), StdNormal())
        yA = randn(MersenneTwister(101), 4)
        let yr = Reactant.to_rarray(yA), yb = Reactant.to_rarray(zeros(4))
            fn = @compile (y -> latent_pfwd(A, y).n)(yr)
            @test _val(fn(yr)) ≈ latent_pfwd(A, yA).n rtol = 1.0e-6        # trailing leaf = coord 4
            frt = @compile ((y, b) -> latent_pback!(b, A, latent_pfwd(A, y)))(yr, yb)
            @test Array(frt(yr, yb)) ≈ yA rtol = 1.0e-6                    # round-trip: pback! threading
        end

        # Matching-base image-prior array node (2x2 = 4 coords, the @view/reshape path) THEN a
        # scalar Normal that must consume coordinate 5 — the hot image-prior layout.
        img = PushforwardDistribution(
            PT.ScaleShift(Float64.(reshape(1:4, 2, 2)), fill(0.5, 2, 2)), StdNormal(2, 2)
        )
        B = transport_to((img = img, n = Normal(99.0, 0.1)), StdNormal())
        yB = randn(MersenneTwister(102), 5)
        let yr = Reactant.to_rarray(yB), yb = Reactant.to_rarray(zeros(5))
            fn = @compile (y -> latent_pfwd(B, y).n)(yr)
            @test _val(fn(yr)) ≈ latent_pfwd(B, yB).n rtol = 1.0e-6        # trailing leaf = coord 5
            frt = @compile ((y, b) -> latent_pback!(b, B, latent_pfwd(B, y)))(yr, yb)
            @test Array(frt(yr, yb)) ≈ yB rtol = 1.0e-6
        end
    end

    @testset "array-base logpdf / insupport are Reactant-safe" begin
        # Guards the `Std*` array `logpdf` (the StdUniform/StdExponential kernels mask via
        # `insupport` on the whole array through `ifelse`) and that `insupport` is consumed the
        # Reactant-safe data-flow way. A regression to a boolean `if`/`&&` form would fail here.
        bases = (
            StdNormal(4), StdUniform(4), StdExponential(4),
            StdInverseGamma(3.0, (4,)), StdTDist(5.0, (4,)),
        )
        xs = (
            randn(MersenneTwister(1), 4), rand(MersenneTwister(2), 4),
            abs.(randn(MersenneTwister(3), 4)) .+ 0.1, abs.(randn(MersenneTwister(4), 4)) .+ 0.5,
            randn(MersenneTwister(5), 4),
        )
        for (d, x0) in zip(bases, xs)
            xr = Reactant.to_rarray(x0)
            fl = @compile (x -> logpdf(d, x))(xr)
            @test _val(fl(xr)) ≈ logpdf(d, x0) rtol = 1.0e-5
            fi = @compile (x -> ifelse(insupport(d, x), one(eltype(x)), -one(eltype(x))))(xr)
            @test _val(fi(xr)) ≈ 1.0          # all x0 are in-support
        end
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
