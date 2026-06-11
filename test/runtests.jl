using ProbabilityTransports
# not exported (clashes with TransformVariables.dimension); import explicitly
using ProbabilityTransports: dimension
using Distributions
using Random
using LinearAlgebra
import TransformVariables as TV
using Test

const PT = ProbabilityTransports

# ---- independent exactness oracle ------------------------------------------
# The package no longer computes a Jacobian for Std-space transports (logpdf_fwd ==
# logpdf(reference) holds by construction), so `logpdf_fwd ≈ logpdf(reference)` would be
# tautological. We instead reconstruct the pulled-back density with an INDEPENDENT
# finite-difference Jacobian of the transport map: a wrong map fails
#     logpdf(start, x) + logabsdet(∂x/∂y) ≈ logpdf_fwd(dto, y).
# This works for any bijective (square) map and for the dimension-reducing
# stick-breaking / simplex maps (K → K-1), which drop the redundant last coordinate.
_flatten(x::Number) = [float(x)]
_flatten(x::AbstractArray) = vec(collect(float.(x)))
_flatten(x::Union{Tuple, NamedTuple}) =
    isempty(x) ? Float64[] : reduce(vcat, map(_flatten, values(x)))

function fd_logjac(dto, y)
    f = yy -> _flatten(transport(dto, yy))
    n = length(y)
    g = length(f(y)) == n ? f : (yy -> f(yy)[1:n])   # dim-reducing: drop redundant coord
    h = 1.0e-6
    J = Matrix{Float64}(undef, n, n)
    for j in 1:n
        yp = collect(float.(y)); yp[j] += h
        ym = collect(float.(y)); ym[j] -= h
        J[:, j] .= (g(yp) .- g(ym)) ./ (2h)
    end
    return first(logabsdet(J))
end

# independent reconstruction of logpdf_fwd; compare to the package's value.
fd_fwd(dto, start, y) = logpdf(start, transport(dto, y)) + fd_logjac(dto, y)

# ---- a data-dependent pushforward map (x = exp.(z)) for the general-map tests ----
# Exercises the non-affine PushforwardDistribution path: `const_logdet(ExpMap()) ==
# false`, so the inverse log-det is computed per call via `with_logabsdet_jacobian`.
struct ExpMap end
struct LogMap end
(::ExpMap)(z) = exp.(z)
(::LogMap)(x) = log.(x)
PT.InverseFunctions.inverse(::ExpMap) = LogMap()
PT.InverseFunctions.inverse(::LogMap) = ExpMap()
PT.ChangesOfVariables.with_logabsdet_jacobian(::ExpMap, z) = (exp.(z), sum(z))
PT.ChangesOfVariables.with_logabsdet_jacobian(::LogMap, x) = (log.(x), -sum(log, x))

@testset "ProbabilityTransports.jl" begin

    @testset "Std distributions round-trip references" begin
        cases = [
            (StdNormal(), Normal(), 0.7),
            (StdUniform(), Uniform(), 0.3),
            (StdExponential(), Exponential(), 1.4),
            (StdTDist(5.0), TDist(5.0), 0.9),
            (StdInverseGamma(3.0), InverseGamma(3.0, 1.0), 0.8),
        ]
        for (std, ref, x) in cases
            @test logpdf(std, x) ≈ logpdf(ref, x)
            @test quantile(std, cdf(std, x)) ≈ x
        end
    end

    @testset "scalar round-trip (transport ∘ pullback)" begin
        rng = MersenneTwister(1)
        for D in (Normal(2.0, 3.0), Gamma(2.0, 1.5), Exponential(1.3), Beta(2.0, 3.0))
            for S in (StdNormal(), StdUniform())
                dto = transport_to(D, S)
                y = rand(rng, dto)
                @test transport(dto, pullback(dto, transport(dto, y))) ≈ transport(dto, y)
                @test pullback(dto, transport(dto, y)) ≈ y
            end
        end
    end

    @testset "equal-dim coincidence: logpdf_fwd == logpdf(StdNormal)" begin
        rng = MersenneTwister(2)
        for D in (Normal(1.0, 2.0), Gamma(2.7, 1.3), Exponential(0.8))
            dto = transport_to(D, StdNormal())
            y = randn(rng, dimension(dto))
            # sampler targets the reference; exactness checked independently (FD)
            @test logpdf(dto, y) ≈ logpdf(StdNormal(dimension(dto)), y)
            @test isapprox(fd_fwd(dto, D, y), logpdf_fwd(dto, y); atol = 1e-5)
        end
    end

    @testset "StdUniform pullback density is flat (== 0)" begin
        rng = MersenneTwister(3)
        for D in (Gamma(2.0, 3.0), Beta(2.0, 2.0), product_distribution([Normal(), Gamma(2.0)]))
            dto = transport_to(D, StdUniform())
            u = rand(rng, dimension(dto))
            @test isapprox(logpdf_fwd(dto, u), 0.0; atol = 1e-10)
            @test isapprox(fd_fwd(dto, D, u), logpdf_fwd(dto, u); atol = 1e-5)   # exactness
        end
    end

    @testset "Product" begin
        rng = MersenneTwister(4)
        P = product_distribution([Normal(), Gamma(2.0), Uniform()])
        for S in (StdNormal(), StdUniform())
            dto = transport_to(P, S)
            @test dimension(dto) == 3
            y = rand(rng, dto)
            x = transport(dto, y)
            @test pullback(dto, x) ≈ y
        end
    end

    @testset "regression: Product nested in a composite advances the index" begin
        rng = MersenneTwister(18)
        # Mixed families yield a genuine `Product` (Normals-only would collapse to
        # MvNormal and dodge this path). The component AFTER the Product must consume
        # fresh latent coordinates, not re-read the Product's.
        P = product_distribution([Gamma(2.0, 1.0), Exponential(0.5)])
        @test P isa Distributions.Product
        nd = (a = P, b = Normal(10.0, 0.1))
        for S in (StdNormal(), StdUniform())
            dto = transport_to(nd, S)
            @test dimension(dto) == 3
            y = rand(rng, dto)
            x = transport(dto, y)
            bref = transport(transport_to(Normal(10.0, 0.1), S), y[3:3])
            @test x.b ≈ bref
            @test pullback(dto, x) ≈ y
        end
        # plain-Tuple composite, Product first
        dto = transport_to(
            (product_distribution([Gamma(2.0, 1.0), Gamma(3.0, 1.0)]), Uniform(2.0, 3.0)),
            StdUniform(),
        )
        u = rand(MersenneTwister(19), dimension(dto))
        xt = transport(dto, u)
        @test xt[2] ≈ 2.0 + u[3]
        @test pullback(dto, xt) ≈ u
    end

    @testset "build-time errors for unsupported transports" begin
        # multivariate with no exact array transport: error at build, not first use
        D = MvLogNormal(MvNormal(zeros(2), Matrix(1.0 * I, 2, 2)))
        @test_throws ArgumentError transport_to(D, StdNormal())
        # Std* bases are transportable distributions, not target spaces
        @test_throws ArgumentError transport_to(Normal(), StdExponential())
        @test_throws ArgumentError transport_to(Normal(), StdTDist(3.0))
    end

    @testset "nested NamedDist / TupleDist (mixed shapes, both spaces)" begin
        rng = MersenneTwister(5)
        nd = NamedDist(a = Normal(), b = (Uniform(), Exponential()), c = Gamma(2.0))
        for S in (StdNormal(), StdUniform())
            dto = transport_to(nd, S)
            @test dimension(dto) == 4
            y = rand(rng, dto)
            x = transport(dto, y)
            @test x isa NamedTuple{(:a, :b, :c)}
            @test pullback(dto, x) ≈ y
        end
        # coincidence under StdNormal
        dto = transport_to(nd, StdNormal())
        y = randn(rng, dimension(dto))
        @test logpdf(dto, y) ≈ logpdf(StdNormal(dimension(dto)), y)
        @test isapprox(fd_fwd(dto, nd, y), logpdf_fwd(dto, y); atol = 1e-5)
    end

    @testset "Affine / LocationScale pushforward" begin
        rng = MersenneTwister(6)
        D = 2.0 * Gamma(2.5, 1.3) + 1.0
        @test D isa Distributions.AffineDistribution
        dto = transport_to(D, StdNormal())
        base = transport_to(Gamma(2.5, 1.3), StdNormal())
        y = randn(rng, 1)
        @test transport(dto, y) ≈ 1.0 + 2.0 * transport(base, y)
        @test isapprox(fd_fwd(dto, D, y), logpdf_fwd(dto, y); atol = 1e-5)
        @test pullback(dto, transport(dto, y)) ≈ y
    end

    @testset "cheap scalar specializations (no erf/erfinv)" begin
        rng = MersenneTwister(7)
        Dn = Normal(3.0, 2.0)
        dn = transport_to(Dn, StdNormal())
        y = randn(rng, 1)
        @test transport(dn, y) ≈ 3.0 + 2.0 * y[1]
        @test isapprox(fd_fwd(dn, Dn, y), logpdf_fwd(dn, y); atol = 1e-5)
        Du = Uniform(2.0, 5.0)
        du = transport_to(Du, StdUniform())
        u = rand(rng, 1)
        @test transport(du, u) ≈ 2.0 + 3.0 * u[1]
        @test isapprox(logpdf_fwd(du, u), 0.0; atol = 1e-12)
        @test isapprox(fd_fwd(du, Du, u), logpdf_fwd(du, u); atol = 1e-5)
    end

    @testset "MvNormal (affine / Cholesky) and DiagNormal" begin
        rng = MersenneTwister(8)
        Σ = [2.0 0.5; 0.5 1.0]
        μ = [1.0, -1.0]
        D = MvNormal(μ, Σ)
        dto = transport_to(D, StdNormal())
        @test dimension(dto) == 2
        y = randn(rng, 2)
        @test transport(dto, y) ≈ μ .+ cholesky(Σ).L * y
        @test isapprox(fd_fwd(dto, D, y), logpdf_fwd(dto, y); atol = 1e-5)   # logdet(L) term
        @test pullback(dto, transport(dto, y)) ≈ y
        # generic space path
        dtoU = transport_to(D, StdUniform())
        u = rand(rng, 2)
        @test isapprox(logpdf_fwd(dtoU, u), 0.0; atol = 1e-8)
        @test isapprox(fd_fwd(dtoU, D, u), logpdf_fwd(dtoU, u); atol = 1e-5)
        @test pullback(dtoU, transport(dtoU, u)) ≈ u
        # diagonal
        Dd = MvNormal([0.5, 1.0, -2.0], Diagonal([1.0, 4.0, 0.25]))
        dd = transport_to(Dd, StdNormal())
        yd = randn(rng, 3)
        @test isapprox(fd_fwd(dd, Dd, yd), logpdf_fwd(dd, yd); atol = 1e-5)
        @test pullback(dd, transport(dd, yd)) ≈ yd
    end

    @testset "Dirichlet (stick-breaking, dim K-1)" begin
        rng = MersenneTwister(9)
        for α in ([2.0, 3.0, 1.5], [1.0, 1.0, 1.0, 1.0], [0.7, 2.0])
            D = Dirichlet(α)
            K = length(α)
            dn = transport_to(D, StdNormal())
            @test dimension(dn) == K - 1
            y = randn(rng, K - 1)
            x = transport(dn, y)
            @test sum(x) ≈ 1.0
            @test all(x .>= -1e-12)
            @test pullback(dn, x) ≈ y
            @test isapprox(fd_fwd(dn, D, y), logpdf_fwd(dn, y); atol = 1e-5)   # dim-reducing
            du = transport_to(D, StdUniform())
            u = rand(rng, K - 1)
            xu = transport(du, u)
            @test sum(xu) ≈ 1.0
            @test pullback(du, xu) ≈ u
            @test isapprox(logpdf_fwd(du, u), 0.0; atol = 1e-8)
            @test isapprox(fd_fwd(du, D, u), logpdf_fwd(du, u); atol = 1e-5)
        end
    end

    @testset "circular cannot be exactly transported to StdNormal/StdUniform" begin
        # No measure-preserving map from a circle to the line → must error (not
        # silently do a projected-normal embedding, which is a different distribution).
        for S in (StdNormal(), StdUniform())
            @test_throws ArgumentError transport_to(VonMises(0.0, 2.0), S)
            @test_throws ArgumentError transport_to(DiagonalVonMises([0.0, 1.0], [2.0, 3.0]), S)
            @test_throws ArgumentError transport_to(WrappedUniform(2π, 2), S)
        end
    end

    @testset "ProjectedNormal: directional, transports EXACTLY" begin
        rng = MersenneTwister(17)
        @test ProjectedNormal([0.0, 2.0]).μ ≈ π / 2
        @test ProjectedNormal([0.0, 2.0]).γ ≈ 2.0
        for (μ, γ) in [(0.0, 3.0), (π / 3, 1.5), (0.0, 0.0)]
            d = ProjectedNormal(μ, γ)
            @test length(d) == 2
            for S in (StdNormal(), StdUniform(), TVFlat())
                dto = transport_to(d, S)
                @test dimension(dto) == 2
                y = S === TVFlat() ? randn(rng, 2) : rand(rng, dto)
                x = transport(dto, y)
                if S !== TVFlat()
                    @test isapprox(fd_fwd(dto, d, y), logpdf_fwd(dto, y); atol = 1e-5)   # EXACT
                end
                @test pullback(dto, x) ≈ y
            end
        end
        # concentrates around μ as γ grows (von-Mises-like)
        function meanres(μ, γ)
            d = ProjectedNormal(μ, γ)
            s = 0.0
            for _ in 1:100_000
                x = rand(rng, d)
                s += cos(atan(x[2], x[1]) - μ)
            end
            s / 100_000
        end
        @test meanres(0.0, 0.0) < 0.05      # γ=0 ⇒ ~uniform
        @test meanres(0.0, 3.0) > 0.9       # γ=3 ⇒ tightly concentrated

        # multivariate: n independent directions (mirrors DiagonalVonMises)
        μv = [0.0, π / 3, -π / 2]
        γv = [2.0, 0.5, 3.0]
        d = ProjectedNormal(μv, γv)
        @test length(d) == 2length(μv)
        # direction i occupies coords (2i-1, 2i); its mean angle is μᵢ
        for i in eachindex(μv)
            @test atan(mean(d)[2i], mean(d)[2i - 1]) ≈ μv[i]
        end
        for S in (StdNormal(), StdUniform(), TVFlat())
            dto = transport_to(d, S)
            @test dimension(dto) == 2length(μv)
            y = S === TVFlat() ? randn(rng, length(d)) : rand(rng, dto)
            x = transport(dto, y)
            if S !== TVFlat()
                @test isapprox(fd_fwd(dto, d, y), logpdf_fwd(dto, y); atol = 1e-5)   # EXACT
            end
            @test pullback(dto, x) ≈ y
        end
        # product_distribution of scalar directions reconstructs the vector prior
        pd = product_distribution([ProjectedNormal(μv[i], γv[i]) for i in eachindex(μv)])
        @test length(pd) == length(d)
        @test mean(pd) ≈ mean(d)
    end

    @testset "flat space (TransformVariables extension)" begin
        rng = MersenneTwister(11)
        # univariate: scalar TV transform, vector-in/scalar-out convention
        for D in (Gamma(2.3, 1.1), Normal(1.0, 2.0), LogNormal(0.0, 1.0), Beta(2.0, 3.0))
            dto = transport_to(D, TVFlat())
            @test dimension(dto) == 1
            y = randn(rng)
            x_pt = transport(dto, [y])
            @test logpdf(dto, [y]) ≈ logpdf_fwd(dto, [y])               # flat coincidence
            @test isapprox(fd_fwd(dto, D, [y]), logpdf_fwd(dto, [y]); atol = 1e-5)  # exact density
            @test pullback(dto, x_pt) ≈ [y]
        end
        # multivariate: matches asflat shape; round-trips
        for D in (MvNormal([1.0, -1.0], [2.0 0.5; 0.5 1.0]), Dirichlet([2.0, 3.0, 1.5]))
            dto = transport_to(D, TVFlat())
            y = randn(rng, dimension(dto))
            x = transport(dto, y)
            @test logpdf(dto, y) ≈ logpdf_fwd(dto, y)
            @test isapprox(fd_fwd(dto, D, y), logpdf_fwd(dto, y); atol = 1e-5)
            @test pullback(dto, x) ≈ y
        end
        # nested NamedDist with a Dirichlet leaf
        nd = NamedDist(a = Normal(), b = Gamma(2.0), c = Dirichlet([1.0, 2.0, 3.0]))
        dto = transport_to(nd, TVFlat())
        y = randn(rng, dimension(dto))
        x = transport(dto, y)
        @test x isa NamedTuple{(:a, :b, :c)}
        @test logpdf(dto, y) ≈ logpdf_fwd(dto, y)
        @test pullback(dto, x) ≈ y
    end

    @testset "heterogeneous Product under TVFlat errors; homogeneous works" begin
        # `as(Vector, t, n)` applies one constraint to every coordinate, so mixed
        # supports must error at build instead of silently mis-transforming.
        Phet = product_distribution([Gamma(2.0, 1.0), Normal(0.0, 1.0)])
        @test_throws ArgumentError transport_to(Phet, TVFlat())
        # same support, different parameters: fine
        Phom = product_distribution([Gamma(2.0, 1.0), Exponential(0.5)])
        dto = transport_to(Phom, TVFlat())
        y = randn(MersenneTwister(20), dimension(dto))
        x = transport(dto, y)
        @test all(>(0), x)
        @test isapprox(fd_fwd(dto, Phom, y), logpdf_fwd(dto, y); atol = 1e-5)
        @test pullback(dto, x) ≈ y
    end

    @testset "PushforwardDistribution density / sampling" begin
        rng = MersenneTwister(21)
        # scalar ScaleShift over StdNormal ≡ Normal(μ, σ)
        d = PushforwardDistribution(PT.ScaleShift(1.0, 2.0), StdNormal())
        ref = Normal(1.0, 2.0)
        for x in (-0.7, 0.3, 2.9)
            @test logpdf(d, x) ≈ logpdf(ref, x)
            @test cdf(d, x) ≈ cdf(ref, x)
        end
        @test quantile(d, 0.3) ≈ quantile(ref, 0.3)
        @test mean(d) ≈ 1.0
        @test var(d) ≈ 4.0
        # array ScaleShift over an array base ≡ iid Normal(μᵢ, sᵢ); the cached
        # lognorm split must reproduce the full density (regression: the split once
        # double-counted lognorm(base)).
        μ = [0.5, -1.0, 2.0]
        s = [1.0, 2.0, 0.5]
        da = PushforwardDistribution(PT.ScaleShift(μ, s), StdNormal(3))
        xa = [0.2, 0.4, 1.9]
        @test logpdf(da, xa) ≈ sum(logpdf(Normal(μ[i], s[i]), xa[i]) for i in 1:3)
        @test logpdf(da, xa) ≈ PT.unnormed_logpdf(da, xa) + PT.lognorm(da)
        # AffineTransform over StdNormal(2) ≡ MvNormal(μ, L·Lᵀ)
        L = LowerTriangular([1.0 0.0; 0.4 0.8])
        μ2 = [1.0, -1.0]
        dm = PushforwardDistribution(PT.AffineTransform(μ2, L), StdNormal(2))
        refm = MvNormal(μ2, Matrix(L * L'))
        x2 = [0.3, -0.2]
        @test logpdf(dm, x2) ≈ logpdf(refm, x2)
        @test cov(dm) ≈ L * L'
        @test mean(dm) ≈ μ2
        # sampling sanity + exact transport over the matching space
        xs = rand(rng, da, 20_000)
        @test vec(sum(xs; dims = 2)) ./ 20_000 ≈ μ atol = 0.05
        dto = transport_to(da, StdNormal())
        y = rand(rng, dto)
        @test isapprox(fd_fwd(dto, da, y), logpdf_fwd(dto, y); atol = 1e-5)
        @test pullback(dto, transport(dto, y)) ≈ y
    end

    @testset "data-dependent pushforward map: exp ∘ StdNormal ≡ LogNormal" begin
        rng = MersenneTwister(23)
        dln = PushforwardDistribution(ExpMap(), StdNormal(3))
        x = [0.5, 1.2, 2.0]
        @test logpdf(dln, x) ≈ sum(logpdf.(LogNormal(), x))
        @test @inferred(logpdf(dln, x)) isa Float64
        # the split convention still holds: the per-call Jacobian lives in unnormed
        @test logpdf(dln, x) ≈ PT.unnormed_logpdf(dln, x) + PT.lognorm(dln)
        @test all(>(0), rand(rng, dln))
        # scalar (0-dim) base
        ds = PushforwardDistribution(ExpMap(), StdNormal())
        @test logpdf(ds, 1.3) ≈ logpdf(LogNormal(), 1.3)
        # exact transport: f over the matching-base identity (no cdf/quantile)
        dto = transport_to(dln, StdNormal())
        y = randn(rng, 3)
        @test transport(dto, y) ≈ exp.(y)
        @test isapprox(fd_fwd(dto, dln, y), logpdf_fwd(dto, y); atol = 1e-5)
        @test pullback(dto, transport(dto, y)) ≈ y
        # flat path: TV threads the data-dependent Jacobian per call
        dtf = transport_to(dln, TVFlat())
        yf = randn(rng, 3)
        @test isapprox(fd_fwd(dtf, dln, yf), logpdf_fwd(dtf, yf); atol = 1e-5)
        @test pullback(dtf, transport(dtf, yf)) ≈ yf
    end

    @testset "type stability and eltype" begin
        rng = MersenneTwister(22)
        td = TupleDist((Normal(), Gamma(2.0)))
        xt = @inferred rand(rng, td)
        @test xt isa Tuple{Float64, Float64}

        nd = NamedDist(a = Normal(), b = (Uniform(), Exponential()), c = Gamma(2.0))
        for S in (StdNormal(), StdUniform())
            dto = transport_to(nd, S)
            y = rand(rng, dto)
            x = @inferred transport(dto, y)
            @inferred pullback(dto, x)
            @inferred logpdf_fwd(dto, y)
            @inferred transport_and_logdensity(dto, y)
        end

        # eltype follows the latent reference
        dto32 = transport_to(Normal(1.0f0, 2.0f0), StdNormal{Float32}())
        @test eltype(dto32) == Float32
        @test eltype(rand(rng, dto32)) == Float32
        @test eltype(transport_to(Normal(), StdNormal())) == Float64
    end

    @testset "affine by reuse: LocationScale over Std bases" begin
        rng = MersenneTwister(12)
        # `loc + scale*Std*()` builds a Distributions.AffineDistribution (LocationScale);
        # PT transports it with no PT-specific affine type.
        affine = [
            (1.0 + 2.0 * StdNormal(), Normal(1.0, 2.0)),
            (2.0 + 3.0 * StdUniform(), Uniform(2.0, 5.0)),
            (2.0 * StdExponential(), Exponential(2.0)),
        ]
        for (d, refd) in affine
            @test d isa Distributions.AffineDistribution
            @test logpdf(d, 1.3) ≈ logpdf(refd, 1.3)
        end
        # transport over all three spaces; check internal consistency
        for d in (1.0 + 2.0 * StdNormal(), 2.0 + 3.0 * StdUniform(), 2.0 * StdExponential(),
                  1.0 + 2.0 * StdTDist(5.0), 3.0 * StdInverseGamma(2.5))
            for S in (StdNormal(), StdUniform(), TVFlat())
                dto = transport_to(d, S)
                y = S === TVFlat() ? randn(rng, dimension(dto)) : rand(rng, dto)
                x = transport(dto, y)
                @test isapprox(fd_fwd(dto, d, y), logpdf_fwd(dto, y); atol = 1e-5)
                @test pullback(dto, x) ≈ y
            end
        end
        # equal-dim coincidence over the Std spaces
        d = 1.0 + 2.0 * StdNormal()
        for S in (StdNormal(), StdUniform())
            dto = transport_to(d, S)
            y = rand(rng, dto)
            ref = S === StdNormal() ? StdNormal(1) : StdUniform(1)
            @test logpdf_fwd(dto, y) ≈ logpdf(ref, y)
        end
    end

    @testset "array Std bases + ScaleShift transport are EXACT" begin
        # exact transport ⇒ logpdf_fwd(y) ≡ logpdf(reference, y) (NOT the tautological
        # logpdf_fwd == logpdf(d, x)+lj). Guards the scalar-over-array n·log|s| Jacobian
        # and element-wise (vs matrix-product) scale.
        rng = MersenneTwister(16)
        for D in (StdNormal(3), StdNormal(2, 2), StdExponential(3), StdUniform(4))
            for S in (StdNormal(), StdUniform())
                dto = transport_to(D, S)
                n = dimension(dto)
                y = rand(rng, dto)
                @test isapprox(fd_fwd(dto, D, y), logpdf_fwd(dto, y); atol = 1e-5)   # exactness
                @test pullback(dto, transport(dto, y)) ≈ y
            end
        end
        # ScaleShift Jacobian: scalar s over an n-vector is n·log|s|; array is Σ log|sᵢ|
        f = PT.ScaleShift(zeros(3), 2.0)
        @test last(PT.with_logabsdet_jacobian(f, ones(3))) ≈ 3 * log(2)
        g = PT.ScaleShift(zeros(3), [2.0, 3.0, 4.0])
        @test last(PT.with_logabsdet_jacobian(g, ones(3))) ≈ log(2) + log(3) + log(4)
    end

    @testset "array-parameter StdTDist / StdInverseGamma" begin
        # Regression: the array-parameter path was unconstructible — the field type was set
        # from `eltype(param)` instead of `typeof(param)`, so a vector param failed to
        # convert — and therefore went entirely untested. Exercise construction, the
        # per-element logpdf split, sampling, moments, and the cdf/quantile kernels.
        rng = MersenneTwister(123)

        @testset "construction, shape, eltype" begin
            for d in (StdTDist([5.0, 8.0, 12.0]), StdInverseGamma([2.5, 3.0, 4.0]))
                @test size(d) == (3,)
                @test length(d) == 3
                @test eltype(d) == Float64
            end
            @test size(StdInverseGamma(fill(3.0, 2, 2))) == (2, 2)
            # integer params still yield a float output eltype (not Int)
            @test eltype(StdTDist([5, 8])) == Float64
            @test eltype(StdInverseGamma([2, 3])) == Float64
        end

        @testset "logpdf == Σ per-element reference, and the unnormed/lognorm split" begin
            νs = [5.0, 8.0, 12.0]; αs = [2.5, 3.0, 4.0]
            dt = StdTDist(νs); di = StdInverseGamma(αs)
            x = [0.3, -0.7, 1.1]; z = [0.4, 0.9, 1.7]
            @test logpdf(dt, x) ≈ sum(logpdf(TDist(ν), xi) for (ν, xi) in zip(νs, x))
            @test logpdf(di, z) ≈ sum(logpdf(InverseGamma(α, 1.0), zi) for (α, zi) in zip(αs, z))
            @test logpdf(dt, x) ≈ PT.unnormed_logpdf(dt, x) + PT.lognorm(dt)
            @test logpdf(di, z) ≈ PT.unnormed_logpdf(di, z) + PT.lognorm(di)
        end

        @testset "moments match the per-element reference" begin
            αs = [3.0, 4.0, 5.0]; di = StdInverseGamma(αs)   # α>2 ⇒ finite mean & var
            @test mean(di) ≈ [mean(InverseGamma(α, 1.0)) for α in αs]
            @test var(di) ≈ [var(InverseGamma(α, 1.0)) for α in αs]
            νs = [6.0, 9.0, 12.0]; dt = StdTDist(νs)          # ν>2 ⇒ zero mean, finite var
            @test mean(dt) ≈ zeros(3)
            @test var(dt) ≈ [ν / (ν - 2) for ν in νs]
        end

        @testset "in-place sampler fills distinct per-element draws" begin
            di = StdInverseGamma([3.0, 4.0, 5.0])
            x = zeros(3)
            Distributions._rand!(rng, di, x)
            @test all(>(0), x)
            @test length(unique(x)) == 3        # guards the old single-draw `x .=` bug
        end

        @testset "sampling mean converges to the analytic mean" begin
            n = 100_000
            di = StdInverseGamma([3.0, 4.0, 5.0])
            Xi = rand(rng, di, n)
            @test vec(sum(Xi; dims = 2)) ./ n ≈ mean(di) rtol = 0.05
            dt = StdTDist([8.0, 10.0, 12.0])
            Xt = rand(rng, dt, n)
            @test all(<(0.05), abs.(vec(sum(Xt; dims = 2)) ./ n))
        end

        @testset "element-wise cdf/quantile kernels round-trip" begin
            for α in (2.5, 3.0, 4.0), z in (0.4, 0.9, 1.7)
                @test PT._ig_elem_quantile(α, PT._ig_elem_cdf(α, z)) ≈ z
            end
        end
    end

    @testset "Truncated (Reactant-friendly), matches Distributions.truncated" begin
        rng = MersenneTwister(13)
        cases = [
            (PT.Truncated(Normal(), -1.0, 2.0), truncated(Normal(), -1.0, 2.0)),
            (PT.Truncated(Normal(); lower = 0.0), truncated(Normal(); lower = 0.0)),
            (PT.Truncated(Normal(); upper = 1.0), truncated(Normal(); upper = 1.0)),
            (PT.Truncated(Gamma(2.0, 1.5), 0.5, 4.0), truncated(Gamma(2.0, 1.5), 0.5, 4.0)),
        ]
        for (d, refd) in cases
            for x in (0.3, 0.7)
                insupport(refd, x) || continue
                @test logpdf(d, x) ≈ logpdf(refd, x)
                @test cdf(d, x) ≈ cdf(refd, x)
            end
            for p in (0.1, 0.5, 0.9)
                @test cdf(d, quantile(d, p)) ≈ p
            end
        end
        # transport over all three spaces (Truncated has quantile → generic scalar path)
        d = PT.Truncated(Normal(), -1.0, 2.0)
        for S in (StdNormal(), StdUniform(), TVFlat())
            dto = transport_to(d, S)
            y = S === TVFlat() ? randn(rng, dimension(dto)) : rand(rng, dto)
            x = transport(dto, y)
            xv = x isa AbstractArray ? x[1] : x
            @test -1.0 <= xv <= 2.0
            @test isapprox(fd_fwd(dto, d, y), logpdf_fwd(dto, y); atol = 1e-5)
            @test pullback(dto, x) ≈ y
        end

        # regression: support endpoints intersect the truncation bounds with the BASE
        # support. A one-sided truncation of a bounded base must not widen its support —
        # `Truncated(Exponential(); upper=1)` once mapped ℝ → (-∞, 1) in flat space, letting
        # samplers reach a logpdf = -Inf region (negative flux + a frozen NUTS chain).
        @testset "one-sided truncation keeps the base support" begin
            dexp = PT.Truncated(Exponential(); upper = 1.0)
            refexp = truncated(Exponential(); upper = 1.0)
            @test minimum(dexp) == minimum(refexp) == 0.0
            @test maximum(dexp) == maximum(refexp) == 1.0
            # an explicit out-of-support bound must not widen the support either
            @test minimum(PT.Truncated(Exponential(), -5.0, 1.0)) == 0.0
            # the flat transform respects the support for any latent value
            dto = transport_to(dexp, TVFlat())
            for y in (-30.0, 0.0, 30.0)
                x = transport(dto, [y])
                @test 0.0 <= x <= 1.0
                @test isfinite(logpdf(dexp, max(x, eps())))
            end
            # scalar ScaleShift pushforward bases report their support (and flip with s < 0)
            dscaled = PT.PushforwardDistribution(PT.ScaleShift(0.0, 0.1), StdExponential())
            @test minimum(dscaled) == 0.0 && maximum(dscaled) == Inf
            dneg = PT.PushforwardDistribution(PT.ScaleShift(0.0, -1.0), StdExponential())
            @test minimum(dneg) == -Inf && maximum(dneg) == 0.0
            # ...so a one-sided truncation of a scaled base is bounded in flat space too
            dts = transport_to(PT.Truncated(dscaled; upper = 1.0), TVFlat())
            for y in (-30.0, 0.0, 30.0)
                x = transport(dts, [y])
                @test 0.0 <= x <= 1.0
            end
        end
    end

    @testset "angular: DiagonalVonMises / WrappedUniform" begin
        rng = MersenneTwister(14)
        # logpdf matches a product of Distributions.VonMises
        d = DiagonalVonMises([0.0, 1.0], [2.0, 3.0])
        @test logpdf(d, [0.1, 0.9]) ≈ logpdf(VonMises(0.0, 2.0), 0.1) + logpdf(VonMises(1.0, 3.0), 0.9)
        @test logpdf(DiagonalVonMises(0.5, 2.0), 0.3) ≈ logpdf(VonMises(0.5, 2.0), 0.3)

        # sampler is type-generic: a Float32 (μ, κ) yields Float32 draws (the
        # Distributions sampler is hardcoded to Float64 and would promote).
        @test rand(rng, DiagonalVonMises(0.5f0, 3.0f0)) isa Float32
        @test eltype(rand(rng, DiagonalVonMises(Float32[0.0, 1.0], Float32[2.0, 5.0]))) == Float32
        @test rand(rng, DiagonalVonMises(0.5, 3.0)) isa Float64
        # statistical sanity: mean resultant length matches Distributions.VonMises
        let μ = 0.7, κ = 4.0, N = 100_000
            ours = [rand(rng, DiagonalVonMises(μ, κ)) for _ in 1:N]
            ref = rand(rng, VonMises(μ, κ), N)
            R(x) = hypot(sum(sin, x), sum(cos, x)) / length(x)
            @test isapprox(R(ours), R(ref); atol = 0.01)
            @test isapprox(atan(sum(sin, ours), sum(cos, ours)), μ; atol = 0.02)
        end

        # TVFlat: AngleTransform per angle (2 reals -> 1 angle); section, not bijection
        dto = transport_to(d, TVFlat())
        @test dimension(dto) == 4
        y = randn(rng, 4)
        x = transport(dto, y)
        # dimension-reducing (2 reals -> 1 angle): no square Jacobian, so use TV directly
        x_tv, lj = TV.transform_and_logjac(PT.transport(dto), y)
        @test x isa AbstractVector && length(x) == 2 && all(-π .< x .<= π)
        @test logpdf(dto, y) ≈ logpdf_fwd(dto, y)                 # flat: stop === nothing
        @test logpdf_fwd(dto, y) ≈ logpdf(d, x_tv) + lj
        @test transport(dto, pullback(dto, x)) ≈ x               # section holds one-way

        # StdNormal / StdUniform: no exact transport for a circular variable → error
        @test_throws ArgumentError transport_to(d, StdNormal())
        @test_throws ArgumentError transport_to(d, StdUniform())

        # WrappedUniform flat
        wu = WrappedUniform(2π, 3)
        dtw = transport_to(wu, TVFlat())
        @test dimension(dtw) == 6
        yw = randn(rng, 6)
        xw = transport(dtw, yw)
        xw_tv, ljw = TV.transform_and_logjac(PT.transport(dtw), yw)
        @test length(xw) == 3
        @test logpdf_fwd(dtw, yw) ≈ logpdf(wu, xw_tv) + ljw
    end

    @testset "DeltaDist (clamped parameter, 0-dim)" begin
        rng = MersenneTwister(15)
        # standalone: consumes no latent coordinates, always returns x0
        for S in (StdNormal(), StdUniform(), TVFlat())
            dto = transport_to(DeltaDist(5.0), S)
            @test dimension(dto) == 0
            @test transport(dto, Float64[]) == 5.0
            @test isapprox(logpdf_fwd(dto, Float64[]), 0.0; atol = 1e-12)
            @test length(pullback(dto, 5.0)) == 0
        end
        # inside a NamedDist: clamps that component, excluded from the latent dim
        nd = NamedDist(a = Normal(), b = DeltaDist([1.0, 2.0]), c = Gamma(2.0))
        for S in (StdNormal(), StdUniform())
            dto = transport_to(nd, S)
            @test dimension(dto) == 2
            y = rand(rng, dto)
            x = transport(dto, y)
            @test x.b == [1.0, 2.0]
            @test pullback(dto, x) ≈ y
        end
    end

end

# Reactant tracing tests, guarded: Reactant is a heavy test dep that frequently
# fails to load/precompile on Julia nightly. Skip with a warning rather than
# failing the whole suite when it is unavailable.
const REACTANT_AVAILABLE = try
    @eval using Reactant
    true
catch err
    @warn "Reactant unavailable — skipping Reactant tracing tests" exception = err
    false
end
if REACTANT_AVAILABLE
    include("reactant.jl")
end
