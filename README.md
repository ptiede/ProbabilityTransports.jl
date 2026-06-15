# ProbabilityTransports

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ptiede.github.io/ProbabilityTransports.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ptiede.github.io/ProbabilityTransports.jl/dev/)
[![Build Status](https://github.com/ptiede/ProbabilityTransports.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ptiede/ProbabilityTransports.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ptiede/ProbabilityTransports.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ptiede/ProbabilityTransports.jl)

ProbabilityTransports maps a *latent reference space* onto a target probability
distribution, so a sampler or optimiser can work in the easy space while you push draws
back to the distribution you actually care about. It is the engine behind reparameterised
priors: build the transport once, sample/score in the reference space, transport to the
model space.


## Quick start

```julia
using ProbabilityTransports, Distributions, Random

# Map a standard normal latent space onto a Gamma target.
dto = transport_to(Gamma(2.0, 1.5), StdNormal())

y    = rand(dto)                       # draw in the latent (StdNormal) space
x    = latent_pfwd(dto, y)               # push to the Gamma's space
y′   = latent_pback(dto, x)         # canonical latent point mapping back to x

# Model-facing primitive: x for the likelihood, ℓ for the sampler.
x, ℓ = latent_pfwd_and_logdensity(dto, y)
```

`transport_to` also accepts composites — `NamedDist`, `TupleDist`, `Tuple`/`NamedTuple` of
distributions — and transports each component, threading latent coordinates automatically.

## Choosing a latent space

The second argument to `transport_to` is the latent space:

| Space            | Latent support | Use when |
|------------------|----------------|----------|
| `StdNormal()`    | ℝⁿ (Gaussian)  | You want an unconstrained Gaussian latent space (e.g. HMC/NUTS); the transport is exact and Jacobian-free. |
| `StdUniform()`   | `[0,1]ⁿ`       | You want a unit-hypercube latent space (e.g. nested sampling); the pulled-back density is flat. |
| `TVFlat()`       | ℝⁿ             | You want to keep the *original* distribution but move its support to ℝⁿ via [TransformVariables](https://github.com/tpapp/TransformVariables.jl); the latent density carries the genuine change-of-variables Jacobian. Requires `TransformVariables` to be loaded. |

For `StdNormal()`/`StdUniform()` the transport is an exact measure transport, so the
latent density is the closed-form reference (no Jacobian is ever formed). `TVFlat()` instead
preserves the target distribution and unconstrains its support.

See the [documentation](https://ptiede.github.io/ProbabilityTransports.jl/dev/) for the full
list of supported distributions and the interface for adding your own.
