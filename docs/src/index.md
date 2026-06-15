```@meta
CurrentModule = ProbabilityTransports
```

# ProbabilityTransports

Documentation for [ProbabilityTransports](https://github.com/ptiede/ProbabilityTransports.jl).

ProbabilityTransports maps a *latent reference space* onto a target probability
distribution. Build the transport once with [`transport_to`](@ref); then sample or score in
the convenient latent space and [`latent_pfwd`](@ref) draws to the model space, or recover a
latent point with [`latent_pback`](@ref). For inference the key primitive is
[`latent_pfwd_and_logdensity`](@ref), which returns the transported point together with the
pulled-back log density for the sampler.

## Latent spaces

- `StdNormal()` — an unconstrained Gaussian latent space (ℝⁿ); the transport is an exact
  measure transport, so the latent density is the closed-form standard-normal reference and
  no Jacobian is formed.
- `StdUniform()` — a unit-hypercube latent space (`[0,1]ⁿ`); the pulled-back density is flat.
- `TVFlat()` — keeps the original distribution but moves its support to ℝⁿ via
  [TransformVariables](https://github.com/tpapp/TransformVariables.jl); here the latent
  density carries the genuine change-of-variables Jacobian. Requires `TransformVariables`.

```julia
using ProbabilityTransports, Distributions

dto  = transport_to(Gamma(2.0, 1.5), StdNormal())
y    = rand(dto)
x, ℓ = latent_pfwd_and_logdensity(dto, y)   # x for the likelihood, ℓ for the sampler
```

## API

```@index
```

```@autodocs
Modules = [ProbabilityTransports]
```
